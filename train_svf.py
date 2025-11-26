import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from network.token_encoder import TokenEncoder

from aug_dataset import AugCephDataset  # <-- YOUR augmented dataset

from losses import (
    huber_landmark_loss,
    jacobian_regularizer,
    inverse_consistency_loss
)


######################################################################
# CUSTOM MRE + SDR CALCULATOR
######################################################################
def compute_mre_sdr(pred, gt, mm_per_pixel):
    """
    pred, gt: numpy arrays (N,19,2)
    Returns:
        mre_mm
        sdr_dict = {2mm: %, 2.5mm: %, 3mm: %, 4mm: %}
    """

    errors_px = np.sqrt(((pred - gt) ** 2).sum(axis=2))   # (N,19)
    errors_mm = errors_px * mm_per_pixel

    mre_mm = errors_mm.mean()

    sdr = {}
    for t in [2.0, 2.5, 3.0, 4.0]:
        sdr[t] = (errors_mm <= t).mean() * 100.0

    return mre_mm, sdr


######################################################################
# WARPING + SVF FUNCTIONS
######################################################################
def warp_with_disp(img, disp):
    B, C, H, W = img.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij"
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    disp_norm = torch.zeros_like(base)
    disp_norm[..., 0] = disp[:, 0] / (W / 2)
    disp_norm[..., 1] = disp[:, 1] / (H / 2)

    grid = base + disp_norm
    return F.grid_sample(img, grid, padding_mode="border", align_corners=False)


def svf_to_disp(v, steps=6):
    disp = v / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_with_disp(disp, disp)
    return disp


def sample_flow_at_points_local(flow, points_px, image_size):
    """
    flow: (B,2,H,W)
    points_px: (B,L,2)
    """
    B, L, _ = points_px.shape
    H, W = image_size

    x_norm = 2 * ((points_px[..., 0] + 0.5) / W) - 1
    y_norm = 2 * ((points_px[..., 1] + 0.5) / H) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B,L,1,2)

    sampled = F.grid_sample(flow, grid,
                            padding_mode="border",
                            align_corners=False)

    sampled = sampled.squeeze(-1)
    sampled = sampled.permute(0, 2, 1)
    return sampled


######################################################################
# TRAINING STEP
######################################################################
def train_step(backbone, svf_head, token_encoder,
               images, gt_landmarks, dt_maps, edge_maps,
               atlas_edges, atlas_lms,
               optimizer, device):

    images = images.float().to(device)
    gt_landmarks = gt_landmarks.float().to(device)
    dt_maps = dt_maps.float().to(device)
    edge_maps = edge_maps.float().to(device)

    B = images.shape[0]

    feats = backbone(images)
    F5 = feats["C5"] if isinstance(feats, dict) else feats

    Hf, Wf = F5.shape[2], F5.shape[3]

    dt_small = F.interpolate(dt_maps, size=(Hf, Wf), mode="bilinear", align_corners=False)
    edge_small = F.interpolate(edge_maps, size=(Hf, Wf), mode="bilinear", align_corners=False)

    v = svf_head(F5, dt_small)

    disp = svf_to_disp(v, steps=6)

    atlas_small = F.interpolate(atlas_edges, size=(Hf, Wf), mode="bilinear", align_corners=False)
    atlas_small = atlas_small.repeat(B, 1, 1, 1)

    warped_edge = warp_with_disp(atlas_small, disp)

    H, W = images.shape[2], images.shape[3]
    disp_full = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)

    disp_lm = sample_flow_at_points_local(disp_full,
                                          atlas_lms.repeat(B, 1, 1),
                                          (H, W))

    pred_lm = atlas_lms.repeat(B, 1, 1) + disp_lm

    L_lm = huber_landmark_loss(pred_lm, gt_landmarks)
    L_edge = F.l1_loss(warped_edge, edge_small)
    L_jac = jacobian_regularizer(disp_full)
    L_inv = inverse_consistency_loss(disp_full, -disp_full)

    L_tok = torch.tensor(0.0, device=device)
    if token_encoder is not None:
        t1 = token_encoder(edge_small.mean(1))
        t2 = token_encoder(warped_edge.mean(1))
        L_tok = F.mse_loss(t1, t2)

    loss = (
        L_lm +
        L_edge +
        0.1 * L_jac +
        0.1 * L_inv +
        1e-4 * L_tok
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), pred_lm.detach(), gt_landmarks.detach()


######################################################################
# EVALUATION LOOP
######################################################################
def evaluate_dataset(loader, backbone, svf_head, atlas_edges, atlas_lms,
                     device, mm_per_pixel):

    all_pred = []
    all_gt = []

    with torch.no_grad():
        for batch in loader:
            images, gt_landmarks, dt_maps, edge_maps,tok = batch

            images = images.to(device)
            dt_maps = dt_maps.to(device)
            edge_maps = edge_maps.to(device)
            gt_landmarks = gt_landmarks.to(device)

            feats = backbone(images)
            F5 = feats["C5"]

            Hf, Wf = F5.shape[2], F5.shape[3]
            dt_small = F.interpolate(dt_maps, size=(Hf, Wf), mode="bilinear", align_corners=False)

            v = svf_head(F5, dt_small)
            disp = svf_to_disp(v, steps=6)

            H, W = images.shape[2], images.shape[3]
            disp_full = F.interpolate(disp, size=(H, W))

            pred_lm = atlas_lms.to(device) + sample_flow_at_points_local(
                disp_full,
                atlas_lms.to(device),
                (H, W)
            )

            all_pred.append(pred_lm.cpu().numpy())
            all_gt.append(gt_landmarks.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    mre, sdr = compute_mre_sdr(all_pred, all_gt, mm_per_pixel)

    return mre, sdr


######################################################################
# MAIN
######################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--atlas-landmarks", type=str, required=True)
    parser.add_argument("--atlas-edge", type=str, required=True)
    parser.add_argument("--test1-dir", type=str, required=True)
    parser.add_argument("--test2-dir", type=str, required=True)
    parser.add_argument("--mm-per-pixel", type=float, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dataset = AugCephDataset("/dgxa_home/se22ucse250/landmark-detection-main/datasets/augmented_ceph")
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    print("Test dataset 1 loaded:", args.test1_dir)
    test1 = AugCephDataset(args.test1_dir)
    test1_loader = DataLoader(test1, batch_size=1, shuffle=False)

    print("Test dataset 2 loaded:", args.test2_dir)
    test2 = AugCephDataset(args.test2_dir)
    test2_loader = DataLoader(test2, batch_size=1, shuffle=False)

    atlas_edges = torch.tensor(np.load(args.atlas_edge)).float().unsqueeze(0).unsqueeze(0).to(device)
    atlas_lms = torch.tensor(np.load(args.atlas_landmarks)).float().unsqueeze(0).to(device)

    backbone = ResNetBackbone("resnet34", pretrained=True).to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    svf_head = SVFHead().to(device)
    token_encoder = None

    optimizer = torch.optim.Adam(svf_head.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        step = 0
        for batch in train_loader:
            images, landmarks, dt_maps, edge_maps, tokens = batch

            loss, pred_lm, gt_lm = train_step(
                backbone, svf_head, token_encoder,
                images, landmarks, dt_maps, edge_maps,
                atlas_edges, atlas_lms,
                optimizer, device
            )

            step += 1
            if step % 200 == 0:
                print(f"Epoch {epoch} step {step}: loss={loss:.4f}")

        ck = {
            "svf": svf_head.state_dict(),
            "epoch": epoch
        }
        torch.save(ck, f"{args.save_dir}/svf_epoch_{epoch}.pth")
        print("✔ Saved checkpoint.")

        print("\n Evaluating Train subset...")
        mre, sdr = evaluate_dataset(
            DataLoader(train_dataset, batch_size=1),
            backbone, svf_head, atlas_edges, atlas_lms, device,
            args.mm_per_pixel
        )
        print("Train MRE:", mre)
        print("Train SDR:", sdr)

        print("\n Evaluating Test1...")
        mre1, sdr1 = evaluate_dataset(
            test1_loader, backbone, svf_head, atlas_edges, atlas_lms,
            device, args.mm_per_pixel
        )
        print("Test1 MRE:", mre1)
        print("Test1 SDR:", sdr1)

        print("\n Evaluating Test2...")
        mre2, sdr2 = evaluate_dataset(
            test2_loader, backbone, svf_head, atlas_edges, atlas_lms,
            device, args.mm_per_pixel
        )
        print("Test2 MRE:", mre2)
        print("Test2 SDR:", sdr2)


if __name__ == "__main__":
    main()
