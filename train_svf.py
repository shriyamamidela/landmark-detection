# train_svf.py — final updated
"""
Train SVF head (Atlas-flow) with token consistency + jacobian + inverse consistency + edge alignment.
Designed to plug into your existing repo structure (data.Dataset, models, losses, preprocessing).
Defaults use paper-like loss weights (good results): L_lm + L_edge + 0.01*L_smooth + 0.1*L_jac + 0.1*L_inv + 1e-4*L_tok
Backbone is frozen by default for stable / fast SVF training; use --finetune-backbone to allow backbone updates.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Dataset
from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead

# losses from your project
from losses import (
    huber_landmark_loss,
    jacobian_regularizer,
    inverse_consistency_loss
)

# token utilities
from preprocessing.topology import (
    extract_arc_tokens_from_edgebank,
    flatten_arc_tokens
)
from network.token_encoder import TokenEncoder

# -------------------------
# Utilities (grid / warping)
# -------------------------
def make_meshgrid(B, H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)


def normalize_disp_for_grid(disp, H, W):
    dx = disp[:, 0] / (W / 2.0)
    dy = disp[:, 1] / (H / 2.0)
    return torch.stack([dx, dy], dim=-1)


def warp_with_disp(img, disp, align_corners=False):
    # img: (B,C,H,W), disp: (B,2,H,W)
    B, C, H, W = img.shape
    base = make_meshgrid(B, H, W, img.device)
    grid = (base + normalize_disp_for_grid(disp, H, W)).clamp(-1, 1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)


def warp_disp(field, disp, align_corners=False):
    return warp_with_disp(field, disp, align_corners)


def svf_to_disp(svf, steps=6, align_corners=False):
    # scaling and squaring exponentiation
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_disp(disp, disp, align_corners)
    return disp


# -------------------------
# Landmark sampler (local sample using grid_sample)
# -------------------------
def sample_flow_at_points_local(flow, points_px, image_size, align_corners=False):
    B, N, _ = points_px.shape
    H, W = image_size

    if align_corners:
        x_norm = 2 * (points_px[..., 0] / (W - 1)) - 1
        y_norm = 2 * (points_px[..., 1] / (H - 1)) - 1
    else:
        x_norm = 2 * ((points_px[..., 0] + 0.5) / W) - 1
        y_norm = 2 * ((points_px[..., 1] + 0.5) / H) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B,N,1,2)

    sampled = F.grid_sample(
        flow, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners
    )  # (B,2,N,1) ALWAYS

    sampled = sampled.squeeze(-1)     # (B,2,N)
    sampled = sampled.permute(0, 2, 1) # (B,N,2)
    return sampled


# -------------------------
# Extra small helper losses
# -------------------------
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def smoothness_loss(v):
    # simple finite-difference smoothness on svf
    return (v[:, :, 1:] - v[:, :, :-1]).abs().mean() + (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()


# -------------------------
# Token extraction (batched)
# -------------------------
def compute_tokens(edge_map_tensor, token_encoder):
    """
    Accepts:
      - edge_map_tensor: torch.Tensor with shape (B,1,H,W) or (B,H,W) float in [0,1] (or other scale)
    Returns:
      - embedding: torch.Tensor (B, token_encoder.out_dim) on same device
    Notes:
      - uses your preprocessing.topology.extract_arc_tokens_from_edgebank which expects HxWx3 uint8 images.
      - conversion to numpy and CPU is done here (the extractor uses scipy/skimage).
    """
    device = edge_map_tensor.device
    arr = edge_map_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]  # (B,H,W)
    B, H, W = arr.shape
    toks = []
    for b in range(B):
        im = arr[b]
        im_u8 = np.clip(im * 255.0, 0, 255).astype(np.uint8)
        im_3ch = np.stack([im_u8, im_u8, im_u8], axis=-1)
        arcs = extract_arc_tokens_from_edgebank(im_3ch)
        flat = flatten_arc_tokens(arcs)   # (243,)
        toks.append(flat)
    toks = torch.tensor(np.stack(toks), dtype=torch.float32, device=device)
    return token_encoder(toks)


# -------------------------
# Single training step
# -------------------------
def train_step(backbone, svf_head, token_encoder,
               images, gt_landmarks, dt_maps,
               atlas_edges, atlas_lms, optimizer, device,
               svf_steps=6, align_corners=False):
    """
    images: (B, C, H, W) [0..255]
    gt_landmarks: (B, L, 2) in pixel coords at resized size (cfg.HEIGHT, cfg.WIDTH)
    dt_maps: (B, 1, H, W)
    atlas_edges: (1,1,h_atlas,w_atlas) float (0..1) or similar
    atlas_lms: (1, L, 2) landmarks in resized coords
    """

    images = images.to(device) / 255.0
    dt_maps = dt_maps.to(device)
    gt_landmarks = gt_landmarks.to(device)

    B, _, H, W = images.shape
    atlas_lms_batch = atlas_lms.to(device).repeat(B, 1, 1)

    # 1) backbone features -> C5
    feats = backbone(images)
    F5 = feats["C5"]

    # 2) resize dt to F5 for conditioning
    target_size = (F5.shape[2], F5.shape[3])
    D_small = F.interpolate(dt_maps, size=target_size, mode="bilinear", align_corners=align_corners)

    # 3) predict svf
    v = svf_head(F5, D_small)  # expected shape (B,2,h,w)

    # 4) exponentiate to displacement
    disp = svf_to_disp(v, steps=svf_steps, align_corners=align_corners)

    # 5) warp atlas edges -> compare with D_small
    atlas_resized = F.interpolate(atlas_edges, size=target_size, mode="bilinear", align_corners=align_corners)
    atlas_rep = atlas_resized.repeat(B, 1, 1, 1)
    warped_edges = warp_with_disp(atlas_rep, disp, align_corners)

    # 6) upsample disp to full resolution and sample landmarks
    disp_full = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=align_corners)
    disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_batch, (H, W), align_corners)

    pred_landmarks = atlas_lms_batch + disp_at_lm
    L_lm = huber_landmark_loss(pred_landmarks, gt_landmarks)

    # other regularizers
    L_edge = edge_alignment_loss(warped_edges, D_small)
    L_smooth = smoothness_loss(v)
    L_jac = jacobian_regularizer(disp_full)

    # inverse consistency (simple symmetric loss)
    disp_bwd = -disp_full
    L_inv = inverse_consistency_loss(disp_full, disp_bwd, align_corners)

    # token consistency: tokens from GT DT (D_small) vs tokens from warped atlas edges
    T_gt = compute_tokens(D_small.detach(), token_encoder)         # (B, out_dim)
    T_pred = compute_tokens(warped_edges.detach(), token_encoder)  # (B, out_dim)
    L_tok = F.mse_loss(T_pred, T_gt)

    # --------------------------
    # LOSS WEIGHTS (paper-like)
    # --------------------------
    loss = (
        L_lm             # 1.0
        + L_edge         # 1.0
        + 0.01 * L_smooth
        + 0.1 * L_jac
        + 0.1 * L_inv
        + 1e-4 * L_tok
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), L_lm.item(), L_edge.item(), L_jac.item(), L_inv.item(), L_tok.item()


# -------------------------
# Epoch loop
# -------------------------
def train_epoch(loader, backbone, svf_head, token_encoder,
                atlas_edges, atlas_lms, optimizer,
                device, svf_steps=6, align_corners=False):
    tot = tot_lm = tot_edge = tot_jac = tot_inv = tot_tok = 0.0
    n = 0
    for images, landmarks, dt_maps in loader:
        L, L_lm, L_edge, L_jac, L_inv, L_tok = train_step(
            backbone, svf_head, token_encoder,
            images, landmarks, dt_maps,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps, align_corners
        )
        tot += L; tot_lm += L_lm; tot_edge += L_edge
        tot_jac += L_jac; tot_inv += L_inv; tot_tok += L_tok
        n += 1

    return (tot / n, tot_lm / n, tot_edge / n, tot_jac / n, tot_inv / n, tot_tok / n)


# -------------------------
# Main entry
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)                # you asked for 30 on GPU
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--svf-steps", type=int, default=6)
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument("--atlas-landmarks", type=str, default="atlas_landmarks_resized.npy")
    parser.add_argument("--atlas-edge", type=str, default="atlas_edge_map.npy")
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/atlas_checkpoints/svf")
    parser.add_argument("--finetune-backbone", action="store_true", help="Allow backbone parameters to be updated (slow but may help).")
    args = parser.parse_args()

    device = cfg.DEVICE

    # dataset + loader
    train_dataset = Dataset("isbi", "train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # atlas files
    atlas_edges = torch.tensor(np.load(args.atlas_edge)).float().unsqueeze(0).unsqueeze(0).to(device)
    atlas_lms = torch.tensor(np.load(args.atlas_landmarks)).float().unsqueeze(0)

    # models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    try:
        in_ch = backbone.layer4[-1].conv2.out_channels
    except Exception:
        in_ch = 512
    svf_head = SVFHead(in_channels=in_ch).to(device)

    token_encoder = TokenEncoder(input_dim=243, hidden=256, out_dim=256).to(device)

    # Freeze backbone by default to stabilize training
    if not args.finetune_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print("📌 Backbone frozen. Use --finetune-backbone to allow backbone updates.")

    # optimizer: train SVF head + token encoder (and optionally backbone if finetune)
    params = list(svf_head.parameters()) + list(token_encoder.parameters())
    if args.finetune_backbone:
        params += [p for p in backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    print("\n🚀 Training SVF Head...\n")

    for epoch in range(1, args.epochs + 1):
        L, L_lm, L_edge, L_jac, L_inv, L_tok = train_epoch(
            train_loader, backbone, svf_head, token_encoder,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps=args.svf_steps, align_corners=args.align_corners
        )

        print(
            f"Epoch {epoch} | Loss={L:.4f} | L_lm={L_lm:.4f} | L_edge={L_edge:.4f} | "
            f"L_jac={L_jac:.4f} | L_inv={L_inv:.4f} | L_tok={L_tok:.6f}"
        )

        torch.save({
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "token_encoder": token_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "loss": L
        }, os.path.join(args.save_dir, f"svf_epoch_{epoch}.pth"))

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
