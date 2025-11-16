# ===============================
# train_svf.py  (FINAL VERSION)
# ===============================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os

from data import Dataset
from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead

# ---- Losses ----
from losses import (
    huber_landmark_loss,
    jacobian_regularizer,
    inverse_consistency_loss
)

# ---- Token extraction utilities ----
from preprocessing.topology import (
    extract_arc_tokens_from_edgebank,
    flatten_arc_tokens
)

from network.token_encoder import TokenEncoder


# ============================================================
# Helper: create grid
# ============================================================
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
    B, C, H, W = img.shape
    base = make_meshgrid(B, H, W, img.device)
    grid = (base + normalize_disp_for_grid(disp, H, W)).clamp(-1, 1)
    return F.grid_sample(
        img, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )


def warp_disp(field, disp, ac=False):
    return warp_with_disp(field, disp, ac)


def svf_to_disp(svf, steps=6, ac=False):
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_disp(disp, disp, ac)
    return disp


# ============================================================
# Landmark sampler
# ============================================================
def sample_flow_at_points_local(flow, points_px, image_size, align_corners=False):
    B, N, _ = points_px.shape
    H, W = image_size

    if align_corners:
        x_norm = 2 * (points_px[..., 0] / (W - 1)) - 1
        y_norm = 2 * (points_px[..., 1] / (H - 1)) - 1
    else:
        x_norm = 2 * ((points_px[..., 0] + 0.5) / W) - 1
        y_norm = 2 * ((points_px[..., 1] + 0.5) / H) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)

    sampled = F.grid_sample(
        flow, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )

    return sampled.squeeze(-1).permute(0, 2, 1)


# ============================================================
# Extra losses
# ============================================================
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def smoothness_loss(v):
    return (v[:, :, 1:] - v[:, :, :-1]).abs().mean() + \
           (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()


# ============================================================
# NEW: Token extraction from an edge map
# ============================================================
def compute_tokens(edge_map_tensor, token_encoder):
    """
    Input:
        edge_map_tensor : (B,1,H,W)
    Output:
        token embedding: (B,256)
    """
    edge_np = edge_map_tensor.detach().cpu().numpy()  # (B,1,H,W)
    B = edge_np.shape[0]

    all_tokens = []
    for b in range(B):
        # Convert to (H,W,1)
        edge_3ch = np.repeat(edge_np[b, 0:1].transpose(1, 2, 0), 3, axis=2)

        arcs = extract_arc_tokens_from_edgebank(edge_3ch)
        flat = flatten_arc_tokens(arcs)   # (243,)

        all_tokens.append(flat)

    tokens = torch.tensor(np.stack(all_tokens), dtype=torch.float32)
    return token_encoder(tokens.to(edge_map_tensor.device))  # (B,256)


# ============================================================
# Training step
# ============================================================
def train_step(backbone, svf_head, token_encoder,
               images, gt_landmarks, dt_maps,
               atlas_edges, atlas_lms, optimizer, device,
               svf_steps=6, ac=False):

    images = images.to(device) / 255.0
    dt_maps = dt_maps.to(device)
    gt_landmarks = gt_landmarks.to(device)

    B, _, H, W = images.shape
    atlas_lms_batch = atlas_lms.to(device).repeat(B, 1, 1)

    feats = backbone(images)
    F5 = feats["C5"]

    target_size = (F5.shape[2], F5.shape[3])
    D_small = F.interpolate(dt_maps, size=target_size, mode="bilinear")

    v = svf_head(F5, D_small)
    disp = svf_to_disp(v, svf_steps, ac)

    atlas_resized = F.interpolate(atlas_edges, size=target_size)
    warped_edges = warp_with_disp(atlas_resized.repeat(B, 1, 1, 1), disp, ac)

    disp_full = F.interpolate(disp, size=(H, W), mode="bilinear")
    disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_batch, (H, W), ac)

    pred_landmarks = atlas_lms_batch + disp_at_lm
    L_lm = huber_landmark_loss(pred_landmarks, gt_landmarks)

    L_edge = edge_alignment_loss(warped_edges, D_small)
    L_smooth = smoothness_loss(v)
    L_jac = jacobian_regularizer(disp_full)

    # Inverse consistency
    disp_bwd = -disp_full
    L_inv = inverse_consistency_loss(disp_full, disp_bwd, ac)

    # ---- NEW: Token consistency ----
    T_gt   = compute_tokens(dt_maps, token_encoder)        # (B,256)
    T_pred = compute_tokens(warped_edges, token_encoder)   # (B,256)
    L_tok  = F.mse_loss(T_pred, T_gt)

    loss = (
        L_lm
      + L_edge
      + 0.01 * L_smooth
      + 0.1 * L_jac
      + 0.1 * L_inv
      + 0.0001 * L_tok    # NEW LOSS
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), L_lm.item(), L_edge.item(), L_jac.item(), L_inv.item(), L_tok.item()


# ============================================================
# Epoch loop
# ============================================================
def train_epoch(loader, backbone, svf_head, token_encoder,
                atlas_edges, atlas_lms, optimizer,
                device, svf_steps=6, ac=False):

    tot = tot_lm = tot_edge = tot_jac = tot_inv = tot_tok = 0
    n = 0

    for images, landmarks, dt_maps in loader:
        L, L_lm, L_edge, L_jac, L_inv, L_tok = train_step(
            backbone, svf_head, token_encoder,
            images, landmarks, dt_maps,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps, ac
        )
        tot += L; tot_lm += L_lm; tot_edge += L_edge
        tot_jac += L_jac; tot_inv += L_inv; tot_tok += L_tok
        n += 1

    return (tot/n, tot_lm/n, tot_edge/n, tot_jac/n, tot_inv/n, tot_tok/n)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--svf-steps", type=int, default=6)
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument("--atlas-landmarks", type=str, default="atlas_landmarks_resized.npy")
    parser.add_argument("--atlas-edge", type=str, default="atlas_edge_map.npy")
    args = parser.parse_args()

    device = cfg.DEVICE

    train_dataset = Dataset("isbi", "train",
                            batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    atlas_edges = torch.tensor(np.load(args.atlas_edge)).float().unsqueeze(0).unsqueeze(0).to(device)
    atlas_lms   = torch.tensor(np.load(args.atlas_landmarks)).float().unsqueeze(0)

    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    in_ch = getattr(backbone.layer4[-1].conv2, "out_channels", 512)

    svf_head = SVFHead(in_channels=in_ch).to(device)

    # ---- NEW: Token Encoder ----
    token_encoder = TokenEncoder(input_dim=243, hidden=256, out_dim=256).to(device)

    optimizer = torch.optim.Adam(svf_head.parameters(), lr=args.lr)

    save_dir = "/content/drive/MyDrive/atlas_checkpoints/svf"
    os.makedirs(save_dir, exist_ok=True)

    print("\n🚀 Training SVF Head...\n")
    for epoch in range(1, args.epochs + 1):
        L, L_lm, L_edge, L_jac, L_inv, L_tok = train_epoch(
            train_loader, backbone, svf_head, token_encoder,
            atlas_edges, atlas_lms,
            optimizer, device,
            args.svf_steps, args.align_corners
        )

        print(
            f"Epoch {epoch} | "
            f"Loss={L:.4f} | L_lm={L_lm:.4f} | L_edge={L_edge:.4f} | "
            f"L_jac={L_jac:.4f} | L_inv={L_inv:.4f} | L_tok={L_tok:.4f}"
        )

        torch.save({
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "opt": optimizer.state_dict(),
            "loss": L
        }, f"{save_dir}/svf_epoch_{epoch}.pth")

    print("\n✅ Training complete.")
