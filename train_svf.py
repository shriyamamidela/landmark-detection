"""
Stage 4 — ATLAS FLOW (SVF Head)
Predict stationary velocity field v(x) from [F, D] and warp canonical atlas.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time

from data import Dataset
from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead

# ---- IMPORT ALL LOSSES ----
from losses import (
    huber_landmark_loss,
    jacobian_regularizer,
    inverse_consistency_loss,
    advanced_edge_loss        # ★ NEW ADVANCED EDGE LOSS ★
)

# ============================================================
#  Utility functions
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
    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode='border',
                         align_corners=align_corners)


def warp_disp(field, disp, align_corners=False):
    return warp_with_disp(field, disp, align_corners)


def svf_to_disp(svf, steps=6, align_corners=False):
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_disp(disp, disp, align_corners)
    return disp

# ============================================================
#  Landmark sampler
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

    sampled = F.grid_sample(flow, grid, mode='bilinear',
                            padding_mode='border',
                            align_corners=align_corners)

    return sampled.squeeze(-1).permute(0, 2, 1)


# ============================================================
#  Smoothness only
# ============================================================
def smoothness_loss(v):
    return (v[:, :, 1:] - v[:, :, :-1]).abs().mean() + \
           (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()


# ============================================================
#  TRAIN STEP
# ============================================================
def train_step(backbone, svf_head, images, gt_landmarks, dt_maps,
               atlas_edges, atlas_lms, optimizer, device,
               svf_steps=6, align_corners=False):

    images = images.to(device) / 255.0
    dt_maps = dt_maps.to(device)
    gt_landmarks = gt_landmarks.to(device)

    B, _, H, W = images.shape
    atlas_lms_batch = atlas_lms.to(device).repeat(B, 1, 1)

    # Backbone → C5
    feats = backbone(images)
    F5 = feats["C5"]

    # Resize DT map
    target_size = (F5.shape[2], F5.shape[3])
    D_small = F.interpolate(dt_maps, size=target_size, mode="bilinear", align_corners=False)

    # Predict SVF
    v = svf_head(F5, D_small)

    # Exponentiate SVF
    disp = svf_to_disp(v, steps=svf_steps, align_corners=align_corners)

    # Warp atlas edges
    atlas_resized = F.interpolate(atlas_edges, size=target_size, mode="bilinear", align_corners=False)
    warped_edges = warp_with_disp(atlas_resized.repeat(B, 1, 1, 1), disp, align_corners)

    # Full res displacement
    disp_full = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)

    # Landmark loss
    disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_batch, (H, W), align_corners)
    pred_landmarks = atlas_lms_batch + disp_at_lm
    L_lm = huber_landmark_loss(pred_landmarks, gt_landmarks)

    # ---- EDGE LOSS (ADVANCED VERSION) ----
    L_edge = advanced_edge_loss(warped_edges, D_small)

    # Smoothness
    L_smooth = smoothness_loss(v)

    # Jacobian regularizer
    L_jac = jacobian_regularizer(disp_full, neg_weight=10.0, dev_weight=1.0)

    # Inverse Consistency
    L_inv = inverse_consistency_loss(disp_full, -disp_full, align_corners)

    # ---- TOTAL LOSS ----
    loss = (
        L_lm
        + L_edge
        + 0.01 * L_smooth
        + 0.1 * L_jac
        + 0.1 * L_inv
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), L_lm.item(), L_edge.item(), L_jac.item(), L_inv.item()


# ============================================================
#  EPOCH LOOP
# ============================================================
def train_epoch(loader, backbone, svf_head, atlas_edges, atlas_lms,
                optimizer, device, svf_steps=6, align_corners=False):

    total = total_lm = total_edge = total_jac = total_inv = 0
    n = 0

    for images, landmarks, dt_maps in loader:
        L, L_lm, L_edge, L_jac, L_inv = train_step(
            backbone, svf_head,
            images, landmarks, dt_maps,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps, align_corners
        )
        total += L
        total_lm += L_lm
        total_edge += L_edge
        total_jac += L_jac
        total_inv += L_inv
        n += 1

    return (
        total / n,
        total_lm / n,
        total_edge / n,
        total_jac / n,
        total_inv / n
    )


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--svf-steps", type=int, default=6)
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument("--atlas-landmarks", type=str, default="atlas_landmarks_clean.npy")
    parser.add_argument("--atlas-edge", type=str, default="atlas_edge_map.npy")

    args = parser.parse_args()
    device = cfg.DEVICE

    # Dataset
    train_dataset = Dataset("isbi", "train",
                            batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    # Atlas data
    atlas_edges = torch.tensor(np.load(args.atlas_edge)).float().unsqueeze(0).unsqueeze(0).to(device)
    atlas_lms = torch.tensor(np.load(args.atlas_landmarks)).float().unsqueeze(0)

    # Models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    try:
        in_ch = backbone.layer4[-1].conv2.out_channels
    except:
        in_ch = 512

    svf_head = SVFHead(in_channels=in_ch).to(device)
    optimizer = torch.optim.Adam(svf_head.parameters(), lr=args.lr)

    # Save directory
    save_dir = "/content/drive/MyDrive/atlas_checkpoints/svf"
    os.makedirs(save_dir, exist_ok=True)

    print("\n🚀 Training SVF Head...\n")
    for epoch in range(1, args.epochs + 1):
        L, L_lm, L_edge, L_jac, L_inv = train_epoch(
            train_loader, backbone, svf_head,
            atlas_edges, atlas_lms,
            optimizer, device,
            args.svf_steps, args.align_corners
        )

        print(
            f"Epoch {epoch} | Loss={L:.4f} | "
            f"L_lm={L_lm:.4f} | L_edge={L_edge:.4f} | "
            f"L_jac={L_jac:.4f} | L_inv={L_inv:.4f}"
        )

        torch.save({
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "opt": optimizer.state_dict(),
            "loss": L
        }, f"{save_dir}/svf_epoch_{epoch}.pth")

    print("\n✅ Training complete.")
