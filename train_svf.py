"""
Stage 4 — ATLAS FLOW (SVF Head)
Predict stationary velocity field v(x) from [F, D] and warp canonical atlas.

This script:
 - loads dataset (Dataset class must return (image, landmarks, dt_map))
 - extracts backbone features (C5)
 - downsamples DT maps to C5 resolution
 - predicts SVF v via SVFHead(F5, D_small)
 - exponentiates SVF using scaling-and-squaring -> displacement field
 - warps atlas edges per-batch and applies losses:
        * Landmark Huber Loss (L_lm)
        * Edge alignment loss (L_edge)
        * Smoothness loss on v (L_smooth)
"""

import torch
import torch.nn as nn
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
from losses import huber_landmark_loss


# ============================================================
#  Utility functions
# ============================================================
def make_meshgrid(B, H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)


def normalize_disp_for_grid(disp, H, W):
    dx = disp[:, 0, :, :] / (W / 2.0)
    dy = disp[:, 1, :, :] / (H / 2.0)
    return torch.stack([dx, dy], dim=-1)


def warp_disp(field, disp, align_corners=False):
    B, C, H, W = field.shape
    device = field.device

    base = make_meshgrid(B, H, W, device)
    disp_norm = normalize_disp_for_grid(disp, H, W)
    grid = (base + disp_norm).clamp(-1.0, 1.0)

    return F.grid_sample(field, grid, mode='bilinear',
                         padding_mode='border',
                         align_corners=align_corners)


def warp_with_disp(img, disp, align_corners=False):
    B, C, H, W = img.shape
    device = img.device

    base = make_meshgrid(B, H, W, device)
    disp_norm = normalize_disp_for_grid(disp, H, W)
    grid = (base + disp_norm).clamp(-1.0, 1.0)

    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode='border',
                         align_corners=align_corners)


def svf_to_disp(svf, steps=6, align_corners=False):
    disp = svf / (2.0 ** steps)
    for _ in range(steps):
        disp = disp + warp_disp(disp, disp, align_corners)
    return disp


# ============================================================
#  Landmark sampler — consistent with align_corners=False
# ============================================================
def sample_flow_at_points_local(flow, points_px, image_size, align_corners=False):
    B, N, _ = points_px.shape
    H, W = image_size

    x = points_px[..., 0]
    y = points_px[..., 1]

    if align_corners:
        x_norm = 2.0 * (x / (W - 1.0)) - 1.0
        y_norm = 2.0 * (y / (H - 1.0)) - 1.0
    else:
        x_norm = 2.0 * ((x + 0.5) / W) - 1.0
        y_norm = 2.0 * ((y + 0.5) / H) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)

    sampled = F.grid_sample(flow, grid, mode='bilinear',
                            padding_mode='border',
                            align_corners=align_corners)

    return sampled.squeeze(-1).permute(0, 2, 1)


# ============================================================
#  Loss functions
# ============================================================
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def smoothness_loss(v):
    dx = (v[:, :, 1:, :] - v[:, :, :-1, :]).abs().mean()
    dy = (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()
    return dx + dy


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

    # 1) Backbone
    feats = backbone(images)
    F5 = feats["C5"]

    # 2) DT at F5 resolution
    target_size = (F5.shape[2], F5.shape[3])
    D_small = F.interpolate(dt_maps, size=target_size,
                            mode="bilinear", align_corners=False)

    # 3) Predict SVF
    v = svf_head(F5, D_small)

    # 4) Exponentiate SVF
    disp = svf_to_disp(v, steps=svf_steps, align_corners=align_corners)

    # 5) Warp atlas edges
    atlas_resized = F.interpolate(atlas_edges, size=target_size,
                                  mode="bilinear", align_corners=False)
    atlas_resized = atlas_resized.repeat(B, 1, 1, 1)
    warped_edges = warp_with_disp(atlas_resized, disp, align_corners)

    # 6) Landmark loss
    disp_full = F.interpolate(disp, size=(H, W),
                              mode="bilinear", align_corners=False)

    disp_at_lm = sample_flow_at_points_local(
        disp_full, atlas_lms_batch, (H, W), align_corners)

    pred_landmarks = atlas_lms_batch + disp_at_lm

    L_lm = huber_landmark_loss(pred_landmarks, gt_landmarks, delta=1.0)
    L_edge = edge_alignment_loss(warped_edges, D_small)
    L_smooth = smoothness_loss(v)

    loss = L_lm + L_edge + 0.01 * L_smooth

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), L_lm.item(), L_edge.item()


# ============================================================
#  EPOCH LOOP
# ============================================================
def train_epoch(loader, backbone, svf_head, atlas_edges, atlas_lms,
                optimizer, device, svf_steps=6, align_corners=False):

    svf_head.train()
    total, total_lm, total_edge = 0, 0, 0
    n = 0

    for batch in loader:
        images, landmarks, dt_maps = batch
        loss, lm, edge = train_step(
            backbone, svf_head,
            images, landmarks, dt_maps,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps, align_corners
        )
        total += loss
        total_lm += lm
        total_edge += edge
        n += 1

    return total / n, total_lm / n, total_edge / n


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

    # CLEAN FILE BY DEFAULT
    parser.add_argument("--atlas-landmarks", type=str, default="atlas_landmarks_clean.npy")
    parser.add_argument("--atlas-edge", type=str, default="atlas_edge_map.npy")

    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    # Load dataset
    train_dataset = Dataset(name="isbi", mode="train",
                        batch_size=args.batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True, num_workers=0)

    print("Loaded training len:", len(train_dataset))

    # Load Atlas edges
    atlas_edges_np = np.load(args.atlas_edge)
    atlas_edges = torch.from_numpy(atlas_edges_np).float().unsqueeze(0).unsqueeze(0).to(device)
    print("Loaded atlas edges:", atlas_edges.shape)

    # Load CLEAN atlas LM file
    atlas_lms_np = np.load(args.atlas_landmarks)       # clean (19,2)
    atlas_lms = torch.tensor(atlas_lms_np).float().unsqueeze(0)
    print("Loaded atlas landmarks:", atlas_lms.shape)

    # Models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    try:
        in_ch = backbone.layer4[-1].conv2.out_channels
    except:
        in_ch = 512

    svf_head = SVFHead(in_channels=in_ch).to(device)
    optimizer = torch.optim.Adam(svf_head.parameters(), lr=args.lr)

    os.makedirs("checkpoints_svf", exist_ok=True)

    print("\n🚀 Training SVF Head...\n")
    for epoch in range(1, args.epochs + 1):
        loss, L_lm, L_edge = train_epoch(
            train_loader, backbone, svf_head,
            atlas_edges, atlas_lms,
            optimizer, device,
            args.svf_steps, args.align_corners
        )

        print(f"Epoch {epoch} | Loss={loss:.4f} | L_lm={L_lm:.4f} | L_edge={L_edge:.4f}")

        torch.save({
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "opt": optimizer.state_dict(),
            "loss": loss
        }, f"checkpoints_svf/svf_epoch_{epoch}.pth")

    print("\n✅ Training complete.")
