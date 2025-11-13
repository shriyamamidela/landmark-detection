"""
Stage 4 — ATLAS FLOW (SVF Head)
Predict stationary velocity field v(x) from [F, D] and warp canonical atlas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os

from data import Dataset
from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead


# ---------------------------------------------------------
#  Scaling and squaring: exp(v)
# ---------------------------------------------------------
def scaling_and_squaring(svf, steps=4):
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_field(disp, disp)
    return disp


def warp_field(field, disp):
    B, C, H, W = field.shape

    dx = disp[:, 0] / (W / 2)
    dy = disp[:, 1] / (H / 2)
    grid = torch.stack([dx, dy], dim=-1)   # [B,H,W,2]
    grid = torch.clamp(grid, -1, 1)

    return F.grid_sample(field, grid, padding_mode="border", align_corners=False)


def warp_with_disp(img, disp):
    B, _, H, W = img.shape
    dx = disp[:, 0] / (W / 2)
    dy = disp[:, 1] / (H / 2)
    grid = torch.stack([dx, dy], dim=-1)
    grid = torch.clamp(grid, -1, 1)
    return F.grid_sample(img, grid, padding_mode="border", align_corners=False)


# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def jacobian_penalty(disp):
    dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    J = dx[:, :, :-1, :] * dy[:, :, :, :-1]
    return torch.relu(-J).mean()


# ---------------------------------------------------------
# SVF Training Step
# ---------------------------------------------------------
def train_step(backbone, svf_head, images, dt_maps, atlas_edges, optimizer, device):
    images = images.to(device) / 255.0
    dt_maps = dt_maps.to(device)

    # 1. backbone features
    feats = backbone(images)
    F5 = feats["C5"]  # [B,512,h,w]
    B = images.size(0)

    # 2. downsample DT
    D_low = F.interpolate(dt_maps, size=F5.shape[2:], mode="bilinear", align_corners=False)

    # 3. SVF prediction: v(F,D)
    v = svf_head(F5, D_low)  # [B,2,h,w]

    # 4. diffeomorphic displacement
    disp = scaling_and_squaring(v)

    # 5. prepare atlas edge map for each sample in batch
    atlas_edges = atlas_edges.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    atlas_edges = atlas_edges.repeat(B, 1, 1, 1)  # → [B,1,H,W]

    # resize atlas to match SVF resolution
    atlas_resized = F.interpolate(atlas_edges, size=F5.shape[2:], mode='bilinear', align_corners=False)

    # 6. warp atlas edges
    warped_edges = warp_with_disp(atlas_resized, disp)

    # 7. losses
    loss_edge = edge_alignment_loss(warped_edges, 0.5 * torch.ones_like(warped_edges))
    loss_jac = jacobian_penalty(disp)
    loss = loss_edge + 0.1 * loss_jac

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ---------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------
def train_epoch(loader, backbone, svf_head, atlas_edges, optimizer, device):
    svf_head.train()
    total_loss = 0

    for images, _, dt_maps in loader:
        total_loss += train_step(backbone, svf_head, images, dt_maps, atlas_edges, optimizer, device)

    return total_loss / len(loader)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    # Dataset
    train_dataset = Dataset("isbi", "train", shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    print("Loaded training:", len(train_dataset))

    # Atlas
    atlas_edges = torch.from_numpy(np.load("atlas_edge_map.npy")).float()
    print("Loaded atlas edges:", atlas_edges.shape)

    # Models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    svf_head = SVFHead(in_channels=512).to(device)

    optimizer = torch.optim.Adam(svf_head.parameters(), lr=1e-4)

    print("\n🚀 Training SVF Head...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(train_loader, backbone, svf_head, atlas_edges, optimizer, device)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        torch.save(svf_head.state_dict(), f"svf_epoch_{epoch}.pth")

    print("\n✅ Training complete.")
