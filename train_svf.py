"""
Stage 4 — ATLAS FLOW (SVF Head)
Predict stationary velocity field v(x) from [F, D] and warp canonical atlas.
This script:
 - loads dataset (Dataset class must return (image, landmarks, dt_map))
 - extracts backbone features (C5)
 - downsamples DT maps to C5 resolution
 - predicts SVF v via SVFHead(F5, D_small)
 - exponentiates SVF using scaling-and-squaring -> displacement field
 - warps atlas edges per-batch and applies losses (edge alignment + smoothness)
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

# ----------------------------
# Utilities: grid / warping
# ----------------------------
def make_meshgrid(B, H, W, device):
    """Return base sampling grid in normalized coords [-1,1] of shape [B,H,W,2]."""
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)           # [H,W,2] where coords = (x,y)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)    # [B,H,W,2]
    return grid


def svf_to_disp(svf, steps=6):
    """
    Exponentiate a stationary velocity field (svf) using scaling-and-squaring.
    svf: [B,2,H,W] representing **pixel** displacements (dx,dy).
    returns disp: [B,2,H,W] (pixel displacements)
    Implementation:
      - divide svf by 2^steps, then iteratively compose using warp
    """
    B, C, H, W = svf.shape
    device = svf.device

    # start with small displacement
    disp = svf / (2.0 ** steps)   # [B,2,H,W]

    for _ in range(steps):
        # warp disp by itself: disp = disp + warp(disp, disp)
        disp = disp + warp_disp(disp, disp)
    return disp


def normalize_disp_for_grid(disp, H, W):
    """
    Convert pixel displacements [B,2,H,W] -> normalized displacement for grid_sample
    normalized x = dx / (W/2), normalized y = dy / (H/2)
    returns [B,H,W,2] (x,y) normalized values
    """
    B = disp.shape[0]
    dx = disp[:, 0, :, :] / (W / 2.0)
    dy = disp[:, 1, :, :] / (H / 2.0)
    # stack into [B,H,W,2] with order (x, y)
    grid_disp = torch.stack([dx, dy], dim=-1)  # [B,H,W,2]
    return grid_disp


def warp_disp(field, disp):
    """
    Warp a vector field 'field' with displacement 'disp'.
    field: [B,C,H,W]
    disp:  [B,2,H,W] (pixel displacements)
    returns warped field [B,C,H,W]
    """
    B, C, H, W = field.shape
    device = field.device

    # base grid in normalized coordinates
    base = make_meshgrid(B, H, W, device=device)   # [B,H,W,2]
    disp_norm = normalize_disp_for_grid(disp, H, W) # [B,H,W,2]

    samp_grid = base + disp_norm
    samp_grid = samp_grid.clamp(-1.0, 1.0)

    return F.grid_sample(field, samp_grid, mode='bilinear', padding_mode='border', align_corners=False)


def warp_with_disp(img, disp):
    """
    Warp image/tensor img [B,C,H,W] by pixel displacement disp [B,2,H,W].
    Returns warped image [B,C,H,W].
    """
    B, C, H, W = img.shape
    device = img.device

    base = make_meshgrid(B, H, W, device=device)
    disp_norm = normalize_disp_for_grid(disp, H, W)
    samp_grid = base + disp_norm
    samp_grid = samp_grid.clamp(-1.0, 1.0)

    return F.grid_sample(img, samp_grid, mode='bilinear', padding_mode='border', align_corners=False)


# ----------------------------
# Loss helpers
# ----------------------------
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def smoothness_loss(v):
    loss_x = (v[:, :, 1:, :] - v[:, :, :-1, :]).abs().mean()
    loss_y = (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()
    return loss_x + loss_y


# ----------------------------
# Training step
# ----------------------------
def train_step(backbone, svf_head, images, dt_maps, atlas_edges, optimizer, device):
    """
    images: [B,3,H,W]
    dt_maps: [B,1,H_full,W_full]  (full image DT maps)
    atlas_edges: [1,1,H_atlas,W_atlas] (will be resized and repeated to batch)
    """
    images = images.to(device) / 255.0
    dt_maps = dt_maps.to(device)

    B = images.size(0)

    # 1) backbone features (C5)
    feats = backbone(images)      # dict with "C5" expected
    if "C5" not in feats:
        raise RuntimeError("Backbone did not return 'C5' feature. Check backbone implementation.")
    F5 = feats["C5"]             # [B, C5_ch, h, w]

    # 2) downsample DT to F5 spatial resolution
    target_size = (F5.shape[2], F5.shape[3])
    D_small = F.interpolate(dt_maps, size=target_size, mode="bilinear", align_corners=False)  # [B,1,h,w]

    # 3) predict SVF (v) from F5 + D_small
    v = svf_head(F5, D_small)    # expect [B,2,h,w]

    # 4) exponentiate SVF -> displacement (pixel units)
    disp = svf_to_disp(v, steps=6)   # [B,2,h,w]

    # 5) resize atlas edges to match grid and expand to batch
    atlas_resized = F.interpolate(atlas_edges, size=target_size, mode="bilinear", align_corners=False)  # [1,1,h,w]
    atlas_resized = atlas_resized.repeat(B, 1, 1, 1)  # [B,1,h,w]

    # 6) warp atlas edges using predicted displacement
    warped_edges = warp_with_disp(atlas_resized, disp)  # [B,1,h,w]

    # 7) losses
    # Edge alignment: warped atlas edges should align to the DT map edges (we use D_small as proxy)
    edge_loss = edge_alignment_loss(warped_edges, D_small)

    # Smoothness on v
    s_loss = smoothness_loss(v)

    loss = edge_loss + 0.01 * s_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # debug prints
    if torch.rand(1).item() < 0.05:   # print occasionally to avoid flooding logs
        print("DEBUG shapes — v:", tuple(v.shape), "disp:", tuple(disp.shape),
              "warped_edges:", tuple(warped_edges.shape), "D_small:", tuple(D_small.shape))
        print(f"edge_loss: {edge_loss.item():.6f}, smooth_loss: {s_loss.item():.6f}, total: {loss.item():.6f}")

    return loss.item()


# ----------------------------
# Epoch loop
# ----------------------------
def train_epoch(loader, backbone, svf_head, atlas_edges, optimizer, device):
    svf_head.train()
    total_loss = 0.0
    for batch in loader:
        # Dataset should return image, landmarks, dt_map
        if len(batch) == 3:
            images, _, dt_maps = batch
        else:
            images, dt_maps = batch[0], batch[1]   # try fallbacks
        total_loss += train_step(backbone, svf_head, images, dt_maps, atlas_edges, optimizer, device)
    return total_loss / len(loader)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--svf-steps", type=int, default=6, help="scaling-and-squaring iterations (power of 2)")
    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    # Dataset
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("Loaded training:", len(train_dataset))

    # Atlas edges (saved earlier)
    atlas_edge_path = "atlas_edge_map.npy"
    if not os.path.exists(atlas_edge_path):
        raise FileNotFoundError(f"Atlas edge map not found at: {atlas_edge_path}. Run tools/make_atlas_edge.py first.")
    atlas_edges_np = np.load(atlas_edge_path)   # (H_atlas, W_atlas)
    atlas_edges = torch.from_numpy(atlas_edges_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    print("Loaded atlas edges:", atlas_edges.shape)

    # Models
    backbone = ResNetBackbone(name="resnet34", pretrained=True, fuse_edges=False).to(device)
    svf_head = SVFHead(in_channels=backbone.layer4[-1].conv2.out_channels if hasattr(backbone, 'layer4') else 512).to(device)

    optimizer = torch.optim.Adam(svf_head.parameters(), lr=args.lr)

    save_dir = "checkpoints_svf"
    os.makedirs(save_dir, exist_ok=True)

    print("\n🚀 Training SVF Head...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(train_loader, backbone, svf_head, atlas_edges, optimizer, device)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        ckpt = os.path.join(save_dir, f"svf_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "svf_state_dict": svf_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, ckpt)

    print("\n✅ Training complete.")
