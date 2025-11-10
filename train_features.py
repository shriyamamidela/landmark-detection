"""
Train DT-Aware Feature Extraction (HRNet + Semantic Fusion Block)
Guided by Distance Transform regression instead of self-supervised loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.backbone import Backbone
from network.semantic_fusion_block import SemanticFusionBlock
from data import Dataset
from config import cfg
from preprocessing.utils import generate_distance_transform
import numpy as np
import os
import argparse


# ---------------------------------------------------------------------- #
# Model Definition
# ---------------------------------------------------------------------- #
class DTAwareFeatureModel(nn.Module):
    """HRNet backbone + SFB + Distance-Transform prediction head"""
    def __init__(self, backbone_name, backbone_weights=None):
        super(DTAwareFeatureModel, self).__init__()

        self.backbone = Backbone(
            name=backbone_name,
            pretrained=False,
            weights_root_path=backbone_weights
        )
        self.sfb = None
        self.initialized = False

        # ✅ DT regression head from high-resolution branch
        # HRNet-W48 → first branch (R4) = 48 channels
        self.dt_head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        _ = self.backbone(x)
        feat_dict = self.backbone.get_feature_dict() or {}

        R4 = feat_dict.get("R4")
        R8 = feat_dict.get("R8")
        R16 = feat_dict.get("R16")
        R32 = feat_dict.get("R32")

        # Initialize SFB dynamically
        if not self.initialized and R8 is not None:
            in_channels = (R8.shape[1], R16.shape[1], R32.shape[1])
            self.sfb = SemanticFusionBlock(num_filters=256, in_channels=in_channels).to(x.device)
            self.initialized = True
            print(f"Initialized SFB with in_channels={in_channels}")

        # Pass through SFB
        if self.sfb is not None:
            P3, P4, P5 = self.sfb([R8, R16, R32])
        else:
            P3, P4, P5 = None, None, None

        # Predict distance transform map
        dt_pred = self.dt_head(R4)
        return dt_pred, {"R4": R4, "P3": P3, "P4": P4, "P5": P5}


# ---------------------------------------------------------------------- #
# Loss Function
# ---------------------------------------------------------------------- #
def distance_transform_loss(pred, target):
    """L1 loss between predicted and ground-truth DT maps"""
    return nn.functional.l1_loss(pred, target)


# ---------------------------------------------------------------------- #
# Utility to generate DT maps from landmarks
# ---------------------------------------------------------------------- #
def prepare_dt_maps(landmarks_batch):
    """Generate DT maps (numpy → torch) from landmark tensors"""
    dt_list = []
    lm_np = landmarks_batch.detach().cpu().numpy()
    if lm_np.ndim == 2:
        lm_np = np.expand_dims(lm_np, 0)
    for lm in lm_np:
        dt = generate_distance_transform(lm, (cfg.HEIGHT, cfg.WIDTH))
        dt = np.array(dt, dtype=np.float32)
        dt_list.append(torch.from_numpy(dt).unsqueeze(0))  # (1, H, W)
    return torch.stack(dt_list)  # (B, 1, H, W)


# ---------------------------------------------------------------------- #
# Training Step (with dynamic resizing fix)
# ---------------------------------------------------------------------- #
def train_step(images, dt_targets, model, optimizer, device):
    images = images.to(device)
    dt_targets = dt_targets.to(device)

    optimizer.zero_grad()
    dt_pred, _ = model(images)

    # ✅ Resize GT DT maps to match HRNet’s low-res output
    if dt_targets.shape[2:] != dt_pred.shape[2:]:
        dt_targets = nn.functional.interpolate(
            dt_targets, size=dt_pred.shape[2:], mode='bilinear', align_corners=False
        )

    loss = distance_transform_loss(dt_pred, dt_targets)
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------- #
# Epoch Training
# ---------------------------------------------------------------------- #
def train_epoch(train_loader, model, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            images, _, dt_maps = batch
        else:
            images, landmarks = batch
            dt_maps = prepare_dt_maps(landmarks)

        images = images / 255.0
        loss = train_step(images, dt_maps, model, optimizer, device)
        total_loss += loss

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"\rEpoch {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {total_loss / (batch_idx+1):.4f}", end="")
    print()
    return total_loss / len(train_loader)


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #
def validate(val_loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, _, dt_maps = batch
            else:
                images, landmarks = batch
                dt_maps = prepare_dt_maps(landmarks)

            images = images.to(device) / 255.0
            dt_maps = dt_maps.to(device)

            dt_pred, _ = model(images)

            # ✅ Resize GT DT maps for consistency
            if dt_maps.shape[2:] != dt_pred.shape[2:]:
                dt_maps = nn.functional.interpolate(
                    dt_maps, size=dt_pred.shape[2:], mode='bilinear', align_corners=False
                )

            loss = distance_transform_loss(dt_pred, dt_maps)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# ---------------------------------------------------------------------- #
# Training Loop
# ---------------------------------------------------------------------- #
def train(train_loader, val_loader, model, optimizer, scheduler, device, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nStarting DT-aware feature training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)

        train_loss = train_epoch(train_loader, model, optimizer, device, epoch)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = validate(val_loader, model, device) if val_loader else float('inf')
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(save_dir, "best_dt_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        if scheduler:
            scheduler.step()

    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------- #
# Main Entry Point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DT-aware HRNet backbone")
    parser.add_argument("--backbone", type=str, default="hrnet_w48")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default="pretrained_weights/hrnet_w48_imagenet.pth")
    parser.add_argument("--save-dir", type=str, default="checkpoints_dt")
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"Device: {device}")

    print("\nLoading datasets...")
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataset = Dataset(name="isbi", mode="valid", batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}")

    print(f"\nCreating model ({args.backbone})...")
    model = DTAwareFeatureModel(backbone_name=args.backbone, backbone_weights=args.pretrained).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train(train_loader, val_loader, model, optimizer, scheduler, device, args.epochs, args.save_dir)
