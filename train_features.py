"""
Stage 1 of ATLAS Pipeline — ResNet + Edge Bank feature extractor.
Trains ONLY on preprocessed & augmented images from augmented_ceph/.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from config import cfg

import numpy as np
import os
import argparse
from datetime import datetime

# 🟨 NEW IMPORT
from aug_dataset import AugCephDataset   # <— you will create this file


# ---------------------------------------------------------------------- #
# Model Definition
# ---------------------------------------------------------------------- #
class ResNetEdgeFusionModel(nn.Module):
    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(backbone_name, pretrained=pretrained)

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fusion_head = nn.Sequential(
            nn.Conv2d(512 + 32, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x, edge_bank):
        feats = self.backbone(x)
        c5 = feats["C5"]

        e = self.edge_encoder(edge_bank)
        e_resized = nn.functional.interpolate(e, size=c5.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([c5, e_resized], dim=1)
        dt_pred = self.fusion_head(fused)
        return dt_pred


# ---------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------- #
def dt_loss(pred, target):
    return nn.functional.l1_loss(pred, target)


# ---------------------------------------------------------------------- #
# Training Step
# ---------------------------------------------------------------------- #
def train_step(images, dt_targets, model, optimizer, device):
    images = images.to(device)
    dt_targets = dt_targets.to(device)

    # edge bank generation
    edge_banks = []
    for img in images:
        np_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        edge = generate_edge_bank(np_img)
        edge_tensor = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        edge_banks.append(edge_tensor)
    edge_banks = torch.cat(edge_banks, dim=0).to(device)

    optimizer.zero_grad()
    dt_pred = model(images, edge_banks)

    if dt_pred.shape[2:] != dt_targets.shape[2:]:
        dt_targets = nn.functional.interpolate(dt_targets, size=dt_pred.shape[2:], mode="bilinear", align_corners=False)

    loss = dt_loss(dt_pred, dt_targets)
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------- #
# Training Loop
# ---------------------------------------------------------------------- #
def train(train_loader, model, optimizer, device, epochs=10, save_dir="checkpoints_resnet_edge"):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, dt_maps) in enumerate(train_loader):
            loss = train_step(images, dt_maps, model, optimizer, device)
            total_loss += loss

            if (batch_idx + 1) % 10 == 0:
                print(f"\rEpoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {total_loss/(batch_idx+1):.4f}", end="")

        epoch_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} Avg Loss: {epoch_loss:.4f}")

        ckpt = os.path.join(save_dir, f"resnet_edge_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_resnet_edge.pth"))
            print(f"✓ Saved best model (loss={best_loss:.4f})")

    print("\nTraining complete.")


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="resnet34")
    args = parser.parse_args()

    device = cfg.DEVICE

    print("\n🟨 Loading Augmented Dataset…")
    train_dataset = AugCephDataset(root="/content/drive/MyDrive/datasets/augmented_ceph")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Total samples: {len(train_dataset)}")

    print("\n🟨 Creating Model…")
    model = ResNetEdgeFusionModel(backbone_name=args.backbone, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(train_loader, model, optimizer, device, epochs=args.epochs)
