"""
Stage 1 of ATLAS Pipeline — ResNet + Edge Bank feature extractor.
Learns to predict Distance-Transform maps (D) from fused (F + E) features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from data import Dataset
from config import cfg
import numpy as np
import os
import cv2
import argparse
from datetime import datetime


# ---------------------------------------------------------------------- #
# Model Definition
# ---------------------------------------------------------------------- #
class ResNetEdgeFusionModel(nn.Module):
    """ResNet backbone fused with Edge Bank (E) for DT supervision."""
    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(backbone_name, pretrained=pretrained)

        # encode edge maps to same feature dimension
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # fuse ResNet (C5) + encoded edges → predict DT
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
def distance_transform_loss(pred, target):
    return nn.functional.l1_loss(pred, target)


# ---------------------------------------------------------------------- #
# Training Step
# ---------------------------------------------------------------------- #
def train_step(images, dt_targets, model, optimizer, device):
    images = images.to(device)
    dt_targets = dt_targets.to(device)

    # generate edge banks on-the-fly
    edge_banks = []
    for img in images:
        np_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        edge = generate_edge_bank(np_img)  # (H, W, 3)
        edge_tensor = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        edge_banks.append(edge_tensor)
    edge_banks = torch.cat(edge_banks, dim=0).to(device)

    optimizer.zero_grad()
    dt_pred = model(images, edge_banks)

    # match target size
    if dt_pred.shape[2:] != dt_targets.shape[2:]:
        dt_targets = nn.functional.interpolate(dt_targets, size=dt_pred.shape[2:], mode="bilinear", align_corners=False)

    loss = distance_transform_loss(dt_pred, dt_targets)
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

        for batch_idx, (images, _, dt_maps) in enumerate(train_loader):
            images = images / 255.0
            loss = train_step(images, dt_maps, model, optimizer, device)
            total_loss += loss

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg = total_loss / (batch_idx + 1)
                print(f"\rEpoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg:.4f}", end="")

        epoch_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} Average Loss: {epoch_loss:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(save_dir, f"resnet_edge_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "timestamp": datetime.now().isoformat()
        }, ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, "best_resnet_edge.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best model so far (loss {best_loss:.4f})")

    print("\n✅ Training complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in → {save_dir}")


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
    print(f"Device: {device}")

    print("\nLoading dataset (ISBI)…")
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Samples: {len(train_dataset)}")

    print(f"\nCreating model ({args.backbone})…")
    model = ResNetEdgeFusionModel(backbone_name=args.backbone, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(train_loader, model, optimizer, device, epochs=args.epochs)
