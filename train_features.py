"""
Train Feature Extraction (Backbone + Semantic Fusion Block)
For HRNet-W48 backbone
Only trains backbone + SFB to learn strong features (no landmark modules)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.backbone import Backbone
from network.semantic_fusion_block import SemanticFusionBlock
from data import Dataset
from config import cfg
import os
import argparse
from utils import rescale_input


# ---------------------------------------------------------------------- #
# Model definition
# ---------------------------------------------------------------------- #
class FeatureExtractionModel(nn.Module):
    """Simplified model for feature extraction training"""
    def __init__(self, backbone_name, backbone_weights=None):
        super(FeatureExtractionModel, self).__init__()
        
        # Backbone
        self.backbone = Backbone(
            name=backbone_name,
            pretrained=False,
            weights_root_path=backbone_weights
        )
        self.backbone_name = backbone_name
        
        # Semantic Fusion Block (initialized dynamically)
        self.sfb = None
        self.initialized = False
    
    def forward(self, x):
        # Forward through backbone
        _ = self.backbone(x)
        
        # Get HRNet feature dictionary
        feat_dict = self.backbone.get_feature_dict() if hasattr(self.backbone, 'get_feature_dict') else {}
        
        # Extract HRNet multi-resolution features
        R4 = feat_dict.get("R4")
        R8 = feat_dict.get("R8")
        R16 = feat_dict.get("R16")
        R32 = feat_dict.get("R32")

        # Alias HRNet branches to "C3, C4, C5" for compatibility
        C3, C4, C5 = R8, R16, R32
        
        # Initialize SFB on first forward pass
        if not self.initialized and C3 is not None:
            in_channels = (C3.shape[1], C4.shape[1], C5.shape[1])
            self.sfb = SemanticFusionBlock(num_filters=256, in_channels=in_channels).to(x.device)
            self.initialized = True
            print(f"Initialized SFB with input channels: {in_channels}")
        
        # Pass through SFB
        if self.sfb is not None and C3 is not None and C4 is not None and C5 is not None:
            P3, P4, P5 = self.sfb([C3, C4, C5])
            return {"R4": R4, "R8": R8, "R16": R16, "R32": R32, "P3": P3, "P4": P4, "P5": P5}
        
        return {"R4": R4, "R8": R8, "R16": R16, "R32": R32}


# ---------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------- #
def feature_reconstruction_loss(features):
    """
    Self-supervised loss:
    Encourages feature diversity (variance) and channel independence (covariance)
    """
    total_loss = 0
    count = 0
    
    for name, feat in features.items():
        if feat is None:
            continue
        
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        
        # Variance term
        variance = feat_flat.var(dim=2).mean()
        var_loss = torch.clamp(1.0 - variance, min=0)
        
        # Covariance term
        feat_normalized = feat_flat - feat_flat.mean(dim=2, keepdim=True)
        cov_matrix = torch.bmm(feat_normalized, feat_normalized.transpose(1, 2)) / (H * W)
        identity = torch.eye(C, device=feat.device).unsqueeze(0).expand(B, -1, -1)
        cov_loss = (cov_matrix - identity).pow(2).mean()
        
        total_loss += var_loss + 0.1 * cov_loss
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)


def contrastive_feature_loss(P3, P4, P5):
    """Contrastive loss to balance feature similarity across pyramid scales"""
    if P3 is None or P4 is None or P5 is None:
        return torch.tensor(0.0)
    
    P3_down = nn.functional.adaptive_avg_pool2d(P3, P5.shape[2:])
    P4_down = nn.functional.adaptive_avg_pool2d(P4, P5.shape[2:])
    
    P3_norm = nn.functional.normalize(P3_down, dim=1)
    P4_norm = nn.functional.normalize(P4_down, dim=1)
    P5_norm = nn.functional.normalize(P5, dim=1)
    
    sim_3_4 = (P3_norm * P4_norm).sum(dim=1).mean()
    sim_4_5 = (P4_norm * P5_norm).sum(dim=1).mean()
    sim_3_5 = (P3_norm * P5_norm).sum(dim=1).mean()
    
    target = 0.5
    loss = (sim_3_4 - target).pow(2) + (sim_4_5 - target).pow(2) + (sim_3_5 - target).pow(2)
    return loss


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train_step(images, model, optimizer, device):
    images = images.to(device)
    optimizer.zero_grad()
    
    features = model(images)
    recon_loss = feature_reconstruction_loss(features)
    contrast_loss = contrastive_feature_loss(
        features.get("P3"), features.get("P4"), features.get("P5")
    )
    
    total_loss = recon_loss + 0.1 * contrast_loss
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), recon_loss.item(), contrast_loss.item()


def train_epoch(train_loader, model, optimizer, device, epoch):
    model.train()
    total_loss = recon_loss_sum = contrast_loss_sum = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = rescale_input(images, scale=(1 / 255), offset=0)
        loss, recon, contrast = train_step(images, model, optimizer, device)
        total_loss += loss
        recon_loss_sum += recon
        contrast_loss_sum += contrast
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"\rEpoch {epoch} [{batch_idx+1}/{len(train_loader)}] - "
                  f"Loss: {total_loss / (batch_idx+1):.4f} "
                  f"(Recon: {recon_loss_sum / (batch_idx+1):.4f}, "
                  f"Contrast: {contrast_loss_sum / (batch_idx+1):.4f})", end="")
    print()
    return total_loss / len(train_loader)


def validate(val_loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = rescale_input(images, scale=(1 / 255), offset=0)
            images = images.to(device)
            features = model(images)
            recon_loss = feature_reconstruction_loss(features)
            contrast_loss = contrastive_feature_loss(
                features.get("P3"), features.get("P4"), features.get("P5")
            )
            loss = recon_loss + 0.1 * contrast_loss
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train(train_loader, val_loader, model, optimizer, scheduler, device, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        train_loss = train_epoch(train_loader, model, optimizer, device, epoch)
        print(f"Training Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss = validate(val_loader, model, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(save_dir, "best_feature_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f"feature_model_epoch_{epoch}.pth"))
        
        if scheduler:
            scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_feature_model.pth')}")


# ---------------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Feature Extraction (Backbone + SFB)")
    parser.add_argument("--backbone", type=str, default="hrnet_w48", help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained backbone weights")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    args = parser.parse_args()
    
    # Device setup
    device = cfg.DEVICE
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    
    # Pretrained weights
    backbone_weights = args.pretrained
    if backbone_weights is None and args.backbone in ["hrnet_w32", "hrnet_w48"]:
        default_path = f"pretrained_weights/{args.backbone}_imagenet.pth"
        if os.path.exists(default_path):
            backbone_weights = default_path
            print(f"Using pretrained weights: {default_path}")
        else:
            print(f"⚠️ No pretrained weights found, using random initialization.")
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    try:
        val_dataset = Dataset(name="isbi", mode="valid", batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}")
    except:
        val_loader = None
        print(f"Train samples: {len(train_dataset)}, No validation set")
    
    # Model
    print("\nCreating model...")
    model = FeatureExtractionModel(
        backbone_name=args.backbone,
        backbone_weights=backbone_weights
    ).to(device)
    
    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train
    train(train_loader, val_loader, model, optimizer, scheduler, device, args.epochs, args.save_dir)
