# train_diffusion_halfres.py
"""
Faster conditional diffusion training using half-resolution DT maps (400x322).
Uses AugCephDataset (images 800x645), but downsamples DT to (400,322) for diffusion UNet.
"""

import os
import argparse
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.diffusion_unet import ConditionalUNet
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from aug_dataset import AugCephDataset
from config import cfg

# ---------------------------
# helpers
# ---------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

def q_sample(x_start, t_idx, noise, alphas_cumprod):
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x_start + torch.sqrt(1.0 - a) * noise

def load_backbone_weights_safe(backbone, path, device):
    if not os.path.exists(path):
        print("Backbone checkpoint not found:", path)
        return backbone

    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        backbone.load_state_dict(state, strict=False)
        print("Loaded backbone weights (strict=False) from:", path)
    except Exception as e:
        print("Warning: failed loading backbone strictly; attempting filtered load:", e)
        filtered = {k: v for k, v in state.items() if k in backbone.state_dict()}
        backbone.state_dict().update(filtered)
        backbone.load_state_dict(backbone.state_dict())
        print("Loaded matching keys into backbone.")
    return backbone

# ---------------------------
# training step
# ---------------------------
def train_step(model, backbone, optimizer, batch, alphas_cumprod, device, Ttimesteps, target_size):
    images, dt_maps_full, tokens = batch

    images = images.to(device)
    tokens = tokens.to(device)

    # Downsample DT to half-res (400,322)
    dt_maps = F.interpolate(dt_maps_full, size=target_size, mode="bilinear", align_corners=False)
    dt_maps = dt_maps.to(device)

    B = images.size(0)

    # 1. Backbone features
    with torch.no_grad():
        feats = backbone(images)
        F_feat = feats["C5"]

    # 2. Edge bank
    edge_banks = []
    for img in images:
        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        edge = generate_edge_bank(np_img)
        edge_tensor = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        edge_banks.append(edge_tensor)

    E = torch.cat(edge_banks, dim=0).to(device)

    # Resize to match backbone feature map resolution
    if E.shape[2:] != F_feat.shape[2:]:
        E_resized = F.interpolate(E, size=F_feat.shape[2:], mode="bilinear", align_corners=False)
    else:
        E_resized = E

    extra_cond = torch.cat([F_feat, E_resized], dim=1)

    # 3. Diffusion forward step
    t_idx = torch.randint(low=0, high=Ttimesteps, size=(B,), device=device, dtype=torch.long)
    noise = torch.randn_like(dt_maps, device=device)
    x_t = q_sample(dt_maps, t_idx, noise, alphas_cumprod)

    # 4. Predict noise
    model.train()
    optimizer.zero_grad()

    noise_pred = model(x_t, t_idx.float().unsqueeze(1), tokens, extra_cond=extra_cond)

    if noise_pred.shape != noise.shape:
        noise_resized = F.interpolate(noise, size=noise_pred.shape[2:], mode="bilinear", align_corners=False)
    else:
        noise_resized = noise

    loss = F.mse_loss(noise_pred, noise_resized)
    loss.backward()
    optimizer.step()

    return loss.item()

# ---------------------------
# training loop
# ---------------------------
def train_diffusion(model, backbone, dataloader, device, epochs=10, Ttimesteps=500,
                    save_dir="/content/drive/MyDrive/atlas_checkpoints/diffusion_halfres", lr=1e-4):

    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    betas = get_beta_schedule(Ttimesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)   # <<< FIXED (moved to GPU)

    # half-resolution target size
    target_size = (400, 322)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            loss = train_step(model, backbone, optimizer, batch, alphas_cumprod, device, Ttimesteps, target_size)
            total_loss += loss
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.6f}")

        ckpt = os.path.join(save_dir, f"diffusion_halfres_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "timestamp": datetime.now().isoformat()
        }, ckpt)

        print("Saved checkpoint →", ckpt)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/atlas_checkpoints/diffusion_halfres")
    parser.add_argument("--backbone-weights", type=str, default="/content/drive/MyDrive/atlas_checkpoints/checkpoints_resnet_edge/best_resnet_edge.pth")
    parser.add_argument("--cond-dim", type=int, default=243)
    parser.add_argument("--in-ch", type=int, default=1)
    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    dataset_root = "/content/landmark-detection/datasets/augmented_ceph"
    train_dataset = AugCephDataset(dataset_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    print("Samples:", len(train_dataset))

    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    backbone = load_backbone_weights_safe(backbone, args.backbone_weights, device)
    backbone.eval()

    model = ConditionalUNet(cond_dim=args.cond_dim, in_ch=args.in_ch,
                            out_ch=1, feat_dim=512).to(device)

    train_diffusion(model, backbone, train_loader, device,
                    epochs=args.epochs, Ttimesteps=args.timesteps,
                    save_dir=args.save_dir, lr=args.lr)
