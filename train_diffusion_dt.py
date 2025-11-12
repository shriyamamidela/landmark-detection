"""
Stage 3 of ATLAS-FLOW-DIFF Pipeline:
Conditional Diffusion model (U-Net) learns to denoise Distance Transform (DT) maps,
conditioned on image features (F), edge maps (E), and topology tokens (T).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import argparse
import numpy as np
from tqdm import tqdm

from models.diffusion_unet import ConditionalUNet
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from data import Dataset
from config import cfg


# ------------------------------------------------------------------------------
# β schedule for noise levels (linear)
# ------------------------------------------------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


# ------------------------------------------------------------------------------
# Forward diffusion: add noise to clean DT map
# ------------------------------------------------------------------------------
def q_sample(x_start, t, noise, alphas_cumprod):
    sqrt_alphas_cumprod_t = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alphas_cumprod_t[:, None, None, None] * x_start + \
           sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * noise


# ------------------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------------------
def train_step(model, optimizer, batch, alphas_cumprod, device):
    images, _, dt_maps = batch
    images, dt_maps = images.to(device) / 255.0, dt_maps.to(device)

    # ---------------------------
    # 1️⃣ Extract feature maps (F)
    # ---------------------------
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    feats = backbone(images)
    F_feat = feats["C5"]

    # ---------------------------
    # 2️⃣ Generate Edge Banks (E)
    # ---------------------------
    edge_banks = []
    for img in images:
        np_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        edge = generate_edge_bank(np_img)
        edge_tensor = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        edge_banks.append(edge_tensor)
    E = torch.cat(edge_banks, dim=0).to(device)

    # ---------------------------
    # 3️⃣ Compute Topology Tokens (T)
    # ---------------------------
    tokens = []
    for img in images:
        np_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        edge_bank = generate_edge_bank(np_img)
        arcs = extract_arc_tokens_from_edgebank(edge_bank)
        T = flatten_arc_tokens(arcs)
        tokens.append(torch.from_numpy(T).unsqueeze(0))
    T_tok = torch.cat(tokens, dim=0).to(device)  # (B, token_dim)

    # ---------------------------
    # 4️⃣ Forward diffusion
    # ---------------------------
    t = torch.randint(0, len(alphas_cumprod), (images.size(0),), device=device)
    noise = torch.randn_like(dt_maps)
    x_t = q_sample(dt_maps, t, noise, alphas_cumprod)

    # ---------------------------
    # 5️⃣ Predict noise with Conditional U-Net
    # ---------------------------
    noise_pred = model(x_t, t.float(), T_tok)
    if noise_pred.shape[2:] != noise.shape[2:]:
      noise = F.interpolate(noise, size=noise_pred.shape[2:], mode="bilinear", align_corners=False)


    # ---------------------------
    # 6️⃣ Loss + Backprop
    # ---------------------------
    loss = F.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------------------
def train_diffusion(model, dataloader, device, epochs=5, T=1000, save_dir="checkpoints_diffusion_dt"):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    beta = get_beta_schedule(T).to(device)
    alphas = 1.0 - beta
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print(f"🚀 Training Conditional Diffusion Model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            total_loss += train_step(model, optimizer, batch, alphas_cumprod, device)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("✅ Training Complete. Model saved.")


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"Device: {device}")

    print("\nLoading dataset...")
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalUNet(in_ch=1, out_ch=1).to(device)

    train_diffusion(model, train_loader, device, epochs=args.epochs, T=args.timesteps)
