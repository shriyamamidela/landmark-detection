# train_diffusion_dt.py
"""
Stage 2 — Conditional Diffusion training for Distance-Transform maps (DT)
Conditions on: F (backbone features), E (edge bank), T (topology tokens), and timestep t.
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

# local modules (make sure PYTHONPATH includes repo root)
from models.diffusion_unet import ConditionalUNet
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from data import Dataset
from config import cfg


# ---------------------------
# beta schedule helpers
# ---------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)


def q_sample(x_start, t_idx, noise, alphas_cumprod):
    """
    Vectorized forward diffusion:
      x_t = sqrt(alpha_cumprod[t]) * x_start + sqrt(1 - alpha_cumprod[t]) * noise
    - t_idx: LongTensor shape (B,) with values in [0, T-1]
    - alphas_cumprod: tensor shape (T,)
    """
    # Gather alpha_cumprod for each sample and reshape to (B,1,1,1)
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x_start + torch.sqrt(1.0 - a) * noise


# ---------------------------
# Single training step
# ---------------------------
def train_step(model, backbone, optimizer, batch, alphas_cumprod, device, T):
    images, _, dt_maps = batch
    images = images.to(device) / 255.0              # (B, C, H, W)
    dt_maps = dt_maps.to(device)                    # (B, 1, H, W)

    B = images.size(0)

    # --- 1) compute backbone features F ---
    with torch.no_grad():
        feats = backbone(images)
        F_feat = feats["C5"]                        # (B, feat_dim, h, w), feat_dim expected 512

    # --- 2) build Edge Bank E (resized as needed) ---
    edge_banks = []
    for img in images:
        # img: (C,H,W) in 0..1
        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        edge = generate_edge_bank(np_img)          # HxWx3 uint8
        edge_tensor = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        edge_banks.append(edge_tensor)
    E = torch.cat(edge_banks, dim=0).to(device)     # (B,3,H,W)

    # resize edge to F_feat spatial size and concat with F for extra_cond
    if E.shape[2:] != F_feat.shape[2:]:
        E_resized = F.interpolate(E, size=F_feat.shape[2:], mode="bilinear", align_corners=False)
    else:
        E_resized = E
    extra_cond = torch.cat([F_feat, E_resized], dim=1)  # (B, feat_dim+3, h, w)

    # --- 3) compute topology tokens T ---
    tokens_list = []
    for img in images:
        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        try:
            edge_bank = generate_edge_bank(np_img)
            arcs = extract_arc_tokens_from_edgebank(edge_bank)
            T_vec = flatten_arc_tokens(arcs)         # numpy (cond_dim,)
        except Exception:
            # fallback zero token if token extraction fails
            T_vec = np.zeros((243,), dtype=np.float32)
        tokens_list.append(torch.from_numpy(T_vec).unsqueeze(0))
    T_tok = torch.cat(tokens_list, dim=0).to(device)    # (B, cond_dim)

    # --- 4) Forward diffusion: sample t and noise, produce x_t ---
    t_idx = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
    noise = torch.randn_like(dt_maps, device=device)
    x_t = q_sample(dt_maps, t_idx, noise, alphas_cumprod)

    # --- 5) Predict noise with conditional UNet ---
    model.train()
    optimizer.zero_grad()

    # The UNet expects x_t (B,1,h,w) — ensure resolution matches
    # If model produces different spatial resolution than noise, we will resize noise in loss
    noise_pred = model(x_t, t_idx.float().unsqueeze(1), T_tok, extra_cond=extra_cond)

    if noise_pred.shape != noise.shape:
        noise_resized = F.interpolate(noise, size=noise_pred.shape[2:], mode="bilinear", align_corners=False)
    else:
        noise_resized = noise

    loss = F.mse_loss(noise_pred, noise_resized)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------
# Training loop
# ---------------------------
def train_diffusion(model, backbone, dataloader, device, epochs=5, T=1000, save_dir="checkpoints_diffusion_dt", lr=1e-4):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    beta = get_beta_schedule(T, device=device)
    alphas = 1.0 - beta
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            loss = train_step(model, backbone, optimizer, batch, alphas_cumprod, device, T)
            total_loss += loss
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch} | Avg loss: {avg_loss:.4f}")

        # save checkpoint (weights only)
        ckpt_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "timestamp": datetime.now().isoformat()
        }, ckpt_path)
        print(f"Saved checkpoint → {ckpt_path}")


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/atlas_checkpoints/diffusion")
    parser.add_argument("--backbone-weights", type=str, default="/content/drive/MyDrive/atlas_checkpoints/resnet34_backbone_only.pth")
    parser.add_argument("--in-ch", type=int, default=1)
    parser.add_argument("--cond-dim", type=int, default=243)
    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    # dataset & loader
    train_dataset = Dataset(name="isbi", mode="train", batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("Samples:", len(train_dataset))

    # load backbone (backbone-only state_dict)
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    if os.path.exists(args.backbone_weights):
        state = torch.load(args.backbone_weights, map_location=device)
        # If state is a dict with keys like 'stem.*' that's fine; load with strict=False to allow mismatch
        if isinstance(state, dict) and any(k.startswith("stem") or k.startswith("layer1") for k in state.keys()):
            backbone.load_state_dict(state, strict=False)
            print("Loaded backbone weights (strict=False) from:", args.backbone_weights)
        else:
            try:
                backbone.load_state_dict(state)
                print("Loaded backbone weights from:", args.backbone_weights)
            except Exception as e:
                print("Warning: unable to load backbone state_dict directly:", e)
    else:
        print("Warning: backbone weights not found at:", args.backbone_weights)

    # diffusion model
    model = ConditionalUNet(cond_dim=args.cond_dim, in_ch=args.in_ch, out_ch=1, feat_dim=512).to(device)

    # train
    train_diffusion(model, backbone, train_loader, device,
                    epochs=args.epochs, T=args.timesteps, save_dir=args.save_dir, lr=args.lr)
