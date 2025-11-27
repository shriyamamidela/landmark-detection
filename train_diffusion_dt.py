"""
train_diffusion_halfres_fixed.py  (DGX version)

Fully fixed half-resolution diffusion trainer:
 - cosine beta schedule
 - EMA (exponential moving average)
 - correct token & conditioning flow
 - correct q_sample (forward diffusion)
 - 400x322 DT training
 - uses your ConditionalUNet + ResNet C5 + edge bank
 - uses AugCephDataset (image, dt, token)
 - saves both model + EMA each epoch
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
from aug_dataset import AugCephDataset
from preprocessing.utils import generate_edge_bank
from config import cfg

# ------------------------------------------------------
# Cosine beta schedule (Nichol & Dhariwal)
# ------------------------------------------------------
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.999)
    return betas

# ------------------------------------------------------
# EMA helper
# ------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}

        for k, v in model.state_dict().items():
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.shadow[k] = v.detach().cpu().clone()

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().cpu(),
                                                    alpha=(1 - self.decay))

    def store(self, path):
        torch.save(self.shadow, path)

    def copy_to(self, model):
        state = model.state_dict()
        for k in self.shadow:
            state[k] = self.shadow[k].to(state[k].device)
        model.load_state_dict(state, strict=False)

# ------------------------------------------------------
# Forward diffusion: q_sample
# ------------------------------------------------------
def q_sample(x_start, t_idx, noise, alphas_cumprod):
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x_start + torch.sqrt(1 - a) * noise

# ------------------------------------------------------
# Single training step
# ------------------------------------------------------
def train_step(model, backbone, optimizer, batch,
               alphas_cumprod, device, Ttimesteps, target_size):

    images, dt_maps_full, tokens = batch

    images = images.to(device)
    tokens = tokens.to(device)

    # Downsample DT (400×322)
    dt_maps = F.interpolate(dt_maps_full.to(device),
                            size=target_size,
                            mode="bilinear",
                            align_corners=False)

    B = images.size(0)

    # Backbone features
    with torch.no_grad():
        feats = backbone(images)
        F_feat = feats["C5"]

    # Edge bank from full-res images
    edge_list = []
    for img in images:
        np_img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        edge = generate_edge_bank(np_img)
        edge = torch.from_numpy(edge).permute(2,0,1).unsqueeze(0).float() / 255.0
        edge_list.append(edge)
    E = torch.cat(edge_list, dim=0).to(device)

    # Resize edges to feature-map size
    if E.shape[2:] != F_feat.shape[2:]:
        E_resized = F.interpolate(E,
                                  size=F_feat.shape[2:],
                                  mode="bilinear",
                                  align_corners=False)
    else:
        E_resized = E

    # Combine backbone + edges
    extra_cond = torch.cat([F_feat, E_resized], dim=1)

    # Diffusion forward: sample t and noise
    t_idx = torch.randint(0, Ttimesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(dt_maps)
    x_t = q_sample(dt_maps, t_idx, noise, alphas_cumprod)

    # Predict noise
    model.train()
    optimizer.zero_grad()
    noise_pred = model(x_t,
                       t_idx.float().unsqueeze(1),
                       tokens,
                       extra_cond=extra_cond)

    # Match shapes for loss
    if noise_pred.shape != noise.shape:
        noise_resized = F.interpolate(noise,
                                      size=noise_pred.shape[2:],
                                      mode='bilinear',
                                      align_corners=False)
    else:
        noise_resized = noise

    loss = F.mse_loss(noise_pred, noise_resized)
    loss.backward()
    optimizer.step()

    return loss.item()

# ------------------------------------------------------
# Training loop
# ------------------------------------------------------
def train_diffusion(model, backbone, dataloader, device,
                    epochs=20, Ttimesteps=500,
                    save_dir="/dgxa_home/.../diffusion_halfres",
                    lr=1e-4, ema_decay=0.9999):

    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    betas = cosine_beta_schedule(Ttimesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    ema = EMA(model, decay=ema_decay)

    target_size = (400, 322)

    for epoch in range(1, epochs + 1):

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        total = 0
        n = 0

        for batch in pbar:
            loss = train_step(model, backbone, optimizer,
                              batch, alphas_cumprod,
                              device, Ttimesteps, target_size)

            total += loss
            n += 1
            ema.update(model)

            pbar.set_postfix(loss=total / n)

        avg = total / max(1, n)
        print(f"\nEpoch {epoch} | Avg Loss: {avg:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir,
                                 f"diffusion_halfres_epoch_{epoch}.pth")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg,
            "time": datetime.now().isoformat()
        }, ckpt_path)

        # Save EMA
        ema_path = os.path.join(save_dir,
                                f"diffusion_halfres_epoch_{epoch}_ema.pth")
        ema.store(ema_path)

        print("Saved:", ckpt_path)
        print("Saved EMA:", ema_path)


# ------------------------------------------------------
# CLI
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str,
                        default="/dgxa_home/se22ucse250/landmark-detection-main/diffusion_halfres")
    parser.add_argument("--backbone-weights", type=str,
                        default="/dgxa_home/se22ucse250/landmark-detection-main/best_resnet_edge.pth")
    parser.add_argument("--cond-dim", type=int, default=243)
    parser.add_argument("--in-ch", type=int, default=1)

    args = parser.parse_args()

    device = cfg.DEVICE
    print("Device:", device)

    dataset_root = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/augmented_ceph"
    train_dataset = AugCephDataset(dataset_root)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    print("Samples:", len(train_dataset))

    # Backbone
    backbone = ResNetBackbone("resnet34",
                               pretrained=False,
                               fuse_edges=False).to(device)

    if os.path.exists(args.backbone_weights):
        state = torch.load(args.backbone_weights, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict(state, strict=False)
        print("Loaded backbone weights:", args.backbone_weights)

    backbone.eval()

    # Model
    model = ConditionalUNet(cond_dim=args.cond_dim,
                            in_ch=args.in_ch,
                            out_ch=1,
                            feat_dim=512).to(device)

    # Train
    train_diffusion(model, backbone, train_loader, device,
                    epochs=args.epochs,
                    Ttimesteps=args.timesteps,
                    save_dir=args.save_dir,
                    lr=args.lr)
