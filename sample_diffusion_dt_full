"""
Sample diffusion denoising process (Stage 3: ATLAS-FLOW-DIFF)
Generates Distance Transform (DT) maps using trained Conditional Diffusion model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from models.diffusion_unet import ConditionalUNet
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from config import cfg

# ------------------------------------------------------------------------------
# Helper: Beta schedule (same as during training)
# ------------------------------------------------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

# ------------------------------------------------------------------------------
# Sampling step (reverse diffusion)
# ------------------------------------------------------------------------------
@torch.no_grad()
def p_sample(model, x_t, t, F_cond, E_cond, T_tok, alphas, alphas_cumprod, betas):
    # Combine conditioning (F + E)
    cond_embed = torch.cat([F_cond, E_cond], dim=1)

    eps_theta = model(x_t, t.float(), T_tok)  # Predict noise
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    sqrt_beta_t = torch.sqrt(betas[t])

    z = torch.randn_like(x_t) if t > 0 else 0
    x_prev = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * eps_theta) + sqrt_beta_t * z
    return x_prev

# ------------------------------------------------------------------------------
# Full sampling loop
# ------------------------------------------------------------------------------
@torch.no_grad()
def sample_diffusion(model, F_cond, E_cond, T_tok, steps=20, shape=(1,1,400,320), device="cuda"):
    betas = get_beta_schedule(steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    x_t = torch.randn(shape, device=device)  # start from pure noise

    imgs = []
    for t in reversed(range(steps)):
        x_t = p_sample(model, x_t, torch.tensor([t], device=device), F_cond, E_cond, T_tok, alphas, alphas_cumprod, betas)
        imgs.append(x_t.detach().cpu().numpy()[0,0])  # store for visualization
    return imgs

# ------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    device = cfg.DEVICE
    print(f"Device: {device}")

    # Load trained model
    ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_5.pth"
    model = ConditionalUNet(in_ch=1, out_ch=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"✅ Loaded model: {ckpt}")

    # Load test image
    img_path = "/content/drive/MyDrive/datasets/ISBI Dataset/Dataset/Testing/Test1/001.bmp"
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0

    # Extract features (F) and edges (E)
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    feats = backbone(img_t)
    F_cond = feats["C5"]
    edge = generate_edge_bank(img_rgb)
    E_cond = torch.from_numpy(edge).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    E_cond = F.interpolate(E_cond, size=F_cond.shape[2:], mode="bilinear", align_corners=False)

    # Topology tokens (T)
    arcs = extract_arc_tokens_from_edgebank(edge)
    T_tok = torch.from_numpy(flatten_arc_tokens(arcs)).unsqueeze(0).to(device)

    # Run sampling
    imgs = sample_diffusion(model, F_cond, E_cond, T_tok, steps=20, shape=(1,1,400,320), device=device)

    # Visualization
    plt.figure(figsize=(14,6))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Input X-ray")
    plt.subplot(1,3,2); plt.imshow(edge[:,:,0], cmap="gray"); plt.title("Edge Bank (E)")
    plt.subplot(1,3,3); plt.imshow(imgs[-1], cmap="magma"); plt.title("Predicted DT (Step 0)")
    plt.tight_layout(); plt.show()

    print("✅ Diffusion sampling complete. Generated DT maps from noise → clean.")
