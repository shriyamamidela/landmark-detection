"""
Final Fixed Sampling Script for Conditional Diffusion (ATLAS-FLOW-DIFF)
Handles mismatched resolutions automatically by up/downsampling internally.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from models.diffusion_unet import ConditionalUNet
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from config import cfg


# ------------------------------------------------------------------------------
# Helper: Beta schedule (same as training)
# ------------------------------------------------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


# ------------------------------------------------------------------------------
# Single reverse diffusion step with dynamic resizing
# ------------------------------------------------------------------------------
@torch.no_grad()
def p_sample(model, x_t, t, T_tok, alphas, alphas_cumprod, betas, latent_shape):
    # Downsample input noise to match model latent resolution if needed
    if x_t.shape[2:] != latent_shape[2:]:
        x_in = F.interpolate(x_t, size=latent_shape[2:], mode="bilinear", align_corners=False)
    else:
        x_in = x_t

    eps_theta = model(x_in, t.float(), T_tok)

    # If model outputs lower-res map, upsample it back to match x_t for consistent math
    if eps_theta.shape[2:] != x_t.shape[2:]:
        eps_theta = F.interpolate(eps_theta, size=x_t.shape[2:], mode="bilinear", align_corners=False)

    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    sqrt_beta_t = torch.sqrt(betas[t])

    z = torch.randn_like(x_t) if t > 0 else 0
    x_prev = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * eps_theta) + sqrt_beta_t * z
    return x_prev


# ------------------------------------------------------------------------------
# Full diffusion sampling
# ------------------------------------------------------------------------------
@torch.no_grad()
def sample_diffusion(model, T_tok, steps=20, shape=(1,1,400,320), device="cuda", latent_shape=(1,1,200,160)):
    betas = get_beta_schedule(steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    x_t = torch.randn(shape, device=device)
    imgs = []

    for t in reversed(range(steps)):
        x_t = p_sample(model, x_t, torch.tensor([t], device=device), T_tok,
                       alphas, alphas_cumprod, betas, latent_shape)
        imgs.append(x_t.detach().cpu().numpy()[0,0])
    return imgs


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    device = cfg.DEVICE
    print("Device:", device)

    # ✅ Load model
    ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_5.pth"
    model = ConditionalUNet(in_ch=1, out_ch=1).to(device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"✅ Loaded model: {ckpt}")

    # ✅ Load test image
    img_path = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ Compute topology tokens (T)
    edge = generate_edge_bank(img_rgb)
    arcs = extract_arc_tokens_from_edgebank(edge)
    T_tok = torch.from_numpy(flatten_arc_tokens(arcs)).unsqueeze(0).to(device)

    # ✅ Infer latent resolution from model
    with torch.no_grad():
        dummy = torch.randn(1, 1, 400, 320).to(device)
        dummy_t = torch.tensor([0.0]).to(device)
        dummy_tok = torch.zeros_like(T_tok)
        latent_out = model(dummy, dummy_t, dummy_tok)
        latent_shape = latent_out.shape
    print(f"ℹ️ Model latent resolution: {latent_shape}")

    # ✅ Run diffusion sampling (auto-aligned)
    imgs = sample_diffusion(model, T_tok, steps=20,
                            shape=(1,1,400,320),
                            latent_shape=latent_shape,
                            device=device)
    print("✅ Diffusion sampling complete.")

    # ✅ Visualize
    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Input X-ray"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(edge[...,0], cmap="gray"); plt.title("Edge Bank (E)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(imgs[-1], cmap="magma"); plt.title("Predicted DT (step 0)"); plt.axis("off")
    plt.tight_layout(); plt.show()
