# sample_diffusion_halfres.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.diffusion_unet import ConditionalUNet
from models.backbone import ResNetBackbone
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from config import cfg


# ----------------------------------------------------------
# Beta schedule
# ----------------------------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)


# ----------------------------------------------------------
# SINGLE REVERSE STEP (FIXED VERSION)
# ----------------------------------------------------------
@torch.no_grad()
def p_sample(model, x_t, t_idx, T_tok, alphas, alphas_cumprod, betas,
             latent_shape, extra_cond=None):

    # t_idx MUST be (B,1)
    if t_idx.dim() == 1:
        t_idx = t_idx.unsqueeze(1)

    # Resize x_t to latent size for UNet if needed
    if x_t.shape[2:] != latent_shape[2:]:
        x_in = F.interpolate(x_t, size=latent_shape[2:], mode="bilinear", align_corners=False)
    else:
        x_in = x_t

    # Predict noise
    eps_theta = model(x_in, t_idx.float(), T_tok, extra_cond=extra_cond)

    # Resize prediction back
    if eps_theta.shape[2:] != x_t.shape[2:]:
        eps_theta = F.interpolate(eps_theta, size=x_t.shape[2:], mode="bilinear", align_corners=False)

    # Gather α, β values
    alpha_t      = alphas[t_idx[:,0]].view(-1,1,1,1)
    alpha_bar_t  = alphas_cumprod[t_idx[:,0]].view(-1,1,1,1)
    beta_t       = betas[t_idx[:,0]].view(-1,1,1,1)

    sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_ab  = torch.sqrt(1 - alpha_bar_t)
    sqrt_beta_t        = torch.sqrt(beta_t)

    # Add noise unless t=0
    z = torch.randn_like(x_t) if t_idx.item() > 0 else 0

    x_prev = (
        sqrt_recip_alpha_t *
        (x_t - ((1 - alpha_t) / sqrt_one_minus_ab) * eps_theta)
    ) + sqrt_beta_t * z

    return x_prev


# ----------------------------------------------------------
# FULL SAMPLING (20–30 STEPS)
# ----------------------------------------------------------
@torch.no_grad()
def sample_halfres(model, backbone, img_rgb, steps=20, device="cuda"):

    # Resize input image to training size (800x645)
    H, W = 800, 645
    img_rs = cv2.resize(img_rgb, (W, H))

    img_t = torch.from_numpy(img_rs).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0

    # 1) Backbone F features
    feats = backbone(img_t)
    F_feat = feats["C5"]

    # 2) Edge bank E
    edge = generate_edge_bank(img_rs)
    edge_t = torch.from_numpy(edge).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0

    if edge_t.shape[2:] != F_feat.shape[2:]:
        edge_res = F.interpolate(edge_t, size=F_feat.shape[2:], mode="bilinear", align_corners=False)
    else:
        edge_res = edge_t

    extra_cond = torch.cat([F_feat, edge_res], dim=1)

    # 3) Topology T tokens
    arcs = extract_arc_tokens_from_edgebank(edge)
    T_vec = flatten_arc_tokens(arcs)
    T_tok = torch.from_numpy(T_vec).unsqueeze(0).to(device).float()

    # 4) Reverse diffusion schedule
    betas = get_beta_schedule(steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 5) Latent UNet resolution
    half_h, half_w = 400, 322

    latent_out = model(
        torch.randn(1,1,half_h,half_w).to(device),
        torch.zeros((1,1), device=device),  # timestep
        T_tok,
        extra_cond=extra_cond,
    )
    latent_shape = latent_out.shape

    # 6) Start from random noise
    x_t = torch.randn(1,1,half_h,half_w).to(device)

    # 7) Reverse diffusion
    for i in reversed(range(steps)):
        t_idx = torch.tensor([i], device=device, dtype=torch.long)
        x_t = p_sample(
            model, x_t, t_idx, T_tok,
            alphas, alphas_cumprod, betas,
            latent_shape, extra_cond=extra_cond
        )

    # Final resize to original DT full size (800x645)
    dt_pred_half = x_t.squeeze().cpu().numpy()
    dt_pred_full = cv2.resize(dt_pred_half, (W, H))

    return dt_pred_full, edge


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    device = cfg.DEVICE
    print("Device:", device)

    # Load diffusion model
    ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion_halfres/diffusion_halfres_epoch_7.pth"
    model = ConditionalUNet(cond_dim=243, in_ch=1, out_ch=1, feat_dim=512).to(device)
    sd = torch.load(ckpt, map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print("Loaded diffusion model:", ckpt)

    # Load backbone
    backbone = ResNetBackbone("resnet34", pretrained=False, fuse_edges=False).to(device)
    bb_ckpt = "/content/drive/MyDrive/atlas_checkpoints/checkpoints_resnet_edge/best_resnet_edge.pth"
    state = torch.load(bb_ckpt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    backbone.load_state_dict(state, strict=False)
    backbone.eval()
    print("Loaded backbone")

    # Load test image
    img_path = "/content/drive/MyDrive/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run sampling
    dt_pred, edge = sample_halfres(model, backbone, img_rgb, steps=20, device=device)

    # Plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(edge[...,0], cmap="gray"); plt.title("Canny"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(dt_pred, cmap="magma"); plt.title("Predicted DT"); plt.axis("off")
    plt.show()
