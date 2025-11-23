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

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

@torch.no_grad()
def p_sample(model, x_t, t_idx, T_tok, alphas, alphas_cumprod, betas, latent_shape, extra_cond=None):
    if x_t.shape[2:] != latent_shape[2:]:
        x_in = F.interpolate(x_t, size=latent_shape[2:], mode="bilinear", align_corners=False)
    else:
        x_in = x_t
    eps_theta = model(x_in, t_idx.float().unsqueeze(1), T_tok, extra_cond=extra_cond)
    if eps_theta.shape[2:] != x_t.shape[2:]:
        eps_theta = F.interpolate(eps_theta, size=x_t.shape[2:], mode="bilinear", align_corners=False)
    alpha_t = alphas[t_idx]
    alpha_bar_t = alphas_cumprod[t_idx]
    sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    sqrt_beta_t = torch.sqrt(betas[t_idx])
    z = torch.randn_like(x_t) if t_idx > 0 else 0
    x_prev = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * eps_theta) + sqrt_beta_t * z
    return x_prev

@torch.no_grad()
def sample_halfres(model, backbone, img_rgb, steps=20, device="cuda"):
    # resize image full->train size
    target_h, target_w = 800, 645
    img_rs = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img_rs).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0

    # F feat
    feats = backbone(img_t)
    F_feat = feats["C5"]

    # edge
    edge = generate_edge_bank(img_rs)
    edge_t = torch.from_numpy(edge).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    if edge_t.shape[2:] != F_feat.shape[2:]:
        edge_res = F.interpolate(edge_t, size=F_feat.shape[2:], mode="bilinear", align_corners=False)
    else:
        edge_res = edge_t
    extra_cond = torch.cat([F_feat, edge_res], dim=1)

    arcs = extract_arc_tokens_from_edgebank(edge)
    T_vec = flatten_arc_tokens(arcs)
    T_tok = torch.from_numpy(T_vec).unsqueeze(0).to(device).float()

    # schedule
    betas = get_beta_schedule(steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # decide half-res target
    half_h, half_w = 400, 322
    latent_out = model(torch.randn(1,1,half_h,half_w).to(device), torch.tensor([0.0]).to(device), T_tok, extra_cond=extra_cond)
    latent_shape = latent_out.shape

    x_t = torch.randn(1,1,half_h,half_w).to(device)
    for i in reversed(range(steps)):
        t_idx = torch.tensor(i, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t_idx, T_tok, alphas, alphas_cumprod, betas, latent_shape, extra_cond=extra_cond)

    # upsample to full DT size if needed
    dt_pred_half = x_t.squeeze().cpu().numpy()
    dt_pred_full = cv2.resize(dt_pred_half, (645,800), interpolation=cv2.INTER_LINEAR)  # note: cv2 expects (w,h)
    return dt_pred_full, edge

if __name__ == "__main__":
    device = cfg.DEVICE
    print("Device:", device)

    # load model
    ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion_halfres/diffusion_halfres_epoch_10.pth"
    model = ConditionalUNet(cond_dim=243, in_ch=1, out_ch=1, feat_dim=512).to(device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    # load backbone
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    bb_ckpt = "/content/drive/MyDrive/atlas_checkpoints/checkpoints_resnet_edge/best_resnet_edge.pth"
    if os.path.exists(bb_ckpt):
        state = torch.load(bb_ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict(state, strict=False)

    img_path = "/content/drive/MyDrive/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dt_pred, edge = sample_halfres(model, backbone, img_rgb, steps=20, device=device)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(edge[...,0], cmap="gray"); plt.title("Canny"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(dt_pred, cmap="magma"); plt.title("Pred DT"); plt.axis("off")
    plt.show()
