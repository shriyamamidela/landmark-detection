# sample_diffusion_dt_full_ddim.py
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

# =========================
# COSINE SCHEDULE (TRAINING)
# =========================
def cosine_beta_schedule(T, s=0.008, device="cpu"):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_cum = torch.cos(((x/T) + s)/(1+s) * np.pi*0.5)**2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return torch.clamp(betas, 1e-4, 0.999)


# =========================
# LOAD MODEL / LOAD EMA
# =========================
def load_model_or_ema(model, path, device):
    sd = torch.load(path, map_location=device)

    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    msd = model.state_dict()
    for k in msd.keys():
        if k in sd:
            msd[k] = sd[k].to(device)
    model.load_state_dict(msd)
    return model


# =========================
# CORRECTED DDIM STEP
# =========================
@torch.no_grad()
def ddim_step(model, x_t, t_idx, t_prev, T_tok,
              alphas, alphas_cum, eta, extra_cond):

    if t_idx.dim() == 0:
        t_idx = t_idx.view(1)
    if t_prev.dim() == 0:
        t_prev = t_prev.view(1)

    # ---- predict noise ----
    eps = model(
        x_t,
        t_idx.float().unsqueeze(1),   # (1,1)
        T_tok,
        extra_cond=extra_cond
    )

    # ---- FIX: resize eps to match x_t ----
    if eps.shape[2:] != x_t.shape[2:]:
        eps = F.interpolate(eps, size=x_t.shape[2:], mode="bilinear", align_corners=False)

    # convert scalars
    a_t  = alphas[t_idx][0]
    ab_t = alphas_cum[t_idx][0]
    ab_prev = alphas_cum[t_prev][0]

    # ---- Compute x0 prediction ----
    x0_hat = (x_t - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
    x0_hat = torch.clamp(x0_hat, -1, 1)

    # ---- DDIM parameters ----
    sigma = eta * torch.sqrt((1 - ab_prev)/(1 - ab_t)) * torch.sqrt(1 - ab_t/ab_prev)
    c = torch.sqrt(1 - ab_prev - sigma*sigma)
    noise = torch.randn_like(x_t) if (t_idx > 0) else 0

    x_prev = torch.sqrt(ab_prev) * x0_hat + c * eps + sigma * noise
    return x_prev


# =========================
# DDIM SAMPLER
# =========================
@torch.no_grad()
def sample_ddim_fast(model, backbone, img_rgb, steps=20, device="cuda", eta=0.0):

    H, W = 800, 645
    hh, ww = 400, 322   # keep as your model was trained

    # image
    img_rs = cv2.resize(img_rgb, (W, H))
    img_t = torch.from_numpy(img_rs).permute(2,0,1).unsqueeze(0).float().to(device)/255.

    # backbone
    feats = backbone(img_t)
    F_feat = feats["C5"]

    # edges
    edge = generate_edge_bank(img_rs)
    edge_t = torch.from_numpy(edge).permute(2,0,1).unsqueeze(0).float().to(device)/255.
    edge_res = F.interpolate(edge_t, size=F_feat.shape[2:], mode="bilinear", align_corners=False)
    extra_cond = torch.cat([F_feat, edge_res], dim=1)

    # tokens
    arcs = extract_arc_tokens_from_edgebank(edge)
    T_vec = flatten_arc_tokens(arcs)
    T_tok = torch.from_numpy(T_vec).unsqueeze(0).float().to(device)

    # schedule
    betas = cosine_beta_schedule(500, device=device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    # time grid
    ts = torch.linspace(499, 0, steps, dtype=torch.long, device=device)

    # start noise
    x_t = torch.randn(1,1,hh,ww).to(device)

    # steps
    for i in range(steps-1):
        t_idx  = ts[i].view(1)
        t_prev = ts[i+1].view(1)

        x_t = ddim_step(model, x_t, t_idx, t_prev, T_tok,
                        alphas, alphas_cum, eta, extra_cond)

    # final
    t_idx = ts[-1].view(1)
    x_t = ddim_step(model, x_t, t_idx, t_idx, T_tok,
                    alphas, alphas_cum, eta, extra_cond)

    # upsample
    dt_half = x_t.squeeze().cpu().numpy()
    dt_full = cv2.resize(dt_half, (W, H))
    return dt_full, edge


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    CKPT_EMA = "/content/diffusion_halfres_epoch_20_ema.pth"
    IMG_PATH = "/content/landmark-detection/datasets/augmented_ceph/image_dir/001_aug0.png"

    device = cfg.DEVICE
    print("Device:", device)

    model = ConditionalUNet(243,1,1,512).to(device)
    print("Loading:", CKPT_EMA)
    model = load_model_or_ema(model, CKPT_EMA, device)
    model.eval()

    # backbone
    backbone = ResNetBackbone("resnet34", pretrained=False, fuse_edges=False).to(device)
    bb_ckpt = "/content/drive/MyDrive/atlas_checkpoints/checkpoints_resnet_edge/best_resnet_edge.pth"
    bb = torch.load(bb_ckpt, map_location=device)
    if "model_state_dict" in bb:
        bb = bb["model_state_dict"]
    backbone.load_state_dict(bb, strict=False)
    backbone.eval()

    # image
    img = cv2.imread(IMG_PATH)
    img_rgxab = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # DDIM sampling
    dt_pred, edge = sample_ddim_fast(model, backbone, img_rgb,
                                     steps=20, device=device, eta=0.0)

    arr = np.array(dt_pred)
    print("\n=== DT RANGE ===")
    print("min:", float(arr.min()))
    print("max:", float(arr.max()))
    print("mean:", float(arr.mean()))
    print("================\n")

    # normalize for display
    vmin, vmax = np.percentile(arr,1), np.percentile(arr,99)
    dt_vis = np.clip((arr-vmin)/(vmax-vmin+1e-9),0,1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(edge[...,0], cmap="gray"); plt.title("Edge"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(dt_vis, cmap="magma"); plt.title("DDIM Predicted DT"); plt.axis("off")
    plt.show()
