"""
ATLAS-FLOW Inference (Final Stable Version)
Uses ONLY your working sample_diffusion() function.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from models.svf_utils import svf_to_disp
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens

# sampler
from sample_diffusion_dt_full import sample_diffusion

# refinement (step 5)
from tools.refine_svf import refine_and_warp
from tools.warp_utils import warp_landmarks_wrapper

device = cfg.DEVICE


# ----------------------------
# Confidence
# ----------------------------
def compute_confidence_from_disp(disp):
    dx = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]

    dx_c = dx[:, :, :, :-1]
    dy_c = dy[:, :, :-1, :]

    J = torch.abs(dx_c).mean(dim=1, keepdim=True) + torch.abs(dy_c).mean(dim=1, keepdim=True)
    conf = torch.exp(-J)
    conf = F.pad(conf, (0,1,0,1))
    return conf


# ----------------------------
# Warp atlas landmarks
# ----------------------------
def warp_landmarks_with_disp(atlas_lm, disp, H, W):
    disp_up = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)
    dx = disp_up[:,0:1]
    dy = disp_up[:,1:2]

    xs = (atlas_lm[:,0] / (W-1)) * 2 - 1
    ys = (atlas_lm[:,1] / (H-1)) * 2 - 1
    grid = torch.tensor(np.stack([xs,ys],axis=-1)).float().to(disp.device)
    grid = grid.unsqueeze(0).unsqueeze(2)

    dx_s = F.grid_sample(dx, grid, align_corners=True).squeeze().cpu().numpy()
    dy_s = F.grid_sample(dy, grid, align_corners=True).squeeze().cpu().numpy()

    warped = np.stack([atlas_lm[:,0] + dx_s,
                       atlas_lm[:,1] + dy_s], axis=-1)
    return warped


# ============================================================
# MAIN PIPELINE
# ============================================================
@torch.no_grad()
def run_atlas_flow(img_path, diffusion_ckpt, svf_ckpt, atlas_lm_path, steps=20):

    # ---------------- Load image ----------------
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"[INFO] Loaded image: {W}x{H}")

    # ---------------- Topology Tokens ----------------
    edge = generate_edge_bank(img_rgb)
    arcs = extract_arc_tokens_from_edgebank(edge)
    T_tok = torch.from_numpy(flatten_arc_tokens(arcs)).unsqueeze(0).float().to(device)

    # ---------------- Diffusion Model ----------------
    from models.diffusion_unet import ConditionalUNet
    diff_model = ConditionalUNet(in_ch=1, out_ch=1).to(device)
    diff_sd = torch.load(diffusion_ckpt, map_location=device)
    if isinstance(diff_sd, dict) and "model_state_dict" in diff_sd:
        diff_sd = diff_sd["model_state_dict"]
    diff_model.load_state_dict(diff_sd)
    diff_model.eval()

    print("[INFO] Sampling DT via diffusion...")

    dummy = torch.randn(1,1,400,320).to(device)
    latent_shape = diff_model(dummy, torch.tensor([0.0]).to(device), T_tok).shape

    imgs = sample_diffusion(
        model=diff_model,
        T_tok=T_tok,
        steps=steps,
        shape=(1,1,400,320),
        device=device,
        latent_shape=latent_shape
    )

    DT_low = imgs[-1]
    DT_low_t = torch.tensor(DT_low).float().unsqueeze(0).unsqueeze(0).to(device)
    print("[INFO] DT predicted successfully.")

    # ---------------- Backbone ----------------
    img_t = torch.tensor(img_rgb).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    backbone.eval()
    F5 = backbone(img_t)["C5"]   # (1,512,75,61)

    # downsample DT
    D_small = F.interpolate(DT_low_t, size=F5.shape[2:], mode="bilinear", align_corners=False)

    # ---------------- SVF ----------------
    svf = SVFHead(in_channels=512).to(device)
    svf_sd = torch.load(svf_ckpt, map_location=device)
    if isinstance(svf_sd, dict) and "svf_state_dict" in svf_sd:
        svf_sd = svf_sd["svf_state_dict"]
    svf.load_state_dict(svf_sd)
    svf.eval()

    v = svf(F5, D_small)
    disp = svf_to_disp(v)

    # ============================================================
    # STEP 1 — DEBUG + BUILD atlas_edges_resized
    # ============================================================
    print("DEBUG: v.shape =", v.shape)
    print("DEBUG: disp_lowres.shape =", disp.shape)
    print("DEBUG: F5.shape =", F5.shape)
    print("DEBUG: D_small.shape =", D_small.shape)

    atlas_edge_full_path = "/content/landmark-detection/atlas_edge_map.npy"
    if os.path.exists(atlas_edge_full_path):
        atlas_edge_full = np.load(atlas_edge_full_path)
        if atlas_edge_full.ndim == 2:
            atlas_edge_full = atlas_edge_full[..., None]
        atlas_edge_full_t = torch.from_numpy(atlas_edge_full.transpose(2,0,1)).unsqueeze(0).float()
        atlas_edges_resized = F.interpolate(atlas_edge_full_t.to(device),
                                            size=F5.shape[2:], mode="bilinear", align_corners=False)
    else:
        print("[WARN] atlas_edge_map.npy not found — using zeros")
        atlas_edges_resized = torch.zeros_like(D_small)

    print("DEBUG: atlas_edges_resized.shape =", atlas_edges_resized.shape)
    print("DEBUG: Ready to run refinement.")

    # ============================================================
    # STEP 5 — LOW-RANK REFINEMENT
    # ============================================================
    from tools.refine_svf import refine_and_warp
    from tools.warp_utils import warp_landmarks_wrapper

    print("[INFO] Running LOW-RANK REFINEMENT...")

    atlas_lm = np.load(atlas_lm_path)

    disp_refined, warped_after, conf_map = refine_and_warp(
        svf_disp=disp,
        svf_v=None,
        F_feat=F5,
        D_small=D_small,
        atlas_edges_resized=atlas_edges_resized,
        atlas_landmarks_px=atlas_lm,
        warp_landmarks_fn=lambda lm, disp_arr: warp_landmarks_wrapper(lm, disp_arr, H, W),
        device=device,
        rank=4,
        n_iter=200,
        lr=1e-2,
        lambda_edge=1.0,
        lambda_jac=1.0,
        lambda_reg=1e-4,
    )

    print("[INFO] Refinement complete.")

    # ---------------- Confidence (original disp) ----------------
    conf = compute_confidence_from_disp(disp)
    conf_up = F.interpolate(conf, size=(H,W), mode="bilinear", align_corners=False)

    xs = (atlas_lm[:,0]/(W-1))*2 - 1
    ys = (atlas_lm[:,1]/(H-1))*2 - 1
    grid = torch.tensor(np.stack([xs,ys],axis=-1)).float().to(device)
    grid = grid.unsqueeze(0).unsqueeze(2)
    conf_vals = F.grid_sample(conf_up, grid, align_corners=True).squeeze().cpu().numpy()

    # ---------------- Visualize ----------------
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    for i,(xy,c) in enumerate(zip(warped_after, conf_vals)):
        x,y = xy
        plt.scatter(x,y,c='lime' if c>0.5 else 'red',s=40)
        plt.text(x+3,y-3,f"{i}:{c:.2f}",color='white',fontsize=8)
    plt.title("ATLAS-FLOW + Refinement")
    plt.axis("off")
    plt.show()

    print("[INFO] COMPLETE.")
    return warped_after, conf_vals


# ---------------- EXECUTE ----------------
if __name__ == "__main__":
    img_path = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    diffusion_ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_5.pth"
    svf_ckpt = "/content/drive/MyDrive/atlas_checkpoints/svf/svf_epoch_29.pth"
    atlas_lm = "/content/landmark-detection/atlas_landmarks_19.npy"

    run_atlas_flow(img_path, diffusion_ckpt, svf_ckpt, atlas_lm)
