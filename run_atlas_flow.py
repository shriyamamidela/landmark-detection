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

# USE ONLY YOUR WORKING SAMPLER
from sample_diffusion_dt_full import sample_diffusion

device = cfg.DEVICE


# ----------------------------
# Confidence from Jacobian
# ----------------------------
def compute_confidence_from_disp(disp):
    """
    disp: [1,2,H,W]
    returns confidence map [1,1,H,W]
    """
    dx = disp[:, :, 1:, :] - disp[:, :, :-1, :]     # (B,2,H-1,W)
    dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]     # (B,2,H,W-1)

    # Crop both to match (B,2,H-1,W-1)
    dx_c = dx[:, :, :, :-1]                         # (B,2,H-1,W-1)
    dy_c = dy[:, :, :-1, :]                         # (B,2,H-1,W-1)

    # Jacobian stability proxy
    J = torch.abs(dx_c).mean(dim=1, keepdim=True) + \
        torch.abs(dy_c).mean(dim=1, keepdim=True)   # (B,1,H-1,W-1)

    conf = torch.exp(-J)

    # Pad back to full resolution
    conf = F.pad(conf, (0,1,0,1))                   # (B,1,H,W)
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
    grid = grid.unsqueeze(0).unsqueeze(2)   # (1,N,1,2)

    dx_s = F.grid_sample(dx, grid, align_corners=True).squeeze().cpu().numpy()
    dy_s = F.grid_sample(dy, grid, align_corners=True).squeeze().cpu().numpy()

    warped = np.stack([atlas_lm[:,0] + dx_s,
                       atlas_lm[:,1] + dy_s], axis=-1)
    return warped


# ----------------------------
# MAIN PIPELINE
# ----------------------------
@torch.no_grad()
def run_atlas_flow(img_path, diffusion_ckpt, svf_ckpt, atlas_lm_path, steps=20):

    # ---------------- Load image ----------------
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    print(f"[INFO] Loaded image: {W}x{H}")

    # ---------------- Topology Tokens (T) ----------------
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

    # ---- Infer latent shape ----
    dummy = torch.randn(1,1,400,320).to(device)
    latent_shape = diff_model(dummy, torch.tensor([0.0]).to(device), T_tok).shape

    # ---- Use YOUR working sampler ----
    imgs = sample_diffusion(
        model=diff_model,
        T_tok=T_tok,
        steps=steps,
        shape=(1,1,400,320),
        device=device,
        latent_shape=latent_shape
    )

    DT_low = imgs[-1]  # numpy array (H,W)
    DT_low_t = torch.tensor(DT_low).float().unsqueeze(0).unsqueeze(0).to(device)

    print("[INFO] DT predicted successfully.")

    # ---------------- Extract F (backbone) ----------------
    img_t = torch.tensor(img_rgb).float().permute(2,0,1).unsqueeze(0).to(device)/255.0
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    backbone.eval()
    F5 = backbone(img_t)["C5"]   # (1,512,25,20)

    # resize DT
    D_small = F.interpolate(DT_low_t, size=F5.shape[2:], mode="bilinear", align_corners=False)

    # ---------------- SVF Prediction ----------------
    # ---------------- SVF Prediction ----------------
    svf = SVFHead(in_channels=512).to(device)

    svf_sd = torch.load(svf_ckpt, map_location=device)

    # Handle full checkpoint dict
    if isinstance(svf_sd, dict) and "svf_state_dict" in svf_sd:
        svf_sd = svf_sd["svf_state_dict"]

    svf.load_state_dict(svf_sd)
    svf.eval()


    v = svf(F5, D_small)
    disp = svf_to_disp(v)

    # ---------------- Warp atlas landmarks ----------------
    atlas_lm = np.load(atlas_lm_path)
    warped = warp_landmarks_with_disp(atlas_lm, disp, H, W)

    # ---------------- Confidence ----------------
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
    for i,(xy,c) in enumerate(zip(warped, conf_vals)):
        x,y = xy
        plt.scatter(x,y,c='lime' if c>0.5 else 'red',s=40)
        plt.text(x+3,y-3,f"{i}:{c:.2f}",color='white',fontsize=8)
    plt.title("ATLAS-FLOW Predicted Landmarks")
    plt.axis("off")
    plt.show()

    print("[INFO] COMPLETE.")
    return warped, conf_vals


# ---------------- Execute ----------------
if __name__ == "__main__":
    img_path = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    diffusion_ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_5.pth"
    svf_ckpt = "/content/drive/MyDrive/atlas_checkpoints/svf/svf_epoch_29.pth"
    atlas_lm = "/content/landmark-detection/atlas_landmarks_19.npy"

    run_atlas_flow(img_path, diffusion_ckpt, svf_ckpt, atlas_lm)
