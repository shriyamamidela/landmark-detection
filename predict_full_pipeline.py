# ============================================================
#   ATLAS-FLOW Full Prediction Pipeline (FINAL VERSION)
#   Compatible with your trained diffusion UNet and SVF model.
#
#   - Runs on ISBI Test1 set
#   - Generates DT via diffusion (NO extra_cond)
#   - Predicts displacement using SVF
#   - Outputs final landmark predictions
#   - Saves CSV, NPY, confidence maps, and overlay images
# ============================================================

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import Dataset
from config import cfg

from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens

from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from models.svf_utils import svf_to_disp

from models.diffusion_unet import ConditionalUNet
from sample_diffusion_dt_full import sample_diffusion   # your working sampler

device = cfg.DEVICE


# ------------------------------------------------------------
# Confidence from displacement Jacobian proxy
# ------------------------------------------------------------
def compute_confidence_from_disp(disp):
    dx = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]

    dx_c = dx[:, :, :, :-1]
    dy_c = dy[:, :, :-1, :]

    J = torch.abs(dx_c).mean(dim=1, keepdim=True) + \
        torch.abs(dy_c).mean(dim=1, keepdim=True)

    conf = torch.exp(-J)
    conf = F.pad(conf, (0,1,0,1))
    return conf


# ------------------------------------------------------------
# Warp atlas landmarks with a displacement field
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# MAIN PIPELINE FOR ALL TEST IMAGES
# ------------------------------------------------------------
@torch.no_grad()
def main():

    # ========= PATHS =========
    DIFF_CKPT = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_30.pth"
    SVF_CKPT  = "/content/drive/MyDrive/atlas_checkpoints/svf/svf_epoch_38.pth"
    ATLAS_LM  = "atlas_landmarks_resized.npy"

    OUT_DIR  = "/content/drive/MyDrive/atlas_results"
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PRED = os.path.join(OUT_DIR, "pred_test_resized.npy")
    OUT_CSV  = os.path.join(OUT_DIR, "predictions.csv")
    VIS_DIR  = os.path.join(OUT_DIR, "visualizations")
    os.makedirs(VIS_DIR, exist_ok=True)

    # ========= LOAD ATLAS =========
    atlas_lm = np.load(ATLAS_LM)      # (19,2)
    atlas_lm = torch.tensor(atlas_lm).float().to(device)

    # ========= LOAD MODELS =========
    print("Loading diffusion model...")
    diff_model = ConditionalUNet(in_ch=1, out_ch=1).to(device)
    sd = torch.load(DIFF_CKPT, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    diff_model.load_state_dict(sd, strict=False)
    diff_model.eval()

    print("Loading SVF...")
    svf = SVFHead(in_channels=512).to(device)
    svf_sd = torch.load(SVF_CKPT, map_location=device)
    svf.load_state_dict(svf_sd["svf"])
    svf.eval()

    backbone = ResNetBackbone("resnet34", pretrained=False).to(device).eval()

    # ========= LOAD TEST SET =========
    ds = Dataset("isbi", "test", batch_size=1, shuffle=False)

    predictions = []
    confidences = []

    # ========= PROCESS EACH TEST IMAGE =========
    for idx in range(len(ds)):
        print(f"\n[INFO] Processing test image {idx+1}/{len(ds)}")

        img, _, dt_map, _ = ds[idx]
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)

        H, W = img_np.shape[:2]

        # ----- STAGE 1: Topology Tokens -----
        edges = generate_edge_bank(img_np)
        arcs = extract_arc_tokens_from_edgebank(edges)
        T_tok = torch.from_numpy(flatten_arc_tokens(arcs)).unsqueeze(0).float().to(device)

        # ----- STAGE 2: DT Diffusion -----
        print("   Sampling diffusion DT...")
        dummy = torch.randn(1,1,400,320).to(device)
        latent_shape = diff_model(dummy, torch.tensor([0.0]).to(device), T_tok).shape

        imgs = sample_diffusion(
            model=diff_model,
            T_tok=T_tok,
            steps=20,
            shape=(1,1,400,320),
            latent_shape=latent_shape,
            device=device
        )

        DT_low = imgs[-1]
        DT_t = torch.tensor(DT_low).float().unsqueeze(0).unsqueeze(0).to(device)

        # ----- STAGE 3: Backbone -----
        img_t = img.unsqueeze(0).float().to(device)
        feats = backbone(img_t)
        F5 = feats["C5"]

        # resize DT to feature shape
        D_small = F.interpolate(DT_t, size=F5.shape[2:], mode="bilinear")

        # ----- STAGE 4: SVF -----
        v = svf(F5, D_small)
        disp = svf_to_disp(v)
        warped_pts = warp_landmarks_with_disp(atlas_lm.cpu().numpy(), disp, H, W)
        predictions.append(warped_pts)

        # confidence
        conf_map = compute_confidence_from_disp(disp)
        confidences.append(conf_map.cpu().numpy())

        # ----- VISUALIZATION -----
        plt.figure(figsize=(6,6))
        plt.imshow(img_np, cmap='gray')
        for (x,y) in warped_pts:
            plt.scatter(x,y,c='red',s=30)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"test_{idx+1:03d}.png"))
        plt.close()

    # ========= SAVE OUTPUTS =========
    predictions = np.stack(predictions)
    confidences = np.stack(confidences)

    np.save(OUT_PRED, predictions)
    np.save(os.path.join(OUT_DIR, "confidence.npy"), confidences)

    # save csv
    with open(OUT_CSV, "w") as f:
        f.write("image,index,x,y\n")
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                x, y = predictions[i,j]
                f.write(f"{i+1},{j+1},{x:.3f},{y:.3f}\n")

    print("\n\n[✔] DONE!")
    print(f"Saved predictions to: {OUT_PRED}")
    print(f"Saved CSV to: {OUT_CSV}")
    print(f"Saved visualizations to: {VIS_DIR}")

    return predictions


if __name__ == "__main__":
    main()
