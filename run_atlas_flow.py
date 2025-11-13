"""
ATLAS-FLOW INFERENCE SCRIPT
----------------------------------------
Stage 4: Warp atlas landmarks using SVF predicted from [F, D]
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from models.svf_utils import svf_to_disp, warp_with_disp
from models.diffusion_unet import ConditionalUNet
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
from config import cfg


device = cfg.DEVICE


# ----------------------------------------------------------
# Helper: Compute Jacobian determinant confidence
# ----------------------------------------------------------
def compute_confidence_from_disp(disp):
    """
    disp: [1,2,Hf,Wf]
    returns confidence map [1,1,Hf,Wf]
    """
    dx = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]

    J = dx[:, :, :, :-1] * dy[:, :, :-1, :]  # coarse jacobian proxy
    conf = torch.exp(-torch.abs(J))          # high jacobian stability → high confidence
    conf = F.pad(conf, (1,0,1,0))            # pad to match disp size
    return conf


# ----------------------------------------------------------
#  Main pipeline function
# ----------------------------------------------------------
@torch.no_grad()
def run_atlas_flow(
    img_path,
    diffusion_ckpt,
    svf_ckpt,
    atlas_lm_path,
    steps=20
):

    # -------------------------------
    # Load image
    # -------------------------------
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    img_t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0

    # -------------------------------
    # Extract features F (ResNet C5)
    # -------------------------------
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    backbone.eval()
    feats = backbone(img_t)
    F5 = feats["C5"]  # (1, 512, 25, 20)

    # -------------------------------
    # Compute topology tokens T
    # -------------------------------
    edge = generate_edge_bank(img_rgb)
    arcs = extract_arc_tokens_from_edgebank(edge)
    T = torch.from_numpy(flatten_arc_tokens(arcs)).unsqueeze(0).float().to(device)

    # -------------------------------
    # Diffusion model → predict DT map
    # -------------------------------
    diff_model = ConditionalUNet(in_ch=1, out_ch=1).to(device)
    diff_model.load_state_dict(torch.load(diffusion_ckpt, map_location=device))
    diff_model.eval()

    # Sample from diffusion (we use 20-step sampler)
    from sample_diffusion_dt_full import sample_diffusion_core  # small helper you already have

    DT_lowres = sample_diffusion_core(diff_model, T, steps=steps, device=device)
    # DT_lowres shape: (1,1,200,160)

    # Resize DT → match F5 (25×20)
    D_small = F.interpolate(DT_lowres, size=F5.shape[2:], mode="bilinear", align_corners=False)

    # -------------------------------
    # Load SVF model & predict v field
    # -------------------------------
    svf = SVFHead(in_channels=512).to(device)
    svf.load_state_dict(torch.load(svf_ckpt, map_location=device))
    svf.eval()

    v = svf(F5, D_small)           # (1,2,25,20)
    disp = svf_to_disp(v)          # (1,2,25,20)

    # -------------------------------
    # Warp atlas landmarks
    # -------------------------------
    atlas_lm = np.load(atlas_lm_path)  # (19,2) pixel
    atlas_lm_t = torch.from_numpy(atlas_lm).float().unsqueeze(0).to(device)

    # convert atlas pixel → normalized grid coords
    ys = (atlas_lm[:,1] / (H-1)) * 2 - 1
    xs = (atlas_lm[:,0] / (W-1)) * 2 - 1
    grid = torch.from_numpy(np.stack([xs,ys],axis=-1)).float().to(device)
    grid = grid.view(1,19,1,2)  # (1,19,1,2)

    # warp the grid by displacement field on low-res then bilinear upsample
    disp_up = F.interpolate(disp, size=(H,W), mode="bilinear", align_corners=False)
    dx = disp_up[:,0]
    dy = disp_up[:,1]

    # sample displacement at each landmark
    dx_lm = F.grid_sample(dx.unsqueeze(1), grid, align_corners=True).squeeze().cpu().numpy()
    dy_lm = F.grid_sample(dy.unsqueeze(1), grid, align_corners=True).squeeze().cpu().numpy()

    warped = np.stack([
        atlas_lm[:,0] + dx_lm,
        atlas_lm[:,1] + dy_lm
    ], axis=-1)

    # -------------------------------
    # Confidence = Jacobian stability + DT curvature
    # -------------------------------
    conf_map = compute_confidence_from_disp(disp)   # (1,1,25,20)
    conf_up = F.interpolate(conf_map, size=(H,W), mode="bilinear", align_corners=False)

    # sample confidence at landmark positions
    conf_vals = F.grid_sample(conf_up, grid, align_corners=True).squeeze().cpu().numpy()

    # -------------------------------
    # Visualization
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    for i,(xy,c) in enumerate(zip(warped, conf_vals)):
        x,y = xy
        plt.scatter(x,y, c='lime' if c>0.5 else 'red', s=40)
        plt.text(x+3,y-3, f"{i}:{c:.2f}", color='white', fontsize=8)
    plt.title("ATLAS-FLOW predicted landmarks")
    plt.show()

    return warped, conf_vals


if __name__ == "__main__":
    img_path = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
    diffusion_ckpt = "/content/drive/MyDrive/atlas_checkpoints/diffusion/diffusion_epoch_5.pth"
    svf_ckpt = "/content/drive/MyDrive/atlas_checkpoints/svf/svf_epoch_29.pth"
    atlas_lm = "/content/landmark-detection/atlas_landmarks_19.npy"

    run_atlas_flow(img_path, diffusion_ckpt, svf_ckpt, atlas_lm)
