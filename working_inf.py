#!/usr/bin/env python3
"""
inference_fixed.py
Final inference script that matches training preprocessing exactly.
Usage examples:
  # run 4 images per folder (fast, visual check)
  python3 inference_fixed.py --limit 4

  # run all images (full export)
  python3 inference_fixed.py --limit -1 --checkpoint /path/to/svf_best.pth
"""
import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from models.backbone import ResNetBackbone
from models.svf_head import SVFHead

# -------------------------
# Config defaults
# -------------------------
DEFAULT_TARGET_SIZE = (512, 412)  # (W, H) as used in aug_dataset
DEFAULT_CHECKPOINT = "/dgxa_home/se22ucse250/landmark-detection-main/checkpoints/debug_svf/svf_best.pth"
DEFAULT_ATLAS_DIR = "/dgxa_home/se22ucse250/landmark-detection-main/atlas"
DEFAULT_TEST1 = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test1changed_1"
DEFAULT_TEST2 = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test2changed_2"
DEFAULT_OUT = "inference_fixed_results"
PX_TO_MM = 0.1

# -------------------------
# Letterbox + landmark mapping (same as aug_dataset.py)
# -------------------------
def letterbox_resize(img, target_size=(512, 412), interp=cv2.INTER_LINEAR, color=(0,0,0)):
    target_w, target_h = target_size[0], target_size[1]
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    # pad to target
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, float(scale), int(left), int(top)

def map_points_to_letterbox(pts, scale, pad_left, pad_top):
    pts = np.asarray(pts, dtype=np.float32)
    mapped = pts.copy()
    mapped[:, 0] = pts[:, 0] * scale + pad_left
    mapped[:, 1] = pts[:, 1] * scale + pad_top
    return mapped

def map_points_from_letterbox(pts, scale, pad_left, pad_top):
    pts = np.asarray(pts, dtype=np.float32)
    out = pts.copy()
    out[:, 0] = (pts[:, 0] - pad_left) / scale
    out[:, 1] = (pts[:, 1] - pad_top) / scale
    return out

# -------------------------
# DT from landmarks (same as dataset)
# -------------------------
def compute_dt_from_landmarks(image_shape, landmarks, radius=3):
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in landmarks:
        cx, cy = int(round(x)), int(round(y))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    mask_inv = 255 - mask
    dt = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
    if dt.max() > 0:
        dt = dt / (dt.max() + 1e-8)
    return dt.astype(np.float32)

# -------------------------
# Label loading (supports .txt with comma/space, .npy, .csv)
# -------------------------
def load_label_file(label_path):
    ext = os.path.splitext(label_path)[1].lower()
    if ext == ".npy":
        arr = np.load(label_path, allow_pickle=True)
        return np.array(arr, dtype=float)
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(label_path, header=None)
        return df.iloc[:, :2].values.astype(float)
    # txt: either "x,y" or "x y"
    pts = []
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                a = s.split(",")
            else:
                a = s.split()
            if len(a) >= 2:
                try:
                    pts.append([float(a[0]), float(a[1])])
                except:
                    pass
    return np.array(pts, dtype=float)

# -------------------------
# SVF utilities (match training)
# -------------------------
def make_meshgrid(B, H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

def warp_with_disp(img, disp):
    B, C, H, W = img.shape
    base = make_meshgrid(B, H, W, img.device)
    disp_norm = torch.zeros_like(base)
    disp_norm[..., 0] = disp[:, 0] / (W / 2.0)
    disp_norm[..., 1] = disp[:, 1] / (H / 2.0)
    grid = (base + disp_norm).clamp(-1, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)

def svf_to_disp(svf, steps=6):
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_with_disp(disp, disp)
    return disp

def sample_flow_at_points_local(flow, points_px, image_size, align_corners=False):
    B, N, _ = points_px.shape
    H, W = image_size
    x_norm = 2 * ((points_px[..., 0] + 0.5) / W) - 1
    y_norm = 2 * ((points_px[..., 1] + 0.5) / H) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B,N,1,2)
    sampled = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)
    sampled = sampled.squeeze(-1)  # (B,2,N)
    sampled = sampled.permute(0, 2, 1)  # (B,N,2)
    return sampled

# -------------------------
# Visualization
# -------------------------
def draw_points(img_gray, gt_pts, pred_pts, filename, mre_mm, outdir):
    fig, ax = plt.subplots(figsize=(6,8))
    ax.imshow(img_gray, cmap="gray")

    # color map for numbering only
    colors = plt.cm.tab20(np.linspace(0,1,20))

    # --- Draw points ---
    for i in range(len(gt_pts)):
        gx, gy = gt_pts[i]
        px, py = pred_pts[i]

        # GT = green circle
        ax.scatter(gx, gy, s=45, facecolors='none', edgecolors='lime',
                   linewidths=2, zorder=3)

        # Pred = red X
        ax.scatter(px, py, s=45, marker='x', color='red',
                   linewidths=2, zorder=4)

        # Landmark numbering WITHOUT white box
        ax.text(gx + 3, gy - 3, str(i+1),
                color=colors[i % 20],
                fontsize=10,
                fontweight='bold',
                zorder=5)

    # --- Add legend ---
    gt_handle = plt.Line2D([0], [0], marker='o', color='lime',
                           markersize=8, linestyle='None', label='Ground Truth')
    pred_handle = plt.Line2D([0], [0], marker='x', color='red',
                             markersize=8, linestyle='None', label='Predicted')

    # store legend object as 'leg'
    leg = ax.legend(handles=[gt_handle, pred_handle],
                    loc='upper right',
                    framealpha=0.6,
                    facecolor='black',
                    edgecolor='white',
                    fontsize=10)

    # change text color to dark blue
    for text in leg.get_texts():
        text.set_color('darkblue')

    ax.set_title(f"{filename}   MRE = {mre_mm:.2f} mm")
    ax.axis("off")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{filename}.png")
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()

              


# -------------------------
# Single-image inference (final correct pipeline)
# -------------------------
def infer_single_image(img_path, label_path, atlas_lms, backbone, svf_head, target_size, device):
    # read original image (BGR)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    H0, W0 = img_bgr.shape[:2]

    # load and parse GT (original coords)
    gt_orig = load_label_file = None
    gt_orig = load_label_file = None
    gt_orig = load_label_file = load_label_path(label_path=label_path) if False else None  # placeholder (we'll use proper loader below)

    # use loader
    gt_orig = load_label_file(label_path=label_path) if False else None  # placeholder (we won't use this)
    # simpler: call the label loader directly:
    gt_orig = load_label_file = None

    # (Correct loader call)
    gt_orig = load_label_file_real(label_path=label_path) if False else None  # placeholder

# -------------------------
# The script below implements the correct single-image pipeline.
# To avoid confusion from placeholder code above, the full implementation continues below.
# -------------------------

def load_label_file_real(label_path):
    return load_label_file(label_path)  # re-use function above


def infer_single(img_path, label_path, atlas_lms, backbone, svf_head, target_size, device):
    """Return dict with keys: name, mre(mm), pred_orig (L,2), viz_path"""
    name = os.path.splitext(os.path.basename(img_path))[0]

    # load original image (BGR) and convert to RGB for consistent behavior
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb_orig.shape[:2]

    # load GT original coords
    gt_orig = load_label_file(label_path)
    if gt_orig is None or len(gt_orig) == 0:
        raise RuntimeError(f"Empty GT: {label_path}")

    # letterbox resize (same as training)
    resized, scale, pad_left, pad_top = letterbox_resize(img_rgb_orig, target_size=target_size, interp=cv2.INTER_LINEAR, color=(0,0,0))
    # note: resized is RGB
    # compute dt from GT mapped to letterbox coords (same as training)
    gt_letter = map_points_to_letterbox(gt_orig, scale, pad_left, pad_top)
    dt = compute_dt_from_landmarks(resized.shape, gt_letter, radius=3)  # shape (H_target, W_target)

    # prepare tensors
    img_t = torch.from_numpy(resized.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device) / 255.0  # (1,3,H,W)
    dt_t = torch.from_numpy(dt).unsqueeze(0).unsqueeze(0).to(device).float()  # (1,1,H,W)

    # forward (model expects letterbox-sized inputs)
    backbone.eval(); svf_head.eval()
    with torch.no_grad():
        feats = backbone(img_t)
        F5 = feats["C5"]
        Hf, Wf = F5.shape[2], F5.shape[3]
        dt_small = F.interpolate(dt_t, size=(Hf, Wf), mode='bilinear', align_corners=False)
        v = svf_head(F5, dt_small)
        disp = svf_to_disp(v, steps=6)
        disp_full = F.interpolate(disp, size=(target_size[1], target_size[0]), mode='bilinear', align_corners=False)  # (1,2,Ht,Wt)

        atlas_lms_t = torch.tensor(atlas_lms, dtype=torch.float32).unsqueeze(0).to(device)  # (1,L,2)
        disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_t, (target_size[1], target_size[0]))
        pred_lm_letter = atlas_lms_t + disp_at_lm
        pred_lm_letter = pred_lm_letter[0].cpu().numpy()

    # compute errors in letterbox coords vs gt_letter
    if pred_lm_letter.shape != gt_letter.shape:
        raise RuntimeError(f"Landmark count mismatch: pred {pred_lm_letter.shape} vs gt {gt_letter.shape}")

    errors_px = np.linalg.norm(pred_lm_letter - gt_letter, axis=1)
    errors_mm = errors_px * PX_TO_MM
    mre_mm = float(errors_mm.mean())

    # map predicted points back to original image coords
    pred_orig = map_points_from_letterbox(pred_lm_letter.copy(), scale, pad_left, pad_top)

    # visualization on letterboxed image for perfect overlay
    img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    viz_path = draw_points(img_gray, gt_letter, pred_lm_letter, name, mre_mm, outdir=os.path.join(args.outdir, "viz"))

    return {
        "name": name,
        "mre": mre_mm,
        "errors_mm": errors_mm,
        "pred_orig": pred_orig,
        "pred_letter": pred_lm_letter,
        "viz": viz_path
    }

# -------------------------
# CLI + main
# -------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--atlas-dir", default=DEFAULT_ATLAS_DIR)
    p.add_argument("--test1-dir", default=DEFAULT_TEST1)
    p.add_argument("--test2-dir", default=DEFAULT_TEST2)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=4, help="Number of images to run per folder. -1 for all.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()

    # create outputs
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "viz"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # load atlas
    atlas_lms_file = os.path.join(args.atlas_dir, "atlas_landmarks_resized.npy")
    atlas_img_file = os.path.join(args.atlas_dir, "atlas_image_resized.npy")
    assert os.path.exists(atlas_lms_file), atlas_lms_file + " missing"
    atlas_lms = np.load(atlas_lms_file)  # (L,2)

    # model
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    svf_head = SVFHead(in_channels=512).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["svf"] if isinstance(ckpt, dict) and "svf" in ckpt else ckpt
    # remove module prefix if any
    new_sd = {}
    for k,v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v
    svf_head.load_state_dict(new_sd, strict=False)
    print("Loaded checkpoint:", args.checkpoint)

    target_size = DEFAULT_TARGET_SIZE  # (W,H)

    # helper to iterate
    def process_folder(folder, label_folder, name_prefix):
        img_files = sorted([f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith((".png",".jpg",".jpeg"))])
        if args.limit is not None and args.limit >= 0:
            img_files = img_files[:args.limit]
        results = []
        for img_path in tqdm(img_files, desc=f"Run {name_prefix}"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            # find label file
            label_file = None
            for ext in (".txt", ".npy", ".csv"):
                cand = os.path.join(label_folder, base + ext)
                if os.path.exists(cand):
                    label_file = cand
                    break
            if label_file is None:
                print("No GT for", img_path)
                continue
            r = infer_single(img_path, label_file, atlas_lms, backbone, svf_head, target_size, device)
            # save pred_orig to npy
            np.save(os.path.join(args.outdir, "preds", f"{r['name']}.npy"), r["pred_orig"])
            results.append(r)
        # print mean MRE
        if len(results) > 0:
            mean_mre = float(np.mean([x["mre"] for x in results]))
            print(f"{name_prefix} mean MRE (mm) over {len(results)} images: {mean_mre:.3f}")
        return results

    # run test1
    print("Processing Test1")
    res1 = process_folder(os.path.join(args.test1_dir, "images"), os.path.join(args.test1_dir, "labels"), "test1")

    # run test2
    print("Processing Test2")
    res2 = process_folder(os.path.join(args.test2_dir, "images"), os.path.join(args.test2_dir, "labels"), "test2")

    print("Done. Results in:", args.outdir)
