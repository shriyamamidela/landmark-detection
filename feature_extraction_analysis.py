#!/usr/bin/env python3
"""
Feature Extraction Analysis (HRNet R-branches only)
- Loads one ISBI test image
- Runs Backbone (HRNet recommended)
- Visualizes R4/R8/R16/R32
- Builds SFB on R8/R16/R32 and visualizes P3/P4/P5
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import argparse

from models.backbone import Backbone      # <— use backbone directly (lighter than Network)
from network.semantic_fusion_block import SemanticFusionBlock
from config import cfg


# ------------------------------ IO helpers ------------------------------
def load_isbi_image_and_landmarks(image_path, annotation_path):
    """Load ISBI image and corresponding landmarks (if file exists)."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks = None
    if os.path.isfile(annotation_path):
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        pts = []
        for i in range(min(19, len(lines))):
            xy = lines[i].strip().split(',')
            if len(xy) == 2:
                pts.append([float(xy[0]), float(xy[1])])
        landmarks = np.array(pts, dtype=np.float32) if pts else None

    return image, landmarks


# ------------------------------ Viz utils ------------------------------
# defaults (overridable by CLI)
MAX_FEATURES = 8
NORMALIZE = 'percentile'   # 'percentile' | 'zscore' | 'minmax'
PCLIP_LOW, PCLIP_HIGH = 2.0, 98.0
INTERP = 'nearest'
UPSAMPLE = True
SHARPEN = 0.2
CMAP = 'gray'
TOPK_BY = 'l2'             # 'l2' | 'l1' | 'mean'


def _unsharp(image, amount=0.0):
    if amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 1)


def visualize_feature_maps(feature_maps, title, save_path=None, max_features=16,
                           upsample_to=None, normalize='minmax', cmap='gray',
                           topk_by='l2', interp='nearest', sharpen=0.0, also_save_composite=False):
    """Grid visualization of top-k channels of a tensor (B,C,H,W) or (C,H,W)."""
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()

    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # (C,H,W)
    if feature_maps.ndim != 3:
        print(f"[WARN] Cannot visualize feature maps with shape: {feature_maps.shape}")
        return

    C, H, W = feature_maps.shape

    # rank channels
    if topk_by == 'l2':
        scores = (feature_maps ** 2).mean(axis=(1, 2))
    elif topk_by == 'l1':
        scores = np.abs(feature_maps).mean(axis=(1, 2))
    else:
        scores = feature_maps.mean(axis=(1, 2))
    idx = np.argsort(-scores)[:min(C, max_features)]
    feature_maps = feature_maps[idx]
    num_features = feature_maps.shape[0]

    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_features):
        r, c = i // cols, i % cols
        fm = feature_maps[i]

        # normalize
        if normalize == 'zscore':
            mu, sigma = fm.mean(), fm.std() + 1e-8
            fm = (fm - mu) / sigma
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        elif normalize == 'percentile':
            lo, hi = np.percentile(fm, PCLIP_LOW), np.percentile(fm, PCLIP_HIGH)
            fm = np.clip((fm - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)

        # upsample
        if upsample_to is not None:
            Ht, Wt = upsample_to
            inter = cv2.INTER_NEAREST if interp == 'nearest' else cv2.INTER_CUBIC
            fm = cv2.resize(fm, (Wt, Ht), interpolation=inter)

        fm = _unsharp(fm, amount=sharpen)
        axes[r, c].imshow(fm, cmap=cmap)
        axes[r, c].set_title(f'Feature {i+1}')
        axes[r, c].axis('off')

    # hide empties
    for i in range(num_features, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')

    plt.suptitle(f"{title}\nShape: {feature_maps.shape}", fontsize=16, weight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps saved to: {save_path}")
    plt.close(fig)


# ------------------------------ Main analysis ------------------------------
def analyze_feature_extraction(backbone_name: str = "hrnet_w32", pretrained_weights: str = None):
    print("Starting Feature Extraction Analysis (R-branches only)...")
    print("=" * 60)

    device = cfg.DEVICE
    print(f"Using device: {device}")

    # dataset paths
    dataset_base = "datasets/ISBI Dataset"
    images_path = f"{dataset_base}/Dataset/Testing/Test1"
    annotations_path = f"{dataset_base}/Annotations/Junior Orthodontist"

    if not os.path.exists(images_path):
        print(f"[ERROR] Images not found at {images_path}")
        return

    image_files = glob.glob(os.path.join(images_path, "*.bmp"))
    if not image_files:
        print(f"[ERROR] No .bmp images found in {images_path}")
        return

    test_image = image_files[0]
    image_name = os.path.splitext(os.path.basename(test_image))[0]
    annotation_file = os.path.join(annotations_path, f"{image_name}.txt")

    print(f"Using image: {image_name}.bmp")

    # load image
    print("\nLOADING INPUT IMAGE")
    try:
        image, _ = load_isbi_image_and_landmarks(test_image, annotation_file)
        print(f"Original image: {image.shape} (H,W,C) -> ({image.shape[0]},{image.shape[1]},{image.shape[2]})")

        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(f"Original ISBI Image: {image_name}\nShape: {image.shape}")
        plt.axis('off')
        plt.savefig("outputs/original_image.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: outputs/original_image.png")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # preprocess
    print("\nIMAGE PREPROCESSING")
    print(f"Network input size: {cfg.WIDTH} x {cfg.HEIGHT}")
    resized = cv2.resize(image, (cfg.WIDTH, cfg.HEIGHT))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    print(f"Tensor: {tensor.shape}")

    plt.figure(figsize=(10, 8))
    plt.imshow(resized)
    plt.title(f"Resized Image for Network Input\nShape: {resized.shape}")
    plt.axis('off')
    plt.savefig("outputs/resized_image.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/resized_image.png")

    # backbone (HRNet recommended)
    print("\nCREATING BACKBONE")
    weights = None
    if pretrained_weights and os.path.exists(pretrained_weights):
        weights = pretrained_weights
        print(f"Using provided weights: {weights}")
    elif backbone_name in ["hrnet_w32", "hrnet_w48"]:
        default_path = f"pretrained_weights/{backbone_name}_imagenet.pth"
        if os.path.exists(default_path):
            weights = default_path
            print(f"Using default pretrained weights: {weights}")
        else:
            print("[WARN] No pretrained weights found. Using random init (features will be weak).")

    backbone = Backbone(name=backbone_name, pretrained=bool(weights), weights_root_path=weights).to(device)
    backbone.eval()
    print("Backbone created.")

    # run backbone
    print("\nBACKBONE FEATURE EXTRACTION (R-branches)")
    with torch.no_grad():
        _ = backbone(tensor)  # forward; HRNet returns R4 by default
        feats = backbone.get_feature_dict() or {}

        for rkey in ["R4", "R8", "R16", "R32"]:
            if rkey in feats:
                fm = feats[rkey]
                print(f"  - {rkey}: {fm.shape}")
                save_path = f"outputs/backbone_{rkey}_features.png"
                visualize_feature_maps(
                    fm, f"Backbone {rkey} Features",
                    save_path=save_path,
                    max_features=MAX_FEATURES,
                    upsample_to=(tensor.shape[2], tensor.shape[3]) if UPSAMPLE else None,
                    normalize=NORMALIZE, cmap=CMAP, topk_by=TOPK_BY,
                    interp=INTERP, sharpen=SHARPEN, also_save_composite=False
                )

    # build SFB on R8/R16/R32
    print("\nSEMANTIC FUSION BLOCK (SFB) on R8/R16/R32")
    R8, R16, R32 = feats.get("R8"), feats.get("R16"), feats.get("R32")
    if R8 is None or R16 is None or R32 is None:
        print("[WARN] Missing R-branches; cannot run SFB.")
    else:
        in_channels = (R8.shape[1], R16.shape[1], R32.shape[1])
        sfb = SemanticFusionBlock(num_filters=256, in_channels=in_channels, name="sfb_r8r16r32").to(device)
        sfb.eval()
        with torch.no_grad():
            P3, P4, P5 = sfb([R8, R16, R32])
        for name, fm in [("P3", P3), ("P4", P4), ("P5", P5)]:
            print(f"  - {name}: {fm.shape}")
            save_path = f"outputs/sfb_{name}_output.png"
            visualize_feature_maps(
                fm, f"SFB Output {name}",
                save_path=save_path,
                max_features=MAX_FEATURES,
                upsample_to=(tensor.shape[2], tensor.shape[3]) if UPSAMPLE else None,
                normalize=NORMALIZE, cmap=CMAP, topk_by=TOPK_BY,
                interp=INTERP, sharpen=SHARPEN, also_save_composite=False
            )

    print("\nDone. Check the 'outputs' folder for visualizations.")


# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="hrnet_w32",
                        choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19", "hrnet_w32", "hrnet_w48"],
                        help="Backbone name")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to backbone weights (e.g., pretrained HRNet ImageNet)")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--normalize", type=str, default=NORMALIZE, choices=["percentile", "zscore", "minmax"])
    parser.add_argument("--percentiles", type=float, nargs=2, default=[PCLIP_LOW, PCLIP_HIGH])
    parser.add_argument("--interp", type=str, default=INTERP, choices=["nearest", "cubic"])
    parser.add_argument("--no-upsample", action="store_true")
    parser.add_argument("--sharpen", type=float, default=SHARPEN)
    parser.add_argument("--cmap", type=str, default=CMAP)
    parser.add_argument("--topk-by", type=str, default=TOPK_BY, choices=["l2", "l1", "mean"])
    args = parser.parse_args()

    MAX_FEATURES = args.max_features
    NORMALIZE = args.normalize
    PCLIP_LOW, PCLIP_HIGH = args.percentiles
    INTERP = args.interp
    UPSAMPLE = not args.no_upsample
    SHARPEN = args.sharpen
    CMAP = args.cmap
    TOPK_BY = args.topk_by

    analyze_feature_extraction(backbone_name=args.backbone, pretrained_weights=args.pretrained)
