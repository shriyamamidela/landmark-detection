#!/usr/bin/env python3
"""
Feature Extraction Analysis Script (HRNet-W48 Compatible)
Visualizes intermediate outputs from input image → HRNet backbone → Semantic Fusion Block (SFB)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from network.model import Network
from network.semantic_fusion_block import SemanticFusionBlock
from config import cfg
import argparse


# --------------------- #
# Load Image + Landmarks
# --------------------- #
def load_isbi_image_and_landmarks(image_path, annotation_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    landmarks = []
    for i in range(19):
        if i < len(lines):
            coords = lines[i].strip().split(',')
            if len(coords) == 2:
                x, y = float(coords[0]), float(coords[1])
                landmarks.append([x, y])
    return image, np.array(landmarks)


# --------------------- #
# Visualization defaults
# --------------------- #
MAX_FEATURES = 8
NORMALIZE = 'percentile'
PCLIP_LOW, PCLIP_HIGH = 2.0, 98.0
INTERP = 'cubic'
UPSAMPLE = True
SHARPEN = 0.5
CMAP = 'gray'
TOPK_BY = 'l2'


def _unsharp(image, amount=0.0):
    if amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 1)


# --------------------- #
# Feature Map Visualizer
# --------------------- #
def visualize_feature_maps(
    feature_maps,
    title,
    save_path=None,
    max_features=16,
    upsample_to=None,
    normalize='minmax',
    cmap='gray',
    topk_by='l2',
    interp='nearest',
    sharpen=0.0,
    also_save_composite=False
):
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()

    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]
    num_channels = feature_maps.shape[0]

    # Rank channels
    if topk_by == 'l2':
        scores = (feature_maps ** 2).mean(axis=(1, 2))
    elif topk_by == 'l1':
        scores = np.abs(feature_maps).mean(axis=(1, 2))
    else:
        scores = feature_maps.mean(axis=(1, 2))

    topk_indices = np.argsort(-scores)[:min(num_channels, max_features)]
    feature_maps = feature_maps[topk_indices]
    num_features = feature_maps.shape[0]

    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(num_features):
        r, c = i // cols, i % cols
        fmap = feature_maps[i]
        if normalize == 'zscore':
            mu, sigma = fmap.mean(), fmap.std() + 1e-8
            fmap = (fmap - mu) / sigma
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        elif normalize == 'percentile':
            lo, hi = np.percentile(fmap, PCLIP_LOW), np.percentile(fmap, PCLIP_HIGH)
            fmap = np.clip((fmap - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

        if upsample_to is not None:
            H, W = upsample_to
            inter = cv2.INTER_CUBIC if interp == 'cubic' else cv2.INTER_NEAREST
            fmap = cv2.resize(fmap, (W, H), interpolation=inter)

        fmap = _unsharp(fmap, amount=sharpen)
        axes[r, c].imshow(fmap, cmap=cmap)
        axes[r, c].set_title(f"Feature {i+1}")
        axes[r, c].axis("off")

    for j in range(num_features, rows * cols):
        r, c = j // cols, j % cols
        axes[r, c].axis("off")

    plt.suptitle(f"{title}\nShape: {feature_maps.shape}", fontsize=16, weight="bold")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()


# --------------------- #
# Main analysis function
# --------------------- #
def analyze_feature_extraction(backbone_name: str = "hrnet_w48", pretrained_weights: str = None):
    print("=" * 70)
    print("🔍 HRNet Feature Extraction Analysis (HRNet-W48)")
    print("=" * 70)

    device = cfg.DEVICE
    print(f"Using device: {device}")

    dataset_base = "datasets/ISBI Dataset"
    images_path = f"{dataset_base}/Dataset/Testing/Test1"
    annotations_path = f"{dataset_base}/Annotations/Junior Orthodontist"

    image_files = glob.glob(f"{images_path}/*.bmp")
    if not image_files:
        print("❌ No test images found.")
        return

    test_image = image_files[0]
    image_name = os.path.splitext(os.path.basename(test_image))[0]
    annotation_file = f"{annotations_path}/{image_name}.txt"

    image, landmarks = load_isbi_image_and_landmarks(test_image, annotation_file)
    resized_image = cv2.resize(image, (cfg.WIDTH, cfg.HEIGHT))

    # Preprocess input
    image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    image_tensor = image_tensor / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    # Create HRNet backbone
    print("\nCreating HRNet-W48 backbone...")
    backbone_weights_path = pretrained_weights if pretrained_weights and os.path.exists(pretrained_weights) else None
    if backbone_weights_path:
        print(f"Using pretrained weights: {backbone_weights_path}")
    else:
        print("⚠️ No pretrained weights found — using random initialization.")

    network = Network(
        backbone_name=backbone_name,
        freeze_backbone=False,
        backbone_weights=backbone_weights_path
    ).to(device)
    network.eval()

    with torch.no_grad():
        _ = network.backbone(image_tensor)
        feat_dict = network.backbone.get_feature_dict() if hasattr(network.backbone, "get_feature_dict") else {}

        if feat_dict:
            print("\n📊 Extracted HRNet multi-resolution features:")
            for rkey in ["R4", "R8", "R16", "R32"]:
                if rkey in feat_dict:
                    fmap = feat_dict[rkey]
                    print(f"   • {rkey}: {tuple(fmap.shape)}")
                    visualize_feature_maps(
                        fmap,
                        f"HRNet {rkey} Features",
                        save_path=f"outputs/hrnet_{rkey}_features.png",
                        max_features=MAX_FEATURES,
                        upsample_to=(image_tensor.shape[2], image_tensor.shape[3]) if UPSAMPLE else None,
                        normalize=NORMALIZE,
                        cmap=CMAP,
                        topk_by=TOPK_BY,
                        interp=INTERP,
                        sharpen=SHARPEN,
                        also_save_composite=True,
                    )

            # Semantic Fusion Block
            R8, R16, R32 = feat_dict.get("R8"), feat_dict.get("R16"), feat_dict.get("R32")
            if all(x is not None for x in [R8, R16, R32]):
                print("\n🚀 Passing through Semantic Fusion Block (SFB)...")
                sfb = SemanticFusionBlock(num_filters=256, in_channels=(R8.shape[1], R16.shape[1], R32.shape[1])).to(device)
                sfb.eval()
                P3, P4, P5 = sfb([R8, R16, R32])

                for name, fmap in zip(["P3", "P4", "P5"], [P3, P4, P5]):
                    visualize_feature_maps(
                        fmap,
                        f"SFB Output {name}",
                        save_path=f"outputs/sfb_{name}_output.png",
                        max_features=MAX_FEATURES,
                        upsample_to=(image_tensor.shape[2], image_tensor.shape[3]) if UPSAMPLE else None,
                        normalize=NORMALIZE,
                        cmap=CMAP,
                        topk_by=TOPK_BY,
                        interp=INTERP,
                        sharpen=SHARPEN,
                        also_save_composite=True,
                    )
        else:
            print("❌ No HRNet features extracted.")

    print("\n✅ Feature Extraction Analysis Completed!")
    print("Check the 'outputs/' directory for visualizations.")


# --------------------- #
# CLI entry point
# --------------------- #
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser(description="Feature Extraction Analysis (HRNet-W48)")
    parser.add_argument("--backbone", type=str, default="hrnet_w48", help="Backbone architecture")
    parser.add_argument("--pretrained", type=str, default="pretrained_weights/hrnet_w48_imagenet.pth", help="Path to pretrained weights")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--normalize", type=str, default=NORMALIZE)
    parser.add_argument("--interp", type=str, default=INTERP)
    parser.add_argument("--sharpen", type=float, default=SHARPEN)
    args = parser.parse_args()

    MAX_FEATURES = args.max_features
    NORMALIZE = args.normalize
    INTERP = args.interp
    SHARPEN = args.sharpen

    analyze_feature_extraction(backbone_name=args.backbone, pretrained_weights=args.pretrained)
