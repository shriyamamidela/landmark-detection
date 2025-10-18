#!/usr/bin/env python3
"""
Feature Extraction Analysis Script
Shows intermediate outputs from input image → backbone → Semantic Fusion Block (SFB)
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


def load_isbi_image_and_landmarks(image_path, annotation_path):
    """Load ISBI image and corresponding landmarks"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load landmarks from annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    # First 19 lines contain x,y coordinates
    landmarks = []
    for i in range(19):
        if i < len(lines):
            coords = lines[i].strip().split(',')
            if len(coords) == 2:
                x, y = float(coords[0]), float(coords[1])
                landmarks.append([x, y])
    landmarks = np.array(landmarks)
    
    return image, landmarks


""" Global visualization defaults (can be overridden later if CLI is added) """
# number of feature channels to display
MAX_FEATURES = 8
# normalization mode: 'percentile' | 'zscore' | 'minmax'
NORMALIZE = 'percentile'
# low/high percentiles for percentile normalization
PCLIP_LOW, PCLIP_HIGH = 2.0, 98.0
# upsample interpolation: 'nearest' | 'cubic'
INTERP = 'nearest'
# upsample to input size?
UPSAMPLE = True
# unsharp mask amount (0 disables)
SHARPEN = 0.2
# matplotlib colormap
CMAP = 'gray'
# ranking criterion for top-k channels: 'l2' | 'l1' | 'mean'
TOPK_BY = 'l2'


def _unsharp(image, amount=0.0):
    if amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 1)


def _composite_maps(feature_maps_np, method='mean'):
    # feature_maps_np: (C,H,W) numpy array in [0,1]
    if feature_maps_np.ndim != 3:
        return None
    if method == 'max':
        comp = feature_maps_np.max(axis=0)
    else:
        comp = feature_maps_np.mean(axis=0)
    return np.clip(comp, 0, 1)


def visualize_feature_maps(feature_maps, title, save_path=None, max_features=16, upsample_to=None, normalize='minmax', cmap='gray', topk_by='l2', interp='nearest', sharpen=0.0, also_save_composite=False):
    """Visualize feature maps with better clarity.
    - Select top-k channels by activation energy.
    - Optionally upsample to a target spatial size for clearer viewing.
    - Use grayscale by default for anatomy-like visualization.
    """
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # Get dimensions
    if len(feature_maps.shape) == 4:  # (batch, channels, height, width)
        batch_size, num_channels, height, width = feature_maps.shape
        feature_maps = feature_maps[0]  # Take first batch
    elif len(feature_maps.shape) == 3:  # (channels, height, width)
        num_channels, height, width = feature_maps.shape
    else:
        print(f"Cannot visualize feature maps with shape: {feature_maps.shape}")
        return

    # Rank channels by activation energy
    if topk_by == 'l2':
        scores = (feature_maps**2).mean(axis=(1, 2))
    elif topk_by == 'l1':
        scores = np.abs(feature_maps).mean(axis=(1, 2))
    else:
        scores = feature_maps.mean(axis=(1, 2))
    topk_indices = np.argsort(-scores)[:min(num_channels, max_features)]
    feature_maps = feature_maps[topk_indices]
    num_features = feature_maps.shape[0]

    # Create subplot grid
    cols = 4
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each feature map
    for i in range(num_features):
        row = i // cols
        col = i % cols
        
        feature_map = feature_maps[i]
        
        # Normalize for visualization
        if normalize == 'zscore':
            mu, sigma = feature_map.mean(), feature_map.std() + 1e-8
            feature_map = (feature_map - mu) / sigma
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        elif normalize == 'percentile':
            lo, hi = np.percentile(feature_map, PCLIP_LOW), np.percentile(feature_map, PCLIP_HIGH)
            feature_map = np.clip((feature_map - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

        # Optional upsample
        if upsample_to is not None:
            H, W = upsample_to
            inter = cv2.INTER_NEAREST if interp == 'nearest' else cv2.INTER_CUBIC
            feature_map = cv2.resize(feature_map, (W, H), interpolation=inter)

        # Optional sharpen
        feature_map = _unsharp(feature_map, amount=sharpen)
        
        axes[row, col].imshow(feature_map, cmap=cmap)
        axes[row, col].set_title(f'Feature {i+1}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_features, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"{title}\nShape: {feature_maps.shape}", fontsize=16, weight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps saved to: {save_path}")
    
    # Also save composite maps (mean and max across selected channels)
    if also_save_composite and save_path:
        base = save_path.rsplit('.', 1)[0]
        comp_mean = feature_maps.mean(axis=0)
        comp_max = feature_maps.max(axis=0)
        # upsampling and sharpening composites similar to channels
        if upsample_to is not None:
            H, W = upsample_to
            inter = cv2.INTER_NEAREST if interp == 'nearest' else cv2.INTER_CUBIC
            comp_mean = cv2.resize(comp_mean, (W, H), interpolation=inter)
            comp_max = cv2.resize(comp_max, (W, H), interpolation=inter)
        comp_mean = _unsharp(comp_mean, amount=sharpen)
        comp_max = _unsharp(comp_max, amount=sharpen)
        for name, comp in [("mean", comp_mean), ("max", comp_max)]:
            plt.figure(figsize=(6, 5))
            plt.imshow(comp, cmap=cmap)
            plt.axis('off')
            out_path = f"{base}_composite_{name}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Composite map saved to: {out_path}")
            plt.close()
    return fig


def analyze_feature_extraction(backbone_name: str = "resnet50", pretrained_weights: str = None):
    """Analyze the complete feature extraction pipeline"""
    print("Starting Feature Extraction Analysis...")
    print("=" * 60)
    
    # Set device
    device = cfg.DEVICE
    print(f"Using device: {device}")
    
    # Find ISBI dataset paths
    dataset_base = "datasets/ISBI Dataset"
    images_path = f"{dataset_base}/Dataset/Testing/Test1"
    annotations_path = f"{dataset_base}/Annotations/Junior Orthodontist"
    
    if not os.path.exists(images_path):
        print(f"Images not found at {images_path}")
        return
    
    # Get first available image
    image_files = glob.glob(f"{images_path}/*.bmp")
    if not image_files:
        print(f"No .bmp images found in {images_path}")
        return
    
    test_image = image_files[0]
    image_name = os.path.splitext(os.path.basename(test_image))[0]
    annotation_file = f"{annotations_path}/{image_name}.txt"
    
    print(f"Using image: {image_name}.bmp")
    
    # Load image
    print("\nLOADING INPUT IMAGE")
    print("-" * 30)
    try:
        image, landmarks = load_isbi_image_and_landmarks(test_image, annotation_file)
        print(f"Original image loaded: {image.shape}")
        print(f"   - Width: {image.shape[1]} pixels")
        print(f"   - Height: {image.shape[0]} pixels")
        print(f"   - Channels: {image.shape[2]} (RGB)")
        
        # Show original image
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(f"Original ISBI Image: {image_name}\nShape: {image.shape}")
        plt.axis('off')
        plt.savefig("outputs/original_image.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Original image saved to: outputs/original_image.png")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Resize image for network input
    print(f"\nIMAGE PREPROCESSING")
    print("-" * 30)
    original_height, original_width = image.shape[:2]
    print(f"Original dimensions: {original_width} x {original_height}")
    print(f"Network input size: {cfg.WIDTH} x {cfg.HEIGHT}")
    
    # Resize image
    resized_image = cv2.resize(image, (cfg.WIDTH, cfg.HEIGHT))
    print(f"Resized image: {resized_image.shape}")
    
    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    print(f"PyTorch tensor: {image_tensor.shape}")
    
    # Show resized image
    plt.figure(figsize=(10, 8))
    plt.imshow(resized_image)
    plt.title(f"Resized Image for Network Input\nShape: {resized_image.shape}")
    plt.axis('off')
    plt.savefig("outputs/resized_image.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Resized image saved to: outputs/resized_image.png")
    
    # Create network
    print(f"\nCREATING NETWORK")
    print("-" * 30)
    
    # Check for pretrained weights
    backbone_weights_path = None
    if pretrained_weights and os.path.exists(pretrained_weights):
        backbone_weights_path = pretrained_weights
        print(f"Loading pretrained weights from: {pretrained_weights}")
    elif backbone_name in ["hrnet_w32", "hrnet_w48"]:
        # Try default path
        default_path = f"pretrained_weights/{backbone_name}_imagenet.pth"
        if os.path.exists(default_path):
            backbone_weights_path = default_path
            print(f"Loading pretrained weights from: {default_path}")
        else:
            print(f"No pretrained weights found at {default_path}")
            print(f"Using random initialization (features will be less clear)")
    
    network = Network(
        backbone_name=backbone_name,
        freeze_backbone=False,
        backbone_weights=backbone_weights_path
    ).to(device)
    print(f"Network created with {sum(p.numel() for p in network.parameters()):,} parameters")
    
    # Run backbone only to get intermediate features
    print(f"\nBACKBONE FEATURE EXTRACTION")
    print("-" * 30)
    network.eval()
    
    with torch.no_grad():
        # Forward pass through backbone
        backbone_output = network.backbone(image_tensor)
        if isinstance(backbone_output, torch.Tensor):
            print(f"Backbone output: {backbone_output.shape}")

        # Get feature dict
        hrnet_feat_dict = None
        if hasattr(network.backbone, 'get_feature_dict'):
            hrnet_feat_dict = network.backbone.get_feature_dict()

        feature_source = hrnet_feat_dict if hrnet_feat_dict else getattr(network, 'backbone_features', {})

        if feature_source:
            print(f"Intermediate features captured:")
            # Visualize C3, C4, C5
            for key in ["C3", "C4", "C5"]:
                if key in feature_source:
                    feature = feature_source[key]
                    print(f"   - {key}: {feature.shape}")
                    save_path = f"outputs/backbone_{key}_features.png"
                    visualize_feature_maps(
                        feature,
                        f"Backbone {key} Features",
                        save_path=save_path,
                        max_features=MAX_FEATURES,
                        upsample_to=(image_tensor.shape[2], image_tensor.shape[3]) if UPSAMPLE else None,
                        normalize=NORMALIZE,
                        cmap=CMAP,
                        topk_by=TOPK_BY,
                        interp=INTERP,
                        sharpen=SHARPEN,
                        also_save_composite=True
                    )
                    plt.close()

            # Visualize HRNet multi-resolution branches
            for rkey in ["R4", "R8", "R16", "R32"]:
                if rkey in feature_source:
                    feature = feature_source[rkey]
                    print(f"   - {rkey}: {feature.shape}")
                    save_path = f"outputs/backbone_{rkey}_features.png"
                    visualize_feature_maps(
                        feature,
                        f"Backbone {rkey} (HRNet) Features",
                        save_path=save_path,
                        max_features=MAX_FEATURES,
                        upsample_to=(image_tensor.shape[2], image_tensor.shape[3]) if UPSAMPLE else None,
                        normalize=NORMALIZE,
                        cmap=CMAP,
                        topk_by=TOPK_BY,
                        interp=INTERP,
                        sharpen=SHARPEN,
                        also_save_composite=True
                    )
                    plt.close()
        else:
            print("No intermediate features captured")
    
    # Create Semantic Fusion Block
    print(f"\nSEMANTIC FUSION BLOCK (SFB)")
    print("-" * 30)
    
    feat_dict = None
    if hasattr(network.backbone, 'get_feature_dict'):
        feat_dict = network.backbone.get_feature_dict()
    if not feat_dict:
        feat_dict = getattr(network, 'backbone_features', {})
    
    if feat_dict:
        C3 = feat_dict.get("C3")
        C4 = feat_dict.get("C4")
        C5 = feat_dict.get("C5")
        
        if C3 is not None and C4 is not None and C5 is not None:
            print(f"Input features to SFB:")
            print(f"   - C3: {C3.shape}")
            print(f"   - C4: {C4.shape}")
            print(f"   - C5: {C5.shape}")
            
            # Create SFB
            in_channels = (C3.shape[1], C4.shape[1], C5.shape[1])
            sfb = SemanticFusionBlock(num_filters=256, in_channels=in_channels, name="test_sfb")
            sfb.to(device)
            sfb.eval()
            
            # Forward pass through SFB
            P3, P4, P5 = sfb([C3, C4, C5])
            
            print(f"SFB output features:")
            print(f"   - P3: {P3.shape}")
            print(f"   - P4: {P4.shape}")
            print(f"   - P5: {P5.shape}")
            
            # Save SFB output visualizations
            for name, feature in [("P3", P3), ("P4", P4), ("P5", P5)]:
                save_path = f"outputs/sfb_{name}_output.png"
                visualize_feature_maps(
                    feature,
                    f"SFB Output {name}",
                    save_path=save_path,
                    max_features=MAX_FEATURES,
                    upsample_to=(image_tensor.shape[2], image_tensor.shape[3]) if UPSAMPLE else None,
                    normalize=NORMALIZE,
                    cmap=CMAP,
                    topk_by=TOPK_BY,
                    interp=INTERP,
                    sharpen=SHARPEN,
                    also_save_composite=True
                )
                plt.close()
            
            # Show feature map comparison
            print(f"\nFEATURE MAP COMPARISON")
            print("-" * 30)
            print(f"Feature map dimensions throughout the pipeline:")
            print(f"   Input Image: {image_tensor.shape}")
            print(f"   Backbone C3: {C3.shape}")
            print(f"   Backbone C4: {C4.shape}")
            print(f"   Backbone C5: {C5.shape}")
            print(f"   SFB P3: {P3.shape}")
            print(f"   SFB P4: {P4.shape}")
            print(f"   SFB P5: {P5.shape}")
            
        else:
            print("Missing intermediate features for SFB")
    else:
        print("No backbone features available")
    
    print(f"\nFeature Extraction Analysis Completed!")
    print(f"Check the 'outputs' folder for all visualizations")
    
    # Create summary visualization
    create_summary_visualization(image_name, resized_image, backbone_name=backbone_name)


def create_summary_visualization(image_name, resized_image, backbone_name: str = "resnet50"):
    """Create a summary visualization showing the complete pipeline"""
    print(f"\nCreating summary visualization...")
    
    # Create a large figure showing the pipeline
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Input and intermediate features
    axes[0, 0].imshow(resized_image)
    axes[0, 0].set_title(f"Input Image\n{resized_image.shape}")
    axes[0, 0].axis('off')
    
    # Load and show feature maps
    feature_files = [
        "outputs/backbone_C3_features.png",
        "outputs/backbone_C4_features.png", 
        "outputs/backbone_C5_features.png"
    ]
    
    for i, feature_file in enumerate(feature_files):
        if os.path.exists(feature_file):
            feature_img = plt.imread(feature_file)
            axes[0, i+1].imshow(feature_img)
            axes[0, i+1].set_title(f"Backbone C{i+3}")
            axes[0, i+1].axis('off')
        else:
            axes[0, i+1].text(0.5, 0.5, f"C{i+3} Features\nNot Available", 
                             ha='center', va='center', transform=axes[0, i+1].transAxes)
            axes[0, i+1].set_title(f"Backbone C{i+3}")
    
    # Row 2: SFB outputs
    sfb_files = [
        "outputs/sfb_P3_output.png",
        "outputs/sfb_P4_output.png",
        "outputs/sfb_P5_output.png"
    ]
    
    for i, sfb_file in enumerate(sfb_files):
        if os.path.exists(sfb_file):
            sfb_img = plt.imread(sfb_file)
            axes[1, i].imshow(sfb_img)
            axes[1, i].set_title(f"SFB P{i+3} Output")
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, f"P{i+3} Output\nNot Available", 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f"SFB P{i+3} Output")
    
    # Pipeline diagram
    pipeline_backbone = backbone_name.upper()
    axes[1, 3].text(0.5, 0.5, f"Feature Extraction Pipeline:\n\nInput → {pipeline_backbone} → C3,C4,C5 → SFB → P3,P4,P5", 
                    ha='center', va='center', transform=axes[1, 3].transAxes, 
                    fontsize=14, weight='bold')
    axes[1, 3].set_title("Pipeline Overview")
    axes[1, 3].axis('off')
    
    plt.suptitle(f"CEPHMark-Net Feature Extraction Analysis\nImage: {image_name}", fontsize=20, weight='bold')
    plt.tight_layout()
    
    # Save summary
    summary_path = "outputs/feature_extraction_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary visualization saved to: {summary_path}")
    plt.close()


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run analysis
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19", "hrnet_w32", "hrnet_w48"], help="Backbone name")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights file")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES, help="number of channels to visualize per grid")
    parser.add_argument("--normalize", type=str, default=NORMALIZE, choices=["percentile", "zscore", "minmax"], help="normalization mode")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[PCLIP_LOW, PCLIP_HIGH], help="low high percentiles for percentile normalization")
    parser.add_argument("--interp", type=str, default=INTERP, choices=["nearest", "cubic"], help="upsample interpolation")
    parser.add_argument("--no-upsample", action="store_true", help="disable upsampling to input size")
    parser.add_argument("--sharpen", type=float, default=SHARPEN, help="unsharp mask amount (0 disables)")
    parser.add_argument("--cmap", type=str, default=CMAP, help="matplotlib colormap")
    parser.add_argument("--topk-by", type=str, default=TOPK_BY, choices=["l2", "l1", "mean"], help="ranking metric for channels")
    args = parser.parse_args()

    # Bind CLI to globals
    MAX_FEATURES = args.max_features
    NORMALIZE = args.normalize
    PCLIP_LOW, PCLIP_HIGH = args.percentiles
    INTERP = args.interp
    UPSAMPLE = not args.no_upsample
    SHARPEN = args.sharpen
    CMAP = args.cmap
    TOPK_BY = args.topk_by

    analyze_feature_extraction(backbone_name=args.backbone, pretrained_weights=args.pretrained)