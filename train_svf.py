#!/usr/bin/env python3
"""
train_svf.py — SVF head training adapted for your augmented dataset (Option A).

- Uses AugCephDataset from /dgxa_home/se22ucse250/landmark-detection-main/datasets/augmented_ceph
- Loads atlas from provided paths
- Evaluates MRE (mm) and SDR@2/2.5/3/4 mm for train and test sets each epoch
- Saves checkpoints and visualization images per epoch

Note: set mm_per_pixel according to your pixel-mm conversion (default 0.1 mm/pixel).
"""
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Repo imports (assumes repo root in PYTHONPATH)
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from network.token_encoder import TokenEncoder  # optional
from losses import huber_landmark_loss, jacobian_regularizer, inverse_consistency_loss
from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens  # optional
from config import cfg
from models.svf_utils import (
    svf_to_disp,
    warp_with_disp,
    sample_flow_at_points_local
)


# ---------------------------
# Augmented training dataset (expects image_dir, label_dir, token_dir)
# ---------------------------
class AugCephDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.img_dir = os.path.join(root, "image_dir")
        self.lbl_dir = os.path.join(root, "label_dir")
        self.tok_dir = os.path.join(root, "token_dir")
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg"))])
        print(f"📦 AugCephDataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def _load_pts(self, path):
        pts = []
        with open(path, "r") as f:
            for ln in f:
                x, y = ln.strip().split(",")
                pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)

    def _compute_dt(self, im_shape, landmarks, radius=3):
        H, W = im_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for (x, y) in landmarks:
            cx, cy = int(x), int(y)
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask_inv = 255 - mask
        dt = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
        if dt.max() > 0:
            dt = dt / dt.max()
        return dt.astype(np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0

        lbl_path = os.path.join(self.lbl_dir, fname.replace(".png", ".txt").replace(".jpg", ".txt"))
        landmarks = self._load_pts(lbl_path)
        landmarks_t = torch.from_numpy(landmarks).float()  # (L,2)

        dt_np = self._compute_dt(img_rgb.shape, landmarks)
        dt_t = torch.from_numpy(dt_np).unsqueeze(0).float()  # (1,H,W)

        edge = cv2.Canny(img_bgr, 80, 160).astype(np.float32) / 255.0
        edge_t = torch.from_numpy(edge).unsqueeze(0).float()

        tok_path = os.path.join(self.tok_dir, fname.replace(".png", ".npy").replace(".jpg", ".npy"))
        token_t = torch.from_numpy(np.load(tok_path).astype(np.float32)) if os.path.exists(tok_path) else torch.zeros(243, dtype=torch.float32)

        return img_t, landmarks_t, dt_t, edge_t, token_t


# ---------------------------
# Small test dataset loader (images + labels only; no tokens)
# ---------------------------
class TestImageFolder(Dataset):
    def __init__(self, root_images: str, root_labels: str):
        self.img_dir = root_images
        self.lbl_dir = root_labels
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg"))])
        print(f"📦 Test dataset loaded: {len(self.files)} samples from {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def _load_pts(self, path):
        pts = []
        with open(path, "r") as f:
            for ln in f:
                x, y = ln.strip().split(",")
                pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0

        lbl_path = os.path.join(self.lbl_dir, fname.replace(".png", ".txt").replace(".jpg", ".txt"))
        landmarks = self._load_pts(lbl_path)
        landmarks_t = torch.from_numpy(landmarks).float()

        # compute DT & edge on the fly
        H, W = img_rgb.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for (x, y) in landmarks:
            cx, cy = int(x), int(y)
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(mask, (cx, cy), 3, 255, -1)
        dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        if dt.max() > 0:
            dt = dt / dt.max()
        dt_t = torch.from_numpy(dt).unsqueeze(0).float()

        edge = cv2.Canny(img_bgr, 80, 160).astype(np.float32) / 255.0
        edge_t = torch.from_numpy(edge).unsqueeze(0).float()

        token_t = torch.zeros(243, dtype=torch.float32)  # placeholder
        return img_t, landmarks_t, dt_t, edge_t, token_t, fname


# ---------------------------
# Utility: ensure (B,1,H,W)
# ---------------------------
def ensure_bchw(t: torch.Tensor, B: int, name: str, device):
    if t is None:
        raise RuntimeError(f"{name} is None")
    t = t.to(device)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        # (B,H,W) or (1,H,W) or (C,H,W)
        if t.shape[0] == B:
            t = t.unsqueeze(1)
        elif t.shape[0] == 1:
            t = t.unsqueeze(1)
        elif t.shape[0] in (1, 3):
            t = t.unsqueeze(0)
            if t.shape[1] > 1:
                t = t.mean(dim=1, keepdim=True)
        else:
            raise RuntimeError(f"Unexpected 3D shape for {name}: {t.shape}")
    elif t.dim() == 4:
        if t.shape[1] > 1:
            t = t.mean(dim=1, keepdim=True)
    else:
        raise RuntimeError(f"Unexpected dims for {name}: {t.dim()}")
    return t


# ---------------------------
# Metrics: MRE (mm) and SDR thresholds (mm)
# ---------------------------
def compute_mre_sdr_batch(pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor, mm_per_pixel=0.1,
                          thresholds_mm=(2.0, 2.5, 3.0, 4.0)) -> Tuple[float, dict]:
    """
    pred_landmarks, gt_landmarks: (B, L, 2) in pixel coords
    Returns: MRE (mm), SDR dict {thr: percent}
    """
    d = torch.norm(pred_landmarks - gt_landmarks, dim=-1)  # (B,L) pixels
    d_mm = d * mm_per_pixel
    mre = d_mm.mean().item()
    sdrs = {}
    total = d_mm.numel()
    for thr in thresholds_mm:
        inside = (d_mm <= thr).sum().item()
        sdrs[thr] = 100.0 * inside / total
    return mre, sdrs


# ---------------------------
# Visualize & save sample predictions
# ---------------------------
def save_visuals(save_dir: str, fname: str, img_rgb: np.ndarray, atlas_img: np.ndarray,
                 atlas_lms: np.ndarray, pred_lms: np.ndarray, warped_atlas_edges: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    h, w = img_rgb.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # input
    vis[:, :w] = img_rgb.astype(np.uint8)

    # atlas image
    a_img = atlas_img.astype(np.uint8)
    vis[:, w:2 * w] = a_img

    # warped atlas edge overlay
    edge_rgb = np.stack([warped_atlas_edges * 255,]*3, axis=-1).astype(np.uint8)
    vis[:, 2 * w:3 * w] = edge_rgb

    # overlay landmarks (pred) on left image
    for (x, y) in pred_lms:
        cv2.circle(vis[:, :w], (int(x), int(y)), 2, (255, 0, 0), -1)
    # atlas lm on middle
    for (x, y) in atlas_lms:
        cv2.circle(vis[:, w:2 * w], (int(x), int(y)), 2, (0, 255, 0), -1)

    outp = os.path.join(save_dir, fname)
    cv2.imwrite(outp, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ---------------------------
# Single training step (same core as your earlier code)
# ---------------------------
def train_step(backbone, svf_head, token_encoder, optim,
               images, landmarks, dt_maps, edge_maps,
               atlas_edges, atlas_lms, device,
               use_token_loss=True, svf_steps=6, align_corners=False):
    if images.max() > 2.0:
        images = images.float().to(device) / 255.0
    else:
        images = images.float().to(device)
    B = images.shape[0]
    landmarks = landmarks.to(device)

    dt_maps = ensure_bchw(dt_maps, B, "dt_maps", device)
    edge_maps = ensure_bchw(edge_maps, B, "edge_maps", device)

    feats = backbone(images)
    F5 = feats.get("C5", None) if isinstance(feats, dict) else feats
    if F5 is None:
        F5 = list(feats.values())[-1]

    Hf, Wf = F5.shape[2], F5.shape[3]
    D_small = F.interpolate(dt_maps, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)
    patient_edge_small = F.interpolate(edge_maps, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)

    v = svf_head(F5, D_small)
    disp = svf_to_disp(v, steps=svf_steps, align_corners=align_corners)

    atlas_resized = F.interpolate(atlas_edges, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)
    atlas_rep = atlas_resized.repeat(B, 1, 1, 1)
    warped_edges = warp_with_disp(atlas_rep, disp, align_corners=align_corners)

    H, W = images.shape[2], images.shape[3]
    disp_full = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=align_corners)

    # sample displacement at atlas landmark positions
    atlas_lms_batch = atlas_lms.to(device).repeat(B, 1, 1)
    disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_batch, (H, W), align_corners=align_corners)
    pred_lms = atlas_lms_batch + disp_at_lm

    # losses
    L_lm = huber_landmark_loss(pred_lms, landmarks)
    L_edge = F.l1_loss(warped_edges, patient_edge_small)
    L_smooth = ((v[:, :, 1:] - v[:, :, :-1]).abs().mean() + (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean())
    L_jac = jacobian_regularizer(disp_full)
    L_inv = inverse_consistency_loss(disp_full, -disp_full, align_corners=align_corners)

    L_tok = torch.tensor(0.0, device=device)
    if use_token_loss and (token_encoder is not None):
        T_gt = compute_tokens(patient_edge_small.detach(), token_encoder)
        T_pred = compute_tokens(warped_edges.detach(), token_encoder)
        L_tok = F.mse_loss(T_pred, T_gt)

    loss = (1.0 * L_lm + 1.0 * L_edge + 0.01 * L_smooth + 0.1 * L_jac + 0.1 * L_inv + 1e-4 * L_tok)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return {
        "loss": loss.item(),
        "L_lm": L_lm.item(),
        "L_edge": L_edge.item(),
        "L_smooth": L_smooth.item(),
        "L_jac": L_jac.item(),
        "L_inv": L_inv.item(),
        "L_tok": L_tok.item(),
        "pred_lms": pred_lms.detach().cpu()
    }


# ---------------------------
# Evaluate on a loader (returns MRE & SDRs aggregated)
# ---------------------------
def evaluate_loader(backbone, svf_head, token_encoder, loader, atlas_edges, atlas_lms, device, mm_per_pixel=0.1, max_samples=None):
    backbone.eval(); svf_head.eval()
    all_dists = []
    thresholds = (2.0, 2.5, 3.0, 4.0)
    with torch.no_grad():
        n = 0
        for batch in loader:
            if isinstance(batch, tuple) and len(batch) == 6:
                images, landmarks, dt_maps, edge_maps, tokens, fnames = batch
            else:
                images, landmarks, dt_maps, edge_maps, tokens = batch
                fnames = None
            # forward small eval pass reusing train_step but no optimizer step
            out = train_step(backbone, svf_head, token_encoder, optim=torch.optim.SGD([], lr=1e-6),
                             images=images, landmarks=landmarks, dt_maps=dt_maps, edge_maps=edge_maps,
                             atlas_edges=atlas_edges, atlas_lms=atlas_lms, device=device,
                             use_token_loss=(token_encoder is not None), svf_steps=6, align_corners=False)
            pred_lms = out["pred_lms"]  # (B,L,2)
            gt_lms = landmarks
            d = torch.norm(pred_lms - gt_lms, dim=-1)  # pixels
            all_dists.append(d.cpu().numpy())
            n += 1
            if (max_samples is not None) and (n >= max_samples):
                break
    all_dists = np.concatenate([x.reshape(-1) for x in all_dists], axis=0)
    all_dists_mm = all_dists * mm_per_pixel
    mre = float(all_dists_mm.mean())
    sdrs = {}
    for thr in thresholds:
        sdrs[thr] = 100.0 * float((all_dists_mm <= thr).sum()) / float(all_dists_mm.size)
    return mre, sdrs


# ---------------------------
# Main training script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--atlas-landmarks", type=str, required=True)
    parser.add_argument("--atlas-edge", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/svf")
    parser.add_argument("--use-token-loss", action="store_true")
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--mm-per-pixel", type=float, default=0.1)
    parser.add_argument("--test1-dir", type=str, default="/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test1changed_1")
    parser.add_argument("--test2-dir", type=str, default="/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test2changed_2")
    args = parser.parse_args()

    device = cfg.DEVICE if hasattr(cfg, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(args.save_dir, exist_ok=True)
    viz_dir = os.path.join(args.save_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Datasets
    train_root = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/augmented_ceph"
    train_ds = AugCephDataset(train_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # small eval loader on trainset (non-shuffled)
    eval_train_loader = DataLoader(train_ds, batch_size=args.eval_batch, shuffle=False, num_workers=2)

    # test loaders (each has images/labels subfolders inside provided path)
    test1_img_dir = os.path.join(args.test1_dir, "images")
    test1_lbl_dir = os.path.join(args.test1_dir, "labels")
    test2_img_dir = os.path.join(args.test2_dir, "images")
    test2_lbl_dir = os.path.join(args.test2_dir, "labels")

    test1_ds = TestImageFolder(test1_img_dir, test1_lbl_dir)
    test2_ds = TestImageFolder(test2_img_dir, test2_lbl_dir)
    test1_loader = DataLoader(test1_ds, batch_size=1, shuffle=False, num_workers=1)
    test2_loader = DataLoader(test2_ds, batch_size=1, shuffle=False, num_workers=1)

    # atlas
    atlas_edges_np = np.load(args.atlas_edge)
    atlas_edges = torch.tensor(atlas_edges_np).float()
    if atlas_edges.max() > 2.0:
        atlas_edges = atlas_edges / 255.0
    atlas_edges = atlas_edges.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    atlas_lms_np = np.load(args.atlas_landmarks)
    atlas_lms = torch.tensor(atlas_lms_np).float().unsqueeze(0).to(device)  # (1,L,2)

    # Models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    in_ch = 512
    svf_head = SVFHead(in_channels=in_ch).to(device)

    token_encoder = None
    if args.use_token_loss:
        token_encoder = TokenEncoder(input_dim=243, hidden=256, out_dim=256).to(device)

    if not args.finetune_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen (use --finetune-backbone to enable).")

    params = list(svf_head.parameters())
    if token_encoder is not None:
        params += list(token_encoder.parameters())
    if args.finetune_backbone:
        params += [p for p in backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        backbone.train(); svf_head.train()
        if token_encoder is not None:
            token_encoder.train()

        epoch_tot = {"loss": 0.0, "L_lm": 0.0, "L_edge": 0.0, "L_smooth": 0.0, "L_jac": 0.0, "L_inv": 0.0, "L_tok": 0.0}
        n = 0
        for batch in train_loader:
            # dataset returns: img_t, landmarks_t, dt_t, edge_t, token_t
            images, landmarks, dt_maps, edge_maps, tokens = batch
            out = train_step(backbone, svf_head, token_encoder, optimizer,
                             images, landmarks, dt_maps, edge_maps,
                             atlas_edges, atlas_lms, device,
                             use_token_loss=(token_encoder is not None),
                             svf_steps=6, align_corners=False)
            for k in ["loss", "L_lm", "L_edge", "L_smooth", "L_jac", "L_inv", "L_tok"]:
                epoch_tot[k] += out.get(k, 0.0)
            n += 1
            if n % 200 == 0:
                print(f"Epoch {epoch} step {n}: loss={epoch_tot['loss']/n:.4f}")

        for k in epoch_tot:
            epoch_tot[k] /= max(1, n)

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "token_encoder": token_encoder.state_dict() if token_encoder is not None else None,
            "opt": optimizer.state_dict(),
            "loss": epoch_tot["loss"]
        }
        ckpt_path = os.path.join(args.save_dir, f"svf_epoch_{epoch}.pth")
        torch.save(ckpt, ckpt_path)
        if epoch_tot["loss"] < best_loss:
            best_loss = epoch_tot["loss"]
            torch.save(ckpt, os.path.join(args.save_dir, "svf_best.pth"))
            print("✅ Saved best checkpoint")

        # evaluation: train subset + both tests
        print(f"\n--- Epoch {epoch} summary ---")
        print(f"Train avg losses: loss={epoch_tot['loss']:.6f} L_lm={epoch_tot['L_lm']:.6f} L_edge={epoch_tot['L_edge']:.6f}")

        # evaluate on small subset of train (max_samples controls number of batches used)
        mre_train, sdr_train = evaluate_loader(backbone, svf_head, token_encoder, eval_train_loader, atlas_edges, atlas_lms, device, mm_per_pixel=args.mm_per_pixel, max_samples=50)
        print(f"Train MRE: {mre_train:.4f} mm | SDRs: {sdr_train}")

        mre_t1, sdr_t1 = evaluate_loader(backbone, svf_head, token_encoder, test1_loader, atlas_edges, atlas_lms, device, mm_per_pixel=args.mm_per_pixel, max_samples=None)
        print(f"Test1 MRE: {mre_t1:.4f} mm | SDRs: {sdr_t1}")

        mre_t2, sdr_t2 = evaluate_loader(backbone, svf_head, token_encoder, test2_loader, atlas_edges, atlas_lms, device, mm_per_pixel=args.mm_per_pixel, max_samples=None)
        print(f"Test2 MRE: {mre_t2:.4f} mm | SDRs: {sdr_t2}")

        # save viz for first few test images from test1
        with torch.no_grad():
            cnt = 0
            for img_t, lm_t, dt_t, edge_t, tok_t, fname in test1_loader:
                # forward to get disp+pred_lms and warped edges
                images = img_t.to(device)
                feats = backbone(images)
                F5 = feats.get("C5", None) if isinstance(feats, dict) else feats
                Hf, Wf = F5.shape[2], F5.shape[3]
                D_small = F.interpolate(dt_t.to(device), size=(Hf, Wf), mode='bilinear')
                v = svf_head(F5, D_small)
                disp = svf_to_disp(v, steps=6)
                atlas_resized = F.interpolate(atlas_edges, size=(Hf, Wf), mode='bilinear')
                atlas_rep = atlas_resized.repeat(1, 1, 1, 1)
                warped_edges = warp_with_disp(atlas_rep, disp)
                H, W = images.shape[2], images.shape[3]
                disp_full = F.interpolate(disp, size=(H, W), mode='bilinear')
                disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms.to(device), (H, W))
                pred_lms = (atlas_lms.to(device) + disp_at_lm)[0].cpu().numpy()
                # prepare visuals
                img_rgb = (img_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                atlas_img = np.load(args.atlas_landmarks.replace("atlas_landmarks_resized.npy", "atlas_image_resized.npy")) if os.path.exists(args.atlas_landmarks.replace("atlas_landmarks_resized.npy", "atlas_image_resized.npy")) else np.zeros_like(img_rgb)
                atlas_img = atlas_img.astype(np.uint8)
                warped_edges_np = warped_edges[0, 0].cpu().numpy()
                atlas_lms_np = atlas_lms[0].cpu().numpy()
                save_visuals(viz_dir, f"epoch{epoch}_test1_{cnt}_{fname[0]}", img_rgb, atlas_img, atlas_lms_np, pred_lms, warped_edges_np)
                cnt += 1
                if cnt >= 4:
                    break

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s\n")

    print("Training finished.")


if __name__ == "__main__":
    main()
