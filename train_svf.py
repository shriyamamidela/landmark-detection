# train_svf.py  — FINAL V6 (token-loss enabled, robust, main() entry included)
"""
Train SVF head (Atlas-flow) with:
  - landmark Huber loss
  - atlas-edge alignment (L1)
  - SVF smoothness
  - jacobian regularizer
  - inverse consistency
  - optional token consistency (topology tokens)

Usage example:
  python train_svf.py \
    --epochs 40 \
    --batch-size 1 \
    --atlas-landmarks atlas_landmarks_resized.npy \
    --atlas-edge atlas_edge_map_resized.npy \
    --use-token-loss

Notes:
 - Expects data.Dataset("isbi","train") to return (image, landmarks, dt_map, edge_map)
 - Images expected in dataset are letterbox-resized to (cfg.HEIGHT, cfg.WIDTH).
 - Methods reference (for traceability): METHODS_PDF points at local file path.
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Dataset
from config import cfg
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from network.token_encoder import TokenEncoder

from losses import (
    huber_landmark_loss,
    jacobian_regularizer,
    inverse_consistency_loss
)

# local methods doc (for your traceability)
METHODS_PDF = "/mnt/data/ATLAS_FLOW_DIFF_methods_summary.pdf"

# optional arc-token utilities (if present)
try:
    from preprocessing.topology import extract_arc_tokens_from_edgebank, flatten_arc_tokens
    HAVE_ARC = True
except Exception:
    HAVE_ARC = False


# -------------------------
# Grid, warping helpers
# -------------------------
def make_meshgrid(B, H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)


def normalize_disp_for_grid(disp, H, W):
    dx = disp[:, 0] / (W / 2.0)
    dy = disp[:, 1] / (H / 2.0)
    return torch.stack([dx, dy], dim=-1)


def warp_with_disp(img, disp, align_corners=False):
    B, C, H, W = img.shape
    base = make_meshgrid(B, H, W, img.device)
    grid = (base + normalize_disp_for_grid(disp, H, W)).clamp(-1, 1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)


def warp_disp(field, disp, align_corners=False):
    return warp_with_disp(field, disp, align_corners)


def svf_to_disp(svf, steps=6, align_corners=False):
    # exponentiate SVF by scaling and squaring
    disp = svf / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_disp(disp, disp, align_corners)
    return disp


# -------------------------
# Landmark sampler (grid_sample)
# -------------------------
def sample_flow_at_points_local(flow, points_px, image_size, align_corners=False):
    B, N, _ = points_px.shape
    H, W = image_size

    if align_corners:
        x_norm = 2 * (points_px[..., 0] / (W - 1)) - 1
        y_norm = 2 * (points_px[..., 1] / (H - 1)) - 1
    else:
        x_norm = 2 * ((points_px[..., 0] + 0.5) / W) - 1
        y_norm = 2 * ((points_px[..., 1] + 0.5) / H) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B,N,1,2)

    sampled = F.grid_sample(
        flow, grid, mode="bilinear", padding_mode="border", align_corners=align_corners
    )  # -> (B,2,N,1)

    sampled = sampled.squeeze(-1)  # (B,2,N)
    sampled = sampled.permute(0, 2, 1)  # (B,N,2)
    return sampled


# -------------------------
# Small helper losses
# -------------------------
def edge_alignment_loss(warped_edges, target_edges):
    return F.l1_loss(warped_edges, target_edges)


def smoothness_loss(v):
    return (v[:, :, 1:] - v[:, :, :-1]).abs().mean() + (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean()


# -------------------------
# Token computation (batched)
# -------------------------
def fast_edge_descriptor(edge_map_tensor, bins=243):
    # Compute a fast gradient-magnitude histogram descriptor as fallback
    arr = edge_map_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    B = arr.shape[0]
    feats = []
    for i in range(B):
        im = arr[i].astype(np.float32)
        gx = np.gradient(im)[1]
        gy = np.gradient(im)[0]
        mag = np.sqrt(gx ** 2 + gy ** 2).ravel()
        maxv = mag.max() if mag.max() > 0 else 1.0
        hist, _ = np.histogram(mag, bins=bins, range=(0.0, maxv))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        feats.append(hist)
    return torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)


def compute_tokens(edge_map_tensor, token_encoder=None):
    """
    Returns either raw descriptor (if token_encoder is None) or encoded embedding (B, out_dim).
    """
    device = edge_map_tensor.device
    if HAVE_ARC:
        # heavy arc-based extractor (matches prior pipeline) - may be slower
        arr = edge_map_tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]
        toks = []
        for im in arr:
            im_u8 = np.clip(im * 255.0, 0, 255).astype(np.uint8)
            im_3ch = np.stack([im_u8, im_u8, im_u8], axis=-1)
            arcs = extract_arc_tokens_from_edgebank(im_3ch)
            flat = flatten_arc_tokens(arcs)
            toks.append(flat)
        desc = torch.tensor(np.stack(toks), dtype=torch.float32, device=device)
    else:
        desc = fast_edge_descriptor(edge_map_tensor).to(device)

    if token_encoder is None:
        return desc
    return token_encoder(desc)


# -------------------------
# Robust normalization to (B,1,H,W)
# -------------------------
def ensure_bchw(t, B, name, device):
    """
    Normalize a torch.Tensor t to shape (B,1,H,W).
    Accepts dims: 2 (H,W), 3 (B,H,W) or (1,H,W) or (C,H,W), 4 (B,C,H,W).
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(t)}")
    t = t.to(device)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif t.dim() == 3:
        # (B,H,W) or (1,H,W) or (C,H,W)
        if t.shape[0] == B:
            t = t.unsqueeze(1)  # (B,1,H,W)
        elif t.shape[0] == 1:
            t = t.unsqueeze(1)  # (1,1,H,W)
        elif t.shape[0] in (1, 3):
            # (C,H,W) -> treat as (1,C,H,W)
            t = t.unsqueeze(0)
            if t.shape[1] > 1:
                t = t.mean(dim=1, keepdim=True)
        else:
            # ambiguous - try to treat as (B,H,W)
            if t.shape[0] == B:
                t = t.unsqueeze(1)
            else:
                raise RuntimeError(f"Unexpected 3D shape for {name}: {t.shape}")
    elif t.dim() == 4:
        if t.shape[1] > 1:
            t = t.mean(dim=1, keepdim=True)
    else:
        raise RuntimeError(f"Unexpected dims for {name}: {t.dim()}")
    if not (t.dim() == 4 and t.shape[1] == 1):
        raise RuntimeError(f"{name} normalization failed; got shape {t.shape}")
    return t


# -------------------------
# Single training step
# -------------------------
def train_step(backbone, svf_head, token_encoder,
               images, gt_landmarks, dt_maps, edge_maps,
               atlas_edges, atlas_lms,
               optimizer, device,
               svf_steps=6, align_corners=False, use_token_loss=True):

    # images normalization
    if images.max() > 2.0:
        images = images.float().to(device) / 255.0
    else:
        images = images.float().to(device)

    gt_landmarks = gt_landmarks.to(device)

    B = images.shape[0]
    atlas_lms_batch = atlas_lms.to(device).repeat(B, 1, 1)

    # ensure dt_maps & edge_maps are (B,1,H,W)
    dt_maps = ensure_bchw(dt_maps, B, "dt_maps", images.device)
    edge_maps = ensure_bchw(edge_maps, B, "edge_maps", images.device)

    # backbone features
    feats = backbone(images)
    F5 = feats.get("C5", None) if isinstance(feats, dict) else feats
    if F5 is None:
        # fallback: take last feature
        if isinstance(feats, dict):
            F5 = list(feats.values())[-1]
        else:
            raise RuntimeError("Backbone returned None for features")

    Hf, Wf = F5.shape[2], F5.shape[3]

    # resize dt + patient edge to feature resolution
    D_small = F.interpolate(dt_maps, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)
    patient_edge_small = F.interpolate(edge_maps, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)

    # predict svf
    v = svf_head(F5, D_small)  # expects (B,2,Hf,Wf) in pixel units

    # exponentiate
    disp = svf_to_disp(v, steps=svf_steps, align_corners=align_corners)

    # warp atlas edges -> compare with patient small edge map
    atlas_resized = F.interpolate(atlas_edges, size=(Hf, Wf), mode='bilinear', align_corners=align_corners)
    atlas_rep = atlas_resized.repeat(B, 1, 1, 1)
    warped_edges = warp_with_disp(atlas_rep, disp, align_corners=align_corners)

    # upsample disp to full resolution and predict landmarks
    H, W = images.shape[2], images.shape[3]
    disp_full = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=align_corners)
    disp_at_lm = sample_flow_at_points_local(disp_full, atlas_lms_batch, (H, W), align_corners=align_corners)
    pred_landmarks = atlas_lms_batch + disp_at_lm

    # losses
    L_lm = huber_landmark_loss(pred_landmarks, gt_landmarks)
    L_edge = edge_alignment_loss(warped_edges, patient_edge_small)
    L_smooth = smoothness_loss(v)
    L_jac = jacobian_regularizer(disp_full)
    L_inv = inverse_consistency_loss(disp_full, -disp_full, align_corners=align_corners)

    L_tok = torch.tensor(0.0, device=device)
    if use_token_loss and (token_encoder is not None):
        T_gt = compute_tokens(patient_edge_small.detach(), token_encoder)
        T_pred = compute_tokens(warped_edges.detach(), token_encoder)
        L_tok = F.mse_loss(T_pred, T_gt)

    loss = (
        1.0 * L_lm +
        1.0 * L_edge +
        0.01 * L_smooth +
        0.1 * L_jac +
        0.1 * L_inv +
        1e-4 * L_tok
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_lm": L_lm.item(),
        "L_edge": L_edge.item(),
        "L_smooth": L_smooth.item(),
        "L_jac": L_jac.item(),
        "L_inv": L_inv.item(),
        "L_tok": L_tok.item()
    }


# -------------------------
# Epoch loop
# -------------------------
def train_epoch(loader, backbone, svf_head, token_encoder,
                atlas_edges, atlas_lms, optimizer,
                device, svf_steps=6, align_corners=False, use_token_loss=True):

    totals = {k: 0.0 for k in ["loss", "L_lm", "L_edge", "L_smooth", "L_jac", "L_inv", "L_tok"]}
    n = 0

    backbone.train()
    svf_head.train()
    if token_encoder is not None:
        token_encoder.train()

    for batch in loader:
        # dataset returns: image, landmarks, dt_map, edge_map
        if len(batch) == 4:
            images, landmarks, dt_maps, edge_maps = batch
        else:
            raise RuntimeError("Dataset must return (image, landmarks, dt_map, edge_map)")

        
        # ---- ROBUST NORMALIZATION FOR dt_maps / edge_maps ----
        # convert possible weird shapes (e.g., (B,1,1,H,W) or (B,1,H,W,1)) -> (B,1,H,W)
        def normalize_to_b1hw(t, B, name):
            # t: torch.Tensor
            if not isinstance(t, torch.Tensor):
                return t
            # if shape already correct, return
            if t.dim() == 4 and t.shape[0] == B and t.shape[1] == 1:
                return t
            # if 5D: try to collapse the middle singleton(s)
            if t.dim() == 5:
                # try last two dims as H,W
                H = t.shape[-2]; W = t.shape[-1]
                try:
                    t = t.reshape(B, -1, H, W)   # combine extra singleton dims into channel axis
                except Exception:
                    # fallback: squeeze any singleton axes except batch
                    dims = [i for i in range(t.dim()) if t.shape[i] == 1 and i != 0]
                    for d in sorted(dims, reverse=True):
                        t = t.squeeze(d)
                    # after squeeze attempt, if still problematic, raise
                    if not (t.dim() == 4 and t.shape[0] == B):
                        raise RuntimeError(f"Unable to normalize {name}; got shape {t.shape}")
                # if we combined multi-channels, reduce to single channel by mean
                if t.shape[1] > 1:
                    t = t.mean(dim=1, keepdim=True)
                return t
            # if shape is (1,H,W) or (H,W)
            if t.dim() == 3 and t.shape[0] == B:
                # treat as (B,H,W) -> add channel
                return t.unsqueeze(1)
            if t.dim() == 3 and t.shape[0] != B and t.shape[0] in (1,3):
                # (C,H,W) -> average channels -> (1,H,W) then unsqueeze batch
                t2 = t.mean(dim=0, keepdim=True)  # (1,H,W)
                return t2.unsqueeze(0) if B != 1 else t2.unsqueeze(0)
            if t.dim() == 2:
                # (H,W) -> (1,1,H,W)
                return t.unsqueeze(0).unsqueeze(0)
            # if already (B, H, W) -> add channel
            if t.dim() == 3 and t.shape[0] == B:
                return t.unsqueeze(1)
            raise RuntimeError(f"Unexpected shape for {name}: {t.shape}")

        # apply normalization (safe guards)
        try:
            dt_maps = normalize_to_b1hw(dt_maps, images.shape[0], "dt_maps")
            edge_maps = normalize_to_b1hw(edge_maps, images.shape[0], "edge_maps")
        except Exception as e:
            # helpful error with shapes
            print("ERROR normalizing dt/edge maps:", e)
            raise

        
        # now call train_step with normalized tensors
        out = train_step(
            backbone, svf_head, token_encoder,
            images, landmarks, dt_maps, edge_maps,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps, align_corners, use_token_loss
        )


        for k in totals:
            totals[k] += out[k]
        n += 1

    for k in totals:
        totals[k] /= max(1, n)
    return totals


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--svf-steps", type=int, default=6)
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument("--atlas-landmarks", type=str, default="atlas_landmarks_resized.npy")
    parser.add_argument("--atlas-edge", type=str, default="atlas_edge_map_resized.npy")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/svf")
    parser.add_argument("--use-token-loss", action="store_true", help="Enable topology token loss (slower).")
    parser.add_argument("--finetune-backbone", action="store_true", help="Allow backbone parameters to be updated.")
    args = parser.parse_args()

    device = cfg.DEVICE if hasattr(cfg, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu")

    # quick startup print (helps verify script actually executed)
    print("📌 Using DEFAULT SVF-safe augmentation for training.")
    print("📌 Methods PDF (for traceability) at:", METHODS_PDF)
    print("Device:", device)

    # Dataset + loader
    train_dataset = Dataset("isbi", "train", batch_size=args.batch_size, shuffle=True)
    if len(train_dataset) == 0:
        raise RuntimeError("Dataset is empty. Check data paths and Dataset implementation.")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # atlas
    if not os.path.exists(args.atlas_landmarks):
        raise FileNotFoundError(f"Atlas landmarks not found: {args.atlas_landmarks}")
    if not os.path.exists(args.atlas_edge):
        raise FileNotFoundError(f"Atlas edge not found: {args.atlas_edge}")

    atlas_edges_np = np.load(args.atlas_edge)
    atlas_edges = torch.tensor(atlas_edges_np).float()
    if atlas_edges.max() > 2.0:
        atlas_edges = atlas_edges / 255.0
    atlas_edges = atlas_edges.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    atlas_lms_np = np.load(args.atlas_landmarks)
    atlas_lms = torch.tensor(atlas_lms_np).float().unsqueeze(0).to(device)  # (1,L,2)

    # Models
    backbone = ResNetBackbone("resnet34", pretrained=True, fuse_edges=False).to(device)
    try:
        in_ch = backbone.layer4[-1].conv2.out_channels
    except Exception:
        in_ch = 512
    svf_head = SVFHead(in_channels=in_ch).to(device)

    token_encoder = None
    if args.use_token_loss:
        token_encoder = TokenEncoder(input_dim=243, hidden=256, out_dim=256).to(device)

    # Freeze backbone unless finetune requested
    if not args.finetune_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print("📌 Backbone frozen. Use --finetune-backbone to train it.")

    # optimizer
    params = list(svf_head.parameters())
    if token_encoder is not None:
        params += list(token_encoder.parameters())
    if args.finetune_backbone:
        params += [p for p in backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    print("\n🚀 Training SVF Head...\n")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = train_epoch(
            train_loader, backbone, svf_head, token_encoder,
            atlas_edges, atlas_lms,
            optimizer, device,
            svf_steps=args.svf_steps, align_corners=args.align_corners, use_token_loss=args.use_token_loss
        )

        print(
            f"Epoch {epoch}/{args.epochs} | Loss={metrics['loss']:.4f} | "
            f"L_lm={metrics['L_lm']:.4f} | L_edge={metrics['L_edge']:.4f} | "
            f"L_jac={metrics['L_jac']:.4f} | L_inv={metrics['L_inv']:.4f} | L_tok={metrics['L_tok']:.6f} "
            f"| time={time.time()-t0:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "svf": svf_head.state_dict(),
            "token_encoder": token_encoder.state_dict() if token_encoder is not None else None,
            "opt": optimizer.state_dict(),
            "loss": metrics["loss"]
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"svf_epoch_{epoch}.pth"))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(ckpt, os.path.join(args.save_dir, "svf_best.pth"))
            print("✅ Saved best checkpoint")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
