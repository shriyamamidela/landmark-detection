# tools/refine_svf.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable

# ---------------------------
# Utilities (strict, tensor-first)
# ---------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_b2hw_tensor(disp):
    """
    Strictly ensure torch.Tensor of shape [B,2,H,W].
    If input is torch.Tensor we do *no* cuda->cpu roundtrip; we permute on-device.
    If input is numpy, we convert to torch on cpu.
    """
    # numpy input
    if isinstance(disp, np.ndarray):
        npd = disp.astype(np.float32)
        if npd.ndim == 3 and npd.shape[-1] == 2:
            return torch.from_numpy(npd.transpose(2, 0, 1)).unsqueeze(0)   # (1,2,H,W)
        if npd.ndim == 3 and npd.shape[0] == 2:
            return torch.from_numpy(npd).unsqueeze(0)                     # (1,2,H,W)
        if npd.ndim == 4 and npd.shape[-1] == 2:
            return torch.from_numpy(npd.transpose(0, 3, 1, 2))           # (B,2,H,W)
        if npd.ndim == 4 and npd.shape[1] == 2:
            return torch.from_numpy(npd)
        raise ValueError(f"Cannot normalize numpy disp with shape {npd.shape}")

    # torch tensor input: operate on-device, preserve device/dtype
    if isinstance(disp, torch.Tensor):
        t = disp
        t = t.float()
        ndim = t.ndim
        shape = tuple(t.shape)
        # already (B,2,H,W)
        if ndim == 4 and shape[1] == 2:
            return t
        # (B,H,W,2)
        if ndim == 4 and shape[-1] == 2:
            return t.permute(0, 3, 1, 2)
        # (B,H,2,W)
        if ndim == 4 and shape[2] == 2:
            return t.permute(0, 2, 1, 3)
        # (2,H,W)
        if ndim == 3 and shape[0] == 2:
            return t.unsqueeze(0)
        # (H,W,2)
        if ndim == 3 and shape[-1] == 2:
            return t.permute(2, 0, 1).unsqueeze(0)
        # fallback: find axis==2 and move to pos 1
        axes = [i for i, s in enumerate(shape) if s == 2]
        if len(axes) >= 1:
            ch = axes[-1]
            if ch != 1:
                perm = list(range(ndim))
                perm.pop(ch)
                perm.insert(1, ch)
                t2 = t.permute(perm)
                if t2.ndim == 3:
                    t2 = t2.unsqueeze(0)
                if t2.ndim == 4 and t2.shape[1] == 2:
                    return t2
        raise ValueError(f"Cannot normalize torch disp with shape {shape}")

    raise TypeError("disp must be torch.Tensor or numpy.ndarray")


# ---------------------------
# warp_with_disp and svf->disp
# ---------------------------
def warp_with_disp(img, disp):
    """
    Warp img (B,C,H,W) with disp (B,2,H,W). Returns (B,C,H,W).
    img can be torch or numpy.

    Uses only differentiable ops so gradients flow to `disp`.
    """
    # convert img to tensor on the same device/dtype as disp
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float().to(disp.device)
    else:
        img = img.to(disp.device).float()

    B, C, H, W = img.shape

    # base grid in normalized coords [-1,1], shape [H,W,2]
    yy = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
    xx = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
    yy_grid, xx_grid = torch.meshgrid(yy, xx, indexing="ij")   # [H,W]
    base_grid = torch.stack((xx_grid, yy_grid), dim=-1)        # [H,W,2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)      # [B,H,W,2]

    # disp: [B,2,H,W] -> permute to [B,H,W,2]
    disp_xy = disp.permute(0, 2, 3, 1)   # [B,H,W,2]

    # normalized offsets
    disp_x_norm = disp_xy[..., 0] / (W / 2)
    disp_y_norm = disp_xy[..., 1] / (H / 2)
    disp_norm = torch.stack((disp_x_norm, disp_y_norm), dim=-1)  # [B,H,W,2]

    grid = base_grid + disp_norm   # [B,H,W,2]

    # grid_sample will produce gradients w.r.t. grid -> disp_norm -> disp
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def svf_to_disp(v, steps=7):
    """
    Scaling-and-squaring: v -> disp. v can be numpy or torch.
    Returns torch tensor [B,2,H,W] on same device as v (if torch) or cpu (if numpy).
    """
    # normalize shape (on-device)
    v_norm = _ensure_b2hw_tensor(v)
    # if tensor, keep device; if numpy, v_norm is CPU tensor
    device = v_norm.device
    v_norm = v_norm.to(device)
    disp = v_norm / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_with_disp(disp, disp)
    return disp


# ---------------------------
# Loss helpers
# ---------------------------
def compute_jacobian_neg_penalty(disp):
    dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]   # [B,2,H,W-1]
    dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]   # [B,2,H-1,W]

    dx_x = dx[:, 0, 1:, :]   # [B,H-1,W-1]
    dy_y = dy[:, 1, :, 1:]   # [B,H-1,W-1]

    J_proxy = dx_x * dy_y
    neg = torch.relu(-J_proxy)
    return neg.mean()


def dt_edge_map_from_dt(dt_tensor):
    """
    dt_tensor: torch [B,1,H,W] or numpy
    Returns normalized edge magnitude [B,1,H,W]
    """
    if isinstance(dt_tensor, np.ndarray):
        dt_tensor = torch.from_numpy(dt_tensor).float()
    gx = dt_tensor[:, :, :, 1:] - dt_tensor[:, :, :, :-1]
    gy = dt_tensor[:, :, 1:, :] - dt_tensor[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    maxv = mag.view(mag.shape[0], -1).max(dim=-1)[0].view(-1, 1, 1, 1)
    return mag / (maxv + 1e-8)


# ---------------------------
# LowRank Correction
# ---------------------------
class LowRankCorrection(torch.nn.Module):
    """
    delta_v(x,y) = sum_k u_k(x) * v_k(y) for each channel separately.
    u: (2, r, H, 1)
    v: (2, r, 1, W)
    Produces delta: [1,2,H,W]
    """
    def __init__(self, H: int, W: int, rank: int = 4, init_scale: float = 1e-3, device: str = "cpu"):
        super().__init__()
        self.H = int(H); self.W = int(W); self.rank = int(rank)
        self.u = torch.nn.Parameter(init_scale * torch.randn(2, self.rank, self.H, 1, device=device))
        self.v = torch.nn.Parameter(init_scale * torch.randn(2, self.rank, 1, self.W, device=device))

    def forward(self):
        # multiply with broadcasting: (2, r, H, 1) * (2, r, 1, W) -> (2, r, H, W)
        outer = self.u * self.v                       # [2, r, H, W]
        summed = outer.sum(dim=1)                     # [2, H, W]
        return summed.unsqueeze(0)                    # [1, 2, H, W]


# ---------------------------
# Main refinement
# ---------------------------
def refine_and_warp(
    svf_disp,
    svf_v,
    F_feat,
    D_small,
    atlas_edges_resized,
    atlas_landmarks_px,
    warp_landmarks_fn: Callable,
    device: str = "cuda",
    rank: int = 4,
    n_iter: int = 200,
    lr: float = 1e-2,
    lambda_edge: float = 1.0,
    lambda_jac: float = 1.0,
    lambda_reg: float = 1e-4,
):
    """
    svf_disp: torch or numpy (various shapes) or None
    svf_v: optional
    F_feat: [1,C,Hf,Wf] torch
    D_small: [1,1,Hf,Wf] torch
    atlas_edges_resized: [1,1,Hf,Wf] torch or numpy
    atlas_landmarks_px: numpy (N,2)
    warp_landmarks_fn: function(atlas_landmarks_px, disp_lowres_numpy) -> warped landmarks (N,2)
    Returns: disp_final_numpy [1,2,Hf,Wf], warped_landmarks (N,2), conf_map_numpy [1,1,Hf,Wf]
    """

    # ---------- prepare disp ----------
    if svf_disp is None:
        if svf_v is None:
            raise ValueError("Either svf_disp or svf_v must be provided")
        disp_t = svf_to_disp(svf_v)
    else:
        disp_t = _ensure_b2hw_tensor(svf_disp)

    # ensure on-device and float
    disp_orig = disp_t.to(device=device).float()
    B, C, Hf, Wf = disp_orig.shape

    # prepare atlas_edges & D_small as device tensors (avoid CPU->GPU leaf issues)
    if isinstance(atlas_edges_resized, np.ndarray):
        atlas_edges_t = torch.tensor(atlas_edges_resized, dtype=torch.float32, device=device)
    else:
        atlas_edges_t = atlas_edges_resized.to(device=device, dtype=torch.float32)

    if isinstance(D_small, np.ndarray):
        D_small_t = torch.tensor(D_small, dtype=torch.float32, device=device)
    else:
        D_small_t = D_small.to(device=device, dtype=torch.float32)

    # ---------- init correction ----------
    corr = LowRankCorrection(H=Hf, W=Wf, rank=rank, device=device).to(device)
    opt = torch.optim.Adam(corr.parameters(), lr=lr)

    target_edge = dt_edge_map_from_dt(D_small_t)

    # ---------- optimization (ensure grads even if caller used no_grad) ----------
    with torch.enable_grad():
        for it in range(n_iter):
            opt.zero_grad()
            delta = corr()                    # [1,2,Hf,Wf], depends on corr.params -> requires_grad True

            # sanity checks
            if delta.ndim != 4 or delta.shape[1] != 2:
                raise RuntimeError(f"LowRankCorrection output shape wrong: {tuple(delta.shape)}")
            if disp_orig.ndim != 4 or disp_orig.shape[1] != 2:
                raise RuntimeError(f"disp_orig shape wrong: {tuple(disp_orig.shape)}")

            disp_ref = disp_orig + delta

            warped_edges = warp_with_disp(atlas_edges_t, disp_ref)  # [1,1,Hf,Wf]
            edge_loss = F.l1_loss(warped_edges, target_edge)

            jac_pen = compute_jacobian_neg_penalty(disp_ref)
            reg = (delta**2).mean()

            loss = lambda_edge * edge_loss + lambda_jac * jac_pen + lambda_reg * reg

            # debug if something is disconnected
            if not loss.requires_grad:
                print("DEBUG: loss.requires_grad = False")
                print("DEBUG: edge_loss.requires_grad =", getattr(edge_loss, "requires_grad", None))
                print("DEBUG: jac_pen.requires_grad  =", getattr(jac_pen, "requires_grad", None))
                print("DEBUG: reg.requires_grad      =", getattr(reg, "requires_grad", None))
                print("DEBUG: delta.requires_grad    =", getattr(delta, "requires_grad", None))
                for i, p in enumerate(corr.parameters()):
                    print(f"DEBUG: param[{i}].requires_grad = {p.requires_grad} device={p.device} shape={tuple(p.shape)}")
                raise RuntimeError("LOSS DOES NOT REQUIRE GRAD — check graph connectivity")

            loss.backward()
            opt.step()

            if (it % 50) == 0 or it == (n_iter - 1):
                print(f"[refine] it={it}/{n_iter} loss={loss.item():.6f} edge={edge_loss.item():.6f} jac={jac_pen.item():.6f} reg={reg.item():.6e}")

    # ---------- final ----------
    disp_final_t = (disp_orig + corr()).detach().cpu().numpy()   # [1,2,Hf,Wf] numpy
    warped_landmarks_px = warp_landmarks_fn(atlas_landmarks_px, disp_final_t)
    conf_map = np.exp(-np.abs(disp_final_t)).mean(axis=1, keepdims=True)  # [1,1,Hf,Wf]

    return disp_final_t, warped_landmarks_px, conf_map
