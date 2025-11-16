import torch
import torch.nn.functional as F


def huber_landmark_loss(pred_landmarks, gt_landmarks, delta=1.0, reduction='mean'):
    """
    pred_landmarks: [B, N, 2]
    gt_landmarks:   [B, N, 2]
    delta: threshold for Huber loss
    """
    diff = pred_landmarks - gt_landmarks          # [B, N, 2]
    abs_diff = diff.abs()

    # quadratic term for |r| <= delta
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))

    # Huber loss element: 0.5*q^2 + delta*(|r|-q)
    loss_elem = 0.5 * (quadratic ** 2) + delta * (abs_diff - quadratic)

    # sum over coordinate axis (x,y)
    loss_per_point = loss_elem.sum(dim=-1)  # [B, N]

    if reduction == 'mean':
        return loss_per_point.mean()
    elif reduction == 'sum':
        return loss_per_point.sum()
    else:
        return loss_per_point  # return [B, N]

# ------------------------------------------------------------------
# Jacobian determinant utilities and regularizer
# ------------------------------------------------------------------
def compute_jacobian_det_central(disp):
    """
    Safe central-difference Jacobian determinant.
    Ensures matching spatial shapes even for odd H/W.
    """
    # disp: [B,2,H,W] → convert to [B,H,W,2]
    u = disp.permute(0, 2, 3, 1)
    ux = u[..., 0]
    uy = u[..., 1]

    # central differences
    ux_x = (ux[:, :, 2:] - ux[:, :, :-2]) * 0.5      # shape [B, H, W-2]
    uy_x = (uy[:, :, 2:] - uy[:, :, :-2]) * 0.5      # shape [B, H, W-2]

    ux_y = (ux[:, 2:, :] - ux[:, :-2, :]) * 0.5      # shape [B, H-2, W]
    uy_y = (uy[:, 2:, :] - uy[:, :-2, :]) * 0.5      # shape [B, H-2, W]

    # crop to common interior region: [B, H-2, W-2]
    ux_x = ux_x[:, 1:-1, :]      # (crop height)
    uy_x = uy_x[:, 1:-1, :]
    ux_y = ux_y[:, :, 1:-1]      # (crop width)
    uy_y = uy_y[:, :, 1:-1]

    # Jacobian matrix entries
    J11 = 1.0 + ux_x
    J12 = ux_y
    J21 = uy_x
    J22 = 1.0 + uy_y

    det = J11 * J22 - J12 * J21   # safe, same shape
    return det



def jacobian_regularizer(disp, neg_weight=10.0, dev_weight=1.0):
    """
    disp: [B,2,H,W] displacement field (pixel units)
    neg_weight: weight for penalizing negative determinants
    dev_weight: weight for penalizing deviation from 1.0 (volume change)
    Returns scalar loss
    """
    det = compute_jacobian_det_central(disp)  # [B, h, w]
    # Penalize negative determinants strongly
    neg_part = F.relu(-det).mean()
    # Penalize deviation from 1 (volume change)
    dev_part = (det - 1.0).abs().mean()

    loss = neg_weight * neg_part + dev_weight * dev_part
    return loss
def inverse_consistency_loss(disp_fwd, disp_bwd, align_corners=False):
    """
    disp_fwd: [B,2,H,W]
    disp_bwd: [B,2,H,W]
    We enforce: disp_fwd + warp(disp_bwd, disp_fwd) ≈ 0
    """
    from torch.nn.functional import grid_sample

    B, C, H, W = disp_fwd.shape
    device = disp_fwd.device

    # Build base grid
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    # Normalize fwd displacement
    disp_norm = torch.stack([
        disp_fwd[:, 0] / (W / 2.0),
        disp_fwd[:, 1] / (H / 2.0),
    ], dim=-1)

    samp_grid = (base + disp_norm).clamp(-1, 1)

    # Warp backward field using forward
    disp_bwd_warped = grid_sample(
        disp_bwd,
        samp_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )

    # Composition: forward + backward ≈ 0
    comp = disp_fwd + disp_bwd_warped

    return comp.abs().mean()
def advanced_edge_loss(warped_edges, D_small, eps=1e-6):
    """
    warped_edges: [B,1,h,w] (soft values, 0..1)
    D_small:       [B,1,h,w] (distance transform; smaller = near GT edge)
    Returns scalar loss: average weighted DT value at predicted edges.
    Implementation:
      - normalize warped_edges to form a distribution per-sample
      - compute expected DT under that distribution
      - also add a small penalty to avoid trivial zero-mass predictions
    """
    # ensure shapes match
    assert warped_edges.shape == D_small.shape, f"shapes mismatch {warped_edges.shape} vs {D_small.shape}"

    # clamp and ensure positive
    w = warped_edges.clamp(min=0.0, max=1.0)

    # per-sample normalization
    mass = w.sum(dim=(2, 3), keepdim=True)              # [B,1,1,1]
    normalized = w / (mass + eps)

    # expected distance where model predicts edges
    expected_dt = (normalized * D_small).sum(dim=(2, 3))  # [B,1]
    loss_dt = expected_dt.mean()

    # small penalty to avoid zero-mass
    mass_penalty = (1.0 / (mass + eps)).mean()

    # final loss: expect-to-be-small + tiny mass regulator
    return loss_dt + 1e-3 * mass_penalty
