import torch
import torch.nn.functional as F

# --------------------------------------------------------------
# 1. Huber Landmark Loss (unchanged)
# --------------------------------------------------------------
def huber_landmark_loss(pred_landmarks, gt_landmarks, delta=1.0, reduction='mean'):
    diff = pred_landmarks - gt_landmarks
    abs_diff = diff.abs()

    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    loss_elem = 0.5 * (quadratic ** 2) + delta * (abs_diff - quadratic)

    loss_per_point = loss_elem.sum(dim=-1)

    if reduction == 'mean':
        return loss_per_point.mean()
    elif reduction == 'sum':
        return loss_per_point.sum()
    else:
        return loss_per_point


# --------------------------------------------------------------
# 2. Improved Jacobian Regularizer (stable forward diff)
# --------------------------------------------------------------
def compute_jacobian_det_safe(disp):
    """
    disp: (B,2,H,W)
    Uses forward differences (stable for small spatial maps).
    """
    u = disp[:, 0]   # (B,H,W)
    v = disp[:, 1]

    du_dx = u[:, :, 1:] - u[:, :, :-1]
    dv_dx = v[:, :, 1:] - v[:, :, :-1]

    du_dy = u[:, 1:, :] - u[:, :-1, :]
    dv_dy = v[:, 1:, :] - v[:, :-1, :]

    du_dx = du_dx[:, :, :-1]
    dv_dx = dv_dx[:, :, :-1]
    du_dy = du_dy[:, :-1, :]
    dv_dy = dv_dy[:, :-1, :]

    J11 = 1 + du_dx
    J12 = du_dy
    J21 = dv_dx
    J22 = 1 + dv_dy

    det = J11 * J22 - J12 * J21
    return det


def jacobian_regularizer(disp, neg_weight=1.0, dev_weight=0.1):
    det = compute_jacobian_det_safe(disp)

    neg_part = F.relu(-det).mean()
    dev_part = (det - 1.0).abs().mean()

    return neg_weight * neg_part + dev_weight * dev_part


# --------------------------------------------------------------
# 3. Improved Inverse Consistency Loss (weaker)
# --------------------------------------------------------------
def inverse_consistency_loss(disp_fwd, disp_bwd, align_corners=False):
    B, C, H, W = disp_fwd.shape
    device = disp_fwd.device

    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)

    disp_norm = torch.stack([
        disp_fwd[:, 0] / (W/2.0),
        disp_fwd[:, 1] / (H/2.0),
    ], dim=-1)

    samp_grid = (base + disp_norm).clamp(-1,1)

    disp_bwd_warped = F.grid_sample(
        disp_bwd, samp_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )

    return (disp_fwd + disp_bwd_warped).abs().mean()


# --------------------------------------------------------------
# 4. NEW: Edge-alignment loss using probability weighting
# --------------------------------------------------------------
def advanced_edge_loss(warped_edges, gt_edges, eps=1e-6):
    w = warped_edges.clamp(0,1)
    mass = w.sum(dim=(2,3), keepdim=True)
    norm = w / (mass + eps)

    loss_dt = (norm * gt_edges).sum(dim=(2,3)).mean()
    mass_pen = (1/(mass+eps)).mean()

    return loss_dt + 1e-3 * mass_pen
