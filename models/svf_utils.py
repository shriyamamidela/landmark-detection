import torch
import torch.nn.functional as F

def svf_to_disp(v, steps=7):
    """
    Scaling and squaring to convert stationary velocity field v
    into displacement field.
    v: [B,2,H,W]
    Returns disp: [B,2,H,W]
    """
    disp = v / (2 ** steps)
    for _ in range(steps):
        disp = disp + warp_with_disp(disp, disp)
    return disp


def warp_with_disp(img, disp):
    """
    Warp img using displacement field.
    img: [B,1,H,W] or [B,C,H,W]
    disp: [B,2,H,W]  (dx, dy)
    Returns warped image: [B,C,H,W]
    """
    B, C, H, W = img.shape

    # normalized grid coordinates
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij"
    )
    base_grid = torch.stack((xx, yy), dim=-1)  # [H,W,2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

    # convert flow from pixels to normalized coords
    disp_norm = torch.zeros_like(base_grid)
    disp_norm[..., 0] = disp[:, 0] / (W / 2)
    disp_norm[..., 1] = disp[:, 1] / (H / 2)

    grid = base_grid + disp_norm  # [B,H,W,2]
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border")
