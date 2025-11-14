# tools/warp_utils.py
import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------
# 1. Upsample displacement (low-res → full-res)
# ---------------------------------------------------------
def upsample_disp_to_fullres(disp_lowres, H_img, W_img):
    """
    disp_lowres : numpy or torch tensor [1,2,Hf,Wf]
    returns disp_up : numpy [H_img, W_img, 2]
    """
    if isinstance(disp_lowres, np.ndarray):
        disp_lowres = torch.from_numpy(disp_lowres).float()

    disp = disp_lowres.float()  # [1,2,Hf,Wf]

    disp_up = F.interpolate(
        disp,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False
    )  # [1,2,H,W]

    disp_up = disp_up[0].permute(1, 2, 0).cpu().numpy()  # -> (H, W, 2)

    return disp_up


# ---------------------------------------------------------
# 2. Warp landmarks using full-res displacement
# ---------------------------------------------------------
def warp_landmarks_fullres(atlas_landmarks_px, disp_up):
    """
    atlas_landmarks_px : (N,2) array of (x,y) in pixel space
    disp_up           : (H,W,2) numpy displacement map
    returns warped landmarks (N,2)
    """

    warped = []
    H, W, _ = disp_up.shape

    for (x, y) in atlas_landmarks_px:
        xi = np.clip(int(x), 0, W - 1)
        yi = np.clip(int(y), 0, H - 1)
        dx, dy = disp_up[yi, xi]
        warped.append([x + dx, y + dy])

    return np.array(warped)


# ---------------------------------------------------------
# 3. Wrapper called by refine_svf.py
# ---------------------------------------------------------
def warp_landmarks_wrapper(atlas_landmarks_px, disp_lowres, H_img, W_img):
    """
    Wraps the two steps:
        - upsample displacement to full resolution
        - warp landmarks using the upsampled displacement

    Returns: numpy array (N,2) of warped landmark pixel coords.
    """

    disp_up = upsample_disp_to_fullres(disp_lowres, H_img, W_img)
    warped = warp_landmarks_fullres(atlas_landmarks_px, disp_up)

    return warped
