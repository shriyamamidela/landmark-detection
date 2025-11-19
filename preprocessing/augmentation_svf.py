"""
# augmentations_svf.py
# Safe augmentations for SVF (Atlas-Flow) training.
# - Photometric-only augmentations (CLAHE, solarize, brightness, saturation, DCP dehaze, low gaussian noise)
# - Optional horizontal flip that flips all paired inputs consistently
#
# Usage:
#   from augmentations_svf import apply_svf_augment
#   img, lms, dt, edge, feats = apply_svf_augment(img, lms, dt_map=dt, edge_map=edge, features=feats)
#
# Dependencies: numpy, cv2
"""
import numpy as np
import cv2

def clahe_rgb(img, clip=2.0, tile=(8,8)):
    # Apply CLAHE on each channel independently (for grayscale images it still works)
    if img is None:
        return None
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        return clahe.apply(img)
    out = np.empty_like(img)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    for c in range(img.shape[2]):
        out[:,:,c] = clahe.apply(img[:,:,c])
    return out

def solarize(img, threshold=128):
    # Simple solarization: invert pixels above a threshold
    if img is None:
        return None
    out = img.copy()
    out[out > threshold] = 255 - out[out > threshold]
    return out

def adjust_brightness(img, delta=0.0):
    # delta in [-1.0, 1.0]; image assumed uint8 [0,255]
    if img is None:
        return None
    out = img.astype(np.float32)
    out = out * (1.0 + delta)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

def adjust_saturation(img, factor=1.0):
    # For grayscale or single channel images this is a no-op
    if img is None:
        return None
    if img.ndim == 2 or img.shape[2] == 1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= factor
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def gaussian_noise(img, sigma=2.0):
    if img is None:
        return None
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

def dcp_dehaze(img, omega=0.95, t_min=0.1, win=15):
    """
    A lightweight DCP dehazing approximation.
    Works on uint8 BGR images. If image is grayscale, returns input.
    This function is intentionally simple and fast; remove or replace with a better
    implementation if you have a dedicated dehazing library.
    """
    if img is None:
        return None
    if img.ndim == 2 or img.shape[2] == 1:
        return img
    # Convert to normalized float [0,1]
    I = img.astype(np.float32) / 255.0
    # Estimate dark channel
    min_channel = np.min(I, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dark = cv2.erode(min_channel, kernel)
    # Estimate atmospheric light (approx: top 0.1% brightest in dark channel)
    flat_dark = dark.ravel()
    numpx = max(1, int(0.001 * flat_dark.size))
    idx = np.argpartition(flat_dark, -numpx)[-numpx:]
    A = np.max(I.reshape(-1,3)[idx], axis=0)
    # Estimate transmission
    transmission = 1 - omega * dark
    transmission = np.clip(transmission, t_min, 1.0)
    # Recover radiance
    t = transmission[..., None]
    J = (I - A) / t + A
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)

def horizontal_flip(image, landmarks):
    """
    Flip image horizontally and adjust landmarks (landmarks in (N,2) with x,y).
    Assumes image width W; new_x = W - 1 - x
    """
    if image is None:
        return None, landmarks
    H, W = image.shape[:2]
    imf = image[:, ::-1].copy()
    if landmarks is None:
        return imf, None
    lms = landmarks.copy()
    lms[:,0] = (W - 1) - lms[:,0]
    return imf, lms

def apply_svf_augment(image, landmarks, dt_map=None, edge_map=None, features=None,
                      p_flip=0.5, do_clahe=True, do_solarize=False,
                      brightness_delta=0.0, saturation_factor=1.0,
                      do_dehaze=False, noise_sigma=0.0):
    """
    Apply safe, consistent augmentations for SVF training.
    Inputs:
      - image: uint8 BGR or grayscale image (H,W) or (H,W,3)
      - landmarks: numpy array (N,2) in pixel coords
      - dt_map: distance transform map (H,W, C) or None
      - edge_map: single channel (H,W) or None
      - features: backbone features (Hf,Wf,Cf) or None
    Returns:
      augmented (image, landmarks, dt_map, edge_map, features)
    Notes:
      - Only horizontal flip changes geometry, and when performed it's applied to ALL inputs consistently.
      - Landmark coordinates are NOT jittered here.
    """
    img = image.copy() if image is not None else None
    lms = landmarks.copy() if landmarks is not None else None
    dt = dt_map.copy() if dt_map is not None else None
    edge = edge_map.copy() if edge_map is not None else None
    feats = features.copy() if features is not None else None

    # Photometric ops (order: dehaze -> clahe -> brightness/saturation -> solarize -> noise)
    if do_dehaze:
        img = dcp_dehaze(img)

    if do_clahe:
        img = clahe_rgb(img)

    if brightness_delta != 0.0:
        img = adjust_brightness(img, brightness_delta)

    if saturation_factor != 1.0:
        img = adjust_saturation(img, saturation_factor)

    if do_solarize:
        img = solarize(img, threshold=128)

    if noise_sigma and noise_sigma > 0:
        img = gaussian_noise(img, sigma=noise_sigma)

    # Horizontal flip (geometry) applied consistently
    if (p_flip > 0) and (np.random.rand() < p_flip):
        img, lms = horizontal_flip(img, lms)
        if dt is not None:
            dt = dt[:, ::-1, ...].copy() if dt.ndim == 3 else dt[:, ::-1].copy()
        if edge is not None:
            edge = edge[:, ::-1].copy()
        if feats is not None:
            # Assuming features are (Hf, Wf, C)
            feats = feats[:, ::-1, :].copy()

    return img, lms, dt, edge, feats
