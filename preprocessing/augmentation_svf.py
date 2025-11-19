"""
augmentation_svf.py
Safe augmentations for SVF (Atlas-Flow) training.

This file provides:
 - Photometric-only augmentations (CLAHE, brightness, saturation, solarize, DCP dehaze)
 - Optional safe horizontal flip
 - AugmentationSVF class used directly in dataset.py
"""

import numpy as np
import cv2


# --------------------------------------------------------
# Photometric transforms
# --------------------------------------------------------
def clahe_rgb(img, clip=2.0, tile=(8,8)):
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
    if img is None:
        return None
    out = img.copy()
    mask = out > threshold
    out[mask] = 255 - out[mask]
    return out


def adjust_brightness(img, delta=0.0):
    if img is None:
        return None
    out = img.astype(np.float32) * (1.0 + delta)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def adjust_saturation(img, factor=1.0):
    if img is None:
        return None
    if img.ndim == 2 or img.shape[2] == 1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_noise(img, sigma=2.0):
    if img is None:
        return None
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def dcp_dehaze(img, omega=0.95, t_min=0.1, win=15):
    if img is None or img.ndim == 2 or img.shape[2] == 1:
        return img
    I = img.astype(np.float32) / 255.0
    min_ch = np.min(I, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dark = cv2.erode(min_ch, kernel)

    flat = dark.ravel()
    numpx = max(1, int(0.001 * flat.size))
    idx = np.argpartition(flat, -numpx)[-numpx:]
    A = np.max(I.reshape(-1, 3)[idx], axis=0)

    transmission = 1 - omega * dark
    transmission = np.clip(transmission, t_min, 1.0)
    
    J = (I - A) / transmission[..., None] + A
    J = np.clip(J, 0, 1)

    return (J * 255).astype(np.uint8)


# --------------------------------------------------------
# Geometry: safe flip
# --------------------------------------------------------
def horizontal_flip(image, landmarks):
    if image is None:
        return None, landmarks
    H, W = image.shape[:2]
    imf = image[:, ::-1].copy()
    if landmarks is None:
        return imf, None
    lms = landmarks.copy()
    lms[:, 0] = (W - 1) - lms[:, 0]
    return imf, lms


# --------------------------------------------------------
# Core augmentation function
# --------------------------------------------------------
def apply_svf_augment(
    image, 
    landmarks, 
    dt_map=None, 
    edge_map=None, 
    features=None,
    p_flip=0.5,
    do_clahe=True,
    do_solarize=False,
    brightness_delta=0.0,
    saturation_factor=1.0,
    do_dehaze=False,
    noise_sigma=0.0,
):
    img = image.copy() if image is not None else None
    lms = landmarks.copy() if landmarks is not None else None
    dt = dt_map.copy() if dt_map is not None else None
    edge = edge_map.copy() if edge_map is not None else None
    feats = features.copy() if features is not None else None

    # ----- Photometric-only (safe) -----
    if do_dehaze:
        img = dcp_dehaze(img)
    if do_clahe:
        img = clahe_rgb(img)
    if brightness_delta != 0.0:
        img = adjust_brightness(img, brightness_delta)
    if saturation_factor != 1.0:
        img = adjust_saturation(img, saturation_factor)
    if do_solarize:
        img = solarize(img)
    if noise_sigma > 0:
        img = gaussian_noise(img, sigma=noise_sigma)

    # ----- Geometry -----
    if np.random.rand() < p_flip:
        img, lms = horizontal_flip(img, lms)
        if dt is not None:
            dt = dt[:, ::-1].copy() if dt.ndim == 2 else dt[:, ::-1, :].copy()
        if edge is not None:
            edge = edge[:, ::-1].copy()
        if feats is not None:
            feats = feats[:, ::-1, :].copy()

    return img, lms, dt, edge, feats


# --------------------------------------------------------
# Wrapper class (used by Dataset)
# --------------------------------------------------------
class AugmentationSVF:
    """
    Minimal safe augmentation wrapper for SVF training.
    Only performs photometric transforms + horizontal flip.
    No landmark jitter. No geometric distortion except flipping.
    """

    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def apply(self, image, landmarks):
        img, lms, _, _, _ = apply_svf_augment(
            image,
            landmarks,
            p_flip=1.0 if self.random_flip else 0.0,
            do_clahe=True,
            do_solarize=False,
            brightness_delta=0.0,
            saturation_factor=1.0,
            do_dehaze=False,
            noise_sigma=0.0,
        )
        return img, lms
