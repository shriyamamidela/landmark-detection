# preprocessing/utils.py
from typing import Union
import torch
from config import cfg
import numpy as np
import math
import cv2
import os
from scipy.ndimage import distance_transform_edt

# -------------------------
# Existing utility functions (kept)
# -------------------------
def craniofacial_region_proposals(
    landmarks: Union[torch.Tensor, np.ndarray],
    image_height: int,
    image_width: int,
    margin: int = cfg.BOX_MARGIN
):
    if isinstance(landmarks, np.ndarray):
        landmarks = torch.from_numpy(landmarks)
    
    landmarks = landmarks.reshape(-1, cfg.NUM_LANDMARKS, 2)

    x_min = landmarks.min(dim=1)[0][:, 0] - margin
    y_min = landmarks.min(dim=1)[0][:, 1] - margin
    x_max = landmarks.max(dim=1)[0][:, 0] + margin
    y_max = landmarks.max(dim=1)[0][:, 1] + margin

    bounding_boxes = torch.stack([
        x_min / image_width,            # x
        y_min / image_height,           # y
        (x_max - x_min) / image_width,  # width
        (y_max - y_min) / image_height  # height
    ], dim=-1)

    return clip_bounding_boxes(bounding_boxes)


def clip_bounding_boxes(boxes):
    boxes = boxes.reshape(-1, 1, 4)

    boxes = transform_bounding_boxes(boxes, mode="xyxy")
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    boxes = transform_bounding_boxes(boxes, mode="xywh")

    return boxes


def decode_bounding_boxes(
    bounding_boxes,
    image_height,
    image_width
):
    bboxes = bounding_boxes.reshape(-1, 1, 4)

    bboxes = torch.stack([
        bboxes[:, :, 0] * image_width,
        bboxes[:, :, 1] * image_height,
        bboxes[:, :, 2] * image_width,
        bboxes[:, :, 3] * image_height],
        dim=2
    )

    return bboxes


def craniofacial_landmark_regions(
    landmarks: torch.Tensor,
    height: int,
    width: int,
    size: int = 3
):
    offset = size / 2

    proposals = torch.stack([
        landmarks[:, :, 0] - offset / width,
        landmarks[:, :, 1] - offset / height,
        landmarks[:, :, 0] + offset / width,
        landmarks[:, :, 1] + offset / height
    ], dim=-1)
    proposals = torch.clamp(proposals, min=0.0, max=1.0)

    return proposals


def encode_cephalometric_landmarks(
    landmarks: torch.Tensor,
    height: int,
    width: int
):
    landmarks = landmarks.reshape(-1, cfg.NUM_LANDMARKS, 2)

    landmarks = torch.stack([
        landmarks[:, :, 0] / width,
        landmarks[:, :, 1] / height
    ], dim=-1)

    return landmarks


def decode_cephalometric_landmarks(
    landmarks: torch.Tensor,
    height: int,
    width: int
):
    landmarks = landmarks.reshape(-1, cfg.NUM_LANDMARKS, 2)

    landmarks = torch.stack([
        landmarks[:, :, 0] * width,
        landmarks[:, :, 1] * height
    ], dim=-1)

    return landmarks


def transform_bounding_boxes(
    boxes: torch.Tensor,
    mode: str
):
    if mode == "xywh":
        x, y, w, h = boxes.split(1, dim=-1)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.cat([x1, y1, x2, y2], dim=-1)
    elif mode == "xyxy":
        x1, y1, x2, y2 = boxes.split(1, dim=-1)
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.cat([x, y, w, h], dim=-1)
    else:
        raise ValueError(f"Mode {mode} not supported")


def rescale_input(
    inputs: torch.Tensor,
    scale: float = 1.0,
    offset: float = 0.0
):
    return inputs * scale + offset


def save_statistics(
    statistics: np.ndarray,
    root_path: str,
    mode: str
):
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, f"{mode}_stats.npy")
    np.save(file_path, statistics)


def load_statistics(
    root_path: str,
    mode: str
):
    file_path = os.path.join(root_path, f"{mode}_stats.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None


def calculate_landmark_accuracy(
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    threshold: float = 2.0
):
    """
    Calculate landmark detection accuracy within a threshold distance
    """
    if isinstance(true_landmarks, np.ndarray):
        true_landmarks = torch.from_numpy(true_landmarks)
    if isinstance(pred_landmarks, np.ndarray):
        pred_landmarks = torch.from_numpy(pred_landmarks)
    
    # Calculate Euclidean distance
    distances = torch.sqrt(
        (true_landmarks[:, :, 0] - pred_landmarks[:, :, 0])**2 +
        (true_landmarks[:, :, 1] - pred_landmarks[:, :, 1])**2
    )
    
    # Calculate accuracy within threshold
    accuracy = (distances <= threshold).float().mean()
    
    return accuracy, distances


def visualize_landmarks(
    image: np.ndarray,
    landmarks: Union[torch.Tensor, np.ndarray],
    save_path: str = None
):
    """
    Visualize landmarks on the image
    """
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(image, str(i), (int(x)+5, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return image

# -------------------------
# New helpers required by the pipeline
# -------------------------
def generate_edge_bank(image: np.ndarray, out_channels: int = 3):
    """
    Create a simple 'edge bank' from the input RGB image.
    Returns HxWx3 uint8 in range [0,255].
    Channels:
      - channel 0: Canny edges
      - channel 1: Sobel magnitude
      - channel 2: Laplacian (or morphological edges fallback)
    This is intentionally simple and fast; replace with steerable / Frangi if you add the dependency.
    """
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny
    canny = cv2.Canny(gray, 50, 150)

    # Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sx**2 + sy**2)
    sobel = np.uint8(np.clip((sobel / (sobel.max()+1e-6)) * 255.0, 0, 255))

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.uint8(np.clip((np.abs(lap) / (np.abs(lap).max()+1e-6)) * 255.0, 0, 255))

    bank = np.stack([canny, sobel, lap], axis=-1)  # H,W,3
    if out_channels != 3:
        # simple channel adjustments (repeat / truncate)
        if out_channels < 3:
            bank = bank[:, :, :out_channels]
        else:
            bank = np.tile(bank[:, :, :1], (1, 1, out_channels))

    return bank  # uint8 HxWxC

def generate_distance_transform(landmarks: np.ndarray, size: tuple):
    """
    landmarks: (N,2) in pixel coordinates relative to target size OR in the same coord system used
    size: (H, W)
    Returns: numpy array shape (1, H, W) float32 normalized [0, 1]
    """
    H, W = size
    heat = np.zeros((H, W), dtype=np.uint8)

    # If landmarks appear normalized (0..1), convert
    lm = np.array(landmarks, dtype=np.float32)
    if lm.max() <= 1.0:
        lm[:, 0] = lm[:, 0] * W
        lm[:, 1] = lm[:, 1] * H

    for (x, y) in lm:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            heat[yi, xi] = 1

    # small gaussian blur so points occupy small area (helps DT stability)
    heat = cv2.GaussianBlur(heat.astype(np.float32), (7, 7), 0)
    # Normalize to [0,1] then invert for distance transform (distance from landmark)
    binary = (heat > 1e-6).astype(np.uint8)
    dt = distance_transform_edt(1 - binary)
    # Normalize to 0..1 (optionally invert so closer = larger value)
    if dt.max() > 0:
        dt = dt.astype(np.float32) / (dt.max() + 1e-8)
        # make closer to landmark have larger values (optional): invert
        dt = 1.0 - dt
    else:
        dt = np.zeros_like(dt, dtype=np.float32)

    return np.expand_dims(dt.astype(np.float32), axis=0)  # shape (1, H, W)
# ================================================================
# AUGMENTATION FUNCTIONS REQUIRED BY augmentation_backbone.py
# ================================================================

def identity(image):
    """Return image unchanged."""
    return image


def inversion(image):
    """Invert image intensities."""
    return 255 - image


def solarization(image, threshold=128):
    """Solarize image like research paper."""
    img = image.copy()
    mask = img < threshold
    img[mask] = 255 - img[mask]
    return img


def high_pass_filtering(image):
    """Apply Laplacian high-pass filtering."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip((lap - lap.min()) / (lap.max() - lap.min() + 1e-8) * 255, 0, 255)
    lap = lap.astype(np.uint8)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def unsharp_masking(image, amount=1.0, radius=5):
    """Sharpen image using Gaussian blur."""
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharp


def contrast_limited_histogram_equalization(image):
    """CLAHE enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def horizontal_flip(image, landmarks):
    """Research-paper style flip with landmarks."""
    h, w = image.shape[:2]
    flipped_img = np.ascontiguousarray(image[:, ::-1])
    flipped_lm = landmarks.copy()
    flipped_lm[:, 0] = w - flipped_lm[:, 0]
    return flipped_img, flipped_lm


def random_cropping(image, landmarks, max_crop=50):
    """Random crop used in research paper."""
    H, W = image.shape[:2]
    crop_x = np.random.randint(0, max_crop)
    crop_y = np.random.randint(0, max_crop)

    cropped = image[crop_y:H, crop_x:W]
    cropped = cv2.resize(cropped, (W, H))

    lm = landmarks.copy()
    lm[:, 0] = (landmarks[:, 0] - crop_x) * (W / (W - crop_x))
    lm[:, 1] = (landmarks[:, 1] - crop_y) * (H / (H - crop_y))
    return cropped, lm


def vertical_shift(image, landmarks, max_shift=20):
    """Vertical jitter."""
    shift = np.random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    shifted_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    lm = landmarks.copy()
    lm[:, 1] += shift
    return shifted_img, lm
