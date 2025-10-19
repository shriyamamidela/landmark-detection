from typing import Union
import torch
from config import cfg
import numpy as np
import math
import cv2
import os


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
