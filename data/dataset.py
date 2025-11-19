# data/dataset.py — FINAL FIXED VERSION (LETTERBOX + EDGE MAP + SAFE SVF AUG)

from data import ISBIDataset, PKUDataset
from preprocessing.augmentation_backbone import AugmentationBackbone
from preprocessing.augmentation_svf import AugmentationSVF
from preprocessing.utils import generate_distance_transform
from paths import Paths

import torch
from torch.utils.data import Dataset as TorchDataset
from config import cfg
import numpy as np
import cv2


# ============================================================
# LETTERBOX RESIZE (preserve aspect ratio)
# ============================================================
def letterbox_resize(image, landmarks, out_h=cfg.HEIGHT, out_w=cfg.WIDTH):
    H, W = image.shape[:2]

    scale = min(out_w / W, out_h / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # pad on bottom-right ONLY
    canvas = np.zeros((out_h, out_w, 3), dtype=resized.dtype)
    canvas[:new_h, :new_w] = resized

    # scale landmarks
    landmarks = landmarks * scale   # scale both x,y

    return canvas, landmarks


# ============================================================
# EDGE MAP GENERATION (safe for SVF)
# ============================================================
def compute_edge_map(image):
    """
    Returns an edge map in shape (H, W) float32 in [0,1].
    Uses Canny but safe parameters for cephalograms.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return edges.astype(np.float32) / 255.0


# ============================================================
# DATASET WRAPPER
# ============================================================
class Dataset(TorchDataset):
    def __init__(self, name, mode, batch_size=1, augmentation=None, shuffle=False):

        if name == "isbi":
            self.dataset = ISBIDataset(Paths.dataset_root_path(name), mode)
        elif name == "pku":
            self.dataset = PKUDataset(Paths.dataset_root_path(name), mode)
        else:
            raise ValueError(f"No dataset '{name}'")

        self.mode = mode.lower().strip()
        self.batch_size = batch_size

        if shuffle:
            self.dataset.shuffle()

        # augmentation choice
        if augmentation is not None:
            self.augmentation = augmentation
            print(f"⚡ Using CUSTOM augmentation for mode '{self.mode}'.")
        elif self.mode == "train":
            print("📌 Using DEFAULT SVF-safe augmentation for training.")
            # SAFE: ONLY flip (no landmark_shift!!)
            self.augmentation = AugmentationSVF(random_flip=True)
        else:
            self.augmentation = None
            print(f"ℹ️ No augmentation applied for mode '{self.mode}'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, landmarks = self.dataset[index]     # numpy (H,W,3), (N,2)

        # ---------- SAFE AUGMENTATION ----------
        if self.augmentation is not None:
            image, landmarks = self.augmentation.apply(image, landmarks)

        # ---------- LETTERBOX RESIZE ----------
        image, landmarks = letterbox_resize(image, landmarks)

        # ---------- FORCE 3-CHANNEL ----------
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        # ---------- EDGE MAP ----------
        edge_map = compute_edge_map(image)    # (H,W)

        # ---------- DT MAP ----------
        dt_map = generate_distance_transform(
            landmarks, (cfg.HEIGHT, cfg.WIDTH)
        ).astype(np.float32)

        # ---------- TO TENSORS ----------
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        landmarks = torch.from_numpy(landmarks).float()
        dt_map = torch.from_numpy(dt_map).float().unsqueeze(0)    # (1,H,W)
        edge_map = torch.from_numpy(edge_map).float().unsqueeze(0)

        # Final output:
        # image: (3,H,W)
        # landmarks: (N,2)
        # dt_map: (1,H,W)
        # edge_map: (1,H,W)
        return image, landmarks, dt_map, edge_map

    def get_batch(self, indices):
        imgs, lms, dts, edges = [], [], [], []
        for i in indices:
            img, lm, dt, edge = self[i]
            imgs.append(img)
            lms.append(lm)
            dts.append(dt)
            edges.append(edge)
        return (
            torch.stack(imgs),
            torch.stack(lms),
            torch.stack(dts),
            torch.stack(edges)
        )
