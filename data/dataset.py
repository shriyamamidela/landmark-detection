# data/dataset.py

from data import ISBIDataset, PKUDataset
from preprocessing.augmentation_backbone import AugmentationBackbone    # NEW (heavy aug)
from preprocessing.augmentation_svf import AugmentationSVF              # NEW (light aug)
from preprocessing.utils import generate_distance_transform
from paths import Paths

import torch
from torch.utils.data import Dataset as TorchDataset
from config import cfg
import numpy as np
import cv2
import os


# ----------------------- #
# Resize utility
# ----------------------- #
def resize(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]
    r_h = image_height / cfg.HEIGHT
    r_w = image_width / cfg.WIDTH

    image = cv2.resize(image, (cfg.WIDTH, cfg.HEIGHT), interpolation=cv2.INTER_CUBIC)

    # scale landmark coordinates
    landmarks = np.stack([
        landmarks[:, 0] / r_w,
        landmarks[:, 1] / r_h
    ], axis=-1)

    return image, landmarks


# ----------------------- #
# Main Dataset Class
# ----------------------- #
class Dataset(TorchDataset):

    def __init__(
        self,
        name: str,
        mode: str,
        batch_size: int = 1,
        augmentation=None,      # <—— now accepts any custom augmentation
        shuffle: bool = False,
    ):
        # pick dataset
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

        # ------------------------------------------- #
        # AUGMENTATION HANDLING
        # ------------------------------------------- #
        if augmentation is not None:
            # user explicitly passed augmentation
            self.augmentation = augmentation
            print(f"⚡ Using CUSTOM augmentation for mode '{self.mode}'.")

        elif self.mode == "train":
            # default behavior → SVF minimal aug (safe)
            print("📌 Using DEFAULT SVF-safe augmentation for training.")
            self.augmentation = AugmentationSVF(
                random_flip=True,
                landmark_shift=True
            )

        else:
            self.augmentation = None
            print(f"ℹ️ No augmentation applied for mode '{self.mode}'.")

    # length
    def __len__(self):
        return len(self.dataset)

    # get one item
    def __getitem__(self, index):
        image, landmarks = self.dataset[index]

        # apply augmentation
        if self.augmentation is not None:
            image, landmarks = self.augmentation.apply(image, landmarks)

        # resize
        image, landmarks = resize(image, landmarks)

        # ensure 3 channels
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        # distance transform
        dt_map = generate_distance_transform(landmarks, (cfg.HEIGHT, cfg.WIDTH))
        dt_map = dt_map.astype(np.float32)

        # convert to tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)     # C,H,W
        landmarks = torch.from_numpy(landmarks).float()             # L,2
        dt_map = torch.from_numpy(dt_map).float()                   # 1,H,W

        return image, landmarks, dt_map

    # batch retrieval
    def get_batch(self, indices):
        imgs, lms, dts = [], [], []
        for i in indices:
            img, lm, dt = self[i]
            imgs.append(img)
            lms.append(lm)
            dts.append(dt)
        return torch.stack(imgs), torch.stack(lms), torch.stack(dts)
