# data/dataset.py  — FIXED VERSION (LETTERBOX RESIZE)

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
# LETTERBOX RESIZE (preserve aspect ratio!!!)
# ============================================================
def letterbox_resize(image, landmarks, out_h=cfg.HEIGHT, out_w=cfg.WIDTH):
    H, W = image.shape[:2]

    scale = min(out_w / W, out_h / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # pad on bottom-right only
    canvas = np.zeros((out_h, out_w, 3), dtype=resized.dtype)
    canvas[:new_h, :new_w] = resized

    # scale landmarks
    landmarks = landmarks * scale   # scale both x,y

    return canvas, landmarks


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
            self.augmentation = AugmentationSVF(random_flip=True, landmark_shift=True)
        else:
            self.augmentation = None
            print(f"ℹ️ No augmentation applied for mode '{self.mode}'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, landmarks = self.dataset[index]

        # augmentation
        if self.augmentation is not None:
            image, landmarks = self.augmentation.apply(image, landmarks)

        # FIXED RESIZE (NO distortion)
        image, landmarks = letterbox_resize(image, landmarks)

        # force 3-channels
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        # distance transform map
        dt_map = generate_distance_transform(landmarks, (cfg.HEIGHT, cfg.WIDTH)).astype(np.float32)

        # convert to tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        landmarks = torch.from_numpy(landmarks).float()
        dt_map = torch.from_numpy(dt_map).float()

        return image, landmarks, dt_map

    def get_batch(self, indices):
        imgs, lms, dts = [], [], []
        for i in indices:
            img, lm, dt = self[i]
            imgs.append(img)
            lms.append(lm)
            dts.append(dt)
        return torch.stack(imgs), torch.stack(lms), torch.stack(dts)
