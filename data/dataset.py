from data import ISBIDataset, PKUDataset
from preprocessing import Augmentation
from paths import Paths
import torch
from torch.utils.data import Dataset as TorchDataset
from config import cfg
import numpy as np
import cv2
import os


def resize(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]
    ratio_height, ratio_width = (image_height / cfg.HEIGHT), (image_width / cfg.WIDTH)

    image = cv2.resize(np.array(image), dsize=(cfg.WIDTH, cfg.HEIGHT), interpolation=cv2.INTER_CUBIC)
    landmarks = np.vstack([
        landmarks[:, 0] / ratio_width,
        landmarks[:, 1] / ratio_height
    ]).T

    return image, landmarks


class Dataset(TorchDataset):
    def __init__(
        self,
        name: str,
        mode: str,
        batch_size: int = 1,
        augmentation: Augmentation = None,
        shuffle: bool = False,
    ):
        # Select dataset type
        if name == "isbi":
            self.dataset = ISBIDataset(Paths.dataset_root_path(name), mode)
        elif name == "pku":
            self.dataset = PKUDataset(Paths.dataset_root_path(name), mode)
        else:
            raise ValueError(f"'{name}' no such dataset exists in your datasets repository.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode.lower().strip()

        if self.shuffle:
            self.dataset.shuffle()

        # ------------------------------------------------------------------ #
        # ✅ Automatically enable augmentation in training mode
        # ------------------------------------------------------------------ #
        if augmentation is not None:
            self.augmentation = augmentation
            print(f"✅ Using custom augmentation for mode '{self.mode}'")
        elif self.mode == "train":
            print("✅ Enabling default augmentation for training mode")
            self.augmentation = Augmentation(
                clahe=True,          # Contrast enhancement
                random_flip=True,    # Random horizontal flips
                random_shift=True,   # Landmark-preserving translation
                unsharp_mask=True    # Edge sharpening
            )
        else:
            self.augmentation = None
            print(f"ℹ️ No augmentation applied for mode '{self.mode}'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, landmarks = self.dataset[index]

        # Apply augmentation if available
        if self.augmentation is not None:
            image, landmarks = self.augmentation.apply(image, landmarks)

        # Resize image and landmarks to cfg-defined size
        image, landmarks = resize(image, landmarks)

        # Convert to PyTorch tensors (channels-first format)
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        landmarks = torch.from_numpy(landmarks).float()

        return image, landmarks

    def get_batch(self, indices):
        """Get a batch of data"""
        images = []
        labels = []

        for idx in indices:
            image, landmarks = self[idx]
            images.append(image)
            labels.append(landmarks)

        return torch.stack(images), torch.stack(labels)
