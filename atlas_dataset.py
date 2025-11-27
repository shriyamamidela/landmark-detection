# atlas_dataset.py
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class AtlasDataset(Dataset):
    """
    Lightweight dataset ONLY for atlas building.
    Loads:
      - images (800×645)
      - landmarks (19×2)
    """
    def __init__(self, root):
        self.img_dir = os.path.join(root, "image_dir")
        self.lbl_dir = os.path.join(root, "label_dir")

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg"))
        ])

        print(f"📦 AtlasDataset loaded: {len(self.files)} samples")

    def load_landmarks(self, path):
        pts = []
        with open(path, "r") as f:
            for line in f:
                x, y = line.strip().split(",")
                pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lm_path = os.path.join(self.lbl_dir, fname.replace(".png",".txt").replace(".jpg",".txt"))
        lm = self.load_landmarks(lm_path)

        return img, lm
