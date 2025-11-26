import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# ---------- DISTANCE TRANSFORM FROM LANDMARKS ----------
def compute_dt_from_landmarks(image_shape, landmarks, radius=3):
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for (x, y) in landmarks:
        cx, cy = int(x), int(y)
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    mask_inv = 255 - mask
    dt = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
    dt = dt / (dt.max() + 1e-8)

    return dt.astype(np.float32)


# ---------- DATASET ----------
class AugCephDataset(Dataset):
    """
    Returns 5 items:
    - image tensor
    - landmark tensor (19,2)
    - dt map (1,H,W)
    - edge map (1,H,W)
    - tokens (243,)
    """

    def __init__(self, root):
        self.root = root
        self.img_dir = os.path.join(root, "image_dir")
        self.lbl_dir = os.path.join(root, "label_dir")
        self.tok_dir = os.path.join(root, "token_dir")

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg"))
        ])

        print(f"📦 AugCephDataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def load_landmarks(self, path):
        pts = []
        with open(path, "r") as f:
            for line in f:
                x, y = line.strip().split(",")
                pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # ------------------ IMAGE ------------------
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0

        # ------------------ LANDMARKS ------------------
        lbl_path = os.path.join(self.lbl_dir,
                                fname.replace(".png", ".txt").replace(".jpg", ".txt"))
        landmarks = self.load_landmarks(lbl_path)
        landmarks_t = torch.from_numpy(landmarks).float()   # (19,2)

        # ------------------ DT MAP ------------------
        dt_np = compute_dt_from_landmarks(img_rgb.shape, landmarks)
        dt_t = torch.from_numpy(dt_np).unsqueeze(0).float()

        # ------------------ EDGE MAP ------------------
        edge = cv2.Canny(img.astype(np.uint8), 80, 160).astype(np.float32) / 255.0
        edge_t = torch.from_numpy(edge).unsqueeze(0)

        # ------------------ TOKENS ------------------
        tok_path = os.path.join(self.tok_dir,
                                fname.replace(".png", ".npy").replace(".jpg", ".npy"))
        tokens = np.load(tok_path).astype(np.float32)
        tok_t = torch.from_numpy(tokens)

        return img_t, landmarks_t, dt_t, edge_t, tok_t
