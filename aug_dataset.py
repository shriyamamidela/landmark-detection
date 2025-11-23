# #writefile aug_dataset.py
# import os
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset
# from PIL import Image


# # ------------------------------- #
# # Helper: compute Distance Transform
# # ------------------------------- #
# def compute_dt_from_landmarks(image_shape, landmarks, radius=3):
#     """
#     Compute a distance transform map D(x,y) where points near landmarks get low values.
#     """
#     h, w = image_shape[0], image_shape[1]

#     # blank mask
#     mask = np.zeros((h, w), dtype=np.uint8)

#     # draw small circles at each landmark location
#     for (x, y) in landmarks:
#         cx, cy = int(x), int(y)
#         if 0 <= cx < w and 0 <= cy < h:
#             cv2.circle(mask, (cx, cy), radius, 255, -1)

#     # invert (foreground=0, background=255)
#     mask_inv = 255 - mask

#     # compute distance transform
#     dt = cv2.distanceTransform(mask_inv, distanceType=cv2.DIST_L2, maskSize=5)

#     # normalize to [0,1]
#     if dt.max() > 0:
#         dt = dt / dt.max()

#     return dt.astype(np.float32)  # H,W


# # ------------------------------- #
# # Augmented Dataset Loader
# # ------------------------------- #
# class AugCephDataset(Dataset):
#     def __init__(self, root):
#         """
#         root/
#             image_dir/
#                 001_aug1.png
#                 002_aug1.png
#             label_dir/
#                 001_aug1.txt
#                 002_aug1.txt
#         """
#         self.root = root
#         self.image_dir = os.path.join(root, "image_dir")
#         self.label_dir = os.path.join(root, "label_dir")

#         self.images = sorted([
#             f for f in os.listdir(self.image_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ])

#     def load_landmarks(self, path):
#         pts = []
#         with open(path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     x, y = line.split(",")
#                     pts.append([float(x), float(y)])
#         return np.array(pts, dtype=np.float32)  # (19,2)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]

#         # load image
#         img_path = os.path.join(self.image_dir, img_name)
#         img = Image.open(img_path).convert("RGB")
#         img_np = np.array(img)  # H,W,3 uint8

#         # load landmarks
#         base = os.path.splitext(img_name)[0]
#         txt_path = os.path.join(self.label_dir, base + ".txt")
#         landmarks = self.load_landmarks(txt_path)

#         # compute DT map
#         dt_map = compute_dt_from_landmarks(img_np.shape, landmarks)  # H,W

#         # convert to tensors
#         img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         dt_t = torch.tensor(dt_map, dtype=torch.float32).unsqueeze(0)  # 1,H,W

#         return img_t, dt_t
# aug_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from preprocessing.utils import generate_distance_transform


class AugCephDataset(Dataset):
    """
    Loads:
    - augmented images  (800x645)
    - corresponding landmarks (.txt)
    - precomputed topology tokens (.npy)
    - generates DT maps on-the-fly from landmarks
    """

    def __init__(self, root):
        super().__init__()
        
        self.root = root
        self.img_dir = os.path.join(root, "image_dir")
        self.lbl_dir = os.path.join(root, "label_dir")
        self.tok_dir = os.path.join(root, "token_dir")

        # All image files
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"📦 AugCephDataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def load_landmarks(self, lbl_path):
        lm = []
        with open(lbl_path, "r") as f:
            for line in f:
                x, y = line.strip().split(",")
                lm.append([float(x), float(y)])
        return np.array(lm, dtype=np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # IMAGE -------------------------
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # (H,W,3) → (3,H,W)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0

        # LANDMARKS ---------------------
        lbl_path = os.path.join(self.lbl_dir, fname.replace(".png", ".txt").replace(".jpg", ".txt"))
        landmarks = self.load_landmarks(lbl_path)

        # DT MAP ------------------------
        dt_np = generate_distance_transform(landmarks, size=img_rgb.shape[:2])
        dt_t = torch.from_numpy(dt_np).float()  # (1, H, W)

        # TOPOLOGY TOKEN ----------------
        tok_path = os.path.join(self.tok_dir, fname.replace(".png",".npy").replace(".jpg",".npy"))
        tokens = np.load(tok_path).astype(np.float32)
        tok_t = torch.from_numpy(tokens)  # shape (243,)

        return img_t, dt_t, tok_t
