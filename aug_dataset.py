import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# ---------- DISTANCE TRANSFORM FROM LANDMARKS ----------
def compute_dt_from_landmarks(image_shape, landmarks, radius=3):
    """
    image_shape: (H,W,3) or (H,W)
    landmarks: Nx2 array (x,y in pixels)
    returns: normalized dt (float32) in range [0,1]
    """
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for (x, y) in landmarks:
        cx, cy = int(round(x)), int(round(y))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    mask_inv = 255 - mask
    # If all zeros, distanceTransform returns zeros
    dt = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
    if dt.max() > 0:
        dt = dt / (dt.max() + 1e-12)
    return dt.astype(np.float32)


class AugCephDataset(Dataset):
    """
    Robust Augmented Ceph Dataset loader.

    Returns always:
      img_t    : torch.FloatTensor (3,H,W) in [0,1]
      lm_t     : torch.FloatTensor (L,2)  landmarks (float pixels)
      dt_t     : torch.FloatTensor (1,H,W) normalized DT
      edge_t   : torch.FloatTensor (1,H,W) edge map (0..1)
      tok_t    : torch.FloatTensor (243,) topology token (zeros if missing)
    """

    def __init__(self, root):
        self.root = root

        # detect image dir (train uses image_dir, tests use images)
        cand_img_dirs = ["image_dir", "images", "img", "images_dir"]
        cand_label_dirs = ["label_dir", "labels", "label", "labels_dir"]
        cand_token_dirs = ["token_dir", "tokens", "token"]

        self.img_dir = None
        self.lbl_dir = None
        self.tok_dir = None

        for d in cand_img_dirs:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                self.img_dir = p
                break
        for d in cand_label_dirs:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                self.lbl_dir = p
                break
        for d in cand_token_dirs:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                self.tok_dir = p
                break

        # fallback: maybe images are directly in root
        if self.img_dir is None:
            # try to find a directory containing many image files
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    # count images
                    imgs = [f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
                    if len(imgs) > 0:
                        # assume this is images dir if labels sibling exists
                        self.img_dir = p
                        break
            # else try root as images
            if self.img_dir is None:
                imgs = [f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
                if len(imgs) > 0:
                    self.img_dir = root

        if self.img_dir is None:
            raise FileNotFoundError(f"No images directory found under {root}. Tried candidates: {cand_img_dirs}")

        # Build file list
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in {self.img_dir}")

        # Info print
        print(f"📦 AugCephDataset loaded: {len(self.files)} samples")
        print(f"  image_dir: {self.img_dir}")
        print(f"  label_dir: {self.lbl_dir if self.lbl_dir is not None else '(none)'}")
        print(f"  token_dir: {self.tok_dir if self.tok_dir is not None else '(none)'}")

        # default token dim (keep consistent across pipeline)
        self.token_dim = 243

    def __len__(self):
        return len(self.files)

    def load_landmarks(self, path):
        pts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # allow "x,y" or "x y"
                if "," in line:
                    x, y = line.split(",")
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = parts[0], parts[1]
                    else:
                        raise ValueError(f"Bad landmark line: {line}")
                pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # image tensor
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0  # (3,H,W)

        # landmarks
        lm_fname = fname
        # map image filename to label filename: replace extension with .txt
        lbl_name = os.path.splitext(lm_fname)[0] + ".txt"
        lm_path = None
        if self.lbl_dir is not None:
            candidate = os.path.join(self.lbl_dir, lbl_name)
            if os.path.exists(candidate):
                lm_path = candidate
        # fallback: maybe labels within image dir (same name .txt)
        if lm_path is None:
            candidate = os.path.join(self.img_dir, lbl_name)
            if os.path.exists(candidate):
                lm_path = candidate

        if lm_path is None:
            raise FileNotFoundError(f"Landmark file not found for image {fname} (expected {lbl_name})")

        landmarks = self.load_landmarks(lm_path)   # (L,2)
        landmarks_t = torch.from_numpy(landmarks).float()

        # DT map
        dt_np = compute_dt_from_landmarks(img_rgb.shape, landmarks)
        dt_t = torch.from_numpy(dt_np).unsqueeze(0).float()  # (1,H,W)

        # Edge map (Canny) - single channel float [0,1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
        edge_t = torch.from_numpy(edge).unsqueeze(0).float()

        # Tokens (optional)
        tok_fname = os.path.splitext(fname)[0] + ".npy"
        tok_path = None
        if self.tok_dir is not None:
            candidate = os.path.join(self.tok_dir, tok_fname)
            if os.path.exists(candidate):
                tok_path = candidate
        # fallback: tokens next to image?
        if tok_path is None:
            candidate = os.path.join(self.img_dir, tok_fname)
            if os.path.exists(candidate):
                tok_path = candidate

        if tok_path is not None:
            tokens = np.load(tok_path).astype(np.float32)
            # if token vector shorter/longer than expected, pad/trim
            if tokens.size != self.token_dim:
                tnew = np.zeros((self.token_dim,), dtype=np.float32)
                n = min(tokens.size, self.token_dim)
                tnew[:n] = tokens.ravel()[:n]
                tokens = tnew
            tok_t = torch.from_numpy(tokens).float()
        else:
            # RETURN ZERO TOKENS so training code won't break (tests typically have no tokens)
            tok_t = torch.zeros((self.token_dim,), dtype=torch.float32)

        return img_t, landmarks_t, dt_t, edge_t, tok_t
