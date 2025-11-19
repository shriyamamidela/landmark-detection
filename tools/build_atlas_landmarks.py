# tools/build_atlas_landmarks.py

import os
import numpy as np

LM_PATH = "/content/landmark-detection/datasets/ISBI Dataset/Annotations/Senior Orthodontist/001.txt"
SAVE_FULL = "atlas_landmarks_fullres.npy"
SAVE_RESIZED = "atlas_landmarks_resized.npy"

HEIGHT, WIDTH = 800, 640   # ← cfg.HEIGHT / cfg.WIDTH

# ----------------------
# Load full-res landmarks
# ----------------------
coords = []
with open(LM_PATH, "r") as f:
    for line in f:
        line = line.strip().replace(",", " ")
        parts = line.split()
        if len(parts) == 2:
            x, y = map(float, parts)
            coords.append([x, y])

coords = np.array(coords)
assert coords.shape == (19, 2)

np.save(SAVE_FULL, coords)
print("Saved:", SAVE_FULL, coords.shape)

# ----------------------
# Resize to (800×640)
# ----------------------
# 001.bmp size:
H0, W0 = 2400, 1935

scale_x = WIDTH / W0
scale_y = HEIGHT / H0

lm_resized = coords.copy()
lm_resized[:, 0] *= scale_x
lm_resized[:, 1] *= scale_y

np.save(SAVE_RESIZED, lm_resized)
print("Saved:", SAVE_RESIZED, lm_resized.shape)
print("min/max:", lm_resized.min(), lm_resized.max())
