# tools/build_atlas_image.py

import os
import cv2
import numpy as np

ATLAS_IMG = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
SAVE_PATH = "atlas_image_fullres.npy"

img = cv2.imread(ATLAS_IMG)
assert img is not None, "Atlas image not found."

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB for consistency
np.save(SAVE_PATH, img)

print("Saved:", SAVE_PATH, img.shape)
