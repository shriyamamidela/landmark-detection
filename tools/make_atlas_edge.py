# tools/build_atlas_edge_map.py

import os
import cv2
import numpy as np
from preprocessing.utils import generate_edge_bank

IMG = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Training/001.bmp"
SAVE_PATH = "atlas_edge_map.npy"

img = cv2.imread(IMG)
assert img is not None
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bank = generate_edge_bank(img)   # (H,W,3)
edge = bank[..., 0]              # Canny channel

np.save(SAVE_PATH, edge)

print("Saved atlas_edge_map.npy:", edge.shape, edge.min(), edge.max())
