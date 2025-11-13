import os
import sys
import numpy as np
import cv2

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from preprocessing.utils import generate_edge_bank

# ---- CONFIGURE ----
atlas_img_path = "/content/drive/MyDrive/datasets/ISBI Dataset/Dataset/Training/001.bmp"
save_path = os.path.join(ROOT, "atlas_edge_map.npy")
# -------------------

img = cv2.imread(atlas_img_path)
assert img is not None, f"Image not found: {atlas_img_path}"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get edge bank → take first channel (Canny)
edge_bank = generate_edge_bank(img_rgb)  # (H,W,3)
atlas_edge = edge_bank[..., 0]           # grayscale edge map

np.save(save_path, atlas_edge)

print("Saved atlas edge map →", save_path)
print("Edge map shape:", atlas_edge.shape)
print("DONE ✔")
