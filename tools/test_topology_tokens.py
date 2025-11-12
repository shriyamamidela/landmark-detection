# quick test (inline)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2, torch, numpy as np
from preprocessing.utils import generate_edge_bank
from preprocessing.topology import extract_arc_tokens_from_edgebank
from matplotlib import pyplot as plt

img_path = "/content/drive/MyDrive/datasets/ISBI Dataset/Dataset/Training/001.bmp"
img = cv2.imread(img_path)
assert img is not None, "image not found: " + img_path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edge = generate_edge_bank(img_rgb)  # (H,W,3)
arcs = extract_arc_tokens_from_edgebank(edge, samples_per_arc=32, curvature_bins=16)

# print summary
for name, info in arcs.items():
    print(name, "arc_length:", info["arc_length"], "curv_hist_sum:", info["curv_hist"].sum())

# viz mandible points
pts = arcs["mandible"]["points"]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(img_rgb); plt.scatter(pts[:,0], pts[:,1], c='r'); plt.title("Mandible spline samples")
plt.subplot(1,2,2); plt.imshow(edge[:,:,0], cmap='gray'); plt.plot(pts[:,0], pts[:,1], '-r'); plt.title("Edges + spline")
plt.show()

