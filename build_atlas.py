# build_atlas_fast.py
import numpy as np
import cv2
import os
from atlas_dataset import AtlasDataset

def procrustes_align(X, Y):
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = (U @ Vt)
    X_aligned = (Xc @ R) + muY
    return X_aligned

def build_atlas_fast(root, out):
    print("📦 Loading AtlasDataset...")
    ds = AtlasDataset(root)
    N = len(ds)
    print(f"Found {N} samples.")

    # ---- MEAN SHAPE ----
    print("📌 Computing mean shape...")
    mean_shape = np.zeros((19, 2), dtype=np.float32)
    for i in range(N):
        _, lm = ds[i]
        mean_shape += lm / N

    # ---- FIND BEST REPRESENTATIVE ----
    print("📌 Selecting best atlas candidate...")
    best_i = -1
    best_score = 1e9

    for i in range(N):
        _, lm = ds[i]
        lm_a = procrustes_align(lm, mean_shape)
        score = np.linalg.norm(lm_a - mean_shape)

        if score < best_score:
            best_score = score
            best_i = i

    # ---- FINAL ATLAS DATA ----
    atlas_img, _ = ds[best_i]
    gray = cv2.cvtColor(atlas_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0

    os.makedirs(out, exist_ok=True)

    np.save(os.path.join(out, "atlas_image_resized.npy"), atlas_img)
    np.save(os.path.join(out, "atlas_landmarks_resized.npy"), mean_shape)
    np.save(os.path.join(out, "atlas_edge_map_resized.npy"), edges)

    print("\n✅ DONE! Saved atlas files:")
    print(" atlas_image_resized.npy")
    print(" atlas_landmarks_resized.npy")
    print(" atlas_edge_map_resized.npy")

if __name__ == "__main__":
    build_atlas_fast(
        "/dgxa_home/se22ucse250/landmark-detection-main/datasets/augmented_ceph",
        "/dgxa_home/se22ucse250/landmark-detection-main/atlas"
    )
