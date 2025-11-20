import numpy as np
import cv2
from data import Dataset
from config import cfg
import os

def procrustes_align(X, Y):
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    normX = np.linalg.norm(Xc)
    normY = np.linalg.norm(Yc)
    if normX == 0 or normY == 0:
        s = 1.0
    else:
        s = normY / normX
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = (U @ Vt).T
    X_aligned = (s * (Xc @ R)) + muY
    return X_aligned

def build_atlas_simple(out_dir="."):
    os.makedirs(out_dir, exist_ok=True)

    ds = Dataset("isbi", "train", batch_size=1, shuffle=False)
    N = len(ds)

    # collect resized images + landmarks
    imgs = []
    lms = []

    for i in range(N):
        img, lm, dt, edge = ds[i]
        img_np = (img.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        imgs.append(img_np)
        lms.append(lm.numpy())

    lms = np.array(lms)  # (N,19,2)

    # compute mean shape
    mean_shape = lms.mean(axis=0)

    # find best atlas candidate = closest to mean shape
    dists = [np.linalg.norm(procrustes_align(lm, mean_shape) - mean_shape) for lm in lms]
    atlas_idx = int(np.argmin(dists))

    atlas_img = imgs[atlas_idx]
    atlas_lms = mean_shape

    # compute atlas edge map
    gray = cv2.cvtColor(atlas_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0

    # save
    np.save(os.path.join(out_dir, "atlas_image_resized.npy"), atlas_img)
    np.save(os.path.join(out_dir, "atlas_landmarks_resized.npy"), atlas_lms)
    np.save(os.path.join(out_dir, "atlas_edge_map_resized.npy"), edges)

    print("DONE!")
    print("Saved:")
    print(" atlas_image_resized.npy")
    print(" atlas_landmarks_resized.npy")
    print(" atlas_edge_map_resized.npy")

build_atlas_simple("/mnt/data")
