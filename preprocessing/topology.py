# preprocessing/topology.py
import numpy as np
import cv2
from scipy import interpolate
from skimage import measure
from typing import Tuple, Dict, List

def _find_prominent_contours(edge_img: np.ndarray, min_length: int = 100) -> List[np.ndarray]:
    """Find contours in a single-channel edge image and return long ones."""
    contours = measure.find_contours(edge_img.astype(np.uint8), level=0.5)
    long_contours = []
    for cnt in contours:
        if cnt.shape[0] >= min_length:
            # find_contours returns (row, col) — convert to (x, y)
            pts = np.stack([cnt[:, 1], cnt[:, 0]], axis=-1)
            long_contours.append(pts)
    return long_contours

def _fit_spline(pts: np.ndarray, n_samples: int = 128, k: int = 3, s: float = 3.0) -> np.ndarray:
    """Parametric spline fit + uniform re-sampling to n_samples points.
       pts: (N,2) array
       returns sampled_pts: (n_samples, 2)
    """
    if pts.shape[0] < (k + 1):
        # fallback: linear interpolation
        t = np.linspace(0, 1, n_samples)
        xs = np.interp(t, np.linspace(0, 1, pts.shape[0]), pts[:, 0])
        ys = np.interp(t, np.linspace(0, 1, pts.shape[0]), pts[:, 1])
        return np.stack([xs, ys], axis=-1)
    # param
    dist = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    u = np.concatenate([[0.0], np.cumsum(dist)])
    u = u / (u[-1] + 1e-8)
    try:
        tck_x = interpolate.splrep(u, pts[:, 0], k=k, s=s)
        tck_y = interpolate.splrep(u, pts[:, 1], k=k, s=s)
        u_fine = np.linspace(0.0, 1.0, n_samples)
        xs = interpolate.splev(u_fine, tck_x)
        ys = interpolate.splev(u_fine, tck_y)
        pts_s = np.stack([xs, ys], axis=-1)
        return pts_s
    except Exception:
        # fallback to linear sample
        t = np.linspace(0, 1, n_samples)
        xs = np.interp(t, np.linspace(0, 1, pts.shape[0]), pts[:, 0])
        ys = np.interp(t, np.linspace(0, 1, pts.shape[0]), pts[:, 1])
        return np.stack([xs, ys], axis=-1)

def _arc_length(pts: np.ndarray) -> float:
    seg = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    return float(seg.sum())

def _curvature(pts: np.ndarray) -> np.ndarray:
    """Compute discrete curvature along sampled pts.
       pts: (N,2). Return curvature array of length N (pad ends).
    """
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-8
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa

def curvature_histogram(pts: np.ndarray, n_bins: int = 16, range_abs: float = 0.1) -> np.ndarray:
    kappa = _curvature(pts)
    hist, _ = np.histogram(np.abs(kappa), bins=n_bins, range=(0.0, range_abs))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (hist.sum() + 1e-8)
    return hist  # shape (n_bins,)

def extract_arc_tokens_from_edgebank(edge_bank: np.ndarray,
                                     search_regions: Dict[str, Tuple[int,int,int,int]] = None,
                                     samples_per_arc: int = 32,
                                     curvature_bins: int = 16) -> Dict[str, np.ndarray]:
    """
    edge_bank: (H, W, C) float in [0,1]. We will use first channel (canny) for contours.
    search_regions: optional dict of bounding rectangles (x, y, w, h) for arcs to restrict contour selection.
                    keys: "mandible", "maxilla", "cranial_base"
    Returns a dict per arc:
       {
         "points": (samples_per_arc, 2),
         "arc_length": float,
         "curv_hist": (curvature_bins,)
       }
    """
    H, W = edge_bank.shape[0:2]
    canny = (edge_bank[..., 0] * 255.0).astype(np.uint8)
    contours = _find_prominent_contours(canny, min_length=80)

    # If search_regions provided, filter contours by overlap with region bbox center.
    def _in_region(cnt, region):
        if region is None:
            return True
        x,y,w,h = region
        cx, cy = cnt.mean(axis=0)
        return (cx >= x and cx <= x + w and cy >= y and cy <= y + h)

    arcs = {"mandible": None, "maxilla": None, "cranial_base": None}
    # Basic heuristic: choose longest contour in region for each arc
    for name in list(arcs.keys()):
        reg = None
        if search_regions and name in search_regions:
            reg = search_regions[name]
        best = None
        best_len = 0
        for cnt in contours:
            if not _in_region(cnt, reg):
                continue
            L = cnt.shape[0]
            if L > best_len:
                best = cnt
                best_len = L
        if best is None and contours:
            # fallback to global longest
            best = max(contours, key=lambda x: x.shape[0])
        if best is None:
            # empty placeholder: straight line across center
            pts = np.stack([np.linspace(W*0.1, W*0.9, samples_per_arc), np.full(samples_per_arc, H/2)], axis=-1)
            sampled = _fit_spline(pts, n_samples=samples_per_arc)
            arcs[name] = {"points": sampled, "arc_length": _arc_length(sampled),
                          "curv_hist": curvature_histogram(sampled, n_bins=curvature_bins)}
        else:
            sampled = _fit_spline(best, n_samples=samples_per_arc)
            arcs[name] = {"points": sampled, "arc_length": _arc_length(sampled),
                          "curv_hist": curvature_histogram(sampled, n_bins=curvature_bins)}
    return arcs

def flatten_arc_tokens(arcs_dict):
    """
    Convert arc dict (output of extract_arc_tokens_from_edgebank)
    into a single flat vector T (np.ndarray, shape [243]).
    """
    feat_list = []
    for name in ["mandible", "maxilla", "cranial_base"]:
        arc = arcs_dict[name]
        pts = arc["points"].flatten()           # (64,)
        arc_len = np.array([arc["arc_length"]]) # (1,)
        curv = arc["curv_hist"].flatten()       # (16,)
        feat = np.concatenate([pts, arc_len, curv])
        feat_list.append(feat)
    T = np.concatenate(feat_list).astype(np.float32)
    return T

