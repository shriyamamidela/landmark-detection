# evaluate_isbi.py  — millimeter-correct evaluation (ISBI 2015 standard)

import numpy as np
import os
import argparse
import json
import csv
import matplotlib.pyplot as plt


# ============================
# Pixel → millimeter conversion
# ============================
# ISBI 2015 cephalograms: 2400 px height ≈ 187.5 mm
# => 1 px = 187.5 / 2400 = 0.078125 mm
PX_TO_MM = 0.078125


def euclidean_errors_mm(pred: np.ndarray, gt: np.ndarray):
    """
    Computes Euclidean error in millimeters.
    pred, gt: (M, L, 2)
    """
    errs_px = np.linalg.norm(pred - gt, axis=-1)   # in pixels
    errs_mm = errs_px * PX_TO_MM                   # convert to mm
    return errs_mm


def mre_and_sdr(errs_mm: np.ndarray, thresholds=(2.0,2.5,3.0,4.0)):
    flat = errs_mm.flatten()
    mre = float(flat.mean())
    sdr = {t: float((flat <= t).sum()) / flat.size for t in thresholds}
    return mre, sdr


def per_landmark_stats(errs_mm: np.ndarray, thresholds=(2.0,2.5,3.0,4.0)):
    M, L = errs_mm.shape
    stats = []
    for l in range(L):
        arr = errs_mm[:, l]
        stats.append({
            "landmark_index": int(l),
            "mean_mm": float(arr.mean()),
            "std_mm": float(arr.std()),
            "median_mm": float(np.median(arr)),
            "sdr": {t: float((arr <= t).sum())/M for t in thresholds}
        })
    return stats


def cumulative_error_curve_mm(errs_mm, out_png, max_x=10.0, step=0.1):
    flat = errs_mm.flatten()
    xs = np.arange(0.0, max_x+1e-9, step)
    ys = [(flat <= x).mean() for x in xs]

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, lw=2)
    plt.xlabel("Error (mm)")
    plt.ylabel("Proportion within threshold")
    plt.grid(True)
    plt.title("Cumulative Error Curve (mm)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_csv(stats, out_csv):
    header = ["landmark_index","mean_mm","std_mm","median_mm",
              "sdr@2mm","sdr@2.5mm","sdr@3mm","sdr@4mm"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in stats:
            w.writerow([
                s["landmark_index"],
                s["mean_mm"],
                s["std_mm"],
                s["median_mm"],
                s["sdr"][2.0],
                s["sdr"][2.5],
                s["sdr"][3.0],
                s["sdr"][4.0]
            ])


def load_landmarks(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".npz":
        d = np.load(path)
        for k in ["pred","gt","landmarks","arr_0"]:
            if k in d:
                return d[k]
        return d[list(d.files)[0]]
    else:
        raise ValueError("Unsupported format: " + path)


def main(args):
    pred = load_landmarks(args.pred)
    gt = load_landmarks(args.gt)

    # compute mm-based errors
    errs_mm = euclidean_errors_mm(pred, gt)

    # summary
    mre, sdr = mre_and_sdr(errs_mm, thresholds=args.thresholds)
    print("🔹 MRE (mm):", mre)
    print("🔹 SDR (mm):", sdr)

    # per-landmark stats
    stats = per_landmark_stats(errs_mm, thresholds=args.thresholds)

    # save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir,"summary.json"), "w") as f:
        json.dump({
            "MRE_mm": mre,
            "SDR_mm": {str(k): v for k,v in sdr.items()}
        }, f, indent=2)

    save_csv(stats, os.path.join(args.out_dir,"per_landmark.csv"))

    cumulative_error_curve_mm(errs_mm,
        os.path.join(args.out_dir,"cumulative_curve.png"),
        max_x=args.max_x,
        step=args.step)

    print("Saved evaluation results to:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.0,2.5,3.0,4.0])
    parser.add_argument("--max-x", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    args.thresholds = tuple(args.thresholds)
    main(args)
