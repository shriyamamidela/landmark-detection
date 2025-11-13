import os
import numpy as np
import glob

# Path to your annotation folder
ANNOT_DIR = "/content/drive/MyDrive/datasets/ISBI Dataset/Annotations/Junior Orthodontist/"

def load_landmarks(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "," not in line:
                continue  # skip extra "2" at end
            x, y = line.split(",")
            pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)[:19]  # ensure only 19 points

all_pts = []

files = sorted(glob.glob(os.path.join(ANNOT_DIR, "*.txt")))
print(f"Found {len(files)} annotation files.")

for fp in files:
    lm = load_landmarks(fp)
    if lm.shape[0] != 19:
        print("Warning: bad file:", fp)
        continue
    all_pts.append(lm)

all_pts = np.stack(all_pts, axis=0)   # (N, 19, 2)
print("Loaded landmark shape:", all_pts.shape)

# Compute mean shape (atlas)
atlas = np.mean(all_pts, axis=0)   # (19,2)

print("Atlas shape:", atlas.shape)
print("First 5 atlas landmarks:")
print(atlas[:5])

# Save atlas
np.save("atlas_landmarks_19.npy", atlas)
print("Saved → atlas_landmarks_19.npy")
