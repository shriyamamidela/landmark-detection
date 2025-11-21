import numpy as np
import json

# Load previous evaluation outputs
stats = json.load(open("eval_fixed/summary.json"))
pl = np.genfromtxt("eval_fixed/per_landmark.csv", delimiter=",", skip_header=1)

# per_landmark.csv columns:
# idx, mean_mm, std_mm, median, SDR2, SDR2.5, SDR3, SDR4

print("\n================= ISBI 2015 Style Table =================\n")
print(f"{'Landmark':<6} {'MRE ± SD (mm)':<18} {'SDR@2mm':<10} {'SDR@2.5mm':<10} {'SDR@3mm':<10} {'SDR@4mm':<10}")

rows = []
for row in pl:
    idx = int(row[0])
    mean_ = row[1]
    std_  = row[2]
    s2, s25, s3, s4 = row[4]*100, row[5]*100, row[6]*100, row[7]*100
    
    print(f"L{idx+1:<5} {mean_:4.2f} ± {std_:4.2f}   {s2:6.2f}   {s25:6.2f}   {s3:6.2f}   {s4:6.2f}")
    rows.append([mean_, std_, s2, s25, s3, s4])

rows = np.array(rows)

# Averages
avg_mean = rows[:,0].mean()
avg_std  = rows[:,1].mean()
avg_s2   = rows[:,2].mean()
avg_s25  = rows[:,3].mean()
avg_s3   = rows[:,4].mean()
avg_s4   = rows[:,5].mean()

print("\nAverage",
      f"{avg_mean:4.2f} ± {avg_std:4.2f}",
      f"{avg_s2:6.2f}",
      f"{avg_s25:6.2f}",
      f"{avg_s3:6.2f}",
      f"{avg_s4:6.2f}")
