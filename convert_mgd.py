"""
Convert .mgd (Mount Wilson merged data) files to the two-column txt format
used by the rest of the codebase.

MGD columns: Star_ID  S-index  JD_rel(from 2444000)  Weight  K/H  CVR  Inst_S  Instr  Date  Time
Output format: JD(= rel + 44000)  S-index
Output files go to data/misc/ as hd{star_id}_caii.txt
"""

import os
import glob

INPUT_DIR  = os.path.join("data", "mount_Wilson_data")
OUTPUT_DIR = os.path.join("data", "misc", "mwd")
JD_OFFSET  = 44000  # converts JD relative to 2444000 → JD relative to 2400000

os.makedirs(OUTPUT_DIR, exist_ok=True)

mgd_files = glob.glob(os.path.join(INPUT_DIR, "*.mgd"))
print(f"Found {len(mgd_files)} .mgd files")

converted = 0
skipped   = 0

for fpath in mgd_files:
    star_id = os.path.splitext(os.path.basename(fpath))[0]
    out_name = f"hd{star_id}_caii.txt"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    rows = []
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                jd_rel = float(parts[2])
                s_index = float(parts[1])
            except ValueError:
                continue
            rows.append((jd_rel + JD_OFFSET, s_index))

    if not rows:
        print(f"  SKIP (no data): {star_id}")
        skipped += 1
        continue

    with open(out_path, "w") as f:
        for jd, s in rows:
            f.write(f"{jd:.5f} {s:.4f}\n")

    converted += 1

print(f"Done: {converted} files converted, {skipped} skipped.")
