#!/usr/bin/env python3
"""Pick the best trial (min no-reg val_loss) of a zs sweep and print its run dir.
Usage: python tools/zs_pick_best.py <results_dir> <runs_exp_dir>
  e.g. python tools/zs_pick_best.py \
       sweeps/pretrain25_zs_phys/pretrain25_zs_phys_002/results runs/pretrain25_zs_phys
Prints: "<best_val_loss> <run_dir>"  (run_dir = <runs_exp_dir>/trial_<hp:04d>)
"""
import glob, json, os, re, sys

results_dir, runs_exp_dir = sys.argv[1], sys.argv[2]
best = None
for f in glob.glob(os.path.join(results_dir, "*.json")):
    m = re.search(r"hp(\d+)_t(\d+)_", os.path.basename(f))
    if not m:
        continue
    try:
        vl = json.load(open(f)).get("val_loss")
    except Exception:
        continue
    if vl is None:
        continue
    hp = int(m.group(1))
    if best is None or vl < best[0]:
        best = (vl, hp, f)

if best is None:
    sys.exit("no valid results")
vl, hp, f = best
run_dir = os.path.join(runs_exp_dir, f"trial_{hp:04d}")
print(f"{vl:.6e} {run_dir} hp={hp}")
