#!/usr/bin/env python3
"""
fix_val_losses.py — One-time migration to correct val_loss in existing result JSONs.

The bug: val_loss was stored as the arithmetic mean of per-dataset no-reg losses.
The fix: replace it with the geometric mean of proc_val_losses (regularized per-dataset
losses stored in the JSON). For trials with small lambda the reg term is negligible,
so geomean(proc_val_losses) ≈ geomean(proc_val_losses_no_reg). For trials with large
lambda the model is bad anyway and won't be selected by the analysis.

Usage:
    python sweep/fix_val_losses.py [--dry-run] [--cell CELL_NAME]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEP_BASE  = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")


def fix_result_file(json_path: str, dry_run: bool) -> str:
    with open(json_path) as f:
        result = json.load(f)

    if "proc_val_losses_no_reg" in result:
        return "skip (already has proc_val_losses_no_reg)"

    pvl = result.get("proc_val_losses")
    if not pvl:
        return "skip (no proc_val_losses)"

    vals = [v for v in pvl.values() if v is not None and v > 0]
    if not vals:
        return "skip (no valid proc_val_losses values)"

    val_loss_corrected = float(np.exp(np.mean(np.log(vals))))
    old_val_loss = result["val_loss"]

    if not dry_run:
        result["val_loss"] = val_loss_corrected
        with open(json_path, "w") as f:
            json.dump(result, f)

    return f"fixed  old={old_val_loss:.5f}  new={val_loss_corrected:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell", default=None, help="process only this cell name")
    args = parser.parse_args()

    if not os.path.isdir(SWEEP_BASE):
        sys.exit(f"SWEEP_BASE not found: {SWEEP_BASE}")

    cell_names = sorted(os.listdir(SWEEP_BASE))
    if args.cell:
        cell_names = [c for c in cell_names if c == args.cell]
        if not cell_names:
            sys.exit(f"Cell not found: {args.cell}")

    total, fixed, skipped = 0, 0, 0
    for cell_name in cell_names:
        if not cell_name.startswith("scaling_p1"):
            continue
        results_dir = os.path.join(SWEEP_BASE, cell_name, "results")
        if not os.path.isdir(results_dir):
            continue
        for fname in sorted(os.listdir(results_dir)):
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(results_dir, fname)
            total += 1
            status = fix_result_file(json_path, args.dry_run)
            if status.startswith("fixed"):
                fixed += 1
                print(f"  {cell_name}/{fname}  {status}")
            else:
                skipped += 1

    print(f"\nDone. total={total}  fixed={fixed}  skipped={skipped}")
    if args.dry_run:
        print("(dry run — no files modified)")


if __name__ == "__main__":
    main()
