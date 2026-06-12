#!/usr/bin/env python3
"""
add_traintime.py — Parse actual training time from run logs and add to result JSONs.

Looks for lines like:
  Finished training: ... in 12.34min (avg ...)
in out_0.log inside each run dir, then writes traintime_hours to all
result JSONs for that trial.

Usage:
    python sweep/add_traintime.py [--dry-run] [--cell CELL_NAME]
"""

import argparse
import json
import os
import re
import sys

import yaml

LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEP_BASE  = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")

_TIME_RE = re.compile(r"Finished training:.*?in\s+([\d.]+)min")


def parse_traintime_hours(run_dir: str) -> float | None:
    if not os.path.isdir(run_dir):
        return None
    for fname in os.listdir(run_dir):
        if not fname.startswith("out_") or not fname.endswith(".log"):
            continue
        try:
            with open(os.path.join(run_dir, fname)) as f:
                for line in f:
                    m = _TIME_RE.search(line)
                    if m:
                        return float(m.group(1)) / 60.0
        except Exception:
            continue
    return None


def process_cell(cell_dir: str, dry_run: bool):
    results_dir = os.path.join(cell_dir, "results")
    if not os.path.isdir(results_dir):
        return

    ckpt_index_path = os.path.join(cell_dir, "checkpoint_index.json")
    if not os.path.exists(ckpt_index_path):
        print(f"  [skip] no checkpoint_index.json")
        return

    with open(ckpt_index_path) as f:
        ckpt_index = json.load(f)

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(results_dir, fname)
        with open(json_path) as f:
            result = json.load(f)

        if "traintime_hours" in result:
            continue

        m = re.match(r"hp(\d+)_t\d+_", fname)
        if not m:
            continue
        hp_idx = int(m.group(1))
        ckpt = ckpt_index.get(str(hp_idx))
        if ckpt is None:
            print(f"  [skip] {fname} — hp_idx={hp_idx} not in checkpoint_index")
            continue

        run_dir = ckpt["run_dir"]
        hours = parse_traintime_hours(run_dir)
        if hours is None:
            print(f"  [skip] {fname} — no training time found in logs at {run_dir}")
            continue

        print(f"  {fname}  {hours*60:.1f}min", end="")
        if not dry_run:
            result["traintime_hours"] = hours
            with open(json_path, "w") as f:
                json.dump(result, f)
            print("  ✓")
        else:
            print("  [dry-run]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell", default=None)
    args = parser.parse_args()

    if not os.path.isdir(SWEEP_BASE):
        sys.exit(f"SWEEP_BASE not found: {SWEEP_BASE}")

    cell_names = sorted(os.listdir(SWEEP_BASE))
    if args.cell:
        cell_names = [c for c in cell_names if c == args.cell]
        if not cell_names:
            sys.exit(f"Cell not found: {args.cell}")

    for cell_name in cell_names:
        if not cell_name.startswith("scaling_p1"):
            continue
        cell_dir = os.path.join(SWEEP_BASE, cell_name)
        if not os.path.isdir(cell_dir):
            continue
        print(f"\n=== {cell_name} ===")
        process_cell(cell_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
