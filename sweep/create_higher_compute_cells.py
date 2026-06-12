#!/usr/bin/env python3
"""
Create new higher-compute cells for all existing (nh, D) combinations.

For each (nh, D), finds the current max t_steps, then computes the next
compute level maintaining √10 spacing, adjusted for the new (doubled) BS.

Run from Jean-Zay:
    python sweep/create_higher_compute_cells.py [--dry-run] [--auto-submit]
"""
import argparse
import math
import os
import re
import subprocess
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from sweep.generate_pretraining_scaling_sweeps import (
    DATASET_SIZES, BS_CAP, _batch_size, needs_32gb, slurm_time_str,
    _subsample_per_ds, DYHPO, RANGE_EXTENSION,
    DATASET_LIST_STR, AMP_ORDERS_STR, SEARCH_SPACE,
)

LUSTRE_BASE  = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEP_BASE   = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")
N_TRIALS     = 10
SETUP_COMMANDS = [
    "module load anaconda-py3/2023.09",
    "conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational",
]
BASE_CLUSTER = {"scheduler": "slurm", "account": "itg@v100",
                "request_gpus": 1, "cpus_per_task": 8}


def bs_old(d_total: int) -> int:
    """Old batch size formula (largest power of 2 < D/2, capped at 8192)."""
    half = d_total / 2
    p = 1
    while p * 2 < half:
        p *= 2
    return min(8192, p)


LOG_GRID = sorted(set(round(10 ** (k / 4)) for k in range(4, 60)))

def round_to_grid(t: float) -> int:
    return min(LOG_GRID, key=lambda x: abs(math.log(x / t)))


def find_max_t() -> dict:
    """Return {(nh, d_key): max_t} from existing cell directories."""
    max_t = {}
    for name in os.listdir(SWEEP_BASE):
        m = re.match(r'scaling_p1(?:ext)?_nh(\d+)_(D[^_]+)_t(\d+)$', name)
        if not m:
            continue
        nh, d_key, t = int(m.group(1)), m.group(2), int(m.group(3))
        if d_key[1:] not in DATASET_SIZES:
            continue
        max_t[(nh, d_key)] = max(max_t.get((nh, d_key), 0), t)
    return max_t


MAX_WALLTIME_H = 20   # Jean-Zay QOS limit (qos_gpu-t3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-submit", action="store_true")
    args = parser.parse_args()

    max_t = find_max_t()
    SQRT10 = 10 ** 0.5

    cells = []
    for (nh, d_key) in sorted(max_t):
        d_total   = DATASET_SIZES[d_key[1:]]
        t_max     = max_t[(nh, d_key)]
        bs_new    = min(_batch_size(d_total), BS_CAP.get(nh, 8192))
        bs_o      = bs_old(d_total)
        t_new     = round_to_grid(t_max * SQRT10 / (bs_new / bs_o))
        cell_name = f"scaling_p1ext_nh{nh}_{d_key}_t{t_new}"
        cell_dir  = os.path.join(SWEEP_BASE, cell_name)

        gpu     = "gpu_p2l" if needs_32gb(nh, bs_new) else "gpu_p2"
        slurm_t = slurm_time_str(t_new, nh, bs_new)

        # Skip cells that would exceed the QOS wall time limit
        h = int(slurm_t.split(":")[0])
        if h >= MAX_WALLTIME_H:
            print(f"  SKIP (>{MAX_WALLTIME_H}h): {cell_name}  [{slurm_t}]")
            continue
        sub     = _subsample_per_ds(d_total)

        cfg = {
            "cluster": {**BASE_CLUSTER, "partition": gpu, "time": slurm_t, "auto_submit": args.auto_submit},
            "paths": {"sweep_dir": SWEEP_BASE, "project_dir": LUSTRE_BASE,
                      "setup_commands": SETUP_COMMANDS},
            "sweep_name": cell_name,
            "n_trials": N_TRIALS,
            "dyhpo": DYHPO,
            "range_extension": RANGE_EXTENSION,
            "fidelity_schedule": {"t_steps": [t_new]},
            "fixed_params": {
                "data.data_path": f"{LUSTRE_BASE}/data/",
                "data.dataset": DATASET_LIST_STR,
                "data.amp_orders": AMP_ORDERS_STR,
                "data.subsample": sub,
                "data.train_test_val": "[0.7, 0.2, 0.1]",
                "model": "lloca",
                "model.net.num_blocks": 8,
                "model.net.num_heads": nh,
                "training.batchsize": bs_new,
                "evaluation.batchsize": 8192,
                "training.get_ID": "false",
                "training.regularization": "L2",
                "training.save_intermediate": "false",
                "training.scheduler": "CosineAnnealingLR",
                "training.loss_aggregation": "geometric_mean",
                "training.validate_frac": 0.01,
                "plot": "true",
            },
            "search_space": SEARCH_SPACE,
        }
        cells.append((cell_name, cell_dir, cfg, gpu, slurm_t, bs_new, t_new, t_max, bs_o))

    print(f"{'Cell':<55} {'BS_old':>6} {'BS_new':>6} {'t_max':>8} {'t_new':>8}  {'GPU':<6} {'time'}")
    print("-" * 110)
    to_create = []
    for cell_name, cell_dir, cfg, gpu, slurm_t, bs_new, t_new, t_max, bs_o in cells:
        exists = os.path.isdir(cell_dir)
        mark = "EXISTS" if exists else "CREATE"
        print(f"  [{mark}] {cell_name:<52} {bs_o:>6} {bs_new:>6} {t_max:>8} {t_new:>8}  {gpu:<6} {slurm_t}")
        if not exists:
            to_create.append((cell_name, cell_dir, cfg))

    print(f"\n{len(to_create)} cells to create.")
    if args.dry_run or not to_create:
        return

    for cell_name, cell_dir, cfg in to_create:
        cfg_path = os.path.join(cell_dir, "sweep_config.yaml")
        cmd = [sys.executable,
               os.path.join(_project_dir, "sweep", "generate_sweep.py"),
               "--config", cfg_path]

        # Write config first so generate_sweep.py can read it
        os.makedirs(cell_dir, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR creating {cell_name}", file=sys.stderr)
        else:
            print(f"  Created: {cell_name}")


if __name__ == "__main__":
    main()
