#!/usr/bin/env python3
"""
generate_pretrain_full_sweeps.py — Full-dataset, time-budgeted pretraining sweeps.

Two independent DyHPO sweeps (one per model width nh ∈ {4, 8}), each:
  * trained on ALL 8 datasets at their FULL size (data.subsample=null, ~100k events),
  * batch size 2**14 = 16384,
  * a SINGLE fidelity level (t_steps) sized so each run lasts ~BUDGET_H hours of
    wall-clock — sized PER nh from a measured s/iter, because nh=8 costs ~2x nh=4
    per step (see bench_pretrain_iter_time.sh),
  * 15 trials with 4 Sobol startup points (dyhpo.n_startup=4),
  * everything else identical to the pretraining-scaling cells (same search space,
    DyHPO params, regularisation, scheduler, loss aggregation, eval, plots).

The point of the campaign: at the full dataset size, does the smaller model (nh=4)
plateau within ~10h, or does the larger one (nh=8) keep improving?

WORKFLOW (run on Jean Zay)
--------------------------
  1. Measure real s/iter at BS=16384 for nh=4 and nh=8:
         sbatch bench_pretrain_iter_time.sh
     -> writes itertime_nh4.log / itertime_nh8.log (and prints a summary).

  2. Generate both sweeps (reads the s/iter from those logs by default):
         python sweep/generate_pretrain_full_sweeps.py
     or pass measured values explicitly:
         python sweep/generate_pretrain_full_sweeps.py --s4 0.13 --s8 0.25
     Add --dry-run to preview t_steps / walltime without writing anything.

  3. Submit each sweep in hard-barrier batches (4 Sobol -> 6 -> 5), interleaved:
         python sweep/submit_staged.py <nh4_dir> <nh8_dir>
     (the generate step prints the exact command with the resolved dirs).
"""

import argparse
import math
import os
import re
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Reuse the exact constants from the scaling-sweep generator so the configs stay
# identical "as in the pretrain scaling" (search space, DyHPO, paths, datasets).
from sweep.generate_pretraining_scaling_sweeps import (
    AMP_ORDERS_STR,
    BASE_CLUSTER,
    BASE_PATHS,
    DATASET_LIST_STR,
    DYHPO,
    LUSTRE_BASE,
    RANGE_EXTENSION,
    SEARCH_SPACE,
)

# ── Campaign knobs ──────────────────────────────────────────────────────────
N_HEADS      = [4, 8]
BATCH_SIZE   = 1 << 14        # 2**14 = 16384
N_TRIALS     = 15
N_STARTUP    = 4              # Sobol startup points
BUDGET_H     = 10.0           # target TRAINING wall-clock per run (hours)
PARTITION    = "gpu_p2"       # V100 32GB; nh=4 ~6.8GB, nh=8 ~12GB at BS=16384 -> fits
# NOTE: Jean Zay forbids --mem/--mem-per-cpu; host RAM scales with --cpus-per-task.
# The earlier OOM was caused by the eval/per-process loaders forking the full 8M-event
# dataset into ~50 persistent workers (now fixed: those loaders use num_workers=0), not
# by too little RAM. If more RAM is ever needed, raise cpus_per_task (not --mem).

# Own sweep root (don't dump into the pretraining_scaling tree).
SWEEP_ROOT = f"{LUSTRE_BASE}/sweeps/pretraining_full"
# Add expandable_segments so the CUDA device allocator doesn't hoard a distinct block per
# distinct (variable) batch size -> bounds the GPU "VRAM alloc" creep over a long run.
PATHS = {
    **BASE_PATHS,
    "sweep_dir": SWEEP_ROOT,
    "setup_commands": list(BASE_PATHS.get("setup_commands", []))
                      + ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
}


def _s_per_iter_from_log(nh: int) -> float | None:
    """Parse the last 'avg <x>s/iter' from itertime_nh{nh}.log (project root)."""
    log = os.path.join(LUSTRE_BASE, f"itertime_nh{nh}.log")
    if not os.path.exists(log):
        # also try cwd (bench writes there when run from the project dir)
        log = f"itertime_nh{nh}.log"
        if not os.path.exists(log):
            return None
    val = None
    with open(log) as f:
        for line in f:
            m = re.search(r"avg ([0-9.]+)s/iter", line)
            if m:
                val = float(m.group(1))
    return val


def _walltime_str(budget_seconds: float) -> str:
    """SLURM HH:MM:SS with headroom: 10% iter-time margin + 60 min for the
    periodic validations (800k val events) and the final eval/plots over all
    ~8M events, rounded up to the next 30 minutes."""
    total = budget_seconds * 1.10 + 3600.0
    minutes = math.ceil(total / 1800.0) * 30        # round up to 30-min blocks
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def make_config(nh: int, t_steps: int, walltime: str) -> dict:
    fixed_params = {
        "data.data_path":            f"{LUSTRE_BASE}/data/",
        "data.dataset":              DATASET_LIST_STR,
        "data.amp_orders":           AMP_ORDERS_STR,
        "data.subsample":            "null",        # FULL datasets, no subsampling
        "data.train_test_val":       "[0.7, 0.2, 0.1]",
        "model":                     "lloca",
        "model.net.num_blocks":      8,
        "model.net.num_heads":       nh,
        "training.batchsize":        BATCH_SIZE,
        "evaluation.batchsize":      8192,
        "training.get_ID":           "false",
        "training.regularization":   "L2",
        "training.save_intermediate":"false",
        "training.scheduler":        "CosineAnnealingLR",
        "training.loss_aggregation": "geometric_mean",
        "training.validate_frac":    0.01,
        "seed":                      42,            # init seed fixed; trials differ only by HPs
        "plot":                      "true",
    }
    cluster = {**BASE_CLUSTER, "partition": PARTITION, "time": walltime, "auto_submit": False}
    return {
        "cluster":            cluster,
        "paths":              PATHS,
        "sweep_name":         f"pretrain_full_nh{nh}",
        "n_trials":           N_TRIALS,
        "dyhpo":              {**DYHPO, "n_startup": N_STARTUP},
        "range_extension":    RANGE_EXTENSION,
        "fidelity_schedule":  {"t_steps": [t_steps]},
        "fixed_params":       fixed_params,
        "search_space":       list(SEARCH_SPACE),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s4", type=float, default=None,
                    help="measured s/iter for nh=4 at BS=16384 (default: read itertime_nh4.log)")
    ap.add_argument("--s8", type=float, default=None,
                    help="measured s/iter for nh=8 at BS=16384 (default: read itertime_nh8.log)")
    ap.add_argument("--budget-hours", type=float, default=BUDGET_H,
                    help=f"target training wall-clock per run (default {BUDGET_H})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; do not write configs or init DyHPO state")
    args = ap.parse_args()

    s_per_iter = {4: args.s4, 8: args.s8}
    for nh in N_HEADS:
        if s_per_iter[nh] is None:
            s_per_iter[nh] = _s_per_iter_from_log(nh)
        if s_per_iter[nh] is None:
            ap.error(f"no s/iter for nh={nh}: pass --s{nh} or run bench_pretrain_iter_time.sh "
                     f"first (expected itertime_nh{nh}.log).")

    budget_s = args.budget_hours * 3600.0

    print(f"=== Full-dataset, time-budgeted pretraining sweeps "
          f"(BS={BATCH_SIZE}, budget≈{args.budget_hours:.1f}h) ===")
    plan = {}
    for nh in N_HEADS:
        s = s_per_iter[nh]
        t_steps  = max(1, round(budget_s / s))
        walltime = _walltime_str(t_steps * s)
        plan[nh] = (t_steps, walltime)
        print(f"  nh={nh}:  {s:.4f} s/iter  ->  t_steps={t_steps}  "
              f"(≈{t_steps*s/3600:.2f}h train)  walltime={walltime}  "
              f"n_trials={N_TRIALS} (Sobol={N_STARTUP})")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    from sweep.generate_sweep import run_generate

    configs_dir = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_full", "configs")
    os.makedirs(configs_dir, exist_ok=True)

    sweep_dirs = []
    for nh in N_HEADS:
        t_steps, walltime = plan[nh]
        cfg = make_config(nh, t_steps, walltime)
        cfg_path = os.path.join(configs_dir, f"{cfg['sweep_name']}.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        # Generate jobs/ + DyHPO state, but DEFER submission: submit_staged.py
        # enforces the 4 -> 6 -> 5 barrier (sweep_manager's soft nice ordering
        # can't stop all 15 starting at once if many GPUs free up together).
        sweep_dir = run_generate(cfg_path, submit=False)
        sweep_dirs.append(sweep_dir)

    print("\nDone generating both sweeps.")
    print("Submit them in hard-barrier batches (4 Sobol -> 6 -> 5), interleaved:")
    print(f"  python sweep/submit_staged.py {' '.join(sweep_dirs)}")


if __name__ == "__main__":
    main()
