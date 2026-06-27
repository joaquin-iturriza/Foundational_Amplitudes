#!/usr/bin/env python3
"""
generate_sampler_ablation_sweeps.py — two nh=4 full-dataset pretraining sweeps to
diagnose the dynamic (deficit-to-solo) ProcessBalancedSampler.

Motivation
----------
The existing `pretrain_full_nh4` sweep searches `training.sampler_alpha_ema` and
`training.sampler_min_alpha_frac` — but the latter is *no longer used by the
weighting* (see config/default.yaml), and NONE of the new sampler's real control
knobs are searched:
    sampler_alpha_min, sampler_sig_k, sampler_deficit_gamma,
    sampler_deficit_cap, sampler_plateau_floor
They sit at hardcoded defaults, so if "the new sampler isn't working as expected"
we can't tell whether the *controller* is wrong or just *mistuned*. These two
sweeps separate those:

  1. pretrain_full_nh4_sampler  — same as pretrain_full_nh4, but the new sampler's
     real knobs are added to the HPO search space (and the dead sampler_min_alpha_frac
     is dropped). Lets DyHPO actually tune the controller.

  2. pretrain_full_nh4_uniform  — NAIVE equal weight per dataset: the dynamic
     controller is turned into a no-op by fixing
         training.sampler_deficit_gamma = 0      (no deficit boost; weight stays 1.0)
         training.sampler_plateau_floor = 1.0    (plateaued datasets also stay at 1.0)
     so ProcessBalancedSampler draws an EQUAL share of every dataset per batch for
     the whole run (no α-fitting feedback). This is the baseline the dynamic sampler
     must beat. No sampler knobs are searched here (they're inert).

Everything else (nh=4, full datasets, batch size, time budget / t_steps, DyHPO
params, regularisation, scheduler, loss aggregation, eval, plots) is IDENTICAL to
generate_pretrain_full_sweeps.py's nh=4 config, so the only varying factor is the
sampler treatment — and both stay comparable to the existing pretrain_full_nh4 run.

WORKFLOW (run on Jean Zay)
--------------------------
  1. (reuse the same s/iter as the pretrain_full campaign)
         sbatch bench_pretrain_iter_time.sh        # writes itertime_nh4.log
  2. Generate AND auto-submit both sweeps (default):
         python sweep/generate_sampler_ablation_sweeps.py
     Each sweep is submitted via submit_staged with its OWN hard-barrier schedule
     (first batch = that sweep's Sobol startup, so DyHPO observes the startup points
     before any surrogate-driven trial runs). Useful variants:
         python sweep/generate_sampler_ablation_sweeps.py --s4 0.13   # explicit s/iter
         python sweep/generate_sampler_ablation_sweeps.py --no-submit # generate only
         python sweep/generate_sampler_ablation_sweeps.py --dry-run   # plan only, write nothing
"""

import argparse
import math
import os
import subprocess
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Shared, identical-to-pretrain-scaling constants.
from sweep.generate_pretraining_scaling_sweeps import (
    AMP_ORDERS_STR,
    BASE_CLUSTER,
    DATASET_LIST_STR,
    DYHPO,
    LUSTRE_BASE,
    RANGE_EXTENSION,
    SEARCH_SPACE,
)
# Reuse the pretrain_full campaign's batch size, time-budget plan, and the
# expandable_segments PATHS so the ablation matches the existing nh=4 run.
from sweep.generate_pretrain_full_sweeps import (
    BATCH_SIZE,
    BUDGET_H,
    PARTITION,
    PATHS as FULL_PATHS,
    _s_per_iter_from_log,
    _walltime_str,
)

NH = 4                       # this ablation is the nh=4 case only
N_TRIALS_SAMPLER = 24        # more search dims (5 base + 6 sampler) → more trials
N_TRIALS_UNIFORM = 15        # only the 5 base HPs → same budget as pretrain_full_nh4
N_STARTUP_SAMPLER = 8        # Sobol startup (scale with the larger space)
N_STARTUP_UNIFORM = 4

# Own sweep root so we don't pollute the pretraining_full tree.
SWEEP_ROOT = f"{LUSTRE_BASE}/sweeps/sampler_ablation"
PATHS = {**FULL_PATHS, "sweep_dir": SWEEP_ROOT}

# Base HPs = the pretrain search space MINUS the sampler entries (we re-add the
# real sampler knobs ourselves for the "sampler" variant only).
BASE_HPS = [e for e in SEARCH_SPACE if not e["name"].startswith("training.sampler")]

# The new sampler's actual control knobs (defaults from config/default.yaml in
# the trailing comments). These are what was never being searched.
SAMPLER_HPS = [
    {"name": "training.sampler_alpha_ema",     "type": "float_uniform", "low": 0.3, "high": 0.95},  # loss-EMA decay (0.7)
    {"name": "training.sampler_alpha_min",     "type": "float_uniform", "low": 0.0, "high": 0.2},   # "flat" floor α_min (0.05)
    {"name": "training.sampler_sig_k",         "type": "float_uniform", "low": 1.0, "high": 3.0},   # plateau confidence k·SE (2.0)
    {"name": "training.sampler_deficit_gamma", "type": "float_uniform", "low": 0.0, "high": 3.0},   # boost strength γ (1.5)
    {"name": "training.sampler_deficit_cap",   "type": "float_uniform", "low": 1.0, "high": 3.0},   # deficit cap (2.0)
    {"name": "training.sampler_plateau_floor", "type": "float_uniform", "low": 0.1, "high": 1.0},   # plateau down-weight (0.3)
]


def _base_fixed_params() -> dict:
    """Identical to generate_pretrain_full_sweeps.make_config's fixed_params (nh=4)."""
    return {
        "data.data_path":            f"{LUSTRE_BASE}/data/",
        "data.dataset":              DATASET_LIST_STR,
        "data.amp_orders":           AMP_ORDERS_STR,
        "data.subsample":            "null",
        "data.train_test_val":       "[0.7, 0.2, 0.1]",
        "model":                     "lloca",
        "model.net.num_blocks":      8,
        "model.net.num_heads":       NH,
        "training.batchsize":        BATCH_SIZE,
        "evaluation.batchsize":      8192,
        "training.get_ID":           "false",
        "training.regularization":   "L2",
        "training.save_intermediate":"false",
        "training.scheduler":        "CosineAnnealingLR",
        "training.loss_aggregation": "geometric_mean",
        "training.validate_frac":    0.01,
        "seed":                      42,
        "plot":                      "true",
    }


def staged_batches(n_trials: int, n_startup: int) -> list[int]:
    """Hard-barrier schedule for submit_staged: the Sobol startup as batch 0 (so the
    surrogate observes it before any BO trial), then the remainder split into two
    roughly-equal BO rounds. Sums to n_trials. E.g. (24,8)->[8,8,8]; (15,4)->[4,6,5]."""
    rest = n_trials - n_startup
    if rest <= 0:
        return [n_trials]
    half = math.ceil(rest / 2)
    return [n_startup, half, rest - half]


def make_config(variant: str, t_steps: int, walltime: str) -> dict:
    """variant ∈ {"sampler", "uniform"}."""
    fixed = _base_fixed_params()
    if variant == "sampler":
        search_space = list(BASE_HPS) + list(SAMPLER_HPS)
        n_trials, n_startup = N_TRIALS_SAMPLER, N_STARTUP_SAMPLER
    elif variant == "uniform":
        # Freeze the dynamic controller → static equal weight per dataset.
        fixed["training.sampler_deficit_gamma"] = 0.0
        fixed["training.sampler_plateau_floor"] = 1.0
        search_space = list(BASE_HPS)              # sampler knobs inert → not searched
        n_trials, n_startup = N_TRIALS_UNIFORM, N_STARTUP_UNIFORM
    else:
        raise ValueError(variant)

    cluster = {**BASE_CLUSTER, "partition": PARTITION, "time": walltime, "auto_submit": False}
    return {
        "cluster":           cluster,
        "paths":             PATHS,
        "sweep_name":        f"pretrain_full_nh{NH}_{variant}",
        "n_trials":          n_trials,
        "dyhpo":             {**DYHPO, "n_startup": n_startup},
        "range_extension":   RANGE_EXTENSION,
        "fidelity_schedule": {"t_steps": [t_steps]},
        "fixed_params":      fixed,
        "search_space":      search_space,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s4", type=float, default=None,
                    help="measured s/iter for nh=4 at BS=16384 (default: read itertime_nh4.log)")
    ap.add_argument("--budget-hours", type=float, default=BUDGET_H,
                    help=f"target training wall-clock per run (default {BUDGET_H})")
    ap.add_argument("--no-submit", action="store_true",
                    help="generate jobs + DyHPO state but do NOT submit to SLURM")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; do not write configs, init DyHPO state, or submit")
    args = ap.parse_args()

    s4 = args.s4 if args.s4 is not None else _s_per_iter_from_log(NH)
    if s4 is None:
        ap.error("no s/iter for nh=4: pass --s4 or run bench_pretrain_iter_time.sh "
                 "first (expected itertime_nh4.log).")

    budget_s = args.budget_hours * 3600.0
    t_steps  = max(1, round(budget_s / s4))
    walltime = _walltime_str(t_steps * s4)

    print(f"=== nh=4 sampler ablation (BS={BATCH_SIZE}, budget≈{args.budget_hours:.1f}h) ===")
    print(f"  s/iter={s4:.4f}  ->  t_steps={t_steps}  (≈{t_steps*s4/3600:.2f}h train)  walltime={walltime}")
    print(f"  sampler variant: {N_TRIALS_SAMPLER} trials (Sobol={N_STARTUP_SAMPLER}), "
          f"{len(BASE_HPS)+len(SAMPLER_HPS)} search dims (5 base + {len(SAMPLER_HPS)} sampler)")
    print(f"  uniform variant: {N_TRIALS_UNIFORM} trials (Sobol={N_STARTUP_UNIFORM}), "
          f"{len(BASE_HPS)} search dims; gamma=0, plateau_floor=1.0 (static equal weights)")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    from sweep.generate_sweep import run_generate

    configs_dir = os.path.join(LUSTRE_BASE, "sweeps", "sampler_ablation", "configs")
    os.makedirs(configs_dir, exist_ok=True)

    generated = []   # (sweep_dir, batches)
    for variant in ("sampler", "uniform"):
        cfg = make_config(variant, t_steps, walltime)
        cfg_path = os.path.join(configs_dir, f"{cfg['sweep_name']}.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        sweep_dir = run_generate(cfg_path, submit=False)
        batches = staged_batches(cfg["n_trials"], cfg["dyhpo"]["n_startup"])
        generated.append((sweep_dir, batches))
        print(f"  wrote {cfg_path}  (batches={batches})")

    print("\nDone generating both sweeps.")

    if args.no_submit:
        print("\n--no-submit: not submitting. Submit each with its own schedule:")
        for sweep_dir, batches in generated:
            print(f"  python sweep/submit_staged.py --batches {','.join(map(str, batches))} {sweep_dir}")
        return

    # Auto-submit: one staged submission per sweep (each needs its own --batches,
    # since the two sweeps have different trial counts / Sobol startups). Every
    # sweep's batch 0 is dependency-free, so they queue together and fill free GPUs
    # across both; later batches are gated per-sweep by afterany barriers.
    submit_staged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submit_staged.py")
    print("\nAuto-submitting both sweeps (hard-barrier Sobol→BO batches)...")
    for sweep_dir, batches in generated:
        cmd = [sys.executable, submit_staged, "--batches", ",".join(map(str, batches)), sweep_dir]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print("\nAll submitted. Watch with:  squeue -u $USER -o '%.18i %.30j %.10T %.20E'")


if __name__ == "__main__":
    main()
