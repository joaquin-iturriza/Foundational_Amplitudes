#!/usr/bin/env python3
"""
scaling_law.py  —  Fit per-dataset solo scaling laws for DyHPO transfer-ratio HPO objective.

For each dataset, runs several short solo training jobs at increasing compute levels
(varying n_data and t_steps), then fits L = C · compute^{-α} in log-log space, where
compute = t_steps × batch_size (total samples seen in solo training).

Outputs a JSON file with (C, α) per dataset used by run_trial.py to compute
transfer ratios:  transfer_ratio_ds = val_loss_joint_ds / (C_ds · compute_ds^{-α_ds})

Usage (from project root, run once before each sweep):
    python sweep/scaling_law.py --config sweep/sweep_config_test.yaml \\
                                --output sweep/scaling_law_params.json
    python sweep/scaling_law.py --config sweep/sweep_config.yaml \\
                                --output /path/to/scaling_law_params.json \\
                                --alpha-prior 0.5   # optional fixed prior for α
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ---------------------------------------------------------------------------
# Compute levels: run at several (n_data, t_steps) combos per dataset.
# We vary t_steps and use a fixed mid-range n_data to keep runs cheap.
# ---------------------------------------------------------------------------

def compute_levels(fidelity_cfg, n_levels=4):
    """
    Return a list of (n_data, t_steps) pairs for solo scaling law runs.

    Uses the largest n_data level for that dataset (most data → cleanest signal)
    and logarithmically spaced t_steps between the min and max of the schedule.
    """
    t_sched = fidelity_cfg["t_steps"]
    t_min, t_max = min(t_sched), max(t_sched)
    # Logarithmically spaced, at least including endpoints
    t_levels = sorted(set(
        [t_min] +
        [int(round(math.exp(math.log(t_min) + i * math.log(t_max / t_min) / (n_levels - 1))))
         for i in range(1, n_levels - 1)] +
        [t_max]
    ))
    return t_levels


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def run_solo_trial(cfg, ds_name, n_data, t_steps, result_path, trial_idx):
    """Run a single solo training job on one dataset."""
    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")

    cmd = [sys.executable, run_script, "local=none"]

    skip_keys = {"training.iterations", "data.subsample", "data.dataset"}
    for key, val in cfg.get("fixed_params", {}).items():
        if key not in skip_keys:
            cmd.append(f"{key}={val}")

    # Solo: single dataset, single n_data value
    cmd.append(f"data.dataset=[{ds_name}]")
    cmd.append(f"data.subsample=[{n_data}]")
    cmd.append(f"training.iterations={t_steps}")
    cmd.append(f"training.increment_steps={t_steps}")
    cmd.append("training.is_dyhpo_run=true")
    cmd.append("warm_start_idx=null")
    cmd.append(f"exp_name=scaling_law_{ds_name}_t{t_steps}_n{n_data}_{trial_idx:04d}")
    cmd.append(f"training.result_path={result_path}")
    cmd.append("plot=false")

    print(f"  [scaling_law] {ds_name}  n_data={n_data}  t_steps={t_steps}")
    print(f"    cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=project_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"run.py exited with code {proc.returncode}")


def fit_scaling_law(computes, losses, alpha_prior=None):
    """
    Fit L = C · compute^{-α} in log-log space.

    If alpha_prior is given, first fits with α fixed to prior (only fit C),
    then refines both using unconstrained least squares.

    Returns (C, alpha, r2) where r2 is the coefficient of determination.
    """
    log_c = np.log(np.array(computes, dtype=float))
    log_l = np.log(np.array(losses,   dtype=float))

    # Unconstrained fit: log_l = intercept - alpha * log_c
    A = np.vstack([np.ones(len(log_c)), log_c]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, log_l, rcond=None)
    log_C = coeffs[0]
    alpha = float(-coeffs[1])
    C     = float(np.exp(log_C))

    # R² in log space
    ss_tot = float(np.var(log_l) * len(log_l))
    log_l_pred = log_C + coeffs[1] * log_c
    ss_res = float(np.sum((log_l - log_l_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return C, alpha, r2


def main():
    parser = argparse.ArgumentParser(
        description="Fit per-dataset solo scaling laws for DyHPO transfer-ratio objective"
    )
    parser.add_argument("--config",      required=True,
                        help="Path to sweep config YAML")
    parser.add_argument("--output",      required=True,
                        help="Path to write scaling law params JSON")
    parser.add_argument("--n-levels",    type=int, default=4,
                        help="Number of compute levels to evaluate per dataset (default: 4)")
    parser.add_argument("--alpha-prior", type=float, default=None,
                        help="Fix α to this value (use 4/DoF from paper); fit C only")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fidelity     = cfg["fidelity_schedule"]
    dataset_names = list(fidelity["n_data"].keys())
    batch_size   = cfg.get("fixed_params", {}).get("training.batchsize", 512)
    if isinstance(batch_size, str):
        batch_size = int(batch_size)

    t_levels = compute_levels(fidelity, n_levels=args.n_levels)
    print(f"[scaling_law] Datasets:     {dataset_names}")
    print(f"[scaling_law] t_steps:      {t_levels}")
    print(f"[scaling_law] Output:       {args.output}")

    # Temporary directory for result JSONs
    tmp_dir = os.path.join(cfg["paths"]["eos_sweep_dir"], "scaling_law_runs")
    os.makedirs(tmp_dir, exist_ok=True)

    params = {}
    trial_idx = 0

    for ds_name in dataset_names:
        # Use the largest n_data level — most signal, cleanest power law
        n_data = max(fidelity["n_data"][ds_name])
        print(f"\n[scaling_law] === Dataset: {ds_name}  (n_data={n_data}) ===")

        computes = []
        losses   = []

        for t_steps in t_levels:
            result_path = os.path.join(
                tmp_dir, f"{ds_name}_t{t_steps}_n{n_data}_{int(time.time())}.json"
            )
            if args.dry_run:
                print(f"  [dry-run] Would run {ds_name} n_data={n_data} t_steps={t_steps}")
                continue

            try:
                run_solo_trial(cfg, ds_name, n_data, t_steps, result_path, trial_idx)
                with open(result_path) as f:
                    result = json.load(f)
                val_loss = float(result["val_loss"])
                compute  = t_steps * batch_size   # total samples = steps × batch_size
                computes.append(compute)
                losses.append(val_loss)
                print(f"    val_loss={val_loss:.6f}  compute={compute:,}")
                trial_idx += 1
            except Exception as e:
                print(f"  [scaling_law] FAILED for {ds_name} t_steps={t_steps}: {e}",
                      file=sys.stderr)

        if args.dry_run:
            # Use prior values as placeholder
            alpha_prior = args.alpha_prior or 0.5
            params[ds_name] = {"C": None, "alpha": alpha_prior, "r2": None, "n_points": 0}
            continue

        if len(computes) < 2:
            print(f"  [scaling_law] Not enough successful runs for {ds_name}, skipping fit.")
            params[ds_name] = {"C": None, "alpha": args.alpha_prior or 0.5,
                               "r2": None, "n_points": len(computes)}
            continue

        C, alpha, r2 = fit_scaling_law(computes, losses, alpha_prior=args.alpha_prior)
        print(f"  [scaling_law] Fit: C={C:.4e}  α={alpha:.4f}  R²={r2:.4f}")
        params[ds_name] = {"C": C, "alpha": alpha, "r2": r2, "n_points": len(computes)}

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n[scaling_law] Params written to: {args.output}")

    # Print summary
    print("\n[scaling_law] Summary:")
    for ds, p in params.items():
        if p["C"] is not None:
            print(f"  {ds.replace('_amplitudes',''):30s}  C={p['C']:.3e}  α={p['alpha']:.3f}"
                  f"  R²={p['r2']:.3f}  (n={p['n_points']})")
        else:
            print(f"  {ds.replace('_amplitudes',''):30s}  α={p['alpha']:.3f} (prior only)")


if __name__ == "__main__":
    main()
