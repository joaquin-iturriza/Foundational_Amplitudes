#!/usr/bin/env python3
"""
test_sweep.py  —  Run a DyHPO sweep locally (no HTCondor, no GPU) to verify the pipeline.

Runs N trials sequentially, using the fidelity schedule and training settings
from the config as-is. Point it at a small test config (e.g. sweep_config_test.yaml)
so each trial finishes in seconds. Tests the full chain:
  DyHPOSampler.suggest() → run.py → result JSON → DyHPOSampler.observe()

The DyHPO state is saved to AFS between every suggest/observe, exactly as in a
real HTCondor sweep, so persistence is also tested.

Usage (from lxplus login node, inside the project root):
    python sweep/test_sweep.py --config sweep/sweep_config_test.yaml
    python sweep/test_sweep.py --config sweep/sweep_config_test.yaml --n-trials 5
    python sweep/test_sweep.py --config sweep/sweep_config_test.yaml --clean
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# Make the project root importable (sweep/ lives one level below the root)
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_sweep_name(cfg):
    # Append _localtest so the test state is isolated from any real production sweep.
    return cfg["sweep_name"] + "_localtest"


def state_path(cfg):
    name    = test_sweep_name(cfg)
    afs_dir = os.path.join(cfg["paths"]["afs_sweep_dir"], name)
    os.makedirs(afs_dir, exist_ok=True)
    return os.path.join(afs_dir, "dyhpo_state.pkl")


def results_dir(cfg):
    name    = test_sweep_name(cfg)
    eos_dir = os.path.join(cfg["paths"]["eos_sweep_dir"], name)
    rdir    = os.path.join(eos_dir, "results")
    os.makedirs(rdir, exist_ok=True)
    return rdir


def open_sampler(cfg, spath):
    """Create a fresh DyHPOSampler and save it, or load if one already exists."""
    from sweep.dyhpo_sampler import DyHPOSampler

    if os.path.exists(spath):
        print(f"[test] Loading existing DyHPO state: {spath}")
        output_path = os.path.join(
            cfg["paths"]["eos_sweep_dir"], test_sweep_name(cfg), "dyhpo_surrogate"
        )
        return DyHPOSampler.load(spath, output_path)

    dyhpo_cfg   = cfg.get("dyhpo", {})
    output_path = os.path.join(
        cfg["paths"]["eos_sweep_dir"], test_sweep_name(cfg), "dyhpo_surrogate"
    )
    os.makedirs(output_path, exist_ok=True)

    sampler = DyHPOSampler(
        hp_space      = cfg["search_space"],
        fidelity_grid = cfg["fidelity_schedule"],
        n_candidates  = dyhpo_cfg.get("n_candidates", 300),
        seed          = dyhpo_cfg.get("seed", 42),
        output_path   = output_path,
        n_startup     = dyhpo_cfg.get("n_startup", 10),
        total_budget  = dyhpo_cfg.get("total_budget", 10_000),
    )
    sampler.save(spath)
    print(f"[test] Initialised new DyHPO state: {spath}")
    return sampler


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def run_trial(cfg, spath, trial_idx):
    from sweep.dyhpo_sampler import DyHPOSampler

    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = test_sweep_name(cfg)
    rdir        = results_dir(cfg)
    output_path = os.path.join(
        cfg["paths"]["eos_sweep_dir"], sweep_name, "dyhpo_surrogate"
    )

    # --- suggest (load → suggest → save, mirrors concurrent HTCondor jobs) ---
    sampler = DyHPOSampler.load(spath, output_path)
    hp_idx, hp_params, n_data_dict, t_steps = sampler.suggest()
    sampler.save(spath)

    print(f"\n[test] Trial {trial_idx}  hp_idx={hp_idx}")
    print(f"[test]   fidelity: n_data={n_data_dict}  t_steps={t_steps}")
    print(f"[test]   HP params: {hp_params}")

    result_path = os.path.join(rdir, f"hp{hp_idx:04d}_t{t_steps}_{int(time.time())}.json")

    # Build command — identical to run_trial.py, always cold-start (no warm-starting in test)
    subsample_vals = list(n_data_dict.values())
    cmd = [sys.executable, run_script, "local=none"]

    skip_keys = {"training.iterations", "data.subsample"}
    for key, val in cfg.get("fixed_params", {}).items():
        if key not in skip_keys:
            cmd.append(f"{key}={val}")

    for key, val in hp_params.items():
        cmd.append(f"{key}={format_value(val)}")

    cmd.append(f"data.subsample=[{','.join(str(n) for n in subsample_vals)}]")
    cmd.append(f"training.iterations={t_steps}")
    cmd.append(f"training.increment_steps={t_steps}")
    cmd.append("training.is_dyhpo_run=true")
    cmd.append("warm_start_idx=null")
    cmd.append(f"exp_name={sweep_name}_trial{trial_idx:04d}")
    cmd.append(f"training.result_path={result_path}")

    print(f"[test]   cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, cwd=project_dir)
        if proc.returncode != 0:
            raise RuntimeError(f"run.py exited with code {proc.returncode}")

        with open(result_path) as f:
            result = json.load(f)
        val_loss        = float(result["val_loss"])
        proc_val_losses = result.get("proc_val_losses")

        from sweep.run_trial import compute_hpo_objective
        hpo_obj      = compute_hpo_objective(result, cfg.get("scaling_law", {}))
        observe_loss = hpo_obj if hpo_obj is not None else val_loss
        print(f"[test]   val_loss={val_loss:.6f}"
              + (f"  hpo_obj={hpo_obj:.4f}" if hpo_obj is not None else "") + "  ✓")

        # --- observe (load → observe → save) ---
        sampler = DyHPOSampler.load(spath, output_path)
        sampler.observe(hp_idx, n_data_dict, t_steps, observe_loss, proc_val_losses)
        sampler.save(spath)

        return True, val_loss

    except Exception as e:
        print(f"[test]   FAILED: {e}", file=sys.stderr)
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Local pipeline test for DyHPO sweep (no HTCondor, no GPU)"
    )
    parser.add_argument("--config",   required=True,
                        help="Path to sweep config YAML (use sweep/sweep_config_test.yaml for local testing)")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials (default: 3)")
    parser.add_argument("--clean",    action="store_true",
                        help="Delete existing DyHPO state before starting (fresh run)")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    spath = state_path(cfg)

    if args.clean and os.path.exists(spath):
        os.remove(spath)
        print(f"[test] Deleted existing state: {spath}")

    fidelity  = cfg["fidelity_schedule"]
    n_combos  = 1
    for sched in fidelity["n_data"].values():
        n_combos *= len(sched)
    n_combos *= len(fidelity["t_steps"])

    print(f"[test] Sweep:         {test_sweep_name(cfg)}")
    print(f"[test] DyHPO state:   {spath}")
    print(f"[test] Datasets:      {list(fidelity['n_data'].keys())}")
    print(f"[test] n_data levels: {[len(s) for s in fidelity['n_data'].values()]} per dataset")
    print(f"[test] t_steps:       {fidelity['t_steps']}")
    print(f"[test] Total combos:  {n_combos}")

    open_sampler(cfg, spath)

    passed, failed = 0, 0
    losses = []
    for i in range(args.n_trials):
        ok, val_loss = run_trial(cfg, spath, trial_idx=i)
        if ok:
            passed += 1
            losses.append(val_loss)
        else:
            failed += 1

    print(f"\n[test] Done: {passed} passed, {failed} failed out of {args.n_trials} trials.")

    if losses:
        from sweep.dyhpo_sampler import DyHPOSampler
        output_path = os.path.join(
            cfg["paths"]["eos_sweep_dir"], test_sweep_name(cfg), "dyhpo_surrogate"
        )
        sampler = DyHPOSampler.load(spath, output_path)
        best = sampler.best_result()
        if best:
            best_params, best_loss = best
            print(f"[test] Best val_loss so far: {best_loss:.6f}")
            print(f"[test]   params: {best_params}")
        results = sampler.all_results()
        print(f"[test] Total observations: {len(results)} (hp_idx, combo) pairs")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
