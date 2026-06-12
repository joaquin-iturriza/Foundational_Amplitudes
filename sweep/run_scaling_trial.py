#!/usr/bin/env python3
"""
run_scaling_trial.py  —  Per-job entrypoint for solo scaling law sweep.

Each HTCondor job runs this script once. It:
  1. Reads the pre-sampled HP candidates from hp_candidates.json by trial-idx.
  2. Runs `python run.py` for a single dataset at a fixed training budget (cold start).
  3. Post-processes the result JSON to inject compute_ds and proc_val_losses
     (run.py does not log these for single-dataset training).
"""

import argparse
import json
import os
import shutil
import sys
import subprocess
import time

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def build_command(cfg, hp_params, dataset, t_steps, result_path, run_dir):
    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = cfg["sweep_name"]

    cmd = [sys.executable, run_script]
    cmd.append("local=none")

    fixed = cfg.get("fixed_params", {})
    skip_keys = {"training.iterations", "data.dataset", "data.data_path"}
    for key, val in fixed.items():
        if key not in skip_keys:
            cmd.append(f"{key}={val}")

    # data path and this specific dataset
    if "data.data_path" in fixed:
        cmd.append(f"data.data_path={fixed['data.data_path']}")
    cmd.append(f"data.dataset=[{dataset}]")

    # HP params from Sobol sample
    for key, val in hp_params.items():
        cmd.append(f"{key}={format_value(val)}")

    cmd.append(f"training.iterations={t_steps}")
    cmd.append("training.is_dyhpo_run=false")

    cmd.append(f"base_dir={project_dir}")
    cmd.append(f"run_dir={run_dir}")
    cmd.append(f"exp_name={sweep_name}")

    cmd.append("warm_start_idx=null")
    cmd.append(f"training.result_path={result_path}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run one solo scaling law trial")
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--t-steps",      type=int, required=True)
    parser.add_argument("--trial-idx",    type=int, required=True)
    args = parser.parse_args()

    cfg         = load_config(args.sweep_config)
    sweep_name  = cfg["sweep_name"]
    project_dir = cfg["paths"]["project_dir"]

    afs_dir, eos_dir = _sweep_dirs(cfg, sweep_name)
    results_dir = os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load pre-sampled HP candidate
    candidates_path = os.path.join(afs_dir, "hp_candidates.json")
    with open(candidates_path) as f:
        candidates = json.load(f)

    if args.trial_idx >= len(candidates):
        raise ValueError(
            f"trial_idx={args.trial_idx} out of range (only {len(candidates)} candidates)"
        )
    hp_params = candidates[args.trial_idx]

    ds_tag  = args.dataset.replace("_amplitudes", "").replace("_", "")
    job_tag = f"{ds_tag}_t{args.t_steps:05d}_trial{args.trial_idx:03d}"

    print(f"[run_scaling_trial] sweep={sweep_name}  dataset={args.dataset}"
          f"  t_steps={args.t_steps}  trial_idx={args.trial_idx}")
    print(f"[run_scaling_trial] HP params: {hp_params}")

    run_dir     = os.path.join(project_dir, "runs", sweep_name, job_tag)
    result_path = os.path.join(results_dir, f"{job_tag}_{int(time.time())}.json")

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    cmd = build_command(cfg, hp_params, args.dataset, args.t_steps,
                        result_path, run_dir)
    print(f"[run_scaling_trial] Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=project_dir)
    if proc.returncode != 0:
        print(f"[run_scaling_trial] FAILED: run.py exited {proc.returncode}", file=sys.stderr)
        sys.exit(1)

    # Post-process: inject compute_ds and proc_val_losses into result JSON.
    # Single-dataset training (train_sampler=None) does not write these fields.
    batchsize = int(cfg.get("fixed_params", {}).get("training.batchsize", 512))
    compute   = args.t_steps * batchsize

    with open(result_path) as f:
        result = json.load(f)

    val_loss  = float(result["val_loss"])
    test_loss = float(result.get("test_loss", val_loss))
    result.setdefault("compute_ds",      {})[args.dataset] = compute
    result.setdefault("proc_val_losses", {})[args.dataset] = test_loss
    result["dataset"]   = args.dataset
    result["t_steps"]   = args.t_steps
    result["trial_idx"] = args.trial_idx

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[run_scaling_trial] Done  test_loss={test_loss:.6f}  compute={compute}")


if __name__ == "__main__":
    main()
