#!/usr/bin/env python3
"""
generate_scaling_sweep.py  —  One DyHPO sweep per (dataset, t_steps_level).

For each cell (dataset × fidelity level) an independent DyHPO surrogate is
initialised and SLURM / HTCondor job files are generated.  Submit one level
at a time; within each cell the surrogate adapts as jobs complete.

Usage:
    python sweep/generate_scaling_sweep.py --config sweep/sweep_config_scaling.yaml
    python sweep/generate_scaling_sweep.py --config sweep/sweep_config_scaling.yaml --dry-run
"""

import argparse
import os
import re
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds_tag(dataset_name):
    return (dataset_name
            .replace('_amplitudes', '')
            .replace(', ', '_')
            .replace('_', ''))


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def _scheduler(cfg):
    return cfg["cluster"].get("scheduler", "htcondor")


def _next_available(base_name, sweep_dir_base):
    def exists(name):
        return os.path.isdir(os.path.join(sweep_dir_base, name))
    if not exists(base_name):
        return base_name
    m = re.search(r"^(.*?)(_(\d+))$", base_name)
    stem  = m.group(1) if m else base_name
    n     = int(m.group(3)) if m else 1
    width = len(m.group(3)) if m else 3
    while True:
        n += 1
        candidate = f"{stem}_{n:0{width}d}"
        if not exists(candidate):
            return candidate


def _make_cell_cfg(outer_cfg, dataset, t_steps, cell_name):
    """Build a generate_sweep.py-compatible config for one (dataset, level) cell."""
    fixed = dict(outer_cfg.get("fixed_params", {}))
    fixed["data.dataset"] = f"[{dataset}]"

    dyhpo = dict(outer_cfg.get("dyhpo", {}))
    dyhpo.setdefault("n_candidates", 200)
    dyhpo.setdefault("seed", outer_cfg.get("seed", 42))
    dyhpo.setdefault("n_startup", 10)
    dyhpo.setdefault("total_budget", outer_cfg["n_trials_per_level"] * 10)

    return {
        "cluster":           outer_cfg["cluster"],
        "paths":             outer_cfg["paths"],
        "sweep_name":        cell_name,
        "n_trials":          outer_cfg["n_trials_per_level"],
        "dyhpo":             dyhpo,
        "fidelity_schedule": {"t_steps": [t_steps]},
        "fixed_params":      fixed,
        "search_space":      outer_cfg["search_space"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate one DyHPO sweep per (dataset, t_steps)")
    parser.add_argument("--config",  required=True, help="Path to sweep config yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    args = parser.parse_args()

    config_abs_path = os.path.abspath(args.config)
    with open(config_abs_path) as f:
        cfg = yaml.safe_load(f)

    # Fix the init seed by default so trials differ only by hyperparameters
    # (a config may override by setting its own fixed_params.seed).
    cfg.setdefault("fixed_params", {}).setdefault("seed", 42)

    from sweep.generate_sweep import (setup_dirs, init_sampler,
                                      write_slurm_script, write_sh, write_sub)

    base_name      = cfg["sweep_name"]
    sweep_dir_base = cfg["paths"].get("sweep_dir", cfg["paths"].get("afs_sweep_dir"))
    sweep_name     = _next_available(base_name, sweep_dir_base)
    if sweep_name != base_name:
        print(f"Name '{base_name}' taken — using '{sweep_name}'.")
    cfg["sweep_name"] = sweep_name

    datasets       = cfg["datasets"]
    t_steps_values = cfg["t_steps_values"]
    n_trials       = cfg["n_trials_per_level"]
    scheduler      = _scheduler(cfg)

    total_jobs = len(datasets) * len(t_steps_values) * n_trials

    print(f"Sweep      : {sweep_name}")
    print(f"Scheduler  : {scheduler}")
    print(f"Datasets   : {datasets}")
    print(f"t_steps    : {t_steps_values}")
    print(f"Trials/cell: {n_trials}  →  {total_jobs} total jobs")

    if args.dry_run:
        print("\n[dry-run] Done.")
        return

    # Save outer config in top-level dir
    top_dir, _ = _sweep_dirs(cfg, sweep_name)
    os.makedirs(top_dir, exist_ok=True)
    saved_outer = os.path.join(top_dir, "sweep_config.yaml")
    with open(saved_outer, "w") as f:
        yaml.dump(cfg, f)
    print(f"\nOuter config: {saved_outer}")

    job_paths_by_level = {i: [] for i in range(len(t_steps_values))}
    cell_jobs_dirs = {}   # (level_idx, ds_tag) -> jobs_dir

    for level_idx, t_steps in enumerate(t_steps_values):
        for dataset in datasets:
            ds_tag    = _ds_tag(dataset)
            cell_name = f"{sweep_name}_{ds_tag}_t{t_steps:05d}"
            cell_cfg  = _make_cell_cfg(cfg, dataset, t_steps, cell_name)

            cell_afs, cell_eos = setup_dirs(cell_cfg, cell_name)
            cell_jobs_dirs[(level_idx, ds_tag)] = os.path.join(cell_afs, "jobs")

            saved_cell = os.path.join(cell_afs, "sweep_config.yaml")
            with open(saved_cell, "w") as f:
                yaml.dump(cell_cfg, f)

            init_sampler(cell_cfg, cell_afs, cell_eos)

            for i in range(n_trials):
                if scheduler == "slurm":
                    path = write_slurm_script(i, cell_cfg, cell_afs, saved_cell)
                else:
                    sh   = write_sh(i, cell_cfg, cell_afs, saved_cell)
                    path = write_sub(i, cell_cfg, cell_afs, sh)
                job_paths_by_level[level_idx].append(path)

    all_job_paths = [p for paths in job_paths_by_level.values() for p in paths]

    # Each cell is its own sweep (own DyHPO state); the sweep_manager interleaves
    # trials round-robin across cells so they don't all start in parallel.
    cell_dirs = [os.path.dirname(jd) for jd in cell_jobs_dirs.values()]
    auto = cfg["cluster"].get("auto_submit", False)
    if auto:
        if scheduler == "slurm":
            from sweep.sweep_manager import submit_sweeps
            submit_sweeps(cell_dirs)
        else:
            import subprocess
            submitted = 0
            for path in all_job_paths:
                try:
                    subprocess.run(["condor_submit", path], check=True)
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"  Failed: {path}: {e}", file=sys.stderr)
            print(f"\nSubmitted {submitted}/{len(all_job_paths)} jobs.")
    elif scheduler == "slurm":
        print("\nSubmit all cells interleaved (round-robin across cells) with:")
        print("  python sweep/sweep_manager.py submit " + " ".join(cell_dirs))
    else:
        print("\nSubmit all cells at once (all levels are independent):")
        for level_idx, t_steps in enumerate(t_steps_values):
            for dataset in datasets:
                ds_tag   = _ds_tag(dataset)
                subs_dir = cell_jobs_dirs[(level_idx, ds_tag)].replace("/jobs", "/subs")
                print(f"  for f in {subs_dir}/*.sub; do condor_submit \"$f\"; done")

    print(f"\nAfter all jobs complete:")
    print(f"  python sweep/fit_scaling_law.py --config {saved_outer}")


if __name__ == "__main__":
    main()
