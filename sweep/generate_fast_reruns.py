#!/usr/bin/env python3
"""
generate_fast_reruns.py — rerun the best trial of each sweep cell with the
current (optimized) code, to measure real val_loss-vs-wall-time.

For every cell under --sweep-base it:
  1. finds the best trial (min val_loss across results/*.json; hp index is read
     from the result filename hpNNNN_tNNNN_*.json),
  2. creates a parallel `<cell><suffix>` cell (default suffix `_fast`) with a copy
     of sweep_config.yaml (sweep_name updated, seed pinned to 42) and the cell's
     dyhpo_state.pkl,
  3. writes a single-trial SLURM job that cold-starts that exact HP via
     `run_trial.py --fixed-hp-idx <best> --fixed-t-steps <t>` — same HP, fresh dir
     (no checkpoint_index → cold start), so traintime_hours is the honest fast
     time to train that config to t_steps.

The `<cell>_fast` cells are normal cells (sweep_config.yaml + results/*.json), so
analyze_pretraining_scaling.py / the plots can read them side by side with the
originals.

RUN ON JEAN ZAY (reads results, copies state, writes jobs):
    python sweep/generate_fast_reruns.py [--sweep-base DIR] [--suffix _fast] [--dry-run]

Then submit them interleaved (round-robin across cells):
    python sweep/sweep_manager.py submit <printed list of *_fast dirs>
"""
import argparse
import json
import os
import re
import shutil
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(_project_dir, "sweeps", "pretraining_scaling")


def best_trial(cell_dir):
    """Return (val_loss, hp_idx, t_steps) for the best trial, or None."""
    rdir = os.path.join(cell_dir, "results")
    if not os.path.isdir(rdir):
        return None
    best = None
    for fn in os.listdir(rdir):
        m = re.match(r"hp(\d+)_t(\d+)_", fn)
        if not m or not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(rdir, fn)) as f:
                vl = json.load(f).get("val_loss")
        except Exception:
            continue
        if vl is None:
            continue
        if best is None or vl < best[0]:
            best = (float(vl), int(m.group(1)), int(m.group(2)))
    return best


def write_job(fast_dir, cfg, fast_config_path, hp_idx, t_steps):
    cluster      = cfg["cluster"]
    project_dir  = cfg["paths"]["project_dir"]
    trial_script = os.path.join(project_dir, "sweep", "run_trial.py")
    setup        = "\n".join(cfg["paths"].get("setup_commands", []))
    name         = os.path.basename(fast_dir)
    content = f"""\
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={cluster["partition"]}
#SBATCH --account={cluster["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{cluster.get("request_gpus", 1)}
#SBATCH --cpus-per-task={cluster.get("cpus_per_task", 8)}
#SBATCH --time={cluster.get("time", "02:00:00")}
#SBATCH --output={fast_dir}/output/trial_0000_%j.out
#SBATCH --error={fast_dir}/error/trial_0000_%j.err

{setup}

python {trial_script} \\
    --sweep-config {fast_config_path} \\
    --trial-idx 0 \\
    --fixed-hp-idx {hp_idx} \\
    --fixed-t-steps {t_steps}
"""
    path = os.path.join(fast_dir, "jobs", "trial_0000.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


def main():
    ap = argparse.ArgumentParser(description="Fast-code reruns of each cell's best trial")
    ap.add_argument("--sweep-base", default=DEFAULT_BASE,
                    help=f"dir containing the cells (default: {DEFAULT_BASE})")
    ap.add_argument("--suffix", default="_fast", help="suffix for the rerun cells")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without creating dirs/jobs")
    args = ap.parse_args()

    base = args.sweep_base
    fast_dirs, skipped = [], []
    for cell in sorted(os.listdir(base)):
        if cell.endswith(args.suffix):
            continue
        cell_dir = os.path.join(base, cell)
        cfg_path = os.path.join(cell_dir, "sweep_config.yaml")
        state    = os.path.join(cell_dir, "dyhpo_state.pkl")
        if not (os.path.isfile(cfg_path) and os.path.isfile(state)):
            continue  # not a cell
        b = best_trial(cell_dir)
        if b is None:
            skipped.append(f"{cell} (no results)")
            continue
        val_loss, hp_idx, t_steps = b
        fast_name = cell + args.suffix
        fast_dir  = os.path.join(base, fast_name)
        print(f"{cell:48s}  best hp_{hp_idx:04d}  t={t_steps:<6d}  "
              f"val_loss={val_loss:.3e}  ->  {fast_name}")
        fast_dirs.append(fast_dir)
        if args.dry_run:
            continue

        for d in ("jobs", "output", "error", "results", "dyhpo_surrogate"):
            os.makedirs(os.path.join(fast_dir, d), exist_ok=True)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["sweep_name"] = fast_name
        cfg.setdefault("fixed_params", {})["seed"] = 42
        fast_cfg_path = os.path.join(fast_dir, "sweep_config.yaml")
        with open(fast_cfg_path, "w") as f:
            yaml.dump(cfg, f)
        shutil.copy2(state, os.path.join(fast_dir, "dyhpo_state.pkl"))
        write_job(fast_dir, cfg, fast_cfg_path, hp_idx, t_steps)

    print(f"\n{len(fast_dirs)} cells" + (" (dry-run, nothing written)" if args.dry_run else " prepared"))
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    if fast_dirs and not args.dry_run:
        print("\nSubmit interleaved with:")
        print("  python sweep/sweep_manager.py submit \\\n    " + " \\\n    ".join(fast_dirs))


if __name__ == "__main__":
    main()
