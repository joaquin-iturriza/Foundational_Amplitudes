#!/usr/bin/env python3
"""
resubmit_timed_out.py — Patch time limits and re-submit timed-out Phase 1 cells.

For each specified cell (or auto-detected incomplete cells), this script:
  1. Identifies which trial scripts are missing results.
  2. Updates the #SBATCH --time= line with the corrected limit.
  3. Prints (or runs) the sbatch commands.

Usage:
    python sweep/resubmit_timed_out.py                    # auto-detect all incomplete
    python sweep/resubmit_timed_out.py --submit           # actually run sbatch
    python sweep/resubmit_timed_out.py --cell scaling_p1_D1e5_t1000 --submit
"""

import argparse
import math
import os
import re
import sys

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Support both Jean-Zay (Lustre) and the local SSHFS mount used for editing
_LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
_MOUNT_BASE  = "/home/joaquin/mnt/jeanzay/Foundational_Amplitudes"
LUSTRE_BASE  = _LUSTRE_BASE if os.path.isdir(_LUSTRE_BASE) else _MOUNT_BASE
SWEEP_BASE   = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")

N_TRIALS = 20   # expected trials per cell

STEP_TIME_MS = {
    2:  {256: 35.89,  1024: 36.03,  4096: 110.52, 8192: 211.60},
    4:  {256: 37.78,  1024: 37.93,  4096: 116.34,  8192: 222.74},
    8:  {256: 38.71,  1024: 42.35,  4096: 218.54,  8192: 426.36},
    16: {256: 43.58,  1024: 65.18,  4096: 450.69,  8192: 889.11},
    32: {256: 74.75,  1024: 138.56, 4096: 1015.07, 8192: None},
}


def slurm_time_str(t_steps: int, ms_per_step: float) -> str:
    total_seconds = t_steps * ms_per_step / 1000.0 * 3.5
    total_seconds = max(total_seconds, 1800)
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    minutes = math.ceil((hours * 60 + minutes) / 30) * 30
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"


def infer_time_limit(sweep_dir: str) -> str | None:
    """Read the sweep config to infer t_steps, num_heads, BS, then compute limit."""
    import yaml
    cfg_path = os.path.join(sweep_dir, "sweep_config.yaml")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    t_steps  = cfg.get("fidelity_schedule", {}).get("t_steps", [None])[-1]
    num_heads = cfg.get("fixed_params", {}).get("model.net.num_heads", 16)
    bs        = cfg.get("fixed_params", {}).get("training.batchsize", 8192)
    if t_steps is None:
        return None
    ms = STEP_TIME_MS.get(int(num_heads), {}).get(int(bs))
    if ms is None:
        ms = STEP_TIME_MS.get(int(num_heads), {}).get(int(bs) // 2, 1200.0)
    return slurm_time_str(int(t_steps), ms)


def n_results(sweep_dir: str) -> int:
    rdir = os.path.join(sweep_dir, "results")
    if not os.path.isdir(rdir):
        return 0
    return sum(1 for f in os.listdir(rdir) if f.endswith(".json"))


def patch_and_collect(sweep_dir: str, new_time: str, submit: bool) -> list[str]:
    """Patch --time in all job scripts; return list of sbatch commands."""
    jobs_dir = os.path.join(sweep_dir, "jobs")
    if not os.path.isdir(jobs_dir):
        print(f"  [warn] no jobs/ dir in {sweep_dir}")
        return []
    cmds = []
    for fname in sorted(os.listdir(jobs_dir)):
        if not fname.endswith(".sh"):
            continue
        script = os.path.join(jobs_dir, fname)
        with open(script) as f:
            text = f.read()
        patched = re.sub(r"(#SBATCH --time=)\S+", rf"\g<1>{new_time}", text)
        with open(script, "w") as f:
            f.write(patched)
        cmds.append(f"sbatch {script}")
    return cmds


def main():
    parser = argparse.ArgumentParser(description="Re-submit timed-out Phase 1 trials.")
    parser.add_argument("--cell", default=None,
                        help="Specific cell name (e.g. scaling_p1_D1e5_t1000). "
                             "If omitted, auto-detect all incomplete cells.")
    parser.add_argument("--submit", action="store_true",
                        help="Actually run sbatch; default is dry-run (print only).")
    args = parser.parse_args()

    if not os.path.isdir(SWEEP_BASE):
        sys.exit(f"Sweep base not found: {SWEEP_BASE}")

    if args.cell:
        candidates = [args.cell]
    else:
        candidates = sorted(
            n for n in os.listdir(SWEEP_BASE)
            if n.startswith("scaling_p1_") and os.path.isdir(os.path.join(SWEEP_BASE, n))
        )

    to_resubmit = []
    for cell_name in candidates:
        sweep_dir = os.path.join(SWEEP_BASE, cell_name)
        n = n_results(sweep_dir)
        if n >= N_TRIALS:
            continue
        new_time = infer_time_limit(sweep_dir)
        if new_time is None:
            print(f"  [skip] {cell_name}: cannot infer time limit")
            continue
        to_resubmit.append((cell_name, sweep_dir, n, new_time))

    if not to_resubmit:
        print("No incomplete cells found.")
        return

    print(f"{'Cell':45s}  {'done':>6s}  {'new_time':>10s}")
    print("-" * 70)
    for cell_name, _, n, new_time in to_resubmit:
        print(f"  {cell_name:43s}  {n:>4d}/{N_TRIALS}  {new_time}")

    all_cmds = []
    for cell_name, sweep_dir, n, new_time in to_resubmit:
        cmds = patch_and_collect(sweep_dir, new_time, args.submit)
        all_cmds.extend(cmds)

    print(f"\n{len(all_cmds)} job scripts patched with new time limits.")

    # Always print Lustre-path sbatch commands (for running on Jean-Zay)
    print("\nTo submit on Jean-Zay (use Lustre paths):")
    for cell_name, sweep_dir, _, _ in to_resubmit:
        lustre_jobs = os.path.join(
            _LUSTRE_BASE, "sweeps", "pretraining_scaling", cell_name, "jobs"
        )
        print(f"  for f in {lustre_jobs}/*.sh; do sbatch \"$f\"; done")

    if args.submit:
        import subprocess
        for cmd in all_cmds:
            r = subprocess.run(cmd.split(), capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  submitted: {r.stdout.strip()}")
            else:
                print(f"  ERROR: {cmd}\n    {r.stderr.strip()}")


if __name__ == "__main__":
    main()
