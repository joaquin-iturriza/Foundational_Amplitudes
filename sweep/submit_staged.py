#!/usr/bin/env python3
"""
submit_staged.py — Submit DyHPO sweep trials in hard-barrier batches.

WHY (vs sweep_manager.py)
-------------------------
sweep_manager stamps jobs with a SLURM `nice` value: SOFT priority ordering. If
many GPUs free up at once (the user's case: ~30 GPUs together) nice does NOT stop
all trials starting in parallel — so every trial calls DyHPO `suggest()` against
the same stale surrogate and the feedback loop is lost.

This tool instead uses SLURM job DEPENDENCIES (`afterany`) to impose HARD barriers
between batches of trials within a sweep:

    batch 0 (the Sobol startup trials) submitted with no dependency
    batch 1 starts only AFTER every job in batch 0 has terminated (observed)
    batch 2 starts only AFTER every job in batch 1 has terminated
    ...

Because `run_trial.py` takes its HP config from `sampler.suggest()` (the
`--trial-idx` is logging-only), gating batch k+1 on batch k's *completion*
guarantees the surrogate has observed batch k before batch k+1 suggests. With
batches = [4, 6, 5] and dyhpo.n_startup=4, batch 0 is exactly the 4 Sobol points,
fully observed before any surrogate-driven trial runs.

`afterany` (not `afterok`) is deliberate: a failed trial still calls
`report_failure()` / frees its slot, so the campaign must not deadlock on it.

MULTIPLE SWEEPS
---------------
Pass several sweep dirs to interleave them: all sweeps' batch 0 are submitted
first (filling free GPUs across sweeps), then each sweep's batch 1 gated on ITS
OWN batch 0, and so on. The sweeps have independent DyHPO states, so each barrier
is per-sweep.

USAGE (run on Jean Zay — shells out to sbatch)
----------------------------------------------
    python sweep/submit_staged.py <sweepA_dir> <sweepB_dir>
    python sweep/submit_staged.py --batches 4,6,5 <sweep_dir>
    python sweep/submit_staged.py --dry-run <sweepA_dir> <sweepB_dir>
"""

import argparse
import os
import subprocess
import sys

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from sweep.sweep_manager import discover_trial_scripts, sweep_name_from_dir


def _sbatch(script, dep_jids=None, dry_run=False):
    cmd = ["sbatch", "--parsable"]
    if dep_jids:
        cmd.append("--dependency=afterany:" + ":".join(dep_jids))
    cmd.append(script)
    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return f"DRY{os.path.basename(script)}"
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout.strip().split(";")[0]


def _batch_slices(scripts, batches):
    """Split [scripts] into consecutive chunks of the given sizes."""
    if sum(batches) != len(scripts):
        raise SystemExit(
            f"batch sizes {batches} sum to {sum(batches)} but sweep has "
            f"{len(scripts)} trial scripts. Pass --batches to match.")
    out, i = [], 0
    for n in batches:
        out.append(scripts[i:i + n])
        i += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="+", help="sweep dirs (each containing jobs/)")
    ap.add_argument("--batches", default="4,6,5",
                    help="comma-separated batch sizes per sweep (default 4,6,5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print sbatch calls without executing")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]

    # Resolve scripts and split into batches per sweep up front (validate sizes).
    per_sweep = []   # [(name, [batch0_scripts, batch1_scripts, ...]), ...]
    for d in args.sweep_dirs:
        d = os.path.abspath(d)
        scripts = [p for _, p in discover_trial_scripts(d)]
        per_sweep.append((sweep_name_from_dir(d), _batch_slices(scripts, batches)))

    print(f"Staged submit: batches={batches} per sweep; barrier=afterany "
          f"(batch k+1 waits for batch k to finish).")

    # Round-by-round across sweeps: all sweeps' batch 0 first, then batch 1, ...
    # Each sweep's batch k+1 depends only on that sweep's own batch k job ids.
    prev_jids = {name: None for name, _ in per_sweep}
    n_rounds = len(batches)
    for k in range(n_rounds):
        print(f"\n=== round {k}  (batch size {batches[k]}) ===")
        for name, sliced in per_sweep:
            dep = prev_jids[name]
            jids = []
            for script in sliced[k]:
                jid = _sbatch(script, dep_jids=dep, dry_run=args.dry_run)
                jids.append(jid)
                dep_note = f"  dep=afterany:{','.join(dep)}" if dep else "  (no dep)"
                print(f"  {name}  {os.path.basename(script):16s} -> job={jid}{dep_note}")
            prev_jids[name] = jids

    print("\nSubmitted. Watch with:  squeue -u $USER -o '%.18i %.30j %.10T %.20E'")


if __name__ == "__main__":
    main()
