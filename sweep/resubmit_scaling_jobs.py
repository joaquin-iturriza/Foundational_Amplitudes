#!/usr/bin/env python3
"""
resubmit_scaling_jobs.py — Cancel all running/pending scaling sweep jobs and resubmit them.

Queries squeue to find jobs whose output file lives under SWEEP_BASE, cancels them,
and resubmits the matching job script from the cell's jobs/ directory.

Usage:
    python sweep/resubmit_scaling_jobs.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys

SWEEP_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/sweeps/pretraining_scaling"


def get_queued_scaling_jobs():
    """Returns list of (job_id, job_name, output_path) for all running/pending jobs
    whose output file is under SWEEP_BASE."""
    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j %o"],
        capture_output=True, text=True, check=True,
    )
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        job_id, job_name, output_path = parts
        if SWEEP_BASE in output_path:
            jobs.append((job_id, job_name, output_path))
    return jobs


def parse_job(job_name, output_path):
    """Extract (cell_name, script_path) from a job's metadata."""
    # output_path: .../SWEEP_BASE/<cell_name>/output/<trial_name>_%j.out
    rel = output_path[len(SWEEP_BASE):].lstrip("/")
    cell_name = rel.split("/")[0]
    # job_name is the trial script name without .sh, e.g. "trial_0000"
    script_path = os.path.join(SWEEP_BASE, cell_name, "jobs", f"{job_name}.sh")
    return cell_name, script_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without cancelling or resubmitting")
    args = parser.parse_args()

    jobs = get_queued_scaling_jobs()
    if not jobs:
        print("No running/pending scaling jobs found.")
        return

    print(f"Found {len(jobs)} running/pending scaling job(s):\n")

    to_cancel  = []
    to_resubmit = []
    errors = []

    for job_id, job_name, output_path in jobs:
        cell_name, script_path = parse_job(job_name, output_path)
        exists = os.path.exists(script_path)
        status = "ok" if exists else "MISSING SCRIPT"
        print(f"  {job_id:>10s}  {cell_name}/{job_name}.sh  [{status}]")
        if exists:
            to_cancel.append(job_id)
            to_resubmit.append((cell_name, script_path))
        else:
            errors.append((job_id, script_path))

    if errors:
        print(f"\nWARNING: {len(errors)} job(s) have missing scripts — will cancel but not resubmit:")
        for job_id, script_path in errors:
            print(f"  {job_id}  {script_path}")

    print()
    if args.dry_run:
        print(f"[dry-run] Would cancel {len(to_cancel)} job(s) and resubmit {len(to_resubmit)}.")
        return

    # Cancel
    if to_cancel:
        print(f"Cancelling {len(to_cancel)} job(s)...")
        subprocess.run(["scancel"] + to_cancel, check=True)
        print("Done.")

    # Resubmit
    print(f"\nResubmitting {len(to_resubmit)} job(s)...")
    submitted, failed = 0, 0
    for cell_name, script_path in to_resubmit:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            new_id = result.stdout.strip().split()[-1]
            print(f"  {cell_name}/{os.path.basename(script_path)}  -> job {new_id}")
            submitted += 1
        else:
            print(f"  FAILED {cell_name}/{os.path.basename(script_path)}: {result.stderr.strip()}")
            failed += 1

    print(f"\nDone. submitted={submitted}  failed={failed}")


if __name__ == "__main__":
    main()
