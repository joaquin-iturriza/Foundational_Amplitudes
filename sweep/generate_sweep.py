#!/usr/bin/env python3
"""
generate_sweep.py  —  Initialise a DyHPO sweep and generate HTCondor job files.

Usage (from lxplus login node or locally):
    python sweep/generate_sweep.py --config sweep/sweep_config.yaml
    python sweep/generate_sweep.py --config sweep/sweep_config.yaml --dry-run
    python sweep/generate_sweep.py --config sweep/sweep_config.yaml --n-trials 60
    python sweep/generate_sweep.py --config sweep/sweep_config.yaml --extend --n-trials 20

Re-running with --extend adds more job files to an existing sweep without
resetting the DyHPO surrogate state or the HP candidate pool.
"""

import argparse
import os
import subprocess
import sys

import yaml

# Make sweep/ importable from the project root
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _sweep_dirs(cfg, sweep_name):
    """Return (sweep_dir, eos_dir). Identical when paths.sweep_dir is set (Jean-Zay / single FS)."""
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def setup_dirs(cfg, sweep_name):
    afs, eos = _sweep_dirs(cfg, sweep_name)
    for d in [
        os.path.join(afs, "jobs"),
        os.path.join(afs, "output"),
        os.path.join(afs, "error"),
        os.path.join(eos, "results"),
        os.path.join(eos, "dyhpo_surrogate"),
    ]:
        os.makedirs(d, exist_ok=True)
    if afs != eos:
        for d in [os.path.join(afs, "subs"), os.path.join(afs, "log")]:
            os.makedirs(d, exist_ok=True)
    return afs, eos


def _scheduler(cfg):
    return cfg["cluster"].get("scheduler", "htcondor")


def init_sampler(cfg, afs_dir, eos_dir):
    """Create and save the initial DyHPOSampler state to AFS."""
    from sweep.dyhpo_sampler import DyHPOSampler

    state_path = os.path.join(afs_dir, "dyhpo_state.pkl")
    if os.path.exists(state_path):
        print(f"  DyHPO state already exists at {state_path} — skipping init.")
        return state_path

    dyhpo_cfg   = cfg.get("dyhpo", {})
    fidelity    = cfg["fidelity_schedule"]
    hp_space    = cfg["search_space"]

    sampler = DyHPOSampler(
        hp_space       = hp_space,
        fidelity_grid  = fidelity,
        n_candidates   = dyhpo_cfg.get("n_candidates", 300),
        seed           = dyhpo_cfg.get("seed", 42),
        output_path    = os.path.join(eos_dir, "dyhpo_surrogate"),
        n_startup      = dyhpo_cfg.get("n_startup", 10),
        total_budget   = dyhpo_cfg.get("total_budget", 10_000),
    )
    sampler.save(state_path)
    print(f"  DyHPO state initialised: {state_path}")
    print(f"  HP candidates pre-sampled: {sampler.n_candidates}")
    return state_path


def write_sh(i, cfg, afs_dir, config_abs_path, t_steps_cap=None):
    project_dir  = cfg["paths"]["project_dir"]
    python_env   = cfg["paths"]["python_env"]
    trial_script = os.path.join(project_dir, "sweep", "run_trial.py")
    cap_flag = f" \\\n    --t-steps-cap {t_steps_cap}" if t_steps_cap is not None else ""

    eos_check = ""
    if "eos_sweep_dir" in cfg.get("paths", {}):
        eos_check = f"""\
_eos_ok=0
for _attempt in 1 2 3 4 5; do
    if [ -f {python_env} ]; then _eos_ok=1; break; fi
    echo "EOS not accessible (attempt ${{_attempt}}/5), retrying in 30s..."
    sleep 30
done
if [ "${{_eos_ok}}" -eq 0 ]; then echo "ERROR: EOS not accessible."; exit 1; fi

"""

    content = f"""\
#!/bin/bash
set -euo pipefail

{eos_check}source {python_env}

python {trial_script} \\
    --sweep-config {config_abs_path} \\
    --trial-idx {i}{cap_flag}
"""
    path = os.path.join(afs_dir, "jobs", f"trial_{i:04d}.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


def _env_setup_lines(cfg):
    """Return shell lines that activate the Python environment."""
    setup_commands = cfg["paths"].get("setup_commands")
    if setup_commands:
        return "\n".join(setup_commands)
    return f"source {cfg['paths']['python_env']}"


def write_slurm_script(i, cfg, sweep_dir, config_abs_path, t_steps_cap=None):
    cluster      = cfg["cluster"]
    project_dir  = cfg["paths"]["project_dir"]
    trial_script = os.path.join(project_dir, "sweep", "run_trial.py")
    cap_flag = f" \\\n    --t-steps-cap {t_steps_cap}" if t_steps_cap is not None else ""

    mem_line = f"#SBATCH --mem={cluster['mem']}\n" if "mem" in cluster else ""
    content = f"""\
#!/bin/bash
#SBATCH --job-name=trial_{i:04d}
#SBATCH --partition={cluster["partition"]}
#SBATCH --account={cluster["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{cluster["request_gpus"]}
#SBATCH --cpus-per-task={cluster.get("cpus_per_task", 8)}
#SBATCH --time={cluster.get("time", "20:00:00")}
{mem_line}#SBATCH --output={sweep_dir}/output/trial_{i:04d}_%j.out
#SBATCH --error={sweep_dir}/error/trial_{i:04d}_%j.err

{_env_setup_lines(cfg)}

python {trial_script} \\
    --sweep-config {config_abs_path} \\
    --trial-idx {i}{cap_flag}
"""
    path = os.path.join(sweep_dir, "jobs", f"trial_{i:04d}.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


def write_sub(i, cfg, afs_dir, sh_path, low_fidelity=False):
    cluster      = cfg["cluster"]
    if low_fidelity:
        requirements = cluster.get("low_fidelity_requirements",
                                   cluster.get("requirements", "True"))
    else:
        requirements = cluster.get("requirements", "True")
    extra        = "\n".join(cluster.get("extra_lines", []))
    if extra:
        extra = extra + "\n"

    gpu_mem_line = ""
    if "gpus_minimum_memory" in cluster:
        gpu_mem_line = f"gpus_minimum_memory     = {cluster['gpus_minimum_memory']}\n"

    priority = cluster.get("priority", None)
    priority_line = f"priority              = {priority}\n" if priority is not None else ""

    request_memory = cluster.get("request_memory", None)
    memory_line = f"request_memory        = {request_memory}\n" if request_memory is not None else ""

    name = f"trial_{i:04d}"
    content = f"""\
# HTCondor submission — sweep: {cfg["sweep_name"]}, trial {i:04d}
# Generated by generate_sweep.py — do not edit manually

executable  = {sh_path}
output      = {afs_dir}/output/{name}.$(ClusterId).$(ProcId).out
error       = {afs_dir}/error/{name}.$(ClusterId).$(ProcId).err
log         = {afs_dir}/log/{name}.$(ClusterId).$(ProcId).log

request_gpus          = {cluster["request_gpus"]}
{gpu_mem_line}{memory_line}+JobFlavour           = "{cluster["job_flavour"]}"
requirements          = {requirements}
max_retries           = 3
{priority_line}{extra}
queue
"""
    path = os.path.join(afs_dir, "subs", f"{name}.sub")
    with open(path, "w") as f:
        f.write(content)
    return path


def prompt_yes_no(msg):
    ans = input(f"\n{msg} (y/N): ").strip().lower()
    return ans == "y"


def next_available_name(base_name, afs_sweep_dir):
    import re
    def state_exists(name):
        return os.path.exists(os.path.join(afs_sweep_dir, name, "dyhpo_state.pkl"))

    if not state_exists(base_name):
        return base_name

    m = re.search(r"^(.*?)(_(\d+))$", base_name)
    if m:
        stem, _, num = m.group(1), m.group(2), m.group(3)
        n = int(num)
        width = len(num)
    else:
        stem  = base_name
        n     = 1
        width = 3

    while True:
        n += 1
        candidate = f"{stem}_{n:0{width}d}"
        if not state_exists(candidate):
            return candidate


def run_generate(config_path, n_trials=None, extend=False, dry_run=False, submit=None):
    """Generate (and optionally submit) one DyHPO sweep.

    submit : None  -> use cluster.auto_submit, else prompt interactively
             True  -> submit now via sweep_manager (interleaved across sweeps)
             False -> generate only (print the sweep_manager command to run later)

    Returns the sweep directory (the one containing jobs/), or None on dry-run.
    """
    config_abs_path = os.path.abspath(config_path)
    cfg = load_config(config_abs_path)
    # Fix the init seed by default so trials differ only by hyperparameters
    # (a config may override by setting its own fixed_params.seed).
    cfg.setdefault("fixed_params", {}).setdefault("seed", 42)
    n_trials = n_trials if n_trials is not None else cfg.get("n_trials", 40)

    base_name     = cfg["sweep_name"]
    afs_sweep_dir = cfg["paths"].get("afs_sweep_dir", cfg["paths"].get("sweep_dir"))
    if extend:
        sweep_name = base_name
        print(f"Extending existing sweep: {sweep_name}")
    else:
        sweep_name = next_available_name(base_name, afs_sweep_dir)
        if sweep_name != base_name:
            print(f"Sweep '{base_name}' already exists — using '{sweep_name}' instead.")

    cfg["sweep_name"] = sweep_name
    print(f"Sweep: {sweep_name}  |  n_trials: {n_trials}")

    afs_dir, eos_dir = setup_dirs(cfg, sweep_name)
    print(f"  Sweep dir : {afs_dir}")

    saved_config = os.path.join(afs_dir, "sweep_config.yaml")
    if not (dry_run and os.path.exists(saved_config)):
        with open(saved_config, "w") as f:
            yaml.dump(cfg, f)
        print(f"  Config saved : {saved_config}")

    if not dry_run:
        init_sampler(cfg, afs_dir, eos_dir)

    # Count existing job files to continue numbering when --extend
    existing_jobs = len([
        f for f in os.listdir(os.path.join(afs_dir, "jobs"))
        if f.startswith("trial_") and f.endswith(".sh")
    ]) if os.path.isdir(os.path.join(afs_dir, "jobs")) else 0

    t_steps_sched   = cfg["fidelity_schedule"]["t_steps"]
    lf_n_trials     = cfg["cluster"].get("low_fidelity_trials", 0)
    lf_t_steps_cap  = max((t for t in t_steps_sched if t < max(t_steps_sched)), default=None)

    scheduler = _scheduler(cfg)
    job_paths = []
    n_lf = 0
    for i in range(existing_jobs, existing_jobs + n_trials):
        low_fidelity = lf_t_steps_cap is not None and (i - existing_jobs) < lf_n_trials
        cap          = lf_t_steps_cap if low_fidelity else None
        if scheduler == "slurm":
            job_path = write_slurm_script(i, cfg, afs_dir, saved_config, t_steps_cap=cap)
        else:
            sh_path  = write_sh(i, cfg, afs_dir, saved_config, t_steps_cap=cap)
            job_path = write_sub(i, cfg, afs_dir, sh_path, low_fidelity=low_fidelity)
        job_paths.append(job_path)
        if low_fidelity:
            n_lf += 1

    lf_summary = f"  ({n_lf} low-fidelity with t_steps_cap={lf_t_steps_cap})" if n_lf else ""
    print(f"\nGenerated {n_trials} jobs (trial_{existing_jobs:04d} … trial_{existing_jobs+n_trials-1:04d}){lf_summary}")
    print(f"  jobs : {os.path.join(afs_dir, 'jobs')}")

    if dry_run:
        print("\n[dry-run] Done.")
        return afs_dir

    # ── Submission ────────────────────────────────────────────────────────────
    # SLURM jobs go through sweep_manager so trials interleave round-robin across
    # all sweeps currently in the queue (one of each first, then the second, ...),
    # which keeps the DyHPO surrogate informed. HTCondor (legacy) keeps the old path.
    if submit is None:
        do_submit = cfg["cluster"].get("auto_submit", False)
        if not do_submit:
            prompt_msg = ("Submit all jobs now (interleaved via sweep_manager)?"
                          if scheduler == "slurm" else "Submit all jobs to HTCondor now?")
            do_submit = prompt_yes_no(prompt_msg)
    else:
        do_submit = submit

    if do_submit:
        if scheduler == "slurm":
            from sweep.sweep_manager import submit_sweeps
            submit_sweeps([afs_dir])
        else:
            submitted = 0
            for job_path in job_paths:
                try:
                    subprocess.run(["condor_submit", job_path], check=True)
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"  Failed to submit {job_path}: {e}", file=sys.stderr)
            print(f"\nSubmitted {submitted}/{n_trials} jobs.")
    elif submit is None:
        # Standalone/interactive use: tell the user how to submit later.
        # (When a caller passes submit=False it batches submission itself, so stay quiet.)
        if scheduler == "slurm":
            print("\nSkipping submission. Submit later (interleaved) with:")
            print(f"  python sweep/sweep_manager.py submit {afs_dir}")
        else:
            print("\nSkipping submission. Submit later with:")
            print(f"  for f in {afs_dir}/subs/trial_*.sub; do condor_submit $f; done")

    return afs_dir


def main():
    parser = argparse.ArgumentParser(description="Generate DyHPO sweep jobs")
    parser.add_argument("--config",    required=True, help="Path to sweep_config.yaml")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Generate files but do not initialise DyHPO state or submit")
    parser.add_argument("--n-trials",  type=int, default=None,
                        help="Override n_trials from config")
    parser.add_argument("--extend",    action="store_true",
                        help="Add more trials to an existing sweep (reuse DyHPO state)")
    args = parser.parse_args()

    run_generate(args.config, n_trials=args.n_trials, extend=args.extend,
                 dry_run=args.dry_run, submit=None)


if __name__ == "__main__":
    main()
