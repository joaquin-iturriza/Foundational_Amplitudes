#!/usr/bin/env python3
"""Generate a clean (non-DyHPO) learning-rate transfer sweep to test μP.

Goal: hold *everything* fixed except width (num_heads) and learning rate, and
scan LR on a grid at each width. If μP works, the loss-vs-LR curves for every
width share (approximately) the same optimum.

Design (per the choices made):
  * Reference cell : scaling_p1_nh16_D1e3_t31623 — supplies the fixed_params
    (datasets, subsample, batchsize, regularization=L2, scheduler, ...) and the
    SHARED non-LR HPs (its DyHPO best, minus training.lr). The same non-LR HPs
    are reused for every width so only LR and width vary.
  * Widths         : num_heads in {4, 8, 16, 32}  (the μP width axis).
  * Budget         : t_steps = 31623 for all widths (the common converged cell).
  * LR grid        : 11 log-spaced points in [1e-4, 3e-2].

Each (width, lr) is one independent `run.py` job (cold start, is_dyhpo_run=true
so warmup-frac / scheduler behave exactly as in the original cell). Results are
written as JSON ({"val_loss": ...}) and collected by analyze_mup_lr_transfer.py.

Run this ON JEAN ZAY (it reads the reference cell's config + summary):

    python sweep/generate_mup_lr_transfer.py
    bash <sweep_dir>/mup_lr_transfer_D1e3_t31623/submit_all.sh   # to launch
"""
import argparse
import json
import os
import shlex
import sys

import numpy as np
import yaml

DEFAULT_REF = (
    "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/"
    "sweeps/pretraining_scaling/scaling_p1_nh16_D1e3_t31623/sweep_config.yaml"
)


def parse_best_params(summary_path):
    """Pull the 'Best params:' block out of a cell summary.txt -> {key: value}."""
    params = {}
    with open(summary_path) as f:
        lines = f.read().splitlines()
    in_block = False
    for ln in lines:
        if ln.startswith("Best params:"):
            in_block = True
            continue
        if in_block:
            if not ln.startswith("  ") or "=" not in ln:
                break
            k, v = ln.strip().split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                v = float(v)
            except ValueError:
                pass
            params[k] = v
    return params


def fmt(v):
    return f"{v:.10e}" if isinstance(v, float) else str(v)


def build_run_cmd(project_dir, fixed_params, shared_hps, num_heads, lr,
                  t_steps, run_dir, result_path):
    run_script = os.path.join(project_dir, "run.py")
    cmd = [sys.executable, run_script, "local=none"]

    # fixed params from the reference cell, but force num_heads per width and
    # never pass training.iterations / training.lr (set explicitly below).
    skip = {"training.iterations", "training.lr", "model.net.num_heads"}
    for k, v in fixed_params.items():
        if k in skip:
            continue
        cmd.append(f"{k}={v}")
    cmd.append(f"model.net.num_heads={num_heads}")

    # shared non-LR HPs (reference cell's best, training.lr already dropped)
    for k, v in shared_hps.items():
        cmd.append(f"{k}={fmt(v)}")

    # swept LR + budget (cold start, dyhpo-style scheduler logic)
    cmd.append(f"training.lr={fmt(lr)}")
    cmd.append(f"training.iterations={t_steps}")
    cmd.append(f"training.increment_steps={t_steps}")
    cmd.append("training.is_dyhpo_run=true")
    cmd.append("warm_start_idx=null")

    cmd.append(f"base_dir={project_dir}")
    cmd.append(f"run_dir={run_dir}")
    cmd.append("exp_name=mup_lr_transfer_D1e3_t31623")
    cmd.append(f"training.result_path={result_path}")
    # shlex.quote each token: some fixed_params (data.dataset, train_test_val)
    # contain spaces and must stay a single shell word for Hydra to parse.
    return shlex.join(cmd)


def slurm_header(cluster, name, out_dir, err_dir, setup_commands, time):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={cluster.get('partition', 'gpu_p2')}",
        f"#SBATCH --account={cluster.get('account', 'itg@v100')}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gres=gpu:{cluster.get('request_gpus', 1)}",
        f"#SBATCH --cpus-per-task={cluster.get('cpus_per_task', 8)}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={out_dir}/{name}_%j.out",
        f"#SBATCH --error={err_dir}/{name}_%j.err",
        "",
    ]
    lines += list(setup_commands) + [""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-config", default=DEFAULT_REF,
                    help="reference cell sweep_config.yaml (fixed_params + paths)")
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=3e-2)
    ap.add_argument("--lr-points", type=int, default=11)
    ap.add_argument("--t-steps", type=int, default=31623)
    ap.add_argument("--time", default="02:00:00", help="SLURM walltime (nh32 is slow)")
    args = ap.parse_args()

    with open(args.ref_config) as f:
        ref = yaml.safe_load(f)

    paths = ref["paths"]
    project_dir = paths["project_dir"]
    sweep_root = paths["sweep_dir"]
    cluster = ref.get("cluster", {})
    setup_commands = paths.get("setup_commands", [])
    fixed_params = ref.get("fixed_params", {})

    summary_path = os.path.join(os.path.dirname(args.ref_config), "summary.txt")
    best = parse_best_params(summary_path)
    shared_hps = {k: v for k, v in best.items() if k != "training.lr"}
    print(f"[gen] reference cell : {os.path.dirname(args.ref_config)}")
    print(f"[gen] shared non-LR HPs (from nh16 best):")
    for k, v in shared_hps.items():
        print(f"        {k} = {v}")

    sweep_name = "mup_lr_transfer_D1e3_t31623"
    out_root = os.path.join(sweep_root, sweep_name)
    jobs_dir = os.path.join(out_root, "jobs")
    res_dir = os.path.join(out_root, "results")
    log_out = os.path.join(out_root, "output")
    log_err = os.path.join(out_root, "error")
    for d in (jobs_dir, res_dir, log_out, log_err):
        os.makedirs(d, exist_ok=True)

    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_points)
    manifest = []
    job_paths = []

    for nh in args.widths:
        for i, lr in enumerate(lrs):
            name = f"nh{nh:02d}_lr{i:02d}"
            run_dir = os.path.join(project_dir, "runs", sweep_name, name)
            result_path = os.path.join(res_dir, f"{name}.json")
            cmd = build_run_cmd(project_dir, fixed_params, shared_hps, nh,
                                float(lr), args.t_steps, run_dir, result_path)
            script = (
                slurm_header(cluster, name, log_out, log_err, setup_commands, args.time)
                + f"\ncd {project_dir}\n{cmd}\n"
            )
            job_path = os.path.join(jobs_dir, f"{name}.sh")
            with open(job_path, "w") as f:
                f.write(script)
            job_paths.append(job_path)
            manifest.append(dict(name=name, num_heads=nh, lr=float(lr),
                                 lr_idx=i, t_steps=args.t_steps,
                                 run_dir=run_dir, result_path=result_path,
                                 job=job_path))

    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(dict(sweep_name=sweep_name, widths=args.widths,
                       lrs=[float(x) for x in lrs], t_steps=args.t_steps,
                       shared_hps=shared_hps, jobs=manifest), f, indent=2)

    submit = os.path.join(out_root, "submit_all.sh")
    with open(submit, "w") as f:
        f.write("#!/bin/bash\n# Independent runs (no shared DyHPO state) — plain sbatch loop is fine.\n")
        for jp in job_paths:
            f.write(f"sbatch {jp}\n")
    os.chmod(submit, 0o755)

    print(f"\n[gen] {len(job_paths)} jobs ({len(args.widths)} widths x {args.lr_points} LRs) "
          f"in {jobs_dir}")
    print(f"[gen] LR grid: {', '.join(f'{x:.2e}' for x in lrs)}")
    print(f"[gen] submit : bash {submit}")
    print(f"[gen] analyze: python sweep/analyze_mup_lr_transfer.py "
          f"--sweep-dir {out_root}")


if __name__ == "__main__":
    main()
