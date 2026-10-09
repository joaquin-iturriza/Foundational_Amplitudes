#!/usr/bin/env python3
"""
run_trial.py  —  Per-job entrypoint for DyHPO-based multi-fidelity HPO.

Each HTCondor job runs this script once. It:
  1. Locks the shared DyHPO state on AFS, calls sampler.suggest(), unlocks.
  2. Checks the checkpoint index to decide whether to warm-start from a previous
     fidelity level for this HP configuration.
  3. Runs `python run.py` with the suggested HP params and fidelity overrides.
  4. Locks the state again, calls sampler.observe() with the result, unlocks.
  5. Registers the new training checkpoint in the checkpoint index.

Fidelity dimension:
  - t_steps : training steps (training.iterations = increment from previous level)
"""

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import yaml
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, so siteconf imports from anywhere
import siteconf


# Make sweep/ importable from the project root
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import subprocess

# ---------------------------------------------------------------
# SIGTERM handler — called when SLURM cancels the job.
# Reports the in-flight trial as failed so DyHPO can reuse the
# slot. Uses os._exit to avoid hanging on Python cleanup.
# ---------------------------------------------------------------
_sigterm_hp_idx   = None
_sigterm_state    = None
_sigterm_eos      = None

def _sigterm_handler(signum, frame):
    if _sigterm_hp_idx is not None and _sigterm_state is not None:
        try:
            from sweep.dyhpo_sampler import DyHPOSampler
            with DyHPOSampler.locked(_sigterm_state, _sigterm_eos) as sampler:
                sampler.report_failure(_sigterm_hp_idx)
            print(f"[run_trial] SIGTERM: reported failure for hp_idx={_sigterm_hp_idx}",
                  file=sys.stderr)
        except Exception as e:
            print(f"[run_trial] SIGTERM: report_failure failed: {e}", file=sys.stderr)
    os._exit(1)

signal.signal(signal.SIGTERM, _sigterm_handler)
from sweep.dyhpo_sampler import DyHPOSampler
from sweep.checkpoint_index import CheckpointIndex


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


RETRY_ATTEMPTS = 3
RETRY_DELAY_S  = 30


def load_config(path):
    with open(path) as f:
        return siteconf.resolve(yaml.safe_load(f))


def compute_hpo_objective(result: dict, scaling_law_cfg: dict) -> float | None:
    """
    Compute the geometric mean of per-dataset transfer ratios as the HPO objective.

    transfer_ratio_ds = val_loss_joint_ds / (C_ds · compute_ds^{-α_ds})

    Values < 1 mean joint training beats the solo scaling law prediction (positive
    transfer).  Returns None if scaling law params are unavailable.
    """
    proc_val_losses = result.get("proc_val_losses", {})
    compute_ds      = result.get("compute_ds", {})

    if not proc_val_losses or not compute_ds:
        return None

    # Load fitted params if available
    sl_params  = {}
    params_file = scaling_law_cfg.get("params_file")
    if params_file and os.path.exists(params_file):
        with open(params_file) as f:
            sl_params = json.load(f)

    alpha_priors = scaling_law_cfg.get("alpha_prior", {})

    ratios = []
    for ds_name, val_loss in proc_val_losses.items():
        compute = compute_ds.get(ds_name, 0)
        if compute <= 0 or val_loss <= 0:
            continue

        p     = sl_params.get(ds_name, {})
        alpha = p.get("alpha", alpha_priors.get(ds_name, 0.5))
        C     = p.get("C")   # None if scaling_law.py hasn't been run yet

        if C is None:
            # Without C we can't compute the transfer ratio — skip this dataset
            continue

        expected = C * (compute ** (-alpha))
        if expected > 0:
            ratios.append(val_loss / expected)

    if not ratios:
        return None

    return float(np.exp(np.mean(np.log(np.clip(ratios, 1e-10, None)))))


def _failure_penalty(sampler, cfg):
    """Worst-case val_loss to impute for a crashed/diverged trial.

    Returns ``max(observed) * margin`` so the penalty stays on the same scale as
    real losses (a fixed huge value would distort the GP's noise/lengthscale fit).
    Falls back to a configured constant when nothing has been observed yet.
    Set ``dyhpo.failure_penalty: null`` to disable imputation entirely.
    """
    dy = cfg.get("dyhpo", {}) or {}
    if "failure_penalty" in dy and dy["failure_penalty"] is None:
        return None
    margin   = float(dy.get("failure_penalty_margin", 1.5))
    fallback = float(dy.get("failure_penalty", 10.0))
    observed = [v for d in sampler._val_loss_history.values() for v in d.values()
                if v is not None and math.isfinite(v)]
    return max(observed) * margin if observed else fallback


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def build_command(cfg, hp_params, run_dir, run_idx, result_path, t_steps,
                  warm_start_idx=None, increment_steps=None):
    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = cfg["sweep_name"]

    cmd = [sys.executable, run_script]
    cmd.append("local=none")

    # Fixed params (excluding training.iterations — set below)
    skip_keys = {"training.iterations"}
    for key, val in cfg.get("fixed_params", {}).items():
        if key not in skip_keys:
            cmd.append(f"{key}={val}")

    # Suggested HP params
    for key, val in hp_params.items():
        cmd.append(f"{key}={format_value(val)}")

    cmd.append(f"training.iterations={increment_steps}")
    cmd.append(f"training.is_dyhpo_run=true")

    # Run dir and index for output organisation
    cmd.append(f"base_dir={project_dir}")
    cmd.append(f"run_dir={run_dir}")
    cmd.append(f"exp_name={sweep_name}")

    # Always pass increment_steps so scheduler logic has it explicitly at every fidelity level
    cmd.append(f"training.increment_steps={increment_steps}")

    if warm_start_idx is not None:
        # Resume from previous fidelity level checkpoint
        cmd.append(f"warm_start_idx={warm_start_idx}")
    else:
        # Cold start
        cmd.append("warm_start_idx=null")

    cmd.append(f"training.result_path={result_path}")

    return cmd


def _drop_last_model(cfg, run_dir, run_idx, t_steps):
    """A trial that ended at its sweep's top fidelity keeps only its best checkpoint: the last model
    (model_run<i>.pt[.gz]) is read only to warm-start a higher fidelity, and every reported value and every
    fine-tune parent is the best checkpoint (CLAUDE.md "Reported values"; the user's call of 2026-10-09: the last
    models were half of the 212 GB of runs/ on CC-IN2P3 with /sps/lpnhe at 98%)."""
    sched = (cfg.get("fidelity_schedule") or {}).get("t_steps") or [t_steps]
    if t_steps < max(sched):
        return
    models = os.path.join(run_dir, "models")
    best = [os.path.join(models, f"model_run{run_idx}_best.pt{z}") for z in (".gz", "")]
    if not any(os.path.exists(f) for f in best):
        return                                     # no best checkpoint: keep the only model there is
    for z in (".gz", ""):
        f = os.path.join(models, f"model_run{run_idx}.pt{z}")
        if os.path.exists(f):
            os.remove(f)
            print(f"[run_trial] removed the last model {f} (the best checkpoint is kept)")


def _write_summary(cfg, sampler, eos_dir):
    results = sampler.all_results()
    best    = sampler.best_result()

    lines = [
        f"Sweep: {cfg['sweep_name']}",
        f"Evaluations completed: {len(results)}",
    ]
    if best:
        params, loss = best
        lines += [f"Best val_loss: {loss:.6g}", "Best params:"]
        for k, v in params.items():
            lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("Top-10 results (across all fidelity levels):")
    for r in results[:10]:
        ps = "  ".join(
            f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
            for k, v in r['params'].items()
        )
        lines.append(
            f"  hp_{r['hp_idx']:04d}  val_loss={r['val_loss']:.6f}"
            f"  t_steps={r['t_steps']}  {ps}"
        )

    os.makedirs(eos_dir, exist_ok=True)
    with open(os.path.join(eos_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run one DyHPO trial on a compute node")
    parser.add_argument("--sweep-config",  required=True)
    parser.add_argument("--trial-idx",     type=int, required=True,
                        help="Submission-time index (for logging only)")
    parser.add_argument("--t-steps-cap",   type=int, default=None,
                        help="Cap the suggested t_steps to this value (for low-fidelity jobs)")
    parser.add_argument("--fixed-hp-idx",  type=int, default=None,
                        help="Skip DyHPO suggest and run this specific HP config")
    parser.add_argument("--fixed-t-steps", type=int, default=None,
                        help="Fixed t_steps for finalization jobs (requires --fixed-hp-idx)")
    # Detached mode (sweep/drive.py): the driver holds the DyHPO state and hands this job
    # its configuration; the job never touches a shared state file and reports back with
    # one RESULT_JSON line on stdout. That is what lets one sweep run on several sites.
    parser.add_argument("--detached", action="store_true",
                        help="No shared state: --hp-idx/--t-steps/--hp given, result printed as RESULT_JSON")
    parser.add_argument("--hp-idx", type=int, default=None)
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                        help="one hyper-parameter (repeatable); values parsed as YAML")
    args = parser.parse_args()
    if args.detached and (args.hp_idx is None or args.t_steps is None):
        parser.error("--detached needs --hp-idx and --t-steps")

    cfg        = load_config(args.sweep_config)
    sweep_name = cfg["sweep_name"]
    project_dir= cfg["paths"]["project_dir"]

    afs_dir, eos_dir = _sweep_dirs(cfg, sweep_name)
    results_dir= os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    state_path  = os.path.join(afs_dir, "dyhpo_state.pkl")
    ckpt_index_path = os.path.join(afs_dir, "checkpoint_index.json")

    print(f"[run_trial] Sweep: {sweep_name}  |  trial_idx: {args.trial_idx}")

    # ---------------------------------------------------------------
    # 1. Ask DyHPO for the next (hp_config, fidelity) to evaluate
    #    — or use fixed values for finalization jobs
    # ---------------------------------------------------------------
    eos_output_path = os.path.join(eos_dir, "dyhpo_surrogate")
    os.makedirs(eos_output_path, exist_ok=True)

    if args.detached:
        hp_idx, t_steps = args.hp_idx, args.t_steps
        hp_params = {}
        for kv in args.hp:
            k, _, v = kv.partition("=")
            hp_params[k] = yaml.safe_load(v)
        print(f"[run_trial] DETACHED hp_idx={hp_idx}  t_steps={t_steps}")
    elif args.fixed_hp_idx is not None:
        t_steps = args.fixed_t_steps
        with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
            hp_idx    = args.fixed_hp_idx
            hp_params = sampler.candidates_raw[hp_idx]
        print(f"[run_trial] FIXED hp_idx={hp_idx}  t_steps={t_steps}")
    else:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
                    hp_idx, hp_params, t_steps = sampler.suggest(max_t_steps=args.t_steps_cap)
                break
            except Exception as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"Failed to get DyHPO suggestion after {RETRY_ATTEMPTS} attempts: {e}") from e
                print(f"[run_trial] suggest() failed (attempt {attempt}): {e}", file=sys.stderr)
                time.sleep(RETRY_DELAY_S)

    print(f"[run_trial] hp_idx={hp_idx}  t_steps={t_steps}")
    print(f"[run_trial] HP params: {hp_params}")

    # Arm SIGTERM handler now that we know which trial to report as failed
    global _sigterm_hp_idx, _sigterm_state, _sigterm_eos
    _sigterm_hp_idx = hp_idx
    _sigterm_state  = None if args.detached else state_path      # detached: nothing shared to report to
    _sigterm_eos    = eos_output_path

    # ---------------------------------------------------------------
    # 2. Check checkpoint index — resume or cold start?
    # ---------------------------------------------------------------
    if args.detached:
        prev = None                                   # detached trials cold-start
    else:
        with CheckpointIndex(ckpt_index_path) as idx:
            prev = idx.lookup(hp_idx)

    run_dir = os.path.join(project_dir, "runs", sweep_name, f"trial_{hp_idx:04d}")

    can_warm_start = prev is not None and prev['t_steps'] < t_steps

    if can_warm_start:
        warm_start_idx  = prev['run_idx']
        prev_t_steps    = prev['t_steps']
        increment_steps = t_steps - prev_t_steps
        run_idx         = warm_start_idx + 1
        print(f"[run_trial] Resuming hp_{hp_idx:04d} from run_idx={warm_start_idx} "
              f"(t={prev_t_steps} → {t_steps}, +{increment_steps} steps)")
    else:
        warm_start_idx  = None
        increment_steps = t_steps
        run_idx         = 0
        print(f"[run_trial] Cold start hp_{hp_idx:04d}")
        # A cold start never reuses a run dir: an earlier attempt at this point (one that failed at start-up, or a
        # trial moved here by rebalance.py) leaves its dir, and run.py refuses an existing one, so the rerun would die
        # in seconds. The old dir is kept (nothing is deleted) and this attempt takes the first free _r<k> name.
        k = 1
        base_run_dir = run_dir
        while os.path.exists(run_dir):
            run_dir = f"{base_run_dir}_r{k}"
            k += 1
        if run_dir != base_run_dir:
            print(f"[run_trial] {base_run_dir} exists: this attempt runs in {run_dir}")

    # Use t_steps for unique result filename (encodes fidelity level without listing all dataset sizes)
    result_path = os.path.join(results_dir, f"hp{hp_idx:04d}_t{t_steps}_{int(time.time())}.json")

    # ---------------------------------------------------------------
    # 3. Run training
    # ---------------------------------------------------------------
    cmd = build_command(
        cfg, hp_params, run_dir, run_idx, result_path,
        t_steps,
        warm_start_idx=warm_start_idx,
        increment_steps=increment_steps,
    )
    print(f"[run_trial] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, cwd=project_dir)
        if proc.returncode != 0:
            raise RuntimeError(f"run.py exited with code {proc.returncode}")

        with open(result_path) as f:
            result = json.load(f)
        val_loss        = float(result["val_loss"])
        proc_val_losses = result.get("proc_val_losses")
        _drop_last_model(cfg, run_dir, run_idx, t_steps)

        # Use transfer-ratio geometric mean as HPO objective when scaling law
        # params are available; fall back to raw combined val_loss otherwise.
        hpo_obj = compute_hpo_objective(result, cfg.get("scaling_law", {}))
        observe_loss = hpo_obj if hpo_obj is not None else val_loss
        print(f"[run_trial] hp_{hp_idx:04d}  t_steps={t_steps}"
              f"  val_loss={val_loss:.6f}"
              + (f"  hpo_obj={hpo_obj:.4f}" if hpo_obj is not None else ""))

        # -----------------------------------------------------------
        # 4. Report result to DyHPO surrogate (detached: to the driver, via stdout)
        # -----------------------------------------------------------
        if args.detached:
            print("RESULT_JSON " + json.dumps({
                "hp_idx": hp_idx, "t_steps": t_steps, "trial_idx": args.trial_idx,
                "val_loss": val_loss, "observe_loss": float(observe_loss),
                "hpo_obj": hpo_obj, "proc_val_losses": proc_val_losses,
                "result_path": result_path, "run_dir": run_dir}), flush=True)
            return
        with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
            sampler.observe(hp_idx, t_steps, observe_loss, proc_val_losses)

            range_ext_cfg = cfg.get("range_extension", {})
            if range_ext_cfg.get("enabled", False):
                ext_kwargs = {k: v for k, v in range_ext_cfg.items() if k != "enabled"}
                extended = sampler.check_and_extend_ranges(**ext_kwargs)
                for name, info in extended.items():
                    print(
                        f"[run_trial] Range extended: {name} {info['direction']}"
                        f"  {info['old_bound']:.3g} -> {info['new_bound']:.3g}"
                        f"  (extension #{info['extension_count']})"
                    )

            _write_summary(cfg, sampler, eos_dir)

        # -----------------------------------------------------------
        # 5. Register new checkpoint in index
        # -----------------------------------------------------------
        with CheckpointIndex(ckpt_index_path) as idx:
            idx.register(hp_idx, run_dir, run_idx, t_steps, t_steps,
                         trial_idx=args.trial_idx)

    except Exception as e:
        print(f"[run_trial] Trial FAILED: {e}", file=sys.stderr)
        if args.detached:
            print("RESULT_JSON " + json.dumps({"hp_idx": hp_idx, "t_steps": t_steps,
                                              "trial_idx": args.trial_idx, "failed": True,
                                              "error": str(e)[:300]}), flush=True)
            sys.exit(1)
        if args.fixed_hp_idx is None:
            try:
                with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
                    # Impute a finite worst-case loss so the surrogate LEARNS this
                    # region is bad. report_failure() alone only drops this exact
                    # hp_idx from the candidate pool; it feeds the GP nothing, so
                    # the GP keeps proposing nearby points that diverge the same
                    # way (this is exactly how a feature arm that diverges in one
                    # HP corner can lose ~all its trials — see the scalar A/B arm,
                    # which crashed 14/15 on a CUDA index-OOB after diverging).
                    penalty = _failure_penalty(sampler, cfg)
                    if penalty is not None:
                        sampler.observe(hp_idx, t_steps, penalty, None)
                        print(f"[run_trial] imputed failure penalty "
                              f"val_loss={penalty:.4f} for hp_{hp_idx:04d}",
                              file=sys.stderr)
                    sampler.report_failure(hp_idx)
            except Exception as cleanup_err:
                print(f"[run_trial] report_failure: {cleanup_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
