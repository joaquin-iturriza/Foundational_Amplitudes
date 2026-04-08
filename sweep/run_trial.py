#!/usr/bin/env python3
"""
run_trial.py  —  Per-job entrypoint for asynchronous Bayesian HPO.

Each HTCondor job runs this script. It:
  1. Connects to the shared Optuna study (via JournalFileStorage on EOS).
  2. Calls study.ask() to get the best hyperparameter suggestion given whatever
     trials have completed by the time this job actually starts.
  3. Runs `python run.py` with those parameters.
  4. Reports the validation loss back to the study via study.tell().

Failed trials (crash, OOM, missing result) are marked FAIL in the study so they
don't pollute the surrogate, but the study keeps running.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)

RETRY_ATTEMPTS = 3
RETRY_DELAY_S  = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def open_study_with_retry(cfg):
    sweep_name = cfg["sweep_name"]
    # Journal is on AFS (not EOS) — AFS has reliable POSIX locking.
    afs_dir      = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    journal_path = os.path.join(afs_dir, "optuna_journal.log")

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            storage = JournalStorage(JournalFileBackend(journal_path))
            study = optuna.load_study(
                study_name=sweep_name,
                storage=storage,
            )
            return study
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"Could not open Optuna study after {RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            print(
                f"[run_trial] Warning: failed to open study (attempt {attempt}/{RETRY_ATTEMPTS}): {e}",
                file=sys.stderr,
            )
            time.sleep(RETRY_DELAY_S)


def suggest_param(trial, entry):
    name = entry["name"]
    t    = entry["type"]
    if t == "float_log":
        return trial.suggest_float(name, entry["low"], entry["high"], log=True)
    elif t == "float_uniform":
        return trial.suggest_float(name, entry["low"], entry["high"], log=False)
    elif t == "int_log":
        return trial.suggest_int(name, entry["low"], entry["high"], log=True)
    elif t == "int_uniform":
        return trial.suggest_int(name, entry["low"], entry["high"], log=False)
    elif t == "categorical":
        return trial.suggest_categorical(name, entry["choices"])
    else:
        raise ValueError(f"Unknown search space type: {t!r}")


def format_value(v):
    """Format a suggested value for Hydra command-line override."""
    if isinstance(v, float):
        # Scientific notation avoids floating-point display issues
        return f"{v:.6e}"
    return str(v)


def build_command(cfg, trial, result_path):
    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = cfg["sweep_name"]

    cmd = [sys.executable, run_script]

    # config/amplitudes.yaml has `local: none` in its defaults, which looks for
    # config/local/none.yaml — that file doesn't exist on lxplus. Override it here.
    cmd.append("local=none")

    # Fixed params
    for key, val in cfg.get("fixed_params", {}).items():
        cmd.append(f"{key}={val}")

    # Suggested params from Optuna
    for key, val in trial.params.items():
        cmd.append(f"{key}={format_value(val)}")

    # base_experiment builds: base_dir/runs/exp_name/run_name
    # With base_dir=project_dir and exp_name=sweep_name/trial_N:
    # → project_dir/runs/sweep_name/trial_N/run_name  ✓
    cmd.append(f"base_dir={project_dir}")
    exp_name = f"{sweep_name}/trial_{trial.number:04d}"
    cmd.append(f"exp_name={exp_name}")

    # Result path for handoff back to this script
    cmd.append(f"training.result_path={result_path}")

    return cmd


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def _write_summary(study, eos_dir):
    """Write a human-readable summary of the sweep so far to {eos_dir}/summary.txt.

    Updated after every completed trial. Check it with:
        cat /eos/user/j/joiturri/sweeps/{sweep_name}/summary.txt
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    total     = len(study.trials)

    lines = [
        f"Sweep: {study.study_name}",
        f"Trials: {len(completed)} completed, {len(failed)} failed, {total} total",
    ]

    if completed:
        best = study.best_trial
        lines += [
            f"Best trial: #{best.number}  val_loss = {best.value:.6f}",
            "Best params:",
        ]
        for k, v in best.params.items():
            lines.append(f"  {k} = {v}")

        lines.append("")
        lines.append(f"Top-10 (of {len(completed)} completed):")
        sorted_trials = sorted(completed, key=lambda t: t.value)
        for t in sorted_trials[:10]:
            param_str = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in t.params.items())
            lines.append(f"  #{t.number:04d}  val_loss={t.value:.6f}  {param_str}")

    summary_path = os.path.join(eos_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run one HPO trial on a compute node")
    parser.add_argument("--sweep-config", required=True, help="Path to sweep_config.yaml")
    parser.add_argument("--trial-idx", type=int, required=True,
                        help="Submission-time index (used for logging only; actual params come from Optuna)")
    args = parser.parse_args()

    cfg        = load_config(args.sweep_config)
    sweep_name = cfg["sweep_name"]
    eos_dir    = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    results_dir = os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[run_trial] Sweep: {sweep_name}  |  submission index: {args.trial_idx}")

    # --- Connect to study and get next trial ---
    study = open_study_with_retry(cfg)
    trial = study.ask()

    print(f"[run_trial] Optuna trial number: {trial.number}")

    # --- Suggest parameters ---
    for entry in cfg.get("search_space", []):
        suggest_param(trial, entry)

    print(f"[run_trial] Parameters: {trial.params}")

    # --- Result file path (unique per trial + timestamp to avoid any collision) ---
    result_path = os.path.join(results_dir, f"trial_{trial.number:04d}_{int(time.time())}.json")

    # --- Run training ---
    cmd = build_command(cfg, trial, result_path)
    project_dir = cfg["paths"]["project_dir"]

    print(f"[run_trial] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, cwd=project_dir)

        if proc.returncode != 0:
            raise RuntimeError(f"run.py exited with code {proc.returncode}")

        # --- Read result ---
        with open(result_path) as f:
            result = json.load(f)
        val_loss = float(result["val_loss"])

        print(f"[run_trial] Trial {trial.number}: val_loss = {val_loss:.6f}")
        study.tell(trial, val_loss)

        # Write a running summary so you can check progress without running analyze_sweep.py
        _write_summary(study, eos_dir)

    except Exception as e:
        print(f"[run_trial] Trial {trial.number} FAILED: {e}", file=sys.stderr)
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
