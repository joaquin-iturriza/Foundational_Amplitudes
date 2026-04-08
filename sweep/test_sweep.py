#!/usr/bin/env python3
"""
test_sweep.py  —  Run a sweep locally (no HTCondor, no GPU) to verify the pipeline.

Runs N trials sequentially in the current process, using a tiny training budget
so each trial finishes in seconds. Tests the full chain:
  journal init → ask() → run.py → result JSON → tell()

Usage (on lxplus login node, inside ~/Foundational_Amplitudes/):
    python test_sweep.py --config sweep_config.yaml
    python test_sweep.py --config sweep_config.yaml --n-trials 5
    python test_sweep.py --config sweep_config.yaml --clean
"""

import argparse
import os
import subprocess
import sys

import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Overrides applied on top of fixed_params during testing.
# These make each trial finish in ~seconds on CPU.
TEST_OVERRIDES = {
    "training.iterations": 10,
    "training.batchsize":  16,
    "training.validate_every_n_steps": 5,
    "training.save_intermediate": "false",
    "training.get_ID": "false",
    "plot": "false",
    # Use only a tiny subsample so data loading is fast
    "data.subsample": "[500,500,500]",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_sweep_name(cfg):
    # Use a separate name so test runs don't pollute the real study's journal.
    return cfg["sweep_name"] + "_localtest"


def journal_path(cfg):
    name = test_sweep_name(cfg)
    afs_dir = os.path.join(cfg["paths"]["afs_sweep_dir"], name)
    os.makedirs(afs_dir, exist_ok=True)
    return os.path.join(afs_dir, "optuna_journal.log")


def open_study(cfg, jpath):
    storage = JournalStorage(JournalFileBackend(jpath))
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=cfg["optuna"]["n_startup_trials"],
        seed=42,
    )
    return optuna.create_study(
        study_name=test_sweep_name(cfg),
        storage=storage,
        direction=cfg["optuna"]["direction"],
        sampler=sampler,
        load_if_exists=True,
    )


def suggest_param(trial, entry):
    name, t = entry["name"], entry["type"]
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
        raise ValueError(f"Unknown type: {t!r}")


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def run_trial(cfg, study, trial_idx):
    import json, time

    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = test_sweep_name(cfg)  # isolated from the real study
    eos_dir     = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    results_dir = os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    trial = study.ask()
    print(f"\n[test] Trial {trial.number} (submission index {trial_idx})")

    for entry in cfg.get("search_space", []):
        suggest_param(trial, entry)

    print(f"[test]   params: {trial.params}")

    result_path = os.path.join(results_dir, f"trial_{trial.number:04d}_{int(time.time())}.json")

    cmd = [sys.executable, run_script, "local=none"]

    # Fixed params, overridden by TEST_OVERRIDES
    merged = dict(cfg.get("fixed_params", {}))
    merged.update(TEST_OVERRIDES)
    for key, val in merged.items():
        cmd.append(f"{key}={val}")

    # Suggested params
    for key, val in trial.params.items():
        cmd.append(f"{key}={format_value(val)}")

    cmd.append(f"exp_name={sweep_name}_test_{trial.number:04d}")
    cmd.append(f"training.result_path={result_path}")

    print(f"[test]   running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, cwd=project_dir)
        if proc.returncode != 0:
            raise RuntimeError(f"run.py exited with code {proc.returncode}")

        with open(result_path) as f:
            result = json.load(f)
        val_loss = float(result["val_loss"])

        print(f"[test]   val_loss = {val_loss:.6f}  ✓")
        study.tell(trial, val_loss)
        return True

    except Exception as e:
        print(f"[test]   FAILED: {e}", file=sys.stderr)
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return False


def main():
    parser = argparse.ArgumentParser(description="Local pipeline test (no HTCondor, no GPU)")
    parser.add_argument("--config",   required=True, help="Path to sweep_config.yaml")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials to run (default: 3)")
    parser.add_argument("--clean",    action="store_true",
                        help="Delete the journal before starting (fresh study)")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    jpath = journal_path(cfg)

    if args.clean and os.path.exists(jpath):
        os.remove(jpath)
        print(f"[test] Deleted existing journal: {jpath}")

    print(f"[test] Sweep: {cfg['sweep_name']}")
    print(f"[test] Journal: {jpath}")
    print(f"[test] Running {args.n_trials} trial(s) on CPU with overrides: {TEST_OVERRIDES}\n")

    study = open_study(cfg, jpath)

    passed, failed = 0, 0
    for i in range(args.n_trials):
        ok = run_trial(cfg, study, trial_idx=i)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n[test] Done: {passed} passed, {failed} failed out of {args.n_trials} trials.")

    if passed > 0:
        best = study.best_trial
        print(f"[test] Best so far: trial {best.number}, val_loss = {best.value:.6f}")
        print(f"[test]   params: {best.params}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
