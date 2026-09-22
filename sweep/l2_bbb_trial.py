#!/usr/bin/env python
"""Per-trial runner for a DyHPO sweep over the L2 online-generation BBB hyperparameters.

Mirrors sweep/run_trial.py's suggest -> run -> observe loop (shared DyHPO state via a file lock), but
the trial is a FULL L2 online-generation BBB run (analysis/divergences/l2_online_uugg.py) instead of
run.py, and the objective is the held-out deep-IR MSE that run writes to --result_path. Single-fidelity
only (fidelity_schedule.t_steps has one value = total_steps): no warm-start / checkpoint index.

Search space (search_space in the sweep config) is over the BBB knobs -- bbb_beta, bbb_sigma_rel, gamma;
fixed_params carries the rest (process, arm=bbb, n_total, rounds, oversample, bbb_ksamples). Each trial
gets a unique --tag (sw<hp_idx>) so run dirs / round-0 pools do not collide.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
from sweep.dyhpo_sampler import DyHPOSampler          # noqa: E402


def _sweep_dirs(cfg, sweep_name):
    d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
    return d, d


def _failure_penalty(sampler, cfg):
    margin = cfg.get("dyhpo", {}).get("failure_margin", 2.0)
    observed = [v for d in sampler._val_loss_history.values() for v in d.values()
                if v is not None and v < 1e30]
    return max(observed) * margin if observed else cfg.get("dyhpo", {}).get("failure_fallback", 1.0)


def build_l2_cmd(cfg, hp_params, t_steps, tag, result_path):
    driver = cfg.get("l2_driver",
                     os.path.join(cfg["paths"]["project_dir"], "analysis/divergences/l2_online_uugg.py"))
    fp = cfg.get("fixed_params", {})
    cmd = [sys.executable, driver,
           "--process", str(fp.get("process", "uugg")),
           "--arm", str(fp.get("arm", "bbb")),
           "--tag", tag, "--seed", str(fp.get("seed", 0)),
           "--total_steps", str(t_steps),
           "--n_total", str(fp.get("n_total", 300000)),
           "--rounds", str(fp.get("rounds", 10)),
           "--oversample", str(fp.get("oversample", 4)),
           "--bbb_ksamples", str(fp.get("bbb_ksamples", 8)),
           "--y_lo", str(fp.get("y_lo", 1e-6)),
           "--mix_ir", str(fp.get("mix_ir", 0.5)),
           "--objective", str(fp.get("objective", "deep")),
           "--heldout_eval", "--result_path", result_path]
    # HPs the sweep varies (bare names -> --<name> <val>)
    for key, val in hp_params.items():
        cmd += [f"--{key}", str(val)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--trial-idx", type=int, required=True)
    ap.add_argument("--t-steps-cap", type=int, default=None)  # accepted for generate_sweep compatibility
    args = ap.parse_args()

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)
    sweep_name = cfg["sweep_name"]
    sweep_dir, eos_dir = _sweep_dirs(cfg, sweep_name)
    state_path = os.path.join(sweep_dir, "dyhpo_state.pkl")
    surrogate_out = os.path.join(eos_dir, "dyhpo_surrogate")
    results_dir = os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. suggest
    with DyHPOSampler.locked(state_path, surrogate_out) as sampler:
        hp_idx, hp_params, t_steps = sampler.suggest(max_t_steps=args.t_steps_cap)
    tag = f"sw{hp_idx:04d}"
    result_path = os.path.join(results_dir, f"hp_{hp_idx:04d}_t{t_steps}.json")
    print(f"[l2_bbb_trial] trial {args.trial_idx} hp_{hp_idx:04d} t={t_steps} params={hp_params}",
          flush=True)

    # 2. run the full L2 BBB run
    try:
        cmd = build_l2_cmd(cfg, hp_params, t_steps, tag, result_path)
        print(f"[l2_bbb_trial] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=cfg["paths"]["project_dir"])
        if proc.returncode != 0:
            raise RuntimeError(f"L2 driver exited {proc.returncode}")
        with open(result_path) as f:
            result = json.load(f)
        val_loss = float(result["val_loss"])
        print(f"[l2_bbb_trial] hp_{hp_idx:04d} val_loss(deep_mse)={val_loss:.6e} "
              f"overall={result.get('overall_mse'):.6e}", flush=True)
        # 3. observe
        with DyHPOSampler.locked(state_path, surrogate_out) as sampler:
            sampler.observe(hp_idx, t_steps, val_loss, None)
    except Exception as e:
        print(f"[l2_bbb_trial] Trial FAILED: {e}", file=sys.stderr)
        try:
            with DyHPOSampler.locked(state_path, surrogate_out) as sampler:
                penalty = _failure_penalty(sampler, cfg)
                sampler.observe(hp_idx, t_steps, penalty, None)
                sampler.report_failure(hp_idx)
                print(f"[l2_bbb_trial] imputed failure penalty {penalty:.4f} for hp_{hp_idx:04d}",
                      file=sys.stderr)
        except Exception as ce:
            print(f"[l2_bbb_trial] report_failure error: {ce}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
