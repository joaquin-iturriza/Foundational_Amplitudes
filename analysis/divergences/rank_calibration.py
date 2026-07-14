#!/usr/bin/env python
"""Re-rank every HETEROSC sweep trial by CALIBRATION, not by the sweep's own objective.

Why: DyHPO optimised val NLL (average log-likelihood). That rewards neither the accuracy we
report (mu-MSE) nor the thing that actually matters here (a reliability slope of 1). So the
sweep's "best" trial is best at neither, and a well-calibrated config may have been passed over
because its NLL was worse. This re-ranks the trials we ALREADY trained -- no new training.

Per trial, on its own val set (preprocessed / log-amplitude space):
  mu-MSE                 accuracy
  reliability slope      fit of log10(actual RMS residual) vs log10(predicted sigma) over
                         sigma-deciles. 1.00 = sigma tracks the true error; >1 = sigma
                         UNDER-dispersed (compresses the error range); <1 = over-dispersed.
  pull std / mean        1.0 / 0.0 if calibrated
  tail ratio             actual/predicted in the TOP sigma decile (the hard events; 1.0 = good)
Ranked by |slope - 1|. GPU.
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from experiment import AmplitudeExperiment  # noqa
from extract_preds import load_finetuned_state  # noqa


def metrics(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_rank_calib_tmp")
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        cfg.model.net.loss = cfg.training.loss
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(
        os.path.join(run_dir, "models", "model_run0_best.pt"))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    with torch.no_grad():
        pred, truth, sig = exp._collect_predictions(exp.val_loader)

    mu = np.asarray(pred, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    s = np.asarray(sig, np.float64).reshape(-1)
    r = y - mu
    pull = r / np.clip(s, 1e-15, None)

    qs = np.quantile(s, np.linspace(0, 1, 11))
    px, py = [], []
    for i in range(10):
        m = (s >= qs[i]) & (s <= qs[i + 1]) if i == 9 else (s >= qs[i]) & (s < qs[i + 1])
        if m.sum() >= 10:
            px.append(s[m].mean()); py.append(np.sqrt(np.mean(r[m] ** 2)))
    slope = float(np.polyfit(np.log10(px), np.log10(py), 1)[0]) if len(px) > 2 else float("nan")
    return dict(
        mse=float(np.mean(r ** 2)),
        slope=slope,
        pull_std=float(np.std(pull)),
        pull_mean=float(np.mean(pull)),
        tail_ratio=float(py[-1] / px[-1]) if px else float("nan"),
        lr=float(cfg.training.lr),
        beta=float(cfg.training.get("heterosc_beta", 0.0) or 0.0),
        lam=float(cfg.training.regularization_lambda),
    )


def main():
    out = []
    for sweep, trials in (("heterosc_datactrl", ["flat_het"]),):
        root = os.path.join(WT, "runs", sweep)
        if not os.path.isdir(root):
            continue
        for t in trials:
            rd = os.path.join(root, t)
            if not os.path.exists(os.path.join(rd, "config.yaml")):
                continue
            try:
                m = metrics(rd); m["trial"] = f"{sweep.replace('heterosc_scratch_','')}/{t}"
                out.append(m)
                print(f"  {m['trial']:22s} slope={m['slope']:5.2f}  mu-MSE={m['mse']:.3g}  "
                      f"pull_std={m['pull_std']:.3f}  tail={m['tail_ratio']:.2f}  "
                      f"(lr={m['lr']:.1e}, b={m['beta']:.3f})", flush=True)
            except Exception as e:
                print(f"  {t}: FAILED {e}", flush=True)

    out.sort(key=lambda d: abs(d["slope"] - 1.0))
    print("\n=== ranked by CALIBRATION |slope - 1| (best first) ===")
    print(f"  {'trial':22s} {'slope':>6s} {'mu-MSE':>10s} {'pull_std':>9s} {'tail':>6s} {'lr':>9s} {'beta':>7s}")
    for d in out:
        print(f"  {d['trial']:22s} {d['slope']:6.2f} {d['mse']:10.3g} {d['pull_std']:9.3f} "
              f"{d['tail_ratio']:6.2f} {d['lr']:9.1e} {d['beta']:7.3f}")
    print(f"\n  (MSE baseline accuracy, swept: 4.72e-5 -- het has no config beating that)")
    with open(os.path.join(WT, "analysis/divergences/calib_rank_datactrl.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
