#!/usr/bin/env python
"""Calibration of the heteroscedastic sigma on the best from-scratch HPO trial.

The question that actually matters: are the predicted uncertainties well calibrated?
Accuracy alone is not the point of the het head.

Reports, on the run's own val set (preprocessed / log-amplitude space):
  * mu-MSE                          -- accuracy, vs the MSE reference (9.19e-5)
  * pull = (y - mu)/sigma           -- should be ~N(0,1) if calibrated
      - std(pull): 1.0 = perfect; >1 = OVER-confident (sigma too small);
                                   <1 = UNDER-confident (sigma too large)
  * coverage  P(|pull|<1,2,3)       -- expect 0.683 / 0.954 / 0.997
  * RELIABILITY (the real test): bin events by predicted sigma, and in each bin compare
    mean predicted sigma against the ACTUAL RMS residual. A calibrated sigma tracks the
    diagonal across the whole range -- i.e. sigma predicts the error it will make, not
    just on average but locally. Global std(pull)~1 can hide a flat/useless sigma, so the
    per-bin table is what decides whether sigma is usable as a signal.
GPU.
"""
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from diag_ckpt_mu import eval_ckpt  # noqa  (builds exp, loads ckpt, returns preds+sigmas)


def report(run_dir, label):
    import torch
    from omegaconf import OmegaConf, open_dict
    from experiment import AmplitudeExperiment
    from extract_preds import load_finetuned_state

    REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_calib_tmp")
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        cfg.model.net.loss = cfg.training.loss
    beta = float(cfg.training.get("heterosc_beta", 0.0) or 0.0)
    lr = float(cfg.training.lr)
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(
        os.path.join(run_dir, "models", "model_run0_best.pt"))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    with torch.no_grad():
        pred, truth, sig = exp._collect_predictions(exp.val_loader)

    mu = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    s = np.asarray(sig, dtype=np.float64).reshape(-1)
    r = y - mu
    pull = r / np.clip(s, 1e-15, None)

    print(f"\n================ {label} ================")
    print(f"  (lr={lr:.2e}, beta={beta:.3f}, N={len(r)})")
    print(f"  mu-MSE            = {np.mean(r**2):.4g}      [MSE reference: 9.19e-5]")
    print(f"  residual RMS      = {np.sqrt(np.mean(r**2)):.4g}")
    print(f"  sigma  median     = {np.median(s):.4g}   mean = {np.mean(s):.4g}")
    print()
    print(f"  pull mean         = {np.mean(pull):+.4f}   (0 = unbiased)")
    print(f"  pull std          = {np.std(pull):.4f}    <-- 1.0 = CALIBRATED "
          f"({'over-confident' if np.std(pull) > 1.15 else 'under-confident' if np.std(pull) < 0.85 else 'OK'})")
    for k, exp_cov in ((1, 0.683), (2, 0.954), (3, 0.997)):
        cov = float(np.mean(np.abs(pull) < k))
        print(f"  coverage |pull|<{k}  = {cov:.3f}   (expect {exp_cov:.3f})")

    # Reliability: does sigma track the ACTUAL error, locally?
    print("\n  RELIABILITY (bin by predicted sigma; calibrated => pred ~ actual):")
    print(f"    {'sigma bin (decile)':22s} {'N':>7s} {'mean pred sigma':>16s} {'actual RMS resid':>18s} {'ratio':>7s}")
    qs = np.quantile(s, np.linspace(0, 1, 11))
    for i in range(10):
        m = (s >= qs[i]) & (s < qs[i + 1] if i < 9 else s <= qs[i + 1])
        if m.sum() < 10:
            continue
        pred_s = s[m].mean()
        act = np.sqrt(np.mean(r[m] ** 2))
        print(f"    [{qs[i]:.4g}, {qs[i+1]:.4g}){'':4s} {m.sum():7d} {pred_s:16.4g} {act:18.4g} "
              f"{act/pred_s:7.2f}")


def main():
    runs = [
        (os.path.join(WT, "runs/heterosc_scratch_hpo/trial_0097"), "BEST het (beta~0.025, plain NLL)"),
        (os.path.join(WT, "runs/heterosc_scratch_hpo/trial_0073"), "2nd  het (beta~0.108)"),
    ]
    for rd, label in runs:
        if not os.path.exists(os.path.join(rd, "config.yaml")):
            print(f"missing {rd}; skip")
            continue
        try:
            report(rd, label)
        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
