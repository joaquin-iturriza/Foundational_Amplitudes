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
    px, py = [], []
    for i in range(10):
        m = (s >= qs[i]) & (s <= qs[i + 1]) if i == 9 else (s >= qs[i]) & (s < qs[i + 1])
        if m.sum() >= 10:
            px.append(s[m].mean()); py.append(np.sqrt(np.mean(r[m] ** 2)))
    slope = np.polyfit(np.log10(px), np.log10(py), 1)[0]
    print(f"\n  >>> RELIABILITY SLOPE = {slope:.2f}   (1.00 = calibrated; >1 = sigma under-dispersed)")
    return r, s, label, beta, float(np.mean(r ** 2))


def make_plot(runs, out_base):
    """Calibration figure: (a) pull vs N(0,1), log-y; (b) reliability, log-log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["C0", "C3"]

    for i, (r, s, label, beta, mse) in enumerate(runs):
        pull = r / np.clip(s, 1e-15, None)
        c = colors[i % len(colors)]
        lab = f"{label}  ($\\beta$={beta:.3f}, $\\mu$-MSE={mse:.2g})"

        # (a) pull distribution vs the unit Gaussian it should be
        bins = np.linspace(-5, 5, 81)
        axs[0].hist(pull, bins=bins, density=True, histtype="step", lw=1.6, color=c,
                    label=f"{lab}\n  std={np.std(pull):.3f}")
        # (b) reliability: predicted sigma vs ACTUAL rms residual, per sigma-decile
        qs = np.quantile(s, np.linspace(0, 1, 11))
        px, py = [], []
        for k in range(10):
            m = (s >= qs[k]) & (s <= qs[k + 1]) if k == 9 else (s >= qs[k]) & (s < qs[k + 1])
            if m.sum() < 10:
                continue
            px.append(s[m].mean())
            py.append(np.sqrt(np.mean(r[m] ** 2)))
        axs[1].plot(px, py, "o-", color=c, lw=1.5, ms=5, label=lab)

    x = np.linspace(-5, 5, 400)
    axs[0].plot(x, norm.pdf(x), "k--", lw=1.4, label="$\\mathcal{N}(0,1)$ (calibrated)")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"pull $=(y-\mu)/\sigma$")
    axs[0].set_ylabel("density")
    axs[0].set_title("Pull distribution")
    axs[0].legend(fontsize=7, frameon=False)
    axs[0].grid(True, which="both", lw=0.4, alpha=0.4)

    lo = min(min(a.get_xdata().min() for a in axs[1].lines),
             min(a.get_ydata().min() for a in axs[1].lines))
    hi = max(max(a.get_xdata().max() for a in axs[1].lines),
             max(a.get_ydata().max() for a in axs[1].lines))
    axs[1].plot([lo, hi], [lo, hi], "k--", lw=1.4, label="perfect calibration")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel(r"predicted $\sigma$  (decile mean)")
    axs[1].set_ylabel("actual RMS residual")
    axs[1].set_title(r"Reliability: does $\sigma$ track the true error?")
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].grid(True, which="both", lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_base}.png and {out_base}.pdf")


def main():
    runs = [
        (os.path.join(WT, "runs/heterosc_scratch_hpo2/trial_0016"), "best het (refined sweep)"),
        (os.path.join(WT, "runs/heterosc_scratch_hpo2/trial_0021"), "2nd het (refined sweep)"),
    ]
    collected = []
    for rd, label in runs:
        if not os.path.exists(os.path.join(rd, "config.yaml")):
            print(f"missing {rd}; skip")
            continue
        try:
            collected.append(report(rd, label))
        except Exception:
            import traceback
            traceback.print_exc()
    if collected:
        figdir = os.path.join(WT, "analysis/divergences/figs")
        os.makedirs(figdir, exist_ok=True)
        make_plot(collected, os.path.join(figdir, "heterosc_calibration"))


if __name__ == "__main__":
    main()
