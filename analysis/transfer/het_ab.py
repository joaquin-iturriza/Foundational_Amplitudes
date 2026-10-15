"""The user's heteroscedastic loss against the MSE baseline at the baseline's HPs (scripts/job_het_ab.sh; docs/results.tex,
the het correction paragraph): e+e- -> t tbar at D=10^4, 8k steps. Inputs, brought home by `site run` into
analysis/transfer/het_ab_data/ (tools/eval_delta.py outputs, test split, V100, TF32 a no-op):
  mse_preds_test.npz   the MSE baseline tp_scr_ee_ttbar_d8/trial_0144 (best of its 8-trial sweep)
  het_preds_test.npz   the het run, seed 2;  het_sigma_test.npz  its mu, truth and sigma (standardized log|M|^2)
Figures (analysis/transfer/figs/):
  het_ab_delta    |Delta| = |A_theta - A|/A, both models
  het_ab_pull     pulls (mu - truth)/sigma against N(0,1), linear and log
  het_ab_calib    the actual |mu - truth| against the predicted sigma, binned in sigma
    python analysis/transfer/het_ab.py
    python analysis/transfer/het_ab.py --sweep    the GP sweep's best trial (tp_het_ee_ttbar_d8/trial_1550, D57) in
        place of the one run: het_ab_data/sweep_eval/<sweep>__<trial>/{preds,sigma}_test.npz, figures het_ab_sweep_*
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "het_ab_data")
OUT = os.path.join(HERE, "figs")
LAB = r"$e^+e^-\to t\bar t$"


def delta(f):
    z = np.load(os.path.join(D, f))
    t, p = z["raw_truth"].reshape(-1).astype(float), z["raw_pred"].reshape(-1).astype(float)
    return (p - t) / t


SWEEP_BEST = os.path.join("sweep_eval", "tp_het_ee_ttbar_d8__trial_1550")


def main():
    os.makedirs(OUT, exist_ok=True)
    sw = "--sweep" in sys.argv
    hp, hs, pre = ((os.path.join(SWEEP_BEST, "preds_test.npz"), os.path.join(SWEEP_BEST, "sigma_test.npz"), "het_ab_sweep_")
                   if sw else ("het_preds_test.npz", "het_sigma_test.npz", "het_ab_"))
    dm, dh = delta("mse_preds_test.npz"), delta(hp)
    bins = np.logspace(-8, 0, 49)
    fig, ax = ps.figure()
    for d, c, lab in ((dm, "k", "MSE"), (dh, ps.C.vermillion, "heteroscedastic")):
        a = np.abs(d)
        h, e = np.histogram(a, bins=bins)
        ax.stairs(h / len(a) / np.diff(np.log10(e)), e, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel(r"$|\Delta| = |A_\theta - A|/A$"); ax.set_ylabel(r"events per unit $\log_{10}|\Delta|$")
    ps.process_label(ax, LAB); ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(OUT, pre + "delta"))

    z = np.load(os.path.join(D, hs))
    mu, tr, sg = (z[k].reshape(-1).astype(float) for k in ("pred", "truth", "sigma"))
    pull = (mu - tr) / sg
    x = np.linspace(-5, 5, 201); g = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, (a1, a2) = ps.figure(ncols=2)
    for ax, log in ((a1, False), (a2, True)):
        hh, ee = np.histogram(pull, bins=np.linspace(-5, 5, 61), density=True)
        ax.stairs(hh, ee, color=ps.C.vermillion, label=f"pulls, std {pull.std():.2f}")
        ax.plot(x, g, "--", color="k", label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel(r"$(\mu-\log|\mathcal{M}|^2)/\sigma$"); ax.set_ylabel("density")
        if log:
            ax.set_yscale("log")
        else:
            ax.plot([0, 0], [0, 1.25 * hh.max()], alpha=0)   # an invisible extent so the axis holds the peak
    ps.process_label(a2, LAB); ps.legend(a1, "upper left")
    ps.save(fig, os.path.join(OUT, pre + "pull"))

    r = np.abs(mu - tr)
    q = np.quantile(sg, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(sg, q[1:-1]), 0, 9)
    cs = np.array([np.median(sg[idx == k]) for k in range(10)])
    rms = np.array([np.sqrt(np.mean(r[idx == k] ** 2)) for k in range(10)])
    fig, ax = ps.figure()
    ax.plot(cs, rms, "o-", color=ps.C.vermillion, label=r"RMS error per $\sigma$ decile")
    lim = [min(cs.min(), rms.min()) / 2, max(cs.max(), rms.max()) * 2]
    ax.plot(lim, lim, "--", color="k", label="calibrated")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"predicted $\sigma$"); ax.set_ylabel("actual error")
    ps.process_label(ax, LAB, loc="lower right"); ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(OUT, pre + "calib"))
    print(f"median |Delta|: MSE {np.median(np.abs(dm)):.3g}, het {np.median(np.abs(dh)):.3g}; "
          f"test mu-MSE (std. units): het {np.mean((mu - tr) ** 2):.3g}; pull mean {pull.mean():.3f} std {pull.std():.3f}; "
          f"median sigma / RMS error {np.median(sg) / np.sqrt(np.mean(r ** 2)):.2f}")


if __name__ == "__main__":
    main()
