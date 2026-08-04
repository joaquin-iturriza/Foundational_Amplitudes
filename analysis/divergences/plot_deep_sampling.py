#!/usr/bin/env python
"""Uniform vs antenna phase-space sampling for the ee->uug deep-IR singularity.

Both fine-tunes start from the same leave-uug-out base22, train on the SAME number of
events (400k) over the SAME (lowered-cut) fiducial region, evaluated on the SAME
held-out deep-IR test set. The ONLY difference is the sampling density:
  uniform : flat RAMBO (deep IR starved — ~flat phase-space measure)
  antenna : dN ∝ 1/(y_ug·y_ubarg) (every IR decade densely filled down to y_min~1e-6)

Left  : MSE Δlog|M|^2 per y_min decade, uniform vs antenna (the extrapolation result —
        does denser IR sampling capture the pole better?).
Right : training events per y_min decade for each sampler (the mechanism — antenna
        moves training mass from the O(1) bulk into the singular decades).
Metric is MSE on de-standardized log|M|^2 (each run's own preprocessing inverted at
eval), so the two samplings compare on the same absolute scale. CPU only.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
import plot_style as ps  # noqa: E402
from extract_ir import ir_observables  # noqa

EDGES = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])          # geometric bin centers
COL = {"uniform": ps.C.blue, "antenna": ps.C.vermillion, "mixture": ps.C.green}
LAB = {"uniform": "uniform", "antenna": r"antenna, $\mathrm{d}N\propto 1/y_{\min}$",
       "mixture": "50/50 mixture"}


def binned_metrics(y, resid):
    """Per y_min decade: MSE and MAE of Δln|M|^2 (log-space L2/L1 = fractional error),
    and median relative error |M2_pred/M2_true - 1| = |exp(Δln)-1| (robust, in %)."""
    mse, mae, medrel, cnt = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (y >= lo) & (y < hi)
        cnt.append(int(m.sum()))
        if m.sum() > 20:
            r = resid[m]
            mse.append(float(np.mean(r ** 2)))
            mae.append(float(np.mean(np.abs(r))))
            medrel.append(float(np.median(np.abs(np.exp(r) - 1.0))))
        else:
            mse.append(np.nan); mae.append(np.nan); medrel.append(np.nan)
    return np.array(mse), np.array(mae), np.array(medrel), np.array(cnt)


def train_coverage(npy_path):
    a = np.load(npy_path, mmap_mode="r")
    P = (a.shape[1] - 1) // 5
    pdg = np.asarray(a[0, P * 4:P * 5]).astype(int)
    mom = np.asarray(a[:, :P * 4], dtype=np.float64).reshape(-1, P, 4)
    y = ir_observables(mom, pdg)["y_min"]
    return np.array([int(((y >= lo) & (y < hi)).sum())
                     for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--npz_prefix", default="deep_eval_")
    ap.add_argument("--data_root", default=REPO)
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "deep_sampling_uniform_vs_antenna"))
    ap.add_argument("--summary_out", default=os.path.join(here, "deep_sampling_summary.json"))
    ap.add_argument("--modes", default="uniform,antenna", help="comma list of sampler tags")
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    res = {}
    for mode in modes:
        d = np.load(os.path.join(args.eval_dir, f"{args.npz_prefix}{mode}.npz"))
        resid = d["pred_logamp"] - d["true_logamp"]
        y = d["y_min"]
        mse, mae, medrel, cnt = binned_metrics(y, resid)
        cov = train_coverage(os.path.join(args.data_root, f"data_deep_{mode}",
                                          "ee_uug_91-1000GeV_amplitudes.npy"))
        res[mode] = dict(mse=mse, mae=mae, medrel=medrel, cnt=cnt, cov=cov,
                         mse_all=float(np.mean(resid ** 2)),
                         mae_all=float(np.mean(np.abs(resid))),
                         medrel_all=float(np.median(np.abs(np.exp(resid) - 1.0))),
                         n=int(resid.size))

    # 2x2, not 1x3. Three panels across \textwidth left each one 1.36x1.09in -- the smallest
    # in the document, barely over the legibility floor, and roughly half the panel every
    # THREE separate panel files, not a 2x2 with the fourth cell blanked out: the hole where
    # the fourth panel would be is the first thing anyone sees. results.tex packs them two per
    # line, the third centred underneath.
    figs = ps.panels(3)
    ax1, ax2, ax3 = (f[1] for f in figs)

    # Panel 1: per-decade MSE of Δln|M|^2 (log-space L2 = squared fractional error)
    for mode in modes:
        ax1.plot(CEN, res[mode]["mse"], "o-", color=COL[mode], label=LAB[mode])
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"$y_{\min}$")
    ax1.set_ylabel(r"MSE$(\Delta\ln|\mathcal{M}|^2)$")
    ps.process_label(ax1, args.process, loc="upper right")

    # Panel 2: per-decade median relative error |M2_pred/M2_true - 1| (robust, in %)
    for mode in modes:
        ax2.plot(CEN, 100 * res[mode]["medrel"], "o-", color=COL[mode], label=LAB[mode])
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"$y_{\min}$")
    ax2.set_ylabel("median rel. error [%]")

    # Panel 3: training coverage per decade (grouped bars, centered per decade)
    nb = len(modes)
    w = 0.8 / nb
    xpos = np.arange(len(CEN))
    for i, mode in enumerate(modes):
        off = (i - (nb - 1) / 2.0) * w
        ax3.bar(xpos + off, np.maximum(res[mode]["cov"], 0.5), width=w,
                color=COL[mode], label=LAB[mode])
    ax3.set_yscale("log")
    ax3.set_xticks(xpos)
    ax3.set_xticklabels([fr"${int(np.log10(lo))}$" for lo in EDGES[:-1]])
    ax3.set_xlabel(r"$\log_{10} y_{\min}$")
    ax3.set_ylabel("training events")

    # Inside the first panel, not a strip over the figure: the arms separate downward to the
    # right, so the lower left is clear.
    ps.legend(ax1, "lower left")
    ps.save_panels(figs, args.out_base)
    print(f"wrote {args.out_base}.png/.pdf")

    def cell(v):
        return None if np.isnan(v) else float(v)
    summ = {}
    for mode in modes:
        r = res[mode]
        summ[mode] = dict(mse_all=r["mse_all"], mae_all=r["mae_all"],
                          medrel_all=r["medrel_all"], n=r["n"],
                          per_decade=[dict(lo=float(EDGES[i]), hi=float(EDGES[i + 1]),
                                           n_test=int(r["cnt"][i]), n_train=int(r["cov"][i]),
                                           mse_log=cell(r["mse"][i]), mae_log=cell(r["mae"][i]),
                                           median_rel_err=cell(r["medrel"][i]))
                                      for i in range(len(CEN))])
    with open(args.summary_out, "w") as f:
        json.dump(summ, f, indent=1)
    print(f"wrote {args.summary_out}")

    # per-decade table: median relative error (%) for each mode — the readable metric
    print("\nmedian relative error |M2_pred/M2_true - 1|  [%]  per y_min decade")
    print("  " + "decade".ljust(16) + "".join(f"{m[:9]:>11}" for m in modes))
    for i in range(len(CEN)):
        cells = "".join(f"{100*res[m]['medrel'][i]:>10.2f}%" for m in modes)
        print(f"  [{EDGES[i]:.0e},{EDGES[i+1]:.0e}) {cells}")
    print("\noverall (MSE-log / MAE-log / median-rel-err%):")
    for m in modes:
        print(f"  {m:>9}: {res[m]['mse_all']:.4g} / {res[m]['mae_all']:.4g} / "
              f"{100*res[m]['medrel_all']:.2f}%")


if __name__ == "__main__":
    main()
