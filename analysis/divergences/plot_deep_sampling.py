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
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_ir import ir_observables  # noqa

EDGES = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])          # geometric bin centers
COL = {"uniform": "#4C72B0", "antenna": "#C44E52"}
LAB = {"uniform": "uniform (flat RAMBO)", "antenna": r"antenna $\propto 1/y_{\min}$"}


def binned_mse(y, resid):
    """MSE of resid per y_min decade + counts."""
    mse, cnt = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (y >= lo) & (y < hi)
        cnt.append(int(m.sum()))
        mse.append(float(np.mean(resid[m] ** 2)) if m.sum() > 20 else np.nan)
    return np.array(mse), np.array(cnt)


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
    args = ap.parse_args()

    modes = ["uniform", "antenna"]
    res = {}
    for mode in modes:
        d = np.load(os.path.join(args.eval_dir, f"{args.npz_prefix}{mode}.npz"))
        resid = d["pred_logamp"] - d["true_logamp"]
        y = d["y_min"]
        mse, cnt = binned_mse(y, resid)
        cov = train_coverage(os.path.join(args.data_root, f"data_deep_{mode}",
                                          "ee_uug_91-1000GeV_amplitudes.npy"))
        res[mode] = dict(mse=mse, cnt=cnt, cov=cov,
                         mse_all=float(np.mean(resid ** 2)),
                         mae_all=float(np.mean(np.abs(resid))), n=int(resid.size))

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.suptitle(r"$e^+e^-\to u\bar u g$ deep-IR sampling A/B  (same base22, same 400k events, "
                 r"same held-out test — only sampling density differs)", fontsize=12)

    # Left: per-decade MSE
    for mode in modes:
        axL.plot(CEN, res[mode]["mse"], "o-", color=COL[mode], lw=1.9, ms=6, label=LAB[mode])
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"$y_{\min}$  (deeper IR $\to$ left)")
    axL.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$ on held-out test (per decade)")
    axL.grid(True, which="both", alpha=0.25); axL.legend(fontsize=10)
    axL.invert_xaxis()

    # Right: training coverage per decade
    w = 0.38
    xpos = np.arange(len(CEN))
    for i, mode in enumerate(modes):
        axR.bar(xpos + (i - 0.5) * w, np.maximum(res[mode]["cov"], 0.5), width=w,
                color=COL[mode], alpha=0.85, label=LAB[mode])
    axR.set_yscale("log")
    axR.set_xticks(xpos)
    axR.set_xticklabels([fr"$10^{{{int(np.log10(lo))}}}$" for lo in EDGES[:-1]])
    axR.set_xlabel(r"$y_{\min}$ decade (lower edge)")
    axR.set_ylabel("training events in decade (of 400k)")
    axR.grid(True, which="both", axis="y", alpha=0.25); axR.legend(fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")

    summ = {}
    for mode in modes:
        r = res[mode]
        summ[mode] = dict(mse_all=r["mse_all"], mae_all=r["mae_all"], n=r["n"],
                          per_decade=[[float(EDGES[i]), float(EDGES[i + 1]),
                                       int(r["cnt"][i]), int(r["cov"][i]),
                                       (None if np.isnan(r["mse"][i]) else float(r["mse"][i]))]
                                      for i in range(len(CEN))])
    with open(args.summary_out, "w") as f:
        json.dump(summ, f, indent=1)
    print(f"wrote {args.summary_out}")
    print(f"\n{'decade':>18} {'uniform MSE':>14} {'antenna MSE':>14}   (test-set MSE per y_min decade)")
    for i in range(len(CEN)):
        u, a = res["uniform"]["mse"][i], res["antenna"]["mse"][i]
        print(f"  [{EDGES[i]:.0e},{EDGES[i+1]:.0e})  {u:>14.4g} {a:>14.4g}")
    print(f"\n  overall  uniform MSE={res['uniform']['mse_all']:.4g}  "
          f"antenna MSE={res['antenna']['mse_all']:.4g}")


if __name__ == "__main__":
    main()
