#!/usr/bin/env python
"""BBB HP-sweep summary: held-out deep-IR MSE vs the two BBB knobs, against the het-sigma / base
baselines. Shows the key structure -- sigma_rel (initial posterior width) dominates and its optimum
runs to the LOWER bound of the search range, i.e. the sweep drives the posterior toward the
deterministic limit to recover point accuracy. Emits .png and .pdf. CPU only."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

HET_SIGMA_G3 = 2.6468e-02      # het-head sigma^gamma=3 baseline (same held-out set + metric)
BASE_UNIFORM = 3.0057e-02      # uniform-keep baseline


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--results", default=os.path.join(
        here, "..", "..", "sweeps", "l2_bbb_uugg", "results"))
    ap.add_argument("--out", default=os.path.join(here, "figs", "l2_bbb_sweep"))
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        d = json.load(open(f))
        rows.append((d["val_loss"], d["bbb_beta"], d["bbb_sigma_rel"], d["gamma"]))
    obj = np.array([r[0] for r in rows]); beta = np.array([r[1] for r in rows])
    srel = np.array([r[2] for r in rows]); gam = np.array([r[3] for r in rows])
    best = int(np.argmin(obj))

    fig, axes = ps.figure(ncols=2, sharey=True)
    for ax, x, lab, logx in ((axes[0], srel, r"$\sigma_{\rm rel}$", True),
                             (axes[1], beta, r"$\beta_{\rm KL}$", True)):
        sc = ax.scatter(x, obj, c=gam, cmap=ps.CMAP, s=40, zorder=3, label="BBB trial")
        ax.scatter([x[best]], [obj[best]], marker="*", s=220, color=ps.C.vermillion,
                   zorder=4, label="best BBB trial")
        ax.axhline(HET_SIGMA_G3, color=ps.C.vermillion, ls="--",
                   label=r"het-head $\sigma^{\gamma=3}$")
        ax.axhline(BASE_UNIFORM, color=ps.C.grey, ls=":", label="uniform")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        if ax is axes[0]:
            ax.set_ylabel(r"deep-IR MSE")
    axes[0].axvline(5e-3, color=ps.C.blue, ls="-.", label=r"$\sigma_{\rm rel}$ lower bound")
    ps.shared_legend(fig, axes[0], ncol=3)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    # Above the bar, not rotated beside it: as a side label the gamma overhung the 6.5in
    # canvas by 0.15in and savefig (bbox=None) cut it off -- on a figure whose subject IS
    # the gamma sweep.
    cb.ax.set_title(r"$\gamma$", pad=6)
    ps.save(fig, args.out)
    print(f"wrote {args.out}.png/.pdf")
    print(f"best: deep_mse={obj[best]:.4e} at sigma_rel={srel[best]:.3e} beta={beta[best]:.3e} "
          f"gamma={gam[best]:.2f}")
    print(f"  vs het-sigma g3 {HET_SIGMA_G3:.4e} ({obj[best]/HET_SIGMA_G3:.3f}x), "
          f"base {BASE_UNIFORM:.4e} ({obj[best]/BASE_UNIFORM:.3f}x)")
    # rank correlation of each knob with the objective (which knob actually matters)
    def rank(a): return np.argsort(np.argsort(a))
    for nm, v in (("sigma_rel", srel), ("beta", beta), ("gamma", gam)):
        print(f"  spearman(obj, {nm}) = {np.corrcoef(rank(obj), rank(v))[0,1]:+.3f}")


if __name__ == "__main__":
    main()
