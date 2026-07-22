#!/usr/bin/env python
"""BBB HP-sweep summary: held-out deep-IR MSE vs the two BBB knobs, against the het-sigma / base
baselines. Shows the key structure -- sigma_rel (initial posterior width) dominates and its optimum
runs to the LOWER bound of the search range, i.e. the sweep drives the posterior toward the
deterministic limit to recover point accuracy. Emits .png and .pdf. CPU only."""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for ax, x, lab, logx in ((axes[0], srel, r"$\sigma_{\rm rel}$ (initial posterior width)", True),
                             (axes[1], beta, r"$\beta_{\rm KL}$ (ELBO KL weight)", True)):
        sc = ax.scatter(x, obj, c=gam, cmap="viridis", s=90, edgecolor="k", linewidth=0.5, zorder=3)
        ax.scatter([x[best]], [obj[best]], s=260, facecolor="none", edgecolor="crimson",
                   linewidth=2.0, zorder=4, label="best BBB trial")
        ax.axhline(HET_SIGMA_G3, color="crimson", ls="--", lw=1.6,
                   label=fr"het-head $\sigma^{{\gamma=3}}$ = {HET_SIGMA_G3:.4f}")
        ax.axhline(BASE_UNIFORM, color="grey", ls=":", lw=1.6,
                   label=fr"base (uniform) = {BASE_UNIFORM:.4f}")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lab); ax.set_ylabel(r"held-out deep-IR MSE $\Delta\log|\mathcal{M}|^2$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    # mark the search lower bound on sigma_rel -- the optimum sits ON it
    axes[0].axvline(5e-3, color="steelblue", ls="-.", lw=1.4)
    axes[0].annotate("search lower bound\n(optimum runs to it →\nposterior driven toward\nthe deterministic limit)",
                     xy=(5e-3, obj[best]), xytext=(0.30, 0.62), textcoords="axes fraction",
                     fontsize=8, color="steelblue",
                     arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2))
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"keep exponent $\gamma$")
    fig.suptitle(r"BBB HP sweep ($ee\to u\bar u gg$, 12 DyHPO trials): tuned epistemic-$\sigma$ "
                 r"beats uniform but not the het-head $\sigma$", fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
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
