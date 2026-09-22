#!/usr/bin/env python
"""Flexible L2 held-out comparison plots. Reads heldout_eval_<label>.npz files (each holds row-aligned
pred_logamp/true_logamp/y_min[/sqrt_s]) and overlays, per y_min IR-resolution decade, the MSE of
Delta log|M|^2 for an arbitrary set of labelled runs. Serves both:

  * Track A (harder divergences): base vs sigma^gamma per process (uug/uugg/uuggg).
  * Track B (Bayesian): epistemic-sigma (bbb) vs het-head sigma vs base on uugg.

With --regions (uug multi-scale) it ALSO draws the sqrt(s) Z-peak / shoulder / continuum breakdown and
the deep-IR x on/off-peak split -- the diagnostic for whether ONE sigma^gamma knob budgets across the
resonance AND the IR. CPU only. Emits BOTH .png and .pdf (project rule).

Usage:
  python plot_l2_compare.py --out figs/l2_uug_multiscale \\
      --npz base=heldout_eval_uug_base_s0 sigma=heldout_eval_uug_sigma_s0 sigma_g3=heldout_eval_uug_g3_s0 \\
      --regions
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DECADES = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
MZ = 91.1876


# heldout_eval_*.npz are written to the MAIN repo (score_and_save uses REPO), not the worktree copy.
NPZ_DIR = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/analysis/divergences"


def load(path):
    if not path.endswith(".npz"):
        path += ".npz"
    if not os.path.isabs(path):
        cand = os.path.join(NPZ_DIR, path)
        path = cand if os.path.exists(cand) else os.path.join(os.path.dirname(__file__), path)
    d = np.load(path)
    err2 = (d["pred_logamp"] - d["true_logamp"]) ** 2
    s = d["sqrt_s"] if ("sqrt_s" in d.files and d["sqrt_s"].size) else None
    return err2, d["y_min"], s


def decade_mse(err2, y_min):
    xs, ys, ns = [], [], []
    for lo, hi in DECADES:
        m = (y_min >= lo) & (y_min < hi)
        if m.sum() == 0:
            xs.append(np.sqrt(lo * hi) if lo > 0 else 5e-7); ys.append(np.nan); ns.append(0); continue
        xs.append(np.sqrt(lo * hi) if lo > 0 else 5e-7); ys.append(err2[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="label=heldout_eval_<...> pairs")
    ap.add_argument("--out", required=True, help="output basename (writes .png + .pdf)")
    ap.add_argument("--process", default="", help="process label drawn inside the axes")
    ap.add_argument("--regions", action="store_true", help="add the sqrt(s) Z-peak/continuum panel (uug)")
    args = ap.parse_args()

    runs = []
    for spec in args.npz:
        # rsplit: paths never contain "=", but labels do (e.g. "$\\gamma=3$")
        label, path = spec.rsplit("=", 1)
        err2, y_min, s = load(path)
        runs.append((label, err2, y_min, s))

    ncol = 2 if args.regions else 1
    fig, axes = ps.figure(ncols=ncol, squeeze=False)
    ax = axes[0, 0]
    colors = ps.sequence(len(runs))
    for (label, err2, y_min, s), c in zip(runs, colors):
        x, y, n = decade_mse(err2, y_min)
        ax.plot(x, y, "o-", color=c, label=label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$y_{\min}$")
    ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    ps.legend(ax, "upper left")
    if args.process:
        ps.process_label(ax, args.process)

    if args.regions:
        ax2 = axes[0, 1]
        SREG = [(r"$<3$", lambda s: np.abs(s - MZ) < 3.0),
                (r"$3\!-\!15$", lambda s: (np.abs(s - MZ) >= 3.0) & (np.abs(s - MZ) < 15.0)),
                (r"$>15$", lambda s: np.abs(s - MZ) >= 15.0),
                (r"$<3$, IR", lambda s: None)]  # special-cased below
        xlab = [r[0] for r in SREG]
        width = 0.8 / max(1, len(runs))
        for j, (label, err2, y_min, s) in enumerate(runs):
            if s is None:
                continue
            vals = []
            for name, fn in SREG:
                if "IR" in name:
                    m = (y_min < 1e-3) & (np.abs(s - MZ) < 3.0)
                else:
                    m = fn(s)
                vals.append(err2[m].mean() if m.sum() else np.nan)
            xpos = np.arange(len(SREG)) + j * width
            ax2.bar(xpos, vals, width=width, label=label, color=colors[j])
        ax2.set_yscale("log")
        ax2.set_xticks(np.arange(len(SREG)) + width * (len(runs) - 1) / 2)
        ax2.set_xticklabels(xlab)
        ax2.set_xlabel(r"$|\sqrt{s}-M_Z|$  [GeV]")
        ax2.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
        ax2.grid(True, axis="y", which="both")
        # No second legend: both panels draw the SAME three arms, so repeating it here just
        # takes up plot area. One legend per figure, inside an axes.

    ps.save(fig, args.out)


if __name__ == "__main__":
    main()
