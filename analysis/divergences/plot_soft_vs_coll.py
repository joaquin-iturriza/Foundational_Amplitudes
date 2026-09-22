#!/usr/bin/env python
"""Soft- vs collinear-cut add-back comparison for ee->uug: which IR limit is harder to
extrapolate INTO? Overlays the two held-out-region MSE(f) curves (soft: x_g<c ; collinear:
y_min<c at hard x_g) built by eval_heldout.py, with f=0 (pure extrapolation) the headline.
Reads heldout_eval_{soft,coll}_f<tag>.npz. CPU only."""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402


def curve(eval_dir, prefix, tags):
    fs, mse = [], []
    for t in tags:
        d = np.load(os.path.join(eval_dir, f"{prefix}{t}.npz"))
        r = d["pred_logamp"] - d["true_logamp"]
        fs.append(int(t) / 100.0); mse.append(float(np.mean(r ** 2)))
    return np.array(fs), np.array(mse)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--tags", default="000,005,015,100")
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "soft_vs_coll_addback"))
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",")]

    fsoft, msoft = curve(args.eval_dir, "heldout_eval_soft_f", tags)
    fcoll, mcoll = curve(args.eval_dir, "heldout_eval_coll_f", tags)
    # f=0 is plotted at a small positive x so it has a place on the log axis; its tick reads "0".
    f0 = 3e-3
    xs = fsoft.copy(); xs[fsoft == 0] = f0
    xc = fcoll.copy(); xc[fcoll == 0] = f0

    fig, ax = ps.figure()
    ax.plot(xs, msoft, "o-", color=ps.C.vermillion, label=r"soft, $x_g<c$")
    ax.plot(xc, mcoll, "s-", color=ps.C.blue, label=r"collinear, $y_{\min}<c$")
    ax.set_xscale("log"); ax.set_yscale("log")
    xt = [f0] + [x for x in sorted(set(list(fsoft) + list(fcoll))) if x > 0]
    ax.set_xticks(xt); ax.set_xticklabels(["0"] + [f"{x:g}" for x in xt[1:]])
    ax.minorticks_off()
    ax.set_xlabel(r"add-back fraction $f$")
    ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    # "collinear cut, $y_{\\min}<c$ at hard $x_g$" made the legend wider than the plot box.
    # That the collinear cut is taken at hard $x_g$ is the definition of the cut and belongs
    # in the caption, not in a legend entry.
    ps.legend(ax, "lower left")
    ps.process_label(ax, r"$e^+e^-\to u\bar u g$", loc="upper right")

    ps.save(fig, args.out_base)
    r0s, r0c = msoft[fsoft == 0][0], mcoll[fcoll == 0][0]
    print(f"  f=0 pure extrapolation: soft MSE={r0s:.4g}  collinear MSE={r0c:.4g}  "
          f"ratio soft/coll={r0s / r0c:.2f}")


if __name__ == "__main__":
    main()
