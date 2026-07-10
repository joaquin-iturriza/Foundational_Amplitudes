#!/usr/bin/env python
"""Soft- vs collinear-cut add-back comparison for ee->uug: which IR limit is harder to
extrapolate INTO? Overlays the two held-out-region MSE(f) curves (soft: x_g<c ; collinear:
y_min<c at hard x_g) built by eval_heldout.py, with f=0 (pure extrapolation) the headline.
Reads heldout_eval_{soft,coll}_f<tag>.npz. CPU only."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    f0 = 3e-3
    xs = fsoft.copy(); xs[fsoft == 0] = f0
    xc = fcoll.copy(); xc[fcoll == 0] = f0

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(xs, msoft, "o-", color="crimson", lw=2.0, label=r"soft cut ($x_g<c$)")
    ax.plot(xc, mcoll, "s-", color="steelblue", lw=2.0, label=r"collinear cut ($y_{\min}<c$, hard $x_g$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    xt = [f0] + [x for x in sorted(set(list(fsoft) + list(fcoll))) if x > 0]
    ax.set_xticks(xt); ax.set_xticklabels(["0"] + [f"{x:g}" for x in xt[1:]])
    ax.axvline(f0 * 2.2, color="grey", ls=":", lw=0.8)
    ax.set_xlabel(r"add-back fraction $f$  (leftmost tick $=f{=}0$, pure extrapolation)")
    ax.set_ylabel(r"held-out-region MSE $\Delta\log|\mathcal{M}|^2$")
    ax.set_title(r"$e^+e^-\to u\bar u g$: extrapolating into the soft vs collinear limit", fontsize=12)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=10)
    # annotate the f=0 headline ratio
    r0s, r0c = msoft[fsoft == 0][0], mcoll[fcoll == 0][0]
    ax.annotate(fr"$f{{=}}0$:  soft {r0s:.3g}  vs  collinear {r0c:.3g}"
                f"\n(harder: {'soft' if r0s > r0c else 'collinear'})",
                xy=(f0, max(r0s, r0c)), xytext=(0.04, 0.92), textcoords="axes fraction",
                fontsize=9, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")
    print(f"  f=0 pure extrapolation: soft MSE={r0s:.4g}  collinear MSE={r0c:.4g}  "
          f"ratio soft/coll={r0s / r0c:.2f}")


if __name__ == "__main__":
    main()
