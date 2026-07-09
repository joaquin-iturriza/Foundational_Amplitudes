#!/usr/bin/env python
"""The add-back curve for the ee->uug hold-out/extrapolation study: held-out
deep-IR-region error vs the add-back fraction f. f=0 is pure extrapolation (the base
never saw uug and the fine-tune never saw the NEAR region), f=1 the in-support
baseline (the NEAR region's add-back pool fully mixed back in). Reads
heldout_eval_summary.json written by eval_heldout.py. CPU only.

Left : RMS Δlog|M|^2 over the whole held-out NEAR region vs f (log-x with an f=0 tick).
Right: RMS vs f split by y_min sub-bin, so the deepest-IR bins (where extrapolation
is hardest) are separated from the shallow tail.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(
        os.path.dirname(__file__), "heldout_eval_summary.json"))
    ap.add_argument("--out_base", default=os.path.join(
        os.path.dirname(__file__), "figs", "addback_curve"))
    args = ap.parse_args()

    S = json.load(open(args.summary))
    S = sorted(S, key=lambda r: r["f"])
    f = np.array([r["f"] for r in S])
    rms = np.array([r["rms"] for r in S])
    mae = np.array([r["mae"] for r in S])
    # f on a log axis needs f=0 mapped to a small positive tick.
    fpos = f.copy()
    f0 = 3e-3
    fpos[f == 0] = f0

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.suptitle(r"$e^+e^-\to u\bar u g$ hold-out / add-back: held-out deep-IR "
                 r"($y_{\min}<c$) error vs add-back fraction $f$", fontsize=12.5)

    axL.plot(fpos, rms, "o-", color="crimson", lw=1.8, label=r"RMS $\Delta\log|\mathcal{M}|^2$")
    axL.plot(fpos, mae, "s--", color="steelblue", lw=1.3, label=r"MAE")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"add-back fraction $f$  (leftmost tick = $f{=}0$, pure extrapolation)")
    axL.set_ylabel(r"held-out-region error")
    axL.axvline(f0 * 2.2, color="grey", ls=":", lw=0.8)
    xt = [f0] + [x for x in f if x > 0]
    axL.set_xticks(xt)
    axL.set_xticklabels(["0"] + [f"{x:g}" for x in f if x > 0])
    axL.grid(True, which="both", alpha=0.25); axL.legend(fontsize=9)

    # per-bin curves: binrms rows are [lo, hi, n, rms]
    bins = S[0]["binrms"]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(bins)))
    for bi, (lo, hi, _, _) in enumerate(bins):
        y = np.array([r["binrms"][bi][3] for r in S])
        axR.plot(fpos, y, "o-", color=cmap[bi], lw=1.5,
                 label=fr"$y_{{\min}}\in[{lo:.0e},{hi:.0e})$")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"add-back fraction $f$")
    axR.set_ylabel(r"RMS $\Delta\log|\mathcal{M}|^2$ (per $y_{\min}$ bin)")
    axR.set_xticks(xt); axR.set_xticklabels(["0"] + [f"{x:g}" for x in f if x > 0])
    axR.grid(True, which="both", alpha=0.25); axR.legend(fontsize=8, title="deeper IR $\\to$ top")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")
    for r in S:
        print(f"  f={r['f']:.2f}: RMS={r['rms']:.4f} MAE={r['mae']:.4f}")


if __name__ == "__main__":
    main()
