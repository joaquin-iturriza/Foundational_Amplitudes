#!/usr/bin/env python
"""The add-back curve for the ee->uug hold-out/extrapolation study: held-out
deep-IR-region error vs the add-back fraction f. f=0 is pure extrapolation (the base
never saw uug and the fine-tune never saw the NEAR region), f=1 the in-support
baseline (the NEAR region's add-back pool fully mixed back in).

Metric: MSE Δlog|M|^2 (the paper's primary metric everywhere; = mean of the squared
log-amplitude residual) plus MAE, computed straight from the per-fraction eval npz
(heldout_eval_ft_f<tag>.npz: true_logamp, pred_logamp, y_min, cut) so the numbers are
exactly the predictions eval_heldout.py produced. Also (re)writes
heldout_eval_summary.json. CPU only.

Left : MSE and MAE Δlog|M|^2 over the whole held-out NEAR region vs f (log-x, f=0 tick).
Right: MSE vs f split by y_min sub-bin, so the deepest-IR bins (where extrapolation is
hardest) are separated from the shallow tail.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YBINS = [(0.0, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2)]   # + a final [1e-2, cut) added per-file


def summarize(npz_path):
    d = np.load(npz_path)
    resid = d["pred_logamp"] - d["true_logamp"]
    y = d["y_min"]
    cut = float(d["cut"])
    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))
    bins = YBINS + [(1e-2, cut)]
    binmse = []
    for lo, hi in bins:
        m = (y >= lo) & (y < hi)
        binmse.append([lo, hi, int(m.sum()),
                       float(np.mean(resid[m] ** 2)) if m.any() else float("nan")])
    return dict(mse=mse, mae=mae, binmse=binmse, n=int(resid.size))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--tags", default="000,005,015,050,100")
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "addback_curve"))
    ap.add_argument("--summary_out", default=os.path.join(here, "heldout_eval_summary.json"))
    args = ap.parse_args()

    S = []
    for tag in [t.strip() for t in args.tags.split(",")]:
        npz = os.path.join(args.eval_dir, f"heldout_eval_ft_f{tag}.npz")
        r = summarize(npz)
        r.update(tag=tag, f=int(tag) / 100.0)
        S.append(r)
    S = sorted(S, key=lambda r: r["f"])
    with open(args.summary_out, "w") as f:
        json.dump(S, f, indent=1)

    f = np.array([r["f"] for r in S])
    mse = np.array([r["mse"] for r in S])
    mae = np.array([r["mae"] for r in S])
    fpos = f.copy()
    f0 = 3e-3
    fpos[f == 0] = f0                      # f=0 mapped to a small positive tick on log-x

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.suptitle(r"$e^+e^-\to u\bar u g$ hold-out / add-back: held-out deep-IR "
                 r"($y_{\min}<c$) error vs add-back fraction $f$", fontsize=12.5)

    axL.plot(fpos, mse, "o-", color="crimson", lw=1.8, label=r"MSE $\Delta\log|\mathcal{M}|^2$")
    axL.plot(fpos, mae, "s--", color="steelblue", lw=1.3, label=r"MAE")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"add-back fraction $f$  (leftmost tick = $f{=}0$, pure extrapolation)")
    axL.set_ylabel(r"held-out-region error")
    axL.axvline(f0 * 2.2, color="grey", ls=":", lw=0.8)
    xt = [f0] + [x for x in f if x > 0]
    axL.set_xticks(xt)
    axL.set_xticklabels(["0"] + [f"{x:g}" for x in f if x > 0])
    axL.grid(True, which="both", alpha=0.25); axL.legend(fontsize=9)

    bins = S[0]["binmse"]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(bins)))
    for bi, (lo, hi, _, _) in enumerate(bins):
        y = np.array([r["binmse"][bi][3] for r in S])
        axR.plot(fpos, y, "o-", color=cmap[bi], lw=1.5,
                 label=fr"$y_{{\min}}\in[{lo:.0e},{hi:.0e})$")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"add-back fraction $f$")
    axR.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$ (per $y_{\min}$ bin)")
    axR.set_xticks(xt); axR.set_xticklabels(["0"] + [f"{x:g}" for x in f if x > 0])
    axR.grid(True, which="both", alpha=0.25); axR.legend(fontsize=8, title="deeper IR $\\to$ top")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")
    for r in S:
        deep = r["binmse"][0][3]
        print(f"  f={r['f']:.2f}: MSE={r['mse']:.4g} MAE={r['mae']:.4g} "
              f"(deepest y_min<1e-3 bin MSE={deep:.4g})")


if __name__ == "__main__":
    main()
