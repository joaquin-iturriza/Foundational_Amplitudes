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
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

YBINS = [(0.0, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2)]   # + a final [1e-2, cut) added per-file


def summarize(npz_path):
    d = np.load(npz_path)
    resid = d["pred_logamp"] - d["true_logamp"]
    y = d["y_min"]
    cut = float(d["cut"])
    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))
    # The trailing [1e-2, cut) bin is only meaningful when the analysis cut sits ABOVE
    # 1e-2. For uugg cut=6e-3, which made an inverted, permanently-empty bin that still
    # claimed a legend entry.
    bins = YBINS + ([(1e-2, cut)] if cut > 1e-2 else [])
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
    ap.add_argument("--npz_prefix", default="heldout_eval_ft_f", help="eval npz basename prefix")
    # The process label is the only text allowed inside the axes; there is deliberately no
    # --title, since what the figure shows belongs in the results.tex caption.
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    S = []
    for tag in [t.strip() for t in args.tags.split(",")]:
        npz = os.path.join(args.eval_dir, f"{args.npz_prefix}{tag}.npz")
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

    fig, (axL, axR) = ps.figure(ncols=2)
    xt = [f0] + [x for x in f if x > 0]
    xtl = ["0"] + [f"{x:g}" for x in f if x > 0]

    axL.plot(fpos, mse, "o-", color=ps.C.vermillion, label=r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    axL.plot(fpos, mae, "s--", color=ps.C.blue, label=r"MAE$(\Delta\log|\mathcal{M}|^2)$")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"add-back fraction $f$")
    axL.set_ylabel("held-out-region error")
    axL.set_xticks(xt); axL.set_xticklabels(xtl); axL.minorticks_off()
    axL.legend(loc="lower left")
    ps.process_label(axL, args.process, loc="upper right")

    bins = S[0]["binmse"]
    ramp = ps.sequence(len(bins))
    def _p10(v):
        """1e-3 -> 10^{-3}; 3e-3 -> 3\times10^{-3}. Rounding the exponent alone turned
        3e-3 into 10^{-3} and printed two identical bin edges. The mantissa is cut to two
        significant digits because the top bin's edge is the analysis cut, whose exact value
        (3.8986e-2) is not a round number and spilled 5 digits into the legend."""
        if v <= 0:
            return "0"
        e = int(np.floor(np.log10(v)))
        m = float(f"{v / 10.0 ** e:.2g}")
        return fr"10^{{{e}}}" if abs(m - 1) < 5e-3 else fr"{m:g}\times 10^{{{e}}}"
    for bi, (lo, hi, _, _) in enumerate(bins):
        y = np.array([r["binmse"][bi][3] for r in S])
        if not np.isfinite(y).any():
            continue
        axR.plot(fpos, y, "o-", color=ramp[bi],
                 label=fr"$y_{{\min}}\in[{_p10(lo)},{_p10(hi)})$")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"add-back fraction $f$")
    axR.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    axR.set_xticks(xt); axR.set_xticklabels(xtl); axR.minorticks_off()
    axR.legend(loc="lower left")

    ps.save(fig, args.out_base)
    for r in S:
        deep = r["binmse"][0][3]
        print(f"  f={r['f']:.2f}: MSE={r['mse']:.4g} MAE={r['mae']:.4g} "
              f"(deepest y_min<1e-3 bin MSE={deep:.4g})")


if __name__ == "__main__":
    main()
