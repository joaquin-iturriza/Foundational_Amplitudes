#!/usr/bin/env python3
"""
zs_hist.py — Overlaid histograms of predicted vs true log-amplitude for the
zero-shot held-out processes. One row per process, one column per model. Each
cell: true log|A| distribution (filled) vs the model's zero-shot prediction
(step outline), same bins. The PID model on ee->ttbar is un-encodable (no top
token) and is drawn as an empty annotated cell.

Reads analysis/zero_shot/<label>__<dataset>.json + ..._arrays.npz (from
tools/zero_shot_eval.py). Usage: python tools/zs_hist.py
"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZS = os.path.join(ROOT, "analysis", "zero_shot")

COLS = [("physzs_short", "physics (20k)"),
        ("pidzs_short",  "PID (20k)"),
        ("phys25_500k_recipe", "physics (500k)")]
ROWS = [("ee_uu_10-1000GeV_test_amplitudes",     "ee -> uu   (flavor swap u<->d)"),
        ("ee_ttbar_346-1000GeV_test_amplitudes", "ee -> ttbar   (novel top quark)")]


def main():
    nC, nR = len(COLS), len(ROWS)
    fig, axes = plt.subplots(nR, nC, figsize=(4.6 * nC, 3.8 * nR), squeeze=False)
    for r, (ds, dlabel) in enumerate(ROWS):
        for c, (lab, mlabel) in enumerate(COLS):
            ax = axes[r][c]
            jpath = os.path.join(ZS, f"{lab}__{ds}.json")
            if not os.path.exists(jpath):
                ax.set_axis_off(); continue
            d = json.load(open(jpath))
            if not d.get("encodable", True):
                ax.text(0.5, 0.5, "cannot encode\n(no token for top,\nPDG ±6)",
                        ha="center", va="center", fontsize=12, color="#b30000",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor("#b30000"); s.set_linestyle("--")
            else:
                npz = os.path.splitext(jpath)[0] + "_arrays.npz"
                a = np.load(npz)
                t, p = a["truth_logamp"], a["pred_logamp"]
                lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
                bins = np.linspace(lo, hi, 60)
                ax.hist(t, bins=bins, color="0.6", alpha=0.85, label="true")
                ax.hist(p, bins=bins, histtype="step", lw=2.0, color="C3",
                        label="predicted (zero-shot)")
                s = d["test"]
                ax.text(0.03, 0.97,
                        f"MSE$_{{std}}$={s['mse_prepd']:.2f}\n"
                        f"med|rel|={s['rel_err_median']*100:.0f}%",
                        ha="left", va="top", transform=ax.transAxes, fontsize=9,
                        bbox=dict(fc="white", ec="0.7", alpha=0.8))
                if r == 0 and c == 0:
                    ax.legend(fontsize=9, loc="upper right")
            if r == 0:
                ax.set_title(mlabel, fontsize=12)
            if c == 0:
                ax.set_ylabel(dlabel + "\n\ncount", fontsize=10)
            if r == nR - 1:
                ax.set_xlabel("log |amplitude|", fontsize=10)
    fig.suptitle("Zero-shot: predicted vs true log-amplitude distribution on held-out processes",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ZS, "zero_shot_histograms.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
