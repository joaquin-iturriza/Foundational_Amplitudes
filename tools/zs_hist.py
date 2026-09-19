#!/usr/bin/env python3
"""
zs_hist.py -- Overlaid histograms of predicted vs true log-amplitude for the
zero-shot held-out processes, one figure per process and one panel per model.
The PID model cannot encode ee->ttbar (no top token), so that figure has two
panels and the histogram figure for ee->uu three.

Reads analysis/zero_shot/<label>__<dataset>.json + ..._arrays.npz (from
tools/zero_shot_eval.py). Usage: python tools/zs_hist.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

ZS = os.path.join(ROOT, "analysis", "zero_shot")

MODELS = [("physzs_short", "quantum numbers, 20k"),
          ("pidzs_short", "one-hot PID, 20k"),
          ("phys25_500k_recipe", "quantum numbers, 500k")]
PROCS = [("ee_uu_10-1000GeV_test_amplitudes", r"$e^+e^-\to u\bar u$", "uu"),
         ("ee_ttbar_346-1000GeV_test_amplitudes", r"$e^+e^-\to t\bar t$", "ttbar")]


def cells(ds):
    out = []
    for lab, mlabel in MODELS:
        jpath = os.path.join(ZS, f"{lab}__{ds}.json")
        if not os.path.exists(jpath):
            continue
        d = json.load(open(jpath))
        if not d.get("encodable", True):
            continue
        a = np.load(os.path.splitext(jpath)[0] + "_arrays.npz")
        out.append((mlabel, a["truth_logamp"], a["pred_logamp"]))
    return out


def draw(ax, mlabel, t, p, proc):
    lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(t, bins=bins, color=ps.C.grey, alpha=0.5, label="true")
    ax.hist(p, bins=bins, histtype="step", color=ps.C.vermillion, label=mlabel)
    ax.set_xlabel(r"$\log|\mathcal{M}|^2$")
    ax.set_ylabel("events")
    ps.legend(ax, "upper left")
    ps.process_label(ax, proc)


def main():
    for ds, proc, tag in PROCS:
        cs = cells(ds)
        base = os.path.join(ZS, f"zero_shot_hist_{tag}")
        if len(cs) == 2:
            fig, axes = ps.figure(ncols=2)
            for ax, (m, t, p) in zip(axes, cs):
                draw(ax, m, t, p, proc)
            ps.save(fig, base)
        else:
            figs = ps.panels(len(cs))
            for (fig, ax), (m, t, p) in zip(figs, cs):
                draw(ax, m, t, p, proc)
            ps.save_panels(figs, base)


if __name__ == "__main__":
    main()
