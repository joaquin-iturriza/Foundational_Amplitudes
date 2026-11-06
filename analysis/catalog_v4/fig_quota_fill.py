#!/usr/bin/env python3
"""How flat the quota's log10|M|^2 coverage is over each class's pp range (plan catalog_coverage step 67, D294): per
quota bin (0.25 dex), the fill = sum over the pool's chunks of min(kept, quota) / sum of the quota, for the old ladder,
the edge ladder and both ladders at budget 50 and both ladders at budget 750; points at the bin centres, the seeds'
mean with their range as error bars; dotted: the class's pp in-support max (bins right of it are the quota's
+0.5 dex margin). Reads data/quota_coverage.json (multichannel_check.py coverage). One panel per class ->
figs/quota_fill_<a..>.{png,pdf}, legend figs/quota_fill_legend.

    python3 analysis/catalog_v4/fig_quota_fill.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402
from fig_flat_quota import LABEL  # noqa: E402

CURVES = (("old", "old ladder", "grey"), ("edge", "edge ladder", "orange"), ("both", "both ladders", "blue"),
          ("both750", "both ladders, 15x budget", "vermillion"))


def main():
    d = json.load(open(os.path.join(HERE, "data", "quota_coverage.json")))["classes"]
    names = [c for c in LABEL if c in d]
    figs = ps.panels(len(names))
    for (fig, ax), c in zip(figs, names):
        o = d[c]
        for k, lab, col in CURVES:
            p = o["policies"].get(k)
            if not p:
                continue
            x = np.array([b["lo"] for b in p["bins"]]) + 0.125
            m = np.array([b["mean"] for b in p["bins"]])
            lo = m - np.array([b["min"] for b in p["bins"]]); hi = np.array([b["max"] for b in p["bins"]]) - m
            ax.errorbar(x, m, yerr=[lo, hi], color=getattr(ps.C, col), marker="o", ms=2.5, lw=1, capsize=1.5,
                        label=lab)
        ax.axvline(o["pp_in_support_max_log10"], color=ps.C.grey, ls=":", lw=0.8, label="pp max")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$")
        ax.set_ylabel("quota fill")
        ps.process_label(ax, LABEL[c], loc="lower left")
    ps.legend_strip(figs[0][1], os.path.join(HERE, "figs", "quota_fill_legend"), ncol=3)
    ps.save_panels(figs, os.path.join(HERE, "figs", "quota_fill"))


if __name__ == "__main__":
    main()
