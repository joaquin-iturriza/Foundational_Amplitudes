#!/usr/bin/env python3
"""How flat the quota's log10|M|^2 coverage is over each class's pp range (plan catalog_coverage step 67, D294): per
quota bin (0.25 dex), the fill = sum over the pool's chunks of min(kept, quota) / sum of the quota, for the old ladder,
the edge ladder and both ladders at budget 50 and both ladders at budget 750; points at the bin centres, the seeds'
mean with their range as error bars; dotted: the class's pp in-support max (bins right of it are the quota's
+0.5 dex margin). Reads data/quota_coverage.json (multichannel_check.py coverage). One panel per class ->
figs/quota_fill_<a..>.{png,pdf}, legend figs/quota_fill_legend.

    python3 analysis/catalog_v4/fig_quota_fill.py
    python3 analysis/catalog_v4/fig_quota_fill.py ee   the four e+e- classes of step (f) (no ladder policy there: the
               quota with the joint-resonance channels at budget 50 and 750; dotted: the class's default-sampled max,
               read from its default-share pool at the fixed collider energies) -> figs/ee_quota_fill_<a..d>, legend
               figs/ee_quota_fill_legend
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


EE_LABEL = {"v4_epem_taptamvevex": r"$e^+e^-\to\tau^+\tau^-\nu_e\bar\nu_e$",
            "v4_epem_bbxvevex": r"$e^+e^-\to b\bar b\nu_e\bar\nu_e$",
            "v4_epem_ahvevex": r"$e^+e^-\to\gamma H\nu_e\bar\nu_e$", "v4_epem_vmvmxz": r"$e^+e^-\to\nu_\mu\bar\nu_\mu Z$"}
EE_CURVES = (("both", "flat quota, budget 50", "blue"), ("both750", "flat quota, budget 750", "vermillion"),
             ("fwdfree750", "budget 750, uncut finals forward", "green"))


def main(ee=False):
    d = json.load(open(os.path.join(HERE, "data", "quota_coverage.json")))["classes"]
    lab_of, curves = (EE_LABEL, EE_CURVES) if ee else (LABEL, CURVES)
    names = [c for c in lab_of if c in d]
    figs = ps.panels(len(names))
    for (fig, ax), c in zip(figs, names):
        o = d[c]
        for k, lab, col in curves:
            p = o["policies"].get(k)
            if not p:
                continue
            x = np.array([b["lo"] for b in p["bins"]]) + 0.125
            m = np.array([b["mean"] for b in p["bins"]])
            lo = m - np.array([b["min"] for b in p["bins"]]); hi = np.array([b["max"] for b in p["bins"]]) - m
            ax.errorbar(x, m, yerr=[lo, hi], color=getattr(ps.C, col), marker="o", ms=2.5, lw=1, capsize=1.5,
                        label=lab)
        ax.axvline(o["pp_in_support_max_log10"], color=ps.C.grey, ls=":", lw=0.8,
                   label="default-sampled max" if ee else "pp max")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$")
        ax.set_ylabel("quota fill")
        ps.process_label(ax, lab_of[c], loc="upper right" if ee else "lower left")
    stem = "ee_quota_fill" if ee else "quota_fill"
    ps.legend_strip(figs[0][1], os.path.join(HERE, "figs", stem + "_legend"), ncol=3)
    ps.save_panels(figs, os.path.join(HERE, "figs", stem))


if __name__ == "__main__":
    main(ee=sys.argv[1:] == ["ee"])
