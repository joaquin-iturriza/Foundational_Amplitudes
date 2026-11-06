#!/usr/bin/env python3
"""Catalog flat part against the 13 TeV pp events in log10|M|^2, per class of the step-58 check (plan catalog_coverage,
D294): the share of each sample per decade of log10|M|^2, in multichannel_check's 0.25-dex bins: the pp events in the
catalog's support (the class's PP4 probe, train + test), the kept flat events of the multichannel policy (step 57) and
of the quota policy (step 58) and, when read, of the quota policy with the ladder and two-resonance channels
(step 58's re-check), seed 42; dashed: the range the quota is meant to fill. Reads data/multichannel_check_reads/ (the per-seed files of `run ...
multichannel+quota <seed>`). One panel per class -> figs/flat_quota_<a..e>.{png,pdf}.

    python3 analysis/catalog_v4/fig_flat_quota.py [seed]
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

READS = os.path.join(HERE, "data", "multichannel_check_reads")
LABEL = {"v4_uux_epemg": r"$u\bar u\to e^+e^-g$", "v4_uux_epemvevex": r"$u\bar u\to e^+e^-\nu_e\bar\nu_e$",
         "v4_uu_bbxuu": r"$uu\to b\bar b uu$", "v4_us_ddxsu": r"$us\to d\bar d su$",
         "v4_udx_csxvevex": r"$u\bar d\to c\bar s\nu_e\bar\nu_e$"}


def per_dex(h, edges):
    h = np.asarray(h, float)
    return h / max(h.sum(), 1) / np.diff(edges)


def main(seed=42):
    runs = {}
    for f in sorted(glob.glob(os.path.join(READS, "*.json"))):
        d = json.load(open(f))
        r = runs.setdefault(d["class"], {"pp": d["pp"], "edges": np.asarray(d["logm_bins"])})
        r.update({k: v for k, v in d["runs"].items() if k.endswith(f"_{seed}")})
    names = [c for c in LABEL if c in runs and f"quota_{seed}" in runs[c]]
    figs = ps.panels(len(names))
    for (fig, ax), c in zip(figs, names):
        r = runs[c]; e = r["edges"]
        ax.stairs(per_dex(r["pp"]["hist_in_support"], e), e, color=ps.C.grey, label="pp 13 TeV")
        ax.stairs(per_dex(r[f"multichannel_{seed}"]["hist_kept"], e), e, color=ps.C.blue, label="multichannel")
        ax.stairs(per_dex(r[f"quota_{seed}"]["hist_kept"], e), e, color=ps.C.vermillion, label="quota")
        if f"quota_ladder_res2_budget500_{seed}" in r:
            ax.stairs(per_dex(r[f"quota_ladder_res2_budget500_{seed}"]["hist_kept"], e), e, color=ps.C.purple,
                      label="same, 10x budget")
        if f"quota_ladder_res2_{seed}" in r:      # step 58 re-check: + ladder and joint-resonance channels
            ax.stairs(per_dex(r[f"quota_ladder_res2_{seed}"]["hist_kept"], e), e, color=ps.C.green,
                      label="quota + ladder, pairs")
        if f"quota_ladder_edge_res2_{seed}" in r:   # step 66: the ladder's pT span and outer end drawn toward the corner
            ax.stairs(per_dex(r[f"quota_ladder_edge_res2_{seed}"]["hist_kept"], e), e, color=ps.C.orange,
                      label="same, ladder at the cut and edge")
        qe = r[f"quota_{seed}"]["quota_chunks"][0]["edges_log10"]     # the quota's target range (pp range +- 0.5 dex)
        for x in (qe[0], qe[-1]):
            ax.axvline(x, color=ps.C.vermillion, ls="--", lw=0.8, label="quota range" if x == qe[0] else None)
        nz = np.nonzero(np.asarray(r["pp"]["hist_in_support"]) + np.asarray(r[f"quota_{seed}"]["hist_kept"]))[0]
        ax.set_xlim(e[nz[0]] - 0.5, e[nz[-1] + 1] + 0.5)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$")
        ax.set_ylabel("share per decade")
        ps.process_label(ax, LABEL[c])
        if c == names[0]:
            ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "flat_quota"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
