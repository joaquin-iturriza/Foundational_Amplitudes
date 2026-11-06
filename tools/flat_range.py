#!/usr/bin/env python3
"""The log|M|^2 range the catalog's flat part covers per class (plan catalog_coverage step 58, Joaquin 2026-11-06:
"uniform sampling in logM^2 that covers the range of the pp sampled process"): the natural-log |M|^2 of the class's
13 TeV pp events in the catalog's support (its PP4 probe's train and test pools, 1.2e4 MadEvent events; those with
sqrt(s_hat) in the class's window and the catalog's cuts passed in the partonic frame, pp_support.py's `sup` mask:
logm_min_in_support / logm_max_in_support of analysis/catalog_v4/data/pp_support_reads.json), min - MARGIN_DEX to
max + MARGIN_DEX decades. Events outside the support are left out: no catalog sampler reaches them, and their outliers
put the pp minimum a median 4.4 decades below the catalog's own (repo-reviewer, 2026-11-06).

    python3 tools/flat_range.py            print the count and the widest ranges
    python3 tools/flat_range.py --write    -> recipes/catalog_v4_flat_range.json {class: [lo, hi]} (natural log)
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "pp_support_reads.json")
OUT = os.path.join(ROOT, "recipes", "catalog_v4_flat_range.json")
# decades of |M|^2 beyond the in-support pp events' min and max: D270 ("sample in a way that covers the range the pp beams
# cover, but wider") asks for a margin; 0.5 decade (two of the sampler's 0.25-dex bins) is the chosen width, not his number
MARGIN_DEX = 0.5


def ranges(reads=READS, margin_dex=MARGIN_DEX):
    """{class: [lo, hi]}: natural-log |M|^2 bounds of the in-support pp events over every probe of the class (a class with several probes takes the
    union), the margin added on both sides"""
    m = margin_dex * np.log(10.0)
    out = {}
    for r in json.load(open(reads))["rows"]:
        d = r.get("default") or {}
        if r.get("class") is None:
            continue
        if "logm_min_in_support" not in d:
            raise SystemExit(f"{reads}: no in-support range ({r['probe']}); re-read with pp_support.py pull")
        if d["logm_min_in_support"] is None:      # no pp event in the support: the class gets no flat range
            continue
        lo, hi = float(d["logm_min_in_support"]) - m, float(d["logm_max_in_support"]) + m
        if r["class"] in out:
            lo, hi = min(lo, out[r["class"]][0]), max(hi, out[r["class"]][1])
        out[r["class"]] = [lo, hi]
    return out


if __name__ == "__main__":
    rg = ranges()
    w = sorted(((hi - lo) / np.log(10.0), c) for c, (lo, hi) in rg.items())
    print(f"{len(rg)} classes; width in decades: median {w[len(w) // 2][0]:.1f}, widest {w[-1][0]:.1f} ({w[-1][1]})")
    if "--write" in sys.argv[1:]:
        with open(OUT, "w") as f:
            json.dump({c: [round(lo, 6), round(hi, 6)] for c, (lo, hi) in sorted(rg.items())}, f, indent=0)
        print("wrote", OUT)
