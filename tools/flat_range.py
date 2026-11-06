#!/usr/bin/env python3
"""The log|M|^2 range the catalog's flat part covers per class (plan catalog_coverage step 58, Joaquin 2026-11-06:
"uniform sampling in logM^2 that covers the range of the pp sampled process"): the natural-log |M|^2 of the class's
13 TeV pp events (its PP4 probe's train and test pools, 1.2e4 MadEvent events, every event: analysis/catalog_v4/data/
pp_support_reads.json, pp_support.py's read of those files), min - MARGIN_DEX to max + MARGIN_DEX decades, the margin
making the range wider than the pp sample's own extremes.

    python3 tools/flat_range.py            print the count and the widest ranges
    python3 tools/flat_range.py --write    -> recipes/catalog_v4_flat_range.json {class: [lo, hi]} (natural log)
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "pp_support_reads.json")
OUT = os.path.join(ROOT, "recipes", "catalog_v4_flat_range.json")
MARGIN_DEX = 0.5        # decades of |M|^2 beyond the pp events' min and max [open]


def ranges(reads=READS, margin_dex=MARGIN_DEX):
    """{class: [lo, hi]}: natural-log |M|^2 bounds over every probe of the class (a class with several probes takes the
    union), the margin added on both sides"""
    m = margin_dex * np.log(10.0)
    out = {}
    for r in json.load(open(reads))["rows"]:
        d = r.get("default") or {}
        if r.get("class") is None or d.get("logm_min") is None or d.get("logm_max") is None:
            continue
        lo, hi = float(d["logm_min"]) - m, float(d["logm_max"]) + m
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
