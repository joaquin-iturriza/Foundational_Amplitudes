#!/usr/bin/env python3
"""MadEvent cost of regenerating catalog_v4's MadEvent pools, from the lhe_rates files (tools/lhe_rates.py, read on CC
from each pool's sidecar): per pool core-s = madevent_s x mg_cores over all its runs. Counted: the MadEvent runs only;
not counted: MadGraph output and compile, the tree and one-loop prebuild, the diagram sidecars. One JSON line: per
prefix and over all, the pools counted, the median, 90% and 99% quantiles and the largest per-pool core-s (its pool,
cores, MadEvent hours summed over its runs and its runs), and the total in core-h.

    python3 tools/regen_time.py [--rates analysis/catalog_v4/data/lhe_rates.json analysis/catalog_v4/data/lhe_rates_dsh_ee4.json]
                                [--prefix pp13_ pp13v4_ dsh_ ee4_]
"""
import argparse, json, os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "analysis", "catalog_v4", "data")


def stats(pools):
    cs = {k: v["madevent_s"] * v["mg_cores"] for k, v in pools.items()}
    if not cs:
        return {"pools": 0}
    x = np.array(list(cs.values()))
    top = max(cs, key=cs.get)
    return {"pools": len(x), "median_core_s": round(float(np.median(x))),
            "q90_core_s": round(float(np.quantile(x, 0.9))), "q99_core_s": round(float(np.quantile(x, 0.99))),
            "max_pool": top, "max_core_s": round(cs[top]), "max_cores": pools[top]["mg_cores"],
            "max_madevent_h": round(pools[top]["madevent_s"] / 3600, 2), "max_runs": pools[top].get("n_runs"), "total_core_h": round(float(x.sum()) / 3600)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", default=[os.path.join(DATA, "lhe_rates.json"),
                                                   os.path.join(DATA, "lhe_rates_dsh_ee4.json")])
    ap.add_argument("--prefix", nargs="+", default=["pp13_", "pp13v4_", "dsh_", "ee4_"])
    a = ap.parse_args()
    d = {}
    for f in a.rates:
        d.update(json.load(open(f)))
    pools = {k: v for k, v in d.items() if k.startswith(tuple(a.prefix)) and v.get("madevent_s") is not None}
    # longest prefix first: pp13v4_ is not pp13_
    by = {p: {} for p in a.prefix}
    for k, v in pools.items():
        by[max((p for p in a.prefix if k.startswith(p)), key=len)][k] = v
    print(json.dumps({"counted": "MadEvent runs only (not: MadGraph output and compile, tree and one-loop prebuild, "
                                 "diagram sidecars)", "prefixes": a.prefix,
                      "all": stats(pools), "by_prefix": {p: stats(v) for p, v in by.items()}}))


if __name__ == "__main__":
    main()
