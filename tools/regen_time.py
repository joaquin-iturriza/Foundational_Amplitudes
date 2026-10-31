#!/usr/bin/env python3
"""MadEvent cost of regenerating catalog_v4's MadEvent pools, from the lhe_rates files (tools/lhe_rates.py, read on CC
from each pool's sidecar): per pool core-s = madevent_s x mg_cores over all its runs. Counted: the MadEvent runs only;
not counted: MadGraph output and compile (reported apart, output_compile, for the pools built in one job), the tree
and one-loop prebuild, the diagram sidecars. One JSON line: per
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


def output_stats(path):
    """MadGraph output and compile per process, wall seconds on one core (lhe_prepare: the MadEvent directory's output
    and the standalone's output and compile, before MadEvent runs), over the pools built in one job"""
    if not path or not os.path.exists(path):
        return None
    x = np.array([float(v) for v in json.load(open(path)).values()])
    return {"pools": len(x), "median_s": round(float(np.median(x)), 1), "q90_s": round(float(np.quantile(x, 0.9)), 1),
            "max_s": round(float(x.max()), 1), "total_core_h": round(float(x.sum()) / 3600, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", default=[os.path.join(DATA, "lhe_rates.json"),
                                                   os.path.join(DATA, "lhe_rates_dsh_ee4.json")])
    ap.add_argument("--prefix", nargs="+", default=["pp13_", "pp13v4_", "dsh_", "ee4_"])
    ap.add_argument("--output-s", default=os.path.join(DATA, "lhe_output_s.json"),
                    help="{pool: seconds of MadGraph output and standalone compile} of the pools built in one job "
                         "(tools/lhe_rates.py output_s of the pp13_ and pp13v4_ pools)")
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
                      "all": stats(pools), "by_prefix": {p: stats(v) for p, v in by.items()},
                      "output_compile": output_stats(a.output_s)}))


if __name__ == "__main__":
    main()
