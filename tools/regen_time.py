#!/usr/bin/env python3
"""MadEvent cost of regenerating catalog_v4's MadEvent pools, from the lhe_rates files (tools/lhe_rates.py, read on CC
from each pool's sidecar): per pool core-s = madevent_s x mg_cores over all its runs. Counted: the MadEvent runs only;
not counted: MadGraph output and compile (reported apart, output_compile, for the pools built in one job), the tree
and one-loop prebuild, the diagram sidecars. One JSON line: per
prefix and over all, the pools counted, the median, 90% and 99% quantiles and the largest per-pool core-s (its pool,
cores, MadEvent hours summed over its runs and its runs), and the total in core-h.

    python3 tools/regen_time.py [--rates analysis/catalog_v4/data/lhe_rates.json analysis/catalog_v4/data/lhe_rates_dsh_ee4.json]
                                [--prefix pp13_ pp13v4_ dsh_ ee4_]
    python3 tools/regen_time.py anchor     the e+e- pools at the collider anchor (recipes/catalog_v4_eea_*.yaml, plan
                                           catalog_coverage step 74), estimated from the old pools they replace
"""
import argparse, json, os, sys

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


SPECTRUM_TIME = (0.67, 1.25)   # time per kept event with the beam spectrum over without, the step-71 probe's range
                               # (analysis/catalog_v4/data/ee_anchor_probe.json, dsh_epem_bbxvevex at the 10 energies)


def _runs(lhe):
    """MadEvent runs of an lhe entry (lhe_runs' count)."""
    if lhe.get("energy_runs"):
        return sum(int(x) for x in lhe["energy_runs"])
    return int(lhe.get("n_runs", 1)) * len(lhe.get("energies") or [1])


def anchor_estimate(rates, gp_times, new_specs, old_specs):
    """Core-hours of the anchor pools (an estimate). Runs: each new pool's runs at its old pool's measured core-seconds
    per run (sidecar madevent_s x cores over its runs; old pool: eea_<x> -> ee4_<x>, dsh_eea_<x> -> dsh_<x>), times
    the spectrum's time factor (low and high of SPECTRUM_TIME). Per run, not per event: a run is a survey plus its
    events, and the survey does not shrink with the events (an eea run asks 204 events where an ee4 run asked 340, so
    the per-run rate errs high there). Gridpacks: the new pools' gridpacks (one per energy of an lhe.gridpack pool) at
    the measured e+e- gridpack tasks' wall x cores (median and mean of gp_times, without spectrum: times the spectrum
    factor as the runs), not in the sidecars' madevent_s. A new pool whose old pool has no measured rate is listed
    (missing_rates), not estimated."""
    import yaml
    old = {}
    for f in old_specs:
        d = yaml.safe_load(open(f)); h = d.get("lhe") or {}
        old.update({k: {**h, **(v.get("lhe") or {})} for k, v in d["processes"].items()})
    new = {}
    for f in new_specs:
        d = yaml.safe_load(open(f)); h = d.get("lhe") or {}
        new.update({k: {**h, **(v.get("lhe") or {})} for k, v in d["processes"].items()})
    g = json.load(open(gp_times))
    gw = np.array(list(g["wall_s"].values()), float) * int(g["cores"])
    per_pool, missing, n_grid, n_runs_new, n_runs_old = {}, [], {"eea_": 0, "dsh_eea_": 0}, 0, 0
    for k, l in new.items():
        o = ("ee4_" + k[len("eea_"):]) if k.startswith("eea_") else ("dsh_" + k[len("dsh_eea_"):])
        r = rates.get(o)
        if not r or o not in old:
            missing.append(k); continue
        per_run = r["madevent_s"] * r["mg_cores"] / len(r["events_per_run"])
        n = _runs(l)
        n_runs_new += n; n_runs_old += len(r["events_per_run"])
        per_pool[k] = {"old": o, "old_core_s": r["madevent_s"] * r["mg_cores"], "per_run_s": per_run, "runs": n,
                       "core_s": per_run * n}
        if l.get("gridpack"):
            n_grid["dsh_eea_" if k.startswith("dsh_eea_") else "eea_"] += len(l.get("energies") or [1])
    x = np.array([v["core_s"] for v in per_pool.values()])
    old_cs = sum(v["old_core_s"] for v in per_pool.values())
    top = max(per_pool, key=lambda k: per_pool[k]["per_run_s"])
    by = lambda p: [v for k, v in per_pool.items() if k.startswith(p)]
    return {"counted": "MadEvent runs at the old pools' measured core-s per run x the spectrum factor, plus gridpacks "
                       "at the measured e+e- gridpack tasks (an estimate; not: output and compile, assembly)",
            "pools": len(per_pool), "missing_rates": missing, "runs_new": n_runs_new, "runs_old": n_runs_old,
            "old_core_h": round(old_cs / 3600, 1),
            "runs_core_h": [round(float(x.sum()) * f / 3600, 1) for f in SPECTRUM_TIME],
            "runs_core_h_by_prefix": {p: [round(sum(v["core_s"] for v in by(p)) * f / 3600, 1) for f in SPECTRUM_TIME]
                                      for p in ("eea_", "dsh_eea_")},
            "gridpacks": n_grid, "gridpack_core_s_median_mean": [round(float(np.median(gw))), round(float(gw.mean()))],
            # low: the median gridpack at the spectrum's low factor; high: the mean at its high factor
            "gridpack_core_h_by_prefix": {p: [round(n * float(np.median(gw)) * SPECTRUM_TIME[0] / 3600, 1),
                                              round(n * float(gw.mean()) * SPECTRUM_TIME[1] / 3600, 1)]
                                          for p, n in n_grid.items()},
            "longest_run": {"pool": top, "old_per_run_s": round(per_pool[top]["per_run_s"]),
                            "with_spectrum_s": [round(per_pool[top]["per_run_s"] * f) for f in SPECTRUM_TIME]},
            "longest_gridpack_wall_s": int(max(g["wall_s"].values()))}


def main():
    if sys.argv[1:2] == ["anchor"]:
        R = os.path.join(ROOT, "recipes")
        rates = {}
        for f in ("lhe_rates.json", "lhe_rates_dsh_ee4.json"):
            rates.update(json.load(open(os.path.join(DATA, f))))
        print(json.dumps(anchor_estimate(rates, os.path.join(DATA, "ee_gridpack_times.json"),
                                         [os.path.join(R, f) for f in ("catalog_v4_eea_processes.yaml",
                                                                       "catalog_v4_eea_default_processes.yaml")],
                                         [os.path.join(R, f) for f in ("catalog_v4_ee_processes.yaml",
                                                                       "catalog_v4_default_processes.yaml")])))
        return
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
