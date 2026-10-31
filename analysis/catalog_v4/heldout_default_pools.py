#!/usr/bin/env python3
"""Which held-out classes (recipes/catalog_v4_heldout_<axis>.yaml) have a default-sampled test pool (MadEvent default,
2000 test events: PP4's pp13v4_/pp13_, EE4's ee4_) and an estimate of what generating the others costs.

Estimate (an estimate, not a probe): a missing class costs the MadEvent core-seconds of its base class's pool scaled to
2000 events times 1.25 (KEPT_SAFETY), core_s_per_event from analysis/catalog_v4/data/lhe_rates.json and
lhe_rates_dsh_ee4.json, floored at the base pool's whole MadEvent time divided by its runs (the integration is paid once
per run); a class with no base pool takes the median of the pools of its beam and final-state multiplicity (e+e- 5-body:
the largest e+e- multiplicity measured, so it is a lower bound). Labels (one-loop, alpha_s 0.13) are not counted.
    python3 analysis/catalog_v4/heldout_default_pools.py  -> data/heldout_default_pools.json
"""
import json, os, statistics

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
HELD = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
N_TEST, SAFETY = 2000, 1.25


def names(f):
    return [p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]]


def main():
    have = {n.split("_", 1)[1] for f in ("catalog_v4_pp_test.yaml", "catalog_v4_ee_test.yaml") for n in names(f)}
    rates = {}
    for f in ("lhe_rates.json", "lhe_rates_dsh_ee4.json"):
        rates.update(json.load(open(os.path.join(DATA, f))))
    by_base = {}
    for k, r in rates.items():
        by_base.setdefault(k.split("_", 1)[1], []).append(r)

    def cost(r):
        per_run = r["madevent_s"] * r["mg_cores"] / max(r["n_runs"], 1)
        return max(r["core_s_per_event"] * N_TEST * SAFETY, per_run)

    med = {}
    for k, r in rates.items():
        med.setdefault((k.startswith("ee4"), r["nfinal"]), []).append(cost(r))
    out = {"axes": {}, "missing": {}}
    tot = 0.0
    for ax in HELD:
        procs = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{ax}.yaml")))["processes"]
                 if p["name"].startswith("v4_")]
        miss = [p for p in procs if p["name"][3:] not in have]
        cs = []
        for p in miss:
            base = p.get("base", p["name"])[3:].replace("_nlo", "").split("__")[0]
            if base in by_base:
                c, how = max(cost(r) for r in by_base[base]), "base pool"
            else:
                ee = base.startswith("epem")
                nf = len(base.split("_")[1]) and max(n for e, n in med if e == ee)
                c, how = statistics.median(med[(ee, nf)]), f"median {'ee' if ee else 'pp'} nfinal {nf}"
            cs.append(c)
            out["missing"][p["name"]] = {"axis": ax, "core_s": round(c, 1), "from": how}
        out["axes"][ax] = {"classes": len(procs), "with_default_pool": len(procs) - len(miss), "missing": len(miss),
                           "core_h_missing": round(sum(cs) / 3600, 2), "max_core_s": round(max(cs), 1) if cs else 0}
        tot += sum(cs)
    out["total"] = {"classes": sum(a["classes"] for a in out["axes"].values()),
                    "with_default_pool": sum(a["with_default_pool"] for a in out["axes"].values()),
                    "missing": sum(a["missing"] for a in out["axes"].values()), "core_h": round(tot / 3600, 2)}
    json.dump(out, open(os.path.join(DATA, "heldout_default_pools.json"), "w"), indent=1)
    print(json.dumps({"axes": out["axes"], "total": out["total"]}))


if __name__ == "__main__":
    main()
