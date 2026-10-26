#!/usr/bin/env python
"""Audit of every catalog_v3 pool generated so far (D102: stages 1, 2 and 3), before any training reads them.
Per recipe entry and role (train in $SCRATCH/amp_data_cache, val and test in $DATA_DIR, as the prebuild writes them):
  rows      the row count against the recipe's n_<role>
  amp       the fraction of |M|^2 non-finite, zero and negative; log10|M|^2 at 0.1%, 50%, 99.9% and its spread
  ids       whether every row carries the same PDG ids
  momenta   on 2000 rows, the largest |sum in - sum out| over sqrt(s_hat), per component
  splits    train against val: the difference of the medians of log10|M|^2 over the val spread (the three roles sample
            one distribution, CLAUDE.md "Every role of a process samples the same distribution")
A pool is flagged when it is missing, short, has a non-finite or zero |M|^2, ids that change, momentum violation above
1e-6, a log10|M|^2 spread below 1e-3 (constant), or train and val medians apart by more than 0.25 val spreads.
    python analysis/catalog_v3/pool_audit.py [--workers 8] [--stages stage1 stage2 stage3] [--out analysis/catalog_v3/pool_audit.json]
"""
import argparse, glob, json, os
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGES = {"stage1": "recipes/catalog_v3_gen_stage1.yaml", "stage2": "recipes/catalog_v3_gen_stage2.yaml",
          "stage3": "recipes/catalog_v3_gen_stage3.yaml",
          # the one-loop sign transfer's extension (gen_catalog_v3.py --write-loops-signext); audited on its own
          # (--stages signext --out analysis/catalog_v3/pool_audit_signext.json)
          "signext": "recipes/catalog_v3_gen_signext.yaml",
          # catalog_v4 (D130), audited on its own (--stages v4_stage1 v4_stage2 v4_stage3 v4_stage4 --out ...): the same
          # checks plus the v4 kinematics, b legs massless (the five-flavour scheme) and sqrt(s_hat) inside the window
          **{f"v4_stage{k}": f"recipes/catalog_v4_gen_stage{k}.yaml" for k in (1, 2, 3, 4)}}
B_MASS_TOL = 1e-2      # GeV: a massless leg's sqrt|E^2 - p^2| in float64 at 20 TeV is ~1e-5 GeV, a 4.7 GeV b is far above
ROLES = ("train", "val", "test")


def find(name, sq, role):
    d = os.path.join(os.environ["SCRATCH"], "amp_data_cache") if role == "train" else os.environ["DATA_DIR"]
    g = sorted(glob.glob(os.path.join(d, f"{name}_{sq[0]}-{sq[1]}GeV_{role}_*amplitudes.npy")))
    return g[0] if g else None


def one(args):
    stage, e = args
    out = {"stage": stage}
    for role in ROLES:
        f = find(e["name"], e["sqrts"], role)
        if f is None:
            out[role] = {"missing": True}
            continue
        a = np.load(f, mmap_mode="r")
        n = (a.shape[1] - 1) // 5
        amp = np.asarray(a[:, -1], dtype=np.float64)
        fin = np.isfinite(amp)
        la = np.log10(np.abs(amp[fin & (amp != 0)]))
        k = min(2000, len(a))
        head = np.asarray(a[:k], dtype=np.float64)
        P = head[:, : 4 * n].reshape(k, n, 4)
        ids = np.asarray(a[:, 4 * n: 5 * n])
        q = P[:, 0] + P[:, 1]
        sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 1e-300, None))
        viol = np.abs(q - P[:, 2:].sum(1)).max(1) / sq
        out[role] = {"file": os.path.basename(f), "rows": int(len(a)), "expected": int(e.get(f"n_{role}", 0)),
                     "nonfinite": float(1 - fin.mean()), "zero": float(np.mean(amp == 0)),
                     "negative": float(np.mean(amp < 0)),
                     "q": [float(x) for x in np.quantile(la, [1e-3, 0.5, 1 - 1e-3])] if len(la) else None,
                     "spread": float(la.std()) if len(la) else 0.0,
                     "ids_constant": bool((ids == ids[0]).all()), "pdg": [int(c) for c in ids[0]],
                     "mom_viol": float(viol.max())}
        if stage.startswith("v4_"):
            m2 = P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1)
            isb = np.abs(ids[0]) == 5
            out[role]["b_mass_max"] = float(np.sqrt(np.abs(m2[:, isb])).max()) if isb.any() else None
            out[role]["sqrts_range"] = [float(sq.min()), float(sq.max())]
    fl = []
    for role in ROLES:
        r = out[role]
        if r.get("missing"):
            fl.append(f"{role} missing"); continue
        if r["rows"] < r["expected"]:
            fl.append(f"{role} {r['rows']}<{r['expected']} rows")
        if r["nonfinite"] or r["zero"]:
            fl.append(f"{role} nonfinite {r['nonfinite']:.2e} zero {r['zero']:.2e}")
        if not r["ids_constant"]:
            fl.append(f"{role} ids change")
        if r["mom_viol"] > 1e-6:
            fl.append(f"{role} momentum {r['mom_viol']:.1e}")
        if r["spread"] < 1e-3:
            fl.append(f"{role} constant |M|^2")
        if (r.get("b_mass_max") or 0.0) > B_MASS_TOL:
            fl.append(f"{role} b mass {r['b_mass_max']:.3g} GeV")
        if r.get("sqrts_range") and not (e["sqrts"][0] * (1 - 1e-9) <= r["sqrts_range"][0]
                                         and r["sqrts_range"][1] <= e["sqrts"][1] * (1 + 1e-9)):
            fl.append(f"{role} sqrt(s_hat) {r['sqrts_range']} outside {e['sqrts']}")
    t, v = out["train"], out["val"]
    if not t.get("missing") and not v.get("missing") and t["q"] and v["q"] and v["spread"] > 0:
        out["split_shift"] = abs(t["q"][1] - v["q"][1]) / v["spread"]
        if out["split_shift"] > 0.25:
            fl.append(f"train/val medians {out['split_shift']:.2f} spreads apart")
    out["flags"] = fl
    return e["name"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis/catalog_v3/pool_audit.json"))
    ap.add_argument("--stages", nargs="+", default=["stage1", "stage2", "stage3"], choices=list(STAGES))
    a = ap.parse_args()
    a.out = os.path.expandvars(a.out)       # a path handed through the job's environment ($SCRATCH/...) arrives unexpanded
    stages = {s: STAGES[s] for s in a.stages}
    jobs = [(s, e) for s, r in stages.items() for e in yaml.safe_load(open(os.path.join(ROOT, r)))["processes"]]
    res = {}
    with Pool(a.workers) as pl:
        for name, r in pl.imap_unordered(one, jobs, chunksize=4):
            res[name] = r
            if r["flags"]:
                print("FLAG", r["stage"], name, "; ".join(r["flags"]), flush=True)
    summ = {}
    for s in stages:
        rs = [r for r in res.values() if r["stage"] == s]
        summ[s] = {"entries": len(rs), "flagged": sum(bool(r["flags"]) for r in rs),
                   "pools_present": sum(not r[x].get("missing") for r in rs for x in ROLES),
                   "signed": sum(r["val"].get("negative", 0) > 0 for r in rs),
                   "max_mom_viol": max((r[x]["mom_viol"] for r in rs for x in ROLES if not r[x].get("missing")),
                                       default=None)}
        print("SUMMARY", s, json.dumps(summ[s]), flush=True)
    json.dump({"summary": summ, "pools": res}, open(a.out, "w"), indent=0)
    for name, r in sorted(res.items()):   # one line per entry, for a reader without the cluster's file
        print("POOL", name, json.dumps({**{x: {k: r[x].get(k) for k in ("rows", "negative", "missing")} for x in ROLES},
                                        "flags": r["flags"]}), flush=True)


if __name__ == "__main__":
    main()
