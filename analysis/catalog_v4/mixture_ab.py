#!/usr/bin/env python3
"""Steps 11 and 14 of plan entry catalog_coverage under the design passed 2026-10-30T17:29 (plan show
catalog_coverage): the pure-default bound (step 11) and the mixture A/B (step 14), read on per-probe zero-shot MSEs.

  extract <run id> <log> [<axis>]   the axis's ZERO_SHOT records of a job log (site logs -n 400000 <run>) ->
                                    data/reads/<run>_<axis>.json {probe: mse_logm2_real}; a record with an error, a
                                    non-finite MSE or a physics status other than ok/allowed is listed apart
  bound                             step 11 -> data/pp_bound.json

The measure (the design's): per probe the MSE of log|M|^2 at the best checkpoint (tools/zero_shot_parent.py,
mse_logm2_real); each side the geometric mean of its two seeds (42, 43); per class (analysis/catalog_v4/
census_classes.json, a probe's class is its catalog_v4 tree's, pp13v4_X and pp13_X both v4_X) the geometric mean of
its probes' arm/control ratios; R = the median over classes; its 95% class bootstrap interval (10000 resamples, seed 0)
widened on each side by |ln R_seed|, R_seed the median over the same classes of control seed 43 over seed 42.
Step 11, on PP4 (recipes/catalog_v4_pp_test.yaml, test split), the classes the bound arm trains on pp events
(a probe of recipes/catalog_v4_pp_mixed_train.yaml in the class) apart from the others:
  YES  R <= 0.90 with the widened interval below 1;  NO  R >= 0.97 or the widened interval contains 1;
  otherwise a small gain, with its interval. The held-out classes' R is reported (expected: a cost or nothing).
"""
import argparse, json, math, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "reads")
# the runs of step 11 (plan catalog_coverage): PP4 reads at the best checkpoint
BOUND = {"arm": ("FA-ccin2p3-a152d7d0", "FA-ccin2p3-a01ad665"),        # REREAD afc96598 (seed 42), d26f13aa (seed 43)
         "control": ("FA-ccin2p3-04253ff7", "FA-ccin2p3-e7e32847")}    # control 339ec100 (42), 380842ed (43) read in eval
YES, NO = 0.90, 0.97


def extract(run, log, axis="PP4"):
    tag = f"{axis} ZERO_SHOT "
    good, bad = {}, {}
    for line in open(log, errors="replace"):
        if not line.startswith(tag):
            continue
        d = json.loads(line[len(tag):])
        p = d.get("probe") or "?"
        x = d.get("mse_logm2_real")
        st = (d.get("physics") or {}).get("status", "unchecked")
        if "error" in d:
            bad[p] = "error: " + str(d["error"])[:200]
        elif x is None or not math.isfinite(x):
            bad[p] = f"non-finite mse {x}"
        elif st not in ("ok", "allowed"):
            bad[p] = f"physics {st}"
        else:
            good[p] = float(x)
    os.makedirs(READS, exist_ok=True)
    out = os.path.join(READS, f"{run}_{axis}.json")
    json.dump({"run": run, "axis": axis, "log_lines_from": os.path.basename(log), "mse": good, "excluded": bad},
              open(out, "w"), indent=0, sort_keys=True)
    print(json.dumps({"out": out, "n": len(good), "excluded": len(bad)}))


def load(run, axis="PP4"):
    return json.load(open(os.path.join(READS, f"{run}_{axis}.json")))["mse"]


def class_of():
    c = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    m = {}
    for rep, members in c.items():
        for x in members:
            m[x] = rep
    return lambda probe: m.get("v4_" + probe.split("_", 1)[1], "v4_" + probe.split("_", 1)[1])


def gmean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def class_ratios(num, den, probes, cls):
    by = {}
    for p in probes:
        by.setdefault(cls(p), []).append(num[p] / den[p])
    return {k: gmean(v) for k, v in by.items()}


def r_with_interval(ratios, r_seed, n_boot=10000, seed=0):
    v = np.array(sorted(ratios.values()))
    r = float(np.median(v))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    w = abs(math.log(r_seed))
    return {"R": round(r, 4), "n_classes": len(v), "boot95": [round(float(lo), 4), round(float(hi), 4)],
            "widened": [round(float(lo) * math.exp(-w), 4), round(float(hi) * math.exp(w), 4)],
            "frac_classes_below_1": round(float(np.mean(v < 1)), 4)}


def bound():
    a42, a43 = (load(r) for r in BOUND["arm"])
    c42, c43 = (load(r) for r in BOUND["control"])
    probes = sorted(set(a42) & set(a43) & set(c42) & set(c43))
    arm = {p: math.sqrt(a42[p] * a43[p]) for p in probes}
    ctl = {p: math.sqrt(c42[p] * c43[p]) for p in probes}
    train = {p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_pp_mixed_train.yaml")))["processes"]}
    cls = class_of()
    trained_cls = {cls(p) for p in probes if p in train}
    out = {"design": "plan catalog_coverage, passed 2026-10-30T17:29", "runs": BOUND, "n_probes": len(probes),
           "probes_missing": sorted((set(a42) | set(a43) | set(c42) | set(c43)) - set(probes))[:50]}
    for part, sel in (("trained", lambda p: cls(p) in trained_cls), ("held_out", lambda p: cls(p) not in trained_cls)):
        ps = [p for p in probes if sel(p)]
        seed_r = class_ratios(c43, c42, ps, cls)
        r_seed = float(np.median(list(seed_r.values())))
        res = r_with_interval(class_ratios(arm, ctl, ps, cls), r_seed)
        res.update(n_probes=len(ps), R_seed=round(r_seed, 4),
                   arm_seed_R=round(float(np.median(list(class_ratios(a43, a42, ps, cls).values()))), 4))
        out[part] = res
    t = out["trained"]
    lo, hi = t["widened"]
    if t["R"] <= YES and hi < 1:
        v = "YES"
    elif t["R"] >= NO or lo <= 1 <= hi:
        v = "NO"
    else:
        v = "small gain"
    out["verdict"] = v
    out["rule"] = f"YES R <= {YES} with the widened interval below 1; NO R >= {NO} or the interval contains 1"
    path = os.path.join(DATA, "pp_bound.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run"); e.add_argument("log"); e.add_argument("axis", nargs="?", default="PP4")
    sub.add_parser("bound")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.run, a.log, a.axis)
    else:
        bound()


if __name__ == "__main__":
    main()
