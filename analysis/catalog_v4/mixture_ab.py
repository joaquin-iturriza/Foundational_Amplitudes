#!/usr/bin/env python3
"""Steps 11 and 14 of plan entry catalog_coverage under the design passed 2026-10-30T17:29 (plan show
catalog_coverage): the pure-default bound (step 11) and the mixture A/B (step 14), read on per-probe zero-shot MSEs.

  extract <run id> <log> [<axis>]   the axis's ZERO_SHOT records of a job log (site logs -n 400000 <run>) ->
                                    data/reads/<run>_<axis>.json {probe: mse_logm2_real}; a record with an error, a
                                    non-finite MSE or a physics status other than ok/allowed is listed apart
  bound                             step 11 -> data/pp_bound.json
  mixture [--partial]               step 14 -> data/mixture_ab.json, only when every read of the design is in;
                                    --partial prints what the reads in hand give (a missing seed left out, no
                                    verdict, nothing written)

The measure (the design's): per probe the MSE of log|M|^2 at the best checkpoint (tools/zero_shot_parent.py,
mse_logm2_real); each side the geometric mean of its two seeds (42, 43); per class (analysis/catalog_v4/
census_classes.json, a probe's class is its catalog_v4 tree's, pp13v4_X and pp13_X both v4_X) the geometric mean of
its probes' arm/control ratios; R = the median over classes; its 95% class bootstrap interval (10000 resamples, seed 0)
widened on each side by |ln R_seed|, R_seed the median over the same classes of control seed 43 over seed 42.
Step 11, on PP4 (recipes/catalog_v4_pp_test.yaml, test split), the classes the bound arm trains on pp events
(a probe of recipes/catalog_v4_pp_mixed_train.yaml in the class) apart from the others:
  YES  R <= 0.90 with the widened interval below 1;  NO  R >= 0.97 or the widened interval contains 1;
  otherwise a small gain, with its interval. The held-out classes' R is reported (expected: a cost or nothing).
Step 14, per default share (0.50, 0.25), arm over the catalog_v4 control on three reads: PP4 on all its 1301 classes
(the classes catalog_v4 trains, recipes/catalog_v4_train.yaml, 797, and the others also reported apart: the design says
"on PP4", reporting the parts apart), EE4 on its 169 probes (split the same way), and the held-out H1..H7c pooled (catalog_v4 probes only, v4_*; each axis
reported as information), with R_seed taken per read:
  GAIN     R <= 0.90 with the widened interval below 1, on PP4 and on EE4
  NO LOSS  the pooled held-out widened interval's upper end below 1.05
  LOSS     the pooled held-out R >= 1.05 with the widened interval above 1;  otherwise the cost is undecided
A share WINS on GAIN and NO LOSS. Verdict: one winner -> it; both -> 0.25 unless 0.50's PP4 and EE4 widened intervals
both lie below 0.25's R on the same read (then 0.50); no winner -> 'keep the current mixture' when no share has GAIN,
else 'trade-off' (a GAIN with a LOSS or an undecided cost: the numbers go to Joaquin with the step-11 bound beside them).
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
# the runs of step 14, (seed 42, seed 43) per read; PP4 and EE4 from the eval re-reads (the arms were cut at their
# 2.5 h limit before those reads), the held-out axes from the arms' and the control's own jobs
HELD = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
MIX = {"default50": {"PP4": ("FA-ccin2p3-8e1cc529", "FA-ccin2p3-52efacaf"),   # REREAD 6c246603, 1d46b938
                     "EE4": ("FA-ccin2p3-8e1cc529", "FA-ccin2p3-52efacaf"),
                     "held": ("FA-ccin2p3-6c246603", "FA-ccin2p3-1d46b938")},
       "default25": {"PP4": ("FA-ccin2p3-7a6a1967", "FA-ccin2p3-897aa5d4"),   # REREAD 11ad944a, 9c6fc7ab
                     "EE4": ("FA-ccin2p3-7a6a1967", "FA-ccin2p3-897aa5d4"),
                     "held": ("FA-ccin2p3-11ad944a", "FA-ccin2p3-9c6fc7ab")},
       "control": {"PP4": ("FA-ccin2p3-04253ff7", "FA-ccin2p3-e7e32847"),     # REREAD 339ec100, 380842ed
                   "EE4": ("FA-ccin2p3-c1863adf", "FA-ccin2p3-def11c5c"),     # REREAD 339ec100, 380842ed
                   "held": ("FA-ccin2p3-339ec100", "FA-ccin2p3-380842ed")}}
GAIN, LOSS = 0.90, 1.05


def extract(run, log, axis="PP4", allow_unchecked=False):
    """allow_unchecked: a read made before the in-job physics check (the control's own held-out reads, 339ec100 and
    380842ed) is kept with status 'unchecked'; used only where the arms read the same pools with the check ok (same
    recipes/catalog_v4_heldout_*.yaml, n_val equal on all 701 probes, var_logm2 within the arm-to-arm spread)."""
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
        elif st not in ("ok", "allowed") and not (allow_unchecked and st == "unchecked"):
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


def read_of(run, read):
    """{probe: mse} of one run on one read (the held-out axes pooled, catalog_v4 probes only: a control job of the
    shared-folder era also read catalog_v3's v3_* probes there), or None when a file is missing."""
    axes = HELD if read == "held" else (read,)
    out = {}
    for ax in axes:
        f = os.path.join(READS, f"{run}_{ax}.json")
        if not os.path.exists(f):
            return None
        out.update({(ax, p) if read == "held" else p: x for p, x in json.load(open(f))["mse"].items()
                    if read != "held" or p.startswith("v4_")})
    return out


def probe_name(k):
    return k[1] if isinstance(k, tuple) else k


def side(pair):
    """Geometric mean over the seeds read; the seeds present."""
    have = [s for s in pair if s is not None]
    keys = set.intersection(*(set(s) for s in have)) if have else set()
    return {k: gmean([s[k] for s in have]) for k in keys}


def compare(arm, ctl, c42, c43, keys, cls):
    cl = lambda k: cls(probe_name(k))
    seed_r = float(np.median(list(class_ratios(c43, c42, keys, cl).values())))
    res = r_with_interval(class_ratios(arm, ctl, keys, cl), seed_r)
    res.update(n_probes=len(keys), R_seed=round(seed_r, 4))
    return res


def mixture(partial=False):
    cls = class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    reads = {a: {r: tuple(read_of(run, r) for run in runs) for r, runs in v.items()} for a, v in MIX.items()}
    missing = sorted(f"{a} {r} {MIX[a][r][i]}" for a, v in reads.items() for r, pair in v.items()
                     for i, s in enumerate(pair) if s is None)
    if missing and not partial:
        sys.exit("mixture: reads missing (extract them first; --partial for what is in hand): " + "; ".join(missing))
    c = reads["control"]
    if any(s is None for pair in c.values() for s in pair):
        sys.exit("mixture: a control read is missing: " + "; ".join(missing))
    out = {"design": "plan catalog_coverage, passed 2026-10-30T17:29", "runs": MIX, "missing_reads": missing,
           "rule": f"GAIN R <= {GAIN} with the widened interval below 1 on PP4 (all classes) and EE4 (all); NO LOSS "
                   f"the pooled held-out widened upper end below {LOSS}; LOSS pooled R >= {LOSS} with the widened "
                   f"interval above 1; WIN = GAIN and NO LOSS", "shares": {}}
    for a in ("default50", "default25"):
        res = {}
        for r in ("PP4", "EE4", "held"):
            arm = side(reads[a][r])
            ctl = side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"all": keys,
                     "trained": [k for k in keys if cls(probe_name(k)) in trained],
                     "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
            if r == "held":
                parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            res[r]["seeds_read"] = [run for run, s in zip(MIX[a][r], reads[a][r]) if s is not None]
            res[r]["probes_unmatched"] = sorted(str(k) for k in set(arm) ^ set(ctl))[:50]
        if not all(k in res[r] for r, k in (("PP4", "all"), ("EE4", "all"), ("held", "pooled"))):
            res.update(gain=None, cost=None, wins=None)      # --partial with no seed of a read in hand
            out["shares"][a] = res
            continue
        pp, ee, h = res["PP4"]["all"], res["EE4"]["all"], res["held"]["pooled"]
        gain = all(x["R"] <= GAIN and x["widened"][1] < 1 for x in (pp, ee))
        if h["widened"][1] < LOSS:
            cost = "NO LOSS"
        elif h["R"] >= LOSS and h["widened"][0] > 1:
            cost = "LOSS"
        else:
            cost = "undecided"
        res.update(gain=gain, cost=cost, wins=gain and cost == "NO LOSS")
        out["shares"][a] = res
    s50, s25 = out["shares"]["default50"], out["shares"]["default25"]
    if s50["wins"] is None or s25["wins"] is None:
        v = None
    elif s50["wins"] and s25["wins"]:
        below = all(s50[r][n]["widened"][1] < s25[r][n]["R"] for r, n in (("PP4", "all"), ("EE4", "all")))
        v = "default50" if below else "default25"
    elif s50["wins"] or s25["wins"]:
        v = "default50" if s50["wins"] else "default25"
    elif not (s50["gain"] or s25["gain"]):
        v = "keep the current mixture"
    else:
        v = "trade-off"
    if partial:
        out["verdict_if_complete"] = v if not missing else None
        out["would_be"] = v
        print(json.dumps(out, indent=1))
        return
    out["verdict"] = v
    out["bound"] = json.load(open(os.path.join(DATA, "pp_bound.json")))["trained"]
    path = os.path.join(DATA, "mixture_ab.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "missing_reads")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run"); e.add_argument("log"); e.add_argument("axis", nargs="?", default="PP4")
    e.add_argument("--allow-unchecked", action="store_true")
    sub.add_parser("bound")
    m = sub.add_parser("mixture")
    m.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.run, a.log, a.axis, a.allow_unchecked)
    elif a.cmd == "mixture":
        mixture(a.partial)
    else:
        bound()


if __name__ == "__main__":
    main()
