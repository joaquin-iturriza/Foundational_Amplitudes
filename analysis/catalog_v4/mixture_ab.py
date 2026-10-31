#!/usr/bin/env python3
"""Steps 11 and 14 of plan entry catalog_coverage under the design passed 2026-10-30T17:29 (plan show
catalog_coverage): the pure-default bound (step 11) and the mixture A/B (step 14), read on per-probe zero-shot MSEs.

  extract <run id> <log> [<axis>]   the axis's ZERO_SHOT records of a job log (site logs -n 3000000 <run>: a
                                    het run's log is ~506k lines) ->
                                    data/reads/<run>_<axis>.json {probe: mse_logm2_real}; a record with an error, a
                                    non-finite MSE or a physics status other than ok/allowed is listed apart
  bound                             step 11 -> data/pp_bound.json
  heldout_default                   step 14's held-out classes on default-sampled events (PP4/EE4 probes of the
                                    held-out classes) -> data/mixture_heldout_default.json
  mixture [--partial]               step 14 -> data/mixture_ab.json, only when every read of the design is in;
                                    --partial prints what the reads in hand give (a missing seed left out, no
                                    verdict, nothing written)
  mixture --het                     step 15 (design passed 2026-10-31T04:56, question 1): the same comparison on the
                                    het-loss runs (MIX_HET, control v4_het_control) -> data/mixture_ab_het.json,
                                    written with what is read: every read or probe not in hand is listed as missing
                                    (never dropped), the verdict only when nothing is missing; beside it het_over_mse,
                                    the het control pair over the MSE control pair on each read (what the het loss
                                    costs mu, information)
  heldout_default [--het]           with --het the het runs, missing reads listed -> data/mixture_heldout_default_het.json

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
# the runs of step 15 (design passed 2026-10-31T04:56): the het-loss twins of MIX, every read from the run's own job
# (its sigma_stage parent; mu frozen there, so mu is the best checkpoint by val_loss_no_reg). The four share arms were
# cut by their limit inside PP4 (772/1055 of 1301 for 466656b2/f0156d88, 908/1129 for 9897833d/df858d33) before EE4:
# those reads wait on their eval-only REREAD (plan step 20) and are reported missing until then
MIX_HET = {"default50": {r: ("FA-ccin2p3-466656b2", "FA-ccin2p3-9897833d") for r in ("PP4", "EE4", "held")},
           "default25": {r: ("FA-ccin2p3-f0156d88", "FA-ccin2p3-df858d33") for r in ("PP4", "EE4", "held")},
           "control": {r: ("FA-ccin2p3-87ea24d0", "FA-ccin2p3-321a3de2") for r in ("PP4", "EE4", "held")}}
# the HD4 eval-only REREADs of the het runs (step 18's 114 default-sampled held-out pools; plan step 20), (seed 42, seed
# 43) per arm, filled when they are submitted; until then the default-sampled held-out read is H1-H3's PP4/EE4 probes
# only and mixture --het reports itself incomplete
HD4_HET = {}


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


def _reads(table):
    """{arm: {read: (seed 42 reads, seed 43 reads)}} and the (arm, read, run) missing, as strings"""
    reads = {a: {r: tuple(read_of(run, r) for run in runs) for r, runs in v.items()} for a, v in table.items()}
    missing = sorted(f"{a} {r} {table[a][r][i]}" for a, v in reads.items() for r, pair in v.items()
                     for i, s in enumerate(pair) if s is None)
    return reads, missing


def _probes_missing(pair, ctl_pair):
    """Per seed of an arm, the control's probes that seed has no read of (a read cut by the job's limit): a count and
    the first names, so a partial read is reported as such, never as complete"""
    ref = set.union(*(set(x) for x in ctl_pair if x is not None)) if any(x is not None for x in ctl_pair) else set()
    out = {}
    for i, x in enumerate(pair):
        m = sorted((str(k) for k in ref - set(x or {})))
        if m:
            out[("seed42", "seed43")[i]] = {"n": len(m), "of": len(ref), "first": m[:10]}
    return out


def mixture(partial=False, het=False):
    cls = class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    table = MIX_HET if het else MIX
    reads, missing = _reads(table)
    if missing and not (partial or het):
        sys.exit("mixture: reads missing (extract them first; --partial for what is in hand): " + "; ".join(missing))
    c = reads["control"]
    if any(s is None for pair in c.values() for s in pair):
        sys.exit("mixture: a control read is missing: " + "; ".join(missing))
    design = "plan catalog_coverage step 15, passed 2026-10-31T04:56 (het loss)" if het else \
        "plan catalog_coverage, passed 2026-10-30T17:29"
    out = {"design": design, "runs": table, "missing_reads": missing, "probes_missing": {},
           "rule": f"GAIN R <= {GAIN} with the widened interval below 1 on PP4 (all classes) and EE4 (all); NO LOSS "
                   f"the pooled held-out widened upper end below {LOSS}; LOSS pooled R >= {LOSS} with the widened "
                   f"interval above 1; WIN = GAIN and NO LOSS" + (
                       "; het: the held-out cost on default-sampled events (held_default: the PP4/EE4 probes of the "
                       "held-out classes, and HD4 once read), the catalog-sampled read (held) beside" if het else ""),
           "shares": {}}
    for a in ("default50", "default25"):
        res = {}
        for r in ("PP4", "EE4", "held"):
            pm = _probes_missing(reads[a][r], c[r])
            if pm:
                out["probes_missing"][f"{a} {r}"] = pm
            arm = side(reads[a][r])
            ctl = side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"all": keys,
                     "trained": [k for k in keys if cls(probe_name(k)) in trained],
                     "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
            if r == "held":
                parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            res[r]["seeds_read"] = [run for run, s in zip(table[a][r], reads[a][r]) if s is not None]
            res[r]["probes_unmatched"] = sorted(str(k) for k in set(arm) ^ set(ctl))[:50]
            res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
        if not all(k in res[r] for r, k in (("PP4", "all"), ("EE4", "all"), ("held", "pooled"))):
            res.update(gain=None, cost=None, wins=None)      # a read with no seed in hand
            out["shares"][a] = res
            continue
        pp, ee, h = res["PP4"]["all"], res["EE4"]["all"], res["held"]["pooled"]
        if het:
            # the design (04:56) and D227: the cost is read on default-sampled held-out events, the catalog-sampled
            # held-out read (res["held"]) beside it as information
            hd, _ = _held_default(reads, a, cls, _held_axes(cls))
            res["held_default"] = hd
            h = hd.get("pooled")
            if h is None:
                res.update(gain=None, cost=None, wins=None)
                out["shares"][a] = res
                continue
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
    complete = not missing and not out["probes_missing"]
    if het:
        # the default-sampled held-out pool is complete only with HD4 (step 18's 114 classes) read on every side
        hd_missing = [f"{a} HD4" for a in table if not HD4_HET.get(a)] + \
            [f"{a} HD4 {run}" for a, runs in HD4_HET.items() for run in runs if read_of(run, "HD4") is None]
        out["held_default_missing"] = hd_missing
        complete = complete and not hd_missing
    if partial and not het:
        out["verdict_if_complete"] = v if not missing else None
        out["would_be"] = v
        print(json.dumps(out, indent=1))
        return
    out["complete"] = complete
    out["verdict"] = v if complete else None
    if het:
        out["verdict_on_reads_in_hand"] = v                 # information only while reads are missing
        out["het_over_mse"] = het_over_mse(cls, trained)
        out["het_over_mse_note"] = "het control over MSE control: loss HETEROSC + sigma stage AND sign_head off together"
        path = os.path.join(DATA, "mixture_ab_het.json")
    else:
        out["bound"] = json.load(open(os.path.join(DATA, "pp_bound.json")))["trained"]
        path = os.path.join(DATA, "mixture_ab.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out))   # the whole result, the last JSON line (tools/verify_claims.py reads its claims from it)


def het_over_mse(cls, trained):
    """What the het setup costs or gains mu (information, design 04:56): the het control pair (MIX_HET) over the MSE
    control pair (MIX) on each read. Two things differ between them, the loss (HETEROSC plus its sigma stage) and
    sign_head (off in the het runs, on in the MSE runs): the ratio is the two together, not the loss alone, R as in mixture with R_seed from the MSE control's seeds"""
    h, _ = _reads({"control": MIX_HET["control"]})
    m, _ = _reads({"control": MIX["control"]})
    out = {}
    for r in ("PP4", "EE4", "held"):
        hp, mp = h["control"][r], m["control"][r]
        if any(x is None for x in hp + mp):
            out[r] = None
            continue
        arm, ctl = side(hp), side(mp)
        keys = sorted(set(arm) & set(ctl), key=str)
        parts = {"all": keys,
                 "trained": [k for k in keys if cls(probe_name(k)) in trained],
                 "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
        if r == "held":
            parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
        out[r] = {n: compare(arm, ctl, mp[0], mp[1], ks, cls) for n, ks in parts.items() if ks}
        out[r]["het_seed_R"] = compare(hp[1], hp[0], mp[0], mp[1], keys, cls)["R"]   # het control seed 43 over 42
        out[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
    return out


def _held_axes(cls):
    """catalog_v4 class -> its held-out axis (recipes/catalog_v4_heldout_<axis>.yaml)"""
    ax_of = {}
    for ax in HELD:
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{ax}.yaml")))["processes"]:
            if p["name"].startswith("v4_"):
                ax_of[cls(p["name"])] = ax
    return ax_of


def _held_default(reads, a, cls, ax_of, hd4=None):
    """R of arm a over the control on the default-sampled events of the held-out classes: the PP4/EE4 probes whose
    class is held out, plus HD4 (hd4 = {arm: (run 42, run 43)}, the het default HD4_HET) where read; per axis and
    pooled, with how many probes the control has and how many were read on every side"""
    hd4 = HD4_HET if hd4 is None else hd4
    tag = lambda r, d: {(r, k): x for k, x in (d or {}).items()}
    arm, ctl, c42, c43 = {}, {}, {}, {}
    srcs = [(r, reads[a][r], reads["control"][r]) for r in ("PP4", "EE4")]
    hd4_missing = [f"{x} HD4" for x in (a, "control") if not hd4.get(x)]
    if not hd4_missing:
        ap = tuple(read_of(x, "HD4") for x in hd4[a])
        cp = tuple(read_of(x, "HD4") for x in hd4["control"])
        hd4_missing = [f"{x} HD4 {run}" for x, runs, rd in ((a, hd4[a], ap), ("control", hd4["control"], cp))
                       for run, y in zip(runs, rd) if y is None]
        if not hd4_missing:                   # HD4 only with all four seeds read, never on one seed
            srcs.append(("HD4", ap, cp))
    used = []
    for r, ap, cp in srcs:
        if any(x is None for x in cp):
            continue
        used.append(r)
        arm.update(tag(r, side(ap))); ctl.update(tag(r, side(cp)))
        c42.update(tag(r, cp[0])); c43.update(tag(r, cp[1]))
    held_ctl = [k for k in ctl if cls(k[1]) in ax_of]
    keys = sorted((k for k in set(arm) & set(ctl) if cls(k[1]) in ax_of), key=str)
    res = {"sources": used, "hd4_missing": hd4_missing, "n_probes_control": len(held_ctl), "n_probes_read": len(keys),
           "probes_missing": sorted(str(k) for k in set(held_ctl) - set(keys))[:20]}
    if keys:
        res["pooled"] = compare(arm, ctl, c42, c43, keys, cls)
    for ax in HELD:
        ks = [k for k in keys if ax_of[cls(k[1])] == ax]
        if ks:
            res[ax] = compare(arm, ctl, c42, c43, ks, cls)
    return res, keys


def heldout_default(het=False):
    """The held-out classes read on default-sampled events (Joaquin, 2026-10-31: the check is against data generated
    the default way): the PP4 and EE4 probes (MadEvent default, test split) whose class is in a held-out axis
    (recipes/catalog_v4_heldout_<axis>.yaml), per axis and pooled, R as in mixture (arm over control, R_seed of the
    same probes), beside the catalog-sampled held-out read on the same classes and on the classes with no
    default-sampled pool. -> data/mixture_heldout_default.json. With het, the het runs (MIX_HET, HD4 from HD4_HET
    once read): a missing read or probe is listed (missing_reads, probes_missing) and the classes read are the ones
    every side has -> data/mixture_heldout_default_het.json"""
    cls = class_of()
    ax_of = _held_axes(cls)
    table = MIX_HET if het else MIX
    reads, missing = _reads(table)
    # every seed of every read this uses, as mixture() without --partial: a missing seed would leave R on one seed
    if missing and not het:
        sys.exit("heldout_default: reads missing (extract them first): " + "; ".join(missing))
    if any(x is None for pair in reads["control"].values() for x in pair):
        sys.exit("heldout_default: a control read is missing: " + "; ".join(missing))
    out = {"design": "plan catalog_coverage step 15 (het), passed 2026-10-31T04:56" if het else
           "plan catalog_coverage step 14, REREAD on default-sampled held-out events",
           "runs": table, "missing_reads": missing, "shares": {}}
    for a in ("default50", "default25"):
        d, keys = _held_default(reads, a, cls, ax_of, None if het else {})
        if not het:          # the MSE file's layout as written at 744fad42
            d = {k: v for k, v in d.items() if k in ("pooled",) + HELD}
        res = {"default": d}
        have = {cls(k[1]) for k in keys}
        harm, hctl = side(reads[a]["held"]), side(reads["control"]["held"])
        hk = sorted(set(harm) & set(hctl), key=str)
        hc = reads["control"]["held"]
        same = [k for k in hk if cls(k[1]) in have]
        res["catalog_sampled"] = {
            "same_classes": compare(harm, hctl, hc[0], hc[1], same, cls) if same else None,
            "no_default_pool": compare(harm, hctl, hc[0], hc[1], [k for k in hk if cls(k[1]) not in have], cls)}
        res["axes_without_default_pool"] = sorted({ax for c, ax in ax_of.items() if c not in have})
        out["shares"][a] = res
    name = "mixture_heldout_default_het.json" if het else "mixture_heldout_default.json"
    json.dump(out, open(os.path.join(DATA, name), "w"), indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run"); e.add_argument("log"); e.add_argument("axis", nargs="?", default="PP4")
    e.add_argument("--allow-unchecked", action="store_true")
    sub.add_parser("bound")
    hd = sub.add_parser("heldout_default")
    hd.add_argument("--het", action="store_true")
    m = sub.add_parser("mixture")
    m.add_argument("--partial", action="store_true")
    m.add_argument("--het", action="store_true")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.run, a.log, a.axis, a.allow_unchecked)
    elif a.cmd == "mixture":
        mixture(a.partial, a.het)
    elif a.cmd == "heldout_default":
        heldout_default(a.het)
    else:
        bound()


if __name__ == "__main__":
    main()
