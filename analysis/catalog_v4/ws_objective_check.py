"""warm_start step 11: which quantity a run logs ranks the warm-start restarts as their zero-shot reads do?

The growth sweep's objective (plan warm_start, next step 11) must track the held-out reads; on C1 neither the het
validation NLL (C1t: NLL-tuned restart, reads R 1.19/1.40 worse) nor the validation MSE of the mean (C1m: tuned on
it, reads at par with the untuned restart) did, and on the growth (C10wm) the mean-MSE-tuned recipe read worse.

For each comparison of WARM (mixture_ab.py) between a restart arm and its same-seed reference, per seed:
  target  ln R_seed on PP4 and EE4: the median over classes of the class geometric mean of the probe MSE ratios
          (mixture_ab.class_ratios), arm over reference; the two-seed R with its widened interval from the
          comparison's data/warm_start_<C>.json says whether the comparison is resolved (interval excludes 1)
  candidate  ln(arm/ref) of each logged quantity (lower is better for all):
          from the run's validation curve (data/reads/<run>_val.json, het NLL every 10 steps):
            nll_final (last of the training block), nll_sigma (last of the sigma stage), nll_best, nll_mean (mean over
            the training block), nll_peak (max after the first read: the restart's excursion)
          from the end-of-run per-process metrics in the job log (`extract`, site side; MSE (prepd), L1, L1r,
            Mean |rel err| on the val split of every trained process): geometric mean over processes, all and split
            by initial state (pp: hadronic beams; ee: e+e-)
          from the zero-shot reads of the held-out axes disjoint from PP4/EE4 (HD4: H2p5, H6, H7E, H7c; HD5: H5)
            and the catalog-sampled H1-H3 (the PP4/EE4 classes, other events): median over classes of the ratio
A candidate tracks the reads when, on every resolved comparison, its ln ratio has the target's sign at both seeds.

    python3 analysis/catalog_v4/ws_objective_check.py extract      -> data/ws_objective_logs.json (site, read-only)
    python3 analysis/catalog_v4/ws_objective_check.py rank         -> data/ws_objective_check.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import mixture_ab as mab  # noqa: E402

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(DATA, "ws_objective_logs.json")
OUT = os.path.join(DATA, "ws_objective_check.json")
TARGETS = ("PP4", "EE4")
READ_CANDS = ("HD4", "HD5", "H1", "H2", "H3", "H2p5", "H5", "H6", "H7E", "H7c")

# (name, warm entry, arm, reference key or ("k", K) for a W at another k of the same entry, json key)
COMPARISONS = [
    ("nll_tuned_vs_untuned", "C1t", ("W", 533), ("ref", "untuned"), "W_over_untuned"),
    ("mse_tuned_vs_untuned", "C1m", ("W", 533), ("ref", "untuned"), "W_over_untuned"),
    ("mse_tuned_vs_nll_tuned", "C1m", ("W", 533), ("ref", "nll_tuned"), "W_over_nll_tuned"),
    ("moment_fill_vs_default", "C1f", ("W", 533), ("ref", "default"), "W_over_default"),
    ("muon_damped_vs_default", "C1u", ("W", 533), ("ref", "default"), "W_over_default"),
    ("growth_mse_tuned_vs_default", "C10wm", ("W", 533), ("ref", "untuned"), "W_over_untuned"),
]
# the check set, not used to pick a candidate: retraining length and carrying the change (key None: resolved when both
# seeds' ln R have one sign on PP4 and EE4 and the two-seed mean |ln R| exceeds CHECK_MIN on either)
CHECKS = [
    ("restart_k1066_vs_k533", "C1", ("W", 1066), ("W", 533), None),
    ("restart_k2132_vs_k1066", "C1", ("W", 2132), ("W", 1066), None),
    ("growth_k1066_vs_k533", "C10w", ("W", 1066), ("W", 533), None),
    ("growth_k2132_vs_k1066", "C10w", ("W", 2132), ("W", 1066), None),
    ("change_vs_no_change_k533", "C1", ("W", 533), ("C", 533), None),
    ("change_vs_no_change_k1066", "C1", ("W", 1066), ("C", 1066), None),
    ("change_vs_no_change_k2132", "C1", ("W", 2132), ("C", 2132), None),
]
CHECK_MIN = 0.05


def _runs(entry, where):
    w = mab.WARM[entry]
    if where[0] in ("W", "C"):
        return tuple(w[where[0]][where[1]])
    return tuple(w["ref"][where[1]])


# ---------------------------------------------------------------- site side: the end-of-run metrics in the job logs
SITE = r'''
import glob, json, math, os, re, sys
out = {}
pat = re.compile(r"^\[[^]]*\] (MSE \(prepd\)|L1  \(prepd\)|L1r \(prepd\)|Mean \|rel err\|) val_(\S+) (\S+): ([-0-9.eE+naif]+)$")
for arg in sys.argv[1:]:
    run, sched = arg.split(":")
    logs = [f for d in ("", "*/", "*/*/") for f in glob.glob(d + "*%s*.out" % sched) if ".venv" not in f][:1]
    if not logs:
        out[run] = {"error": "no log for %s" % sched}
        continue
    m = {}
    for line in open(logs[0], errors="replace"):
        x = pat.match(line.rstrip("\n"))
        if x and x.group(2) == x.group(3):
            m.setdefault(x.group(1), {})[x.group(2)] = float(x.group(4))   # the last block wins: the final model
    out[run] = {"log": logs[0], "metrics": m}
print("WSOBJ " + json.dumps(out))
'''


def _sched(run):
    import sqlite3
    c = sqlite3.connect(os.path.expanduser("~/.local/share/ccorch/runs.db"))
    return c.execute("SELECT sched_id FROM runs WHERE id=?", (run,)).fetchone()[0]


def all_runs():
    rs = set()
    for _, e, a, r, _k in COMPARISONS + CHECKS:
        rs.update(_runs(e, a))
        rs.update(_runs(e, r))
    return sorted(rs)


def extract():
    from analysis.common.readers import site_python
    runs = all_runs()
    got = json.load(open(LOGS)) if os.path.exists(LOGS) else {}
    todo = [r for r in runs if r not in got or "error" in got[r] or "L1  (prepd)" not in got[r]["metrics"]]
    for i in range(0, len(todo), 6):
        part = todo[i:i + 6]
        got.update(site_python(SITE, [f"{r}:{_sched(r)}" for r in part], "WSOBJ", timeout=1500))
        json.dump(got, open(LOGS, "w"), indent=0, sort_keys=True)
        print(json.dumps({r: (got[r].get("error") or {k: len(v) for k, v in got[r]["metrics"].items()}) for r in part}))


# ---------------------------------------------------------------- laptop side: the candidates against the reads
def _gm(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


def _is_pp(proc):
    return not proc.startswith("v4_epem")


def val_cands(run):
    d = json.load(open(os.path.join(mab.READS, f"{run}_val.json")))
    tr = [v for _, v in d["blocks"][0]["vals"]]
    fin = [v for v in tr if math.isfinite(v)]
    c = {"nll_final": tr[-1], "nll_best": min(fin), "nll_mean": float(np.mean(np.clip(fin, None, 10.0))),
         "nll_peak": max(fin[1:])}
    if len(d["blocks"]) > 1 and d["blocks"][1]["vals"]:
        c["nll_sigma"] = d["blocks"][1]["vals"][-1][1]
    return c                    # NLL values: compared as differences, not ratios


def log_cands(run, logs):
    r = logs.get(run) or {}
    out = {}
    for name, key in (("mse", "MSE (prepd)"), ("l1", "L1  (prepd)"), ("l1r", "L1r (prepd)"), ("relerr", "Mean |rel err|")):
        per = (r.get("metrics") or {}).get(key) or {}
        if not per:
            continue
        out[f"val_{name}"] = _gm(per.values())
        out[f"val_{name}_pp"] = _gm([v for p, v in per.items() if _is_pp(p)])
        out[f"val_{name}_ee"] = _gm([v for p, v in per.items() if not _is_pp(p)])
    return out


def read_ratio(arm, ref, axis, cls):
    """ln of the median over classes of the class geometric mean of probe MSE ratios, arm over ref (one seed)"""
    try:
        a, b = mab.load(arm, axis), mab.load(ref, axis)
    except FileNotFoundError:
        return None
    probes = sorted(set(a) & set(b))
    if not probes:
        return None
    r = mab.class_ratios(a, b, probes, cls)
    return math.log(float(np.median(list(r.values()))))


def _resolved(entry, key, axis):
    f = os.path.join(DATA, f"warm_start_{entry}.json")
    d = json.load(open(f))["k"]["533"][key][axis]
    x = d.get("all") or d.get("pooled")
    lo, hi = x["widened"]
    return {"R": x["R"], "widened": [lo, hi], "sign": 1 if lo > 1 else (-1 if hi < 1 else 0)}


def rank():
    logs = json.load(open(LOGS)) if os.path.exists(LOGS) else {}
    cls = mab.class_of()
    rows = {}
    for name, entry, arm_w, ref_w, key in COMPARISONS + CHECKS:
        arm, ref = _runs(entry, arm_w), _runs(entry, ref_w)
        row = {"arm": arm, "ref": ref, "target": {}, "cand": {}}
        for ax in TARGETS:
            ln = [read_ratio(a, b, ax, cls) for a, b in zip(arm, ref)]
            if key:
                row["target"][ax] = dict(_resolved(entry, key, ax), ln_seed=ln)
            else:
                same = all(x > 0 for x in ln) or all(x < 0 for x in ln)
                big = abs(float(np.mean(ln))) > CHECK_MIN
                row["target"][ax] = {"R": round(math.exp(float(np.mean(ln))), 4), "ln_seed": ln,
                                     "sign": int(np.sign(np.mean(ln))) if same and big else 0}
            row["set"] = "select" if key else "check"
        per = {}
        for a, b in zip(arm, ref):
            va, vb = val_cands(a), val_cands(b)
            la, lb = log_cands(a, logs), log_cands(b, logs)
            for k in va:
                if k in vb:
                    per.setdefault(k, []).append(va[k] - vb[k])
            for k in la:
                if k in lb and la[k] and lb[k]:
                    per.setdefault(k, []).append(math.log(la[k] / lb[k]))
            for ax in READ_CANDS:
                x = read_ratio(a, b, ax, cls)
                if x is not None:
                    per.setdefault(f"read_{ax}", []).append(x)
        row["cand"] = {k: [round(x, 4) for x in v] for k, v in per.items() if len(v) == len(arm)}
        rows[name] = row
    # a candidate tracks a set when on every resolved comparison of it it has the target's sign at both seeds;
    # chosen on the select set, confirmed on the check set
    cands = sorted(set.intersection(*[set(r["cand"]) for r in rows.values()]))
    verdict = {}
    for c in cands:
        v = {}
        for st in ("select", "check"):
            hits, misses = [], []
            rs = {n: r for n, r in rows.items() if r["set"] == st}
            for name, r in rs.items():
                signs = {r["target"][ax]["sign"] for ax in TARGETS} - {0}
                if len(signs) != 1:
                    continue
                sg = signs.pop()
                (hits if all(x * sg > 0 for x in r["cand"][c]) else misses).append(name)
            corr = {}
            for ax in TARGETS:      # Pearson r over the set's comparisons and seeds, candidate against target
                x = [y for r in rs.values() for y in r["cand"][c]]
                y = [y for r in rs.values() for y in r["target"][ax]["ln_seed"]]
                corr[ax] = round(float(np.corrcoef(x, y)[0, 1]), 3)
            v[st] = {"tracks": not misses and bool(hits), "agree": hits, "disagree": misses, "pearson": corr}
        verdict[c] = v
    out = {"design": "warm_start step 11: logged quantities against the zero-shot reads (ws_objective_check.py)",
           "comparisons": rows, "verdict": verdict,
           "tracking_select": [c for c, v in verdict.items() if v["select"]["tracks"]],
           "tracking_both": [c for c, v in verdict.items() if v["select"]["tracks"] and v["check"]["tracks"]]}
    json.dump(out, open(OUT, "w"), indent=1)
    for name, r in rows.items():
        t = " ".join(f"{ax} R {r['target'][ax]['R']} {r['target'][ax].get('widened', '')} sign {r['target'][ax]['sign']}"
                     for ax in TARGETS)
        print(f"{r['set']} {name}: {t}")
    for c, v in verdict.items():
        print(f"  {c:16s} " + "  ".join(
            f"{st}: r {v[st]['pearson']['PP4']:+.2f}/{v[st]['pearson']['EE4']:+.2f} tracks={v[st]['tracks']!s:5s} "
            f"miss={','.join(v[st]['disagree']) or '-'}" for st in ("select", "check")))
    print(json.dumps({"out": OUT, "tracking_select": out["tracking_select"], "tracking_both": out["tracking_both"]}))


if __name__ == "__main__":
    {"extract": extract, "rank": rank}[sys.argv[1]]()
