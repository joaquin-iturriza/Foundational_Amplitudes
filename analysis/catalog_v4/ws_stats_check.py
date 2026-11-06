#!/usr/bin/env python3
"""plan warm_start step 7, the superset design's equivalence check (passed 2026-11-04T21:51, its `fixed` line): each
warm start W run under fine_tune.target_stats=parent_shared (C5 v4_mse_warmpp, C6 v4_mse_warmsx, C8 v4_het_warmmag)
must hold, on every pool it shares unchanged with its parent, the parent's transform string (amp_trafos_pp) and the
parent's shared mu/sigma (prepd_mean, prepd_std) to 1e-12; its matched control C_k runs with target_stats=parent, so a
W run failing it differs from C_k in more than the change, and its C_k is rerun with parent_shared before W and C are
compared (mixture_ab.py warm and signext_compare.py warm withhold the verdict until this check passes).

A pool is shared unchanged when the W run's recipe entry equals the parent's (the whole entry), and, for C8, the
pool is not one the change re-targets (a parent transform starting with 'signedlog': the signed one-loop pools, on
their magnitude in C8). Read on CC from each run's data_stats.json and config.yaml (data.processes_file).

    python3 analysis/catalog_v4/ws_stats_check.py C5 C6 C8   -> data/ws_stats_check.json (merged per arm)
"""
import json, os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from analysis.common import site_python  # noqa: E402

OUT = os.path.join(HERE, "data", "ws_stats_check.json")
K = (533, 1066, 2132)
ARMS = {"C5": ("v4_mse_warmpp", ("", "_seed2"), "v4_dimensionless_muon"),
        "C6": ("v4_mse_warmsx", ("", "_seed2", "_seed3", "_seed4"), "v4_dimensionless_muon"),
        "C8": ("v4_het_warmmag", ("", "_seed2"), "v4_het_control")}
TOL = 1e-12
# the floor s of an abslog transform (log max(|x|, s)) is a float the transform string carries to 7 digits, fitted on
# the pool's train split: the parent and a parent_shared W run on the same 1e4 subsample give the same mu/sigma to 1e-12
# but floors that differ from the 5th digit on (2026-11-06, C5/C6: 39-45 of the compared pools, relative difference up to
# 1.2e-4), a float roundoff, not a different transform. The floor moves the target only on events below it, and there by
# delta = |ln(s_w / s_p)| in log|M|^2, so it adds at most delta^2 to a probe's MSE of log|M|^2: at a relative tolerance of
# 1e-3, 1e-6, under the smallest class MSE the reads resolve (about 1e-4, figs/ws_c1m CDFs)
FLOOR_RTOL = 1e-3

SCRIPT = r'''
import json, os, sys, yaml
pairs, retarget = json.loads(sys.argv[1]), set(json.loads(sys.argv[2]))
def same_trafo(a, b, rtol=%r):
    """equal transform lists, an abslog floor equal to rtol (relative); returns (equal, relative floor difference)"""
    if len(a) != len(b):
        return False, None
    worst = 0.0
    for x, y in zip(a, b):
        x, y = str(x), str(y)
        if x.startswith("abslog:") and y.startswith("abslog:"):
            fx, fy = float(x.split(":", 1)[1]), float(y.split(":", 1)[1])
            r = abs(fx - fy) / max(abs(fy), 1e-300)
            worst = max(worst, r)
            if r > rtol:
                return False, r
        elif x != y:
            return False, None
    return True, worst
base = "runs/catalog_v4_short"
def load(arm, recipe=None):
    d = os.path.join(base, arm)
    st = os.path.join(d, "data_stats.json")
    if not os.path.exists(st):
        return None
    if recipe is None:
        recipe = yaml.safe_load(open(os.path.join(d, "config.yaml")))["data"]["processes_file"]
    rec = yaml.safe_load(open(recipe))["processes"]
    return json.load(open(st)), rec
out = {}
for arm, parent in pairs:
    # the parents (v4_het_control, v4_dimensionless_muon[_seedN]) trained on the catalog-sampled set, whose file was
    # recipes/catalog_v4_train.yaml then (their config.yaml) and is recipes/catalog_v4_base_train.yaml since e3f37300
    # (catalog_v4_train.yaml now holds the share-0.25 catalog): the parent's pools are read from the latter
    w, p = load(arm), load(parent, "recipes/catalog_v4_base_train.yaml")
    if w is None or p is None:
        out[arm] = {"ok": None, "why": "no data_stats.json yet" if w is None else "parent unreadable"}
        continue
    (ws, wr), (ps, pr) = w, p
    # a name can recur in a recipe (one entry per window or sampling): an entry is matched by its whole content
    pidx = {json.dumps(e, sort_keys=True): i for i, e in enumerate(pr)}
    n, bad, floor_rel = 0, [], 0.0
    for i, e in enumerate(wr):
        j = pidx.get(json.dumps(e, sort_keys=True))
        if j is None:
            continue
        pt = ps["amp_trafos_pp"][j]
        if arm in retarget and str(pt[0]).startswith("signedlog"):
            continue
        n += 1
        wt = ws["amp_trafos_pp"][i]
        dm = abs(ws["prepd_mean"][i] - ps["prepd_mean"][j]); ds = abs(ws["prepd_std"][i] - ps["prepd_std"][j])
        eq, rel = same_trafo(wt, pt)
        floor_rel = max(floor_rel, rel or 0.0)
        if not eq or dm > %r or ds > %r:
            bad.append({"pool": e["name"], "w": wt, "parent": pt, "d_mean": dm, "d_std": ds, "floor_rel": rel})
    out[arm] = {"ok": not bad, "n_compared": n, "n_mismatch": len(bad), "mismatch_first": bad[:10],
                "n_pools_w": len(wr), "n_pools_parent": len(pr), "max_floor_rel": floor_rel}
print("WSSTATS " + json.dumps(out))
''' % (FLOOR_RTOL, TOL, TOL)


def arms_of(change):
    pre, sfx, par = ARMS[change]
    return [(f"{pre}_k{k}{s}", par + s) for k in K for s in sfx]


def status(change, res=None):
    """True when every W run of the change passed, False when one failed, None while one is unread."""
    res = res if res is not None else (json.load(open(OUT)) if os.path.exists(OUT) else {})
    oks = [res.get(a, {}).get("ok") for a, _ in arms_of(change)]
    return False if False in oks else None if None in oks else True


def main():
    changes = sys.argv[1:] or list(ARMS)
    pairs = [p for c in changes for p in arms_of(c)]
    retarget = [a for a, _ in arms_of("C8")] if "C8" in changes else []
    got = site_python(SCRIPT, [shlex.quote(json.dumps(pairs)), shlex.quote(json.dumps(retarget))], "WSSTATS")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.update(got)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({c: status(c, res) for c in changes}))
    print(json.dumps({a: {k: v for k, v in x.items() if k != "mismatch_first"} for a, x in got.items()}))


if __name__ == "__main__":
    main()
