#!/usr/bin/env python3
"""Why e+e- gains most from the default share (plan catalog_coverage step 43(2)).

The share-0.25 het arm (default25_1e4) against the share-0 het control reads R = 0.18 on EE4 and 0.34 on PP4
(arms_vs_het_control.json, trained classes). The reading tested here: the default share puts training events exactly
where the default-sampled test events sit in sqrt(s), and the catalog's own draw (log-uniform sqrt(s) over the class's
window) puts few there. On e+e- the test energies are a handful of fixed points (91.19, 240, 365, 500, 1000, 3000 GeV),
a measure-zero set for the catalog's draw; on pp sqrt(s_hat) is spread, partly covered by the catalog.

  pull   on CC-IN2P3 (one read-only site run, analysis/common site_python): per probe of EE4 and PP4 whose class is
         trained, sqrt(s) of the probe's test pool events (beams 0 and 1) and of the class's catalog val pool (the catalog sampling, every role alike; the train pools are not kept on CC;
         control's data: the first 1e4 events, as the arms' train_subsample); per test event the share of catalog
         events within a relative window W of its sqrt(s), c, and of the test pool's own events, t (the default share is
         generated the default way, as the test events [D227]); per probe the median over its test events of
         log10(m / c), m = 0.75 c + 0.25 t the share-0.25 mixture's share in the window (c floored at half an event),
         and the share of test events with c = 0 -> data/ee_gain.json
  read   per class (the census classes, as mixture_ab) log R of the arm over the control (each seed's probe ratio, the
         seeds' geometric mean) against the class's density gain log10(m/c); medians per probe set, Spearman correlation
         within PP4, within EE4 and pooled, and the EE4-PP4 gap at matched density gain (classes binned in it)
  pull_kin / read_kin   the next readings (the sqrt(s) one did not order the gain): the same density gain, per test
         event, in four variables, each in the partonic centre-of-mass frame: log10 sqrt(s) (window +-log10 1.02),
         the most forward final's log10(1 - |cos theta|) against the beam, the most collinear final pair's
         log10(1 - cos theta_ij), and log10 |M|^2 (windows +-0.1); per probe the median gain per variable ->
         data/ee_gain_kin.json; read as `read`, per variable -> data/ee_gain_kin_read.json
  read_level   the per-class target level: log R against the control's own MSE of the class (does the default
         share help most where the control is worst, and is EE4 still ahead at matched control MSE)
         -> data/ee_gain_level.json
  beams   (step 62, D295: how could pp-beam data help e+e- events) what the default share contains per beam, from the
         recipes alone: the training classes with a default source, split by the MadEvent pool their dsh_ entry reuses
         (ee4_: e+e- at fixed collider energies; pp13v4_/pp13_: 13 TeV pp), their events per class (f_default x the
         train_subsample of the short arms), the e+e- energies, whether a class's default events come from the same
         generator setup as its EE4 test pool (cards, energies, model); then R (default25 over the het control,
         mixture_ab.compare, MIX_HET runs) split four ways: EE4 and PP4 probes of trained classes (their own default
         events in training) and of held-out classes (none of their own; only other classes' default events)
         -> data/ee_gain_beams.json
    python3 analysis/catalog_v4/ee_gain.py pull | read | pull_kin | read_kin | read_level | beams
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "analysis", "common")]
W = 0.02                       # relative sqrt(s) window around each test event
SHARE = 0.25
N_CAT = 10000
OUT = os.path.join(DATA, "ee_gain.json")

PULL = r'''
import json, sys, numpy as np
sys.path.insert(0, "analysis/common")
from readers import find_pool
W, SHARE, N_CAT = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
pairs = json.load(open(sys.argv[4]))
def rs(path, n):
    a = np.load(path, mmap_mode="r")[:n]
    k = (a.shape[1] - 1) // 5
    p = np.asarray(a[:, :8], dtype=np.float64).reshape(-1, 2, 4).sum(1)
    return np.sort(np.sqrt(np.clip(p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(1), 0, None)))
out = {}
for probe, cls in pairs:
    tp, cp = find_pool(probe, role="test"), find_pool(cls, role="val")
    if tp is None or cp is None:
        out[probe] = {"missing": [tp is None, cp is None]}; continue
    t, c = rs(tp, 2000), rs(cp, N_CAT)
    lo, hi = t * (1 - W), t * (1 + W)
    cs = (np.searchsorted(c, hi) - np.searchsorted(c, lo)) / len(c)
    ts = (np.searchsorted(t, hi) - np.searchsorted(t, lo)) / len(t)
    cf = np.maximum(cs, 0.5 / len(c))
    g = np.log10(((1 - SHARE) * cs + SHARE * ts) / cf)
    out[probe] = {"cls": cls, "gain_median": float(np.median(g)), "c_zero": float(np.mean(cs == 0)),
                  "sqrts_test_median": float(np.median(t)), "sqrts_cat_median": float(np.median(c)), "n_test": len(t)}
print("EEGAIN " + json.dumps(out))
'''


PULL_KIN = r'''
import json, sys, numpy as np
sys.path.insert(0, "analysis/common")
from readers import find_pool
SHARE, N_CAT = float(sys.argv[1]), int(sys.argv[2])
pairs = json.load(open(sys.argv[3]))
WIN = {"sqrts": float(np.log10(1.02)), "fwd": 0.1, "coll": 0.1, "logm2": 0.1}
def boost(p, b):
    b2 = np.clip((b ** 2).sum(-1), 1e-300, None)
    g = 1 / np.sqrt(np.clip(1 - b2, 1e-300, None))
    bp = (b[:, None, :] * p[..., 1:]).sum(-1)
    e = g[:, None] * (p[..., 0] - bp)
    v = p[..., 1:] + (((g - 1) / b2)[:, None] * bp - g[:, None] * p[..., 0])[..., None] * b[:, None, :]
    return np.concatenate([e[..., None], v], -1)
def feats(path, n):
    a = np.asarray(np.load(path, mmap_mode="r")[:n], dtype=np.float64)
    k = (a.shape[1] - 1) // 5
    p = a[:, :4 * k].reshape(-1, k, 4)
    P = p[:, 0] + p[:, 1]
    q = boost(p, P[:, 1:] / P[:, :1])
    ax = q[:, 0, 1:] / np.linalg.norm(q[:, 0, 1:], axis=-1, keepdims=True)
    f = q[:, 2:, 1:]
    u = f / np.clip(np.linalg.norm(f, axis=-1, keepdims=True), 1e-300, None)
    c = (u * ax[:, None, :]).sum(-1)
    fwd = np.log10(np.clip(1 - np.abs(c), 1e-12, None)).min(1)
    cij = np.einsum("nid,njd->nij", u, u)
    iu = np.triu_indices(k - 2, 1)
    coll = np.log10(np.clip(1 - cij[:, iu[0], iu[1]], 1e-12, None)).min(1) if k > 3 else np.zeros(len(a))
    s = np.sqrt(np.clip(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1), 1e-300, None))
    m = np.abs(a[:, -1])
    return {"sqrts": np.log10(s), "fwd": fwd, "coll": coll, "logm2": np.log10(np.clip(m, 1e-300, None))}
out = {}
for probe, cls in pairs:
    tp, cp = find_pool(probe, role="test"), find_pool(cls, role="val")
    if tp is None or cp is None:
        out[probe] = {"missing": [tp is None, cp is None]}; continue
    T, C = feats(tp, 2000), feats(cp, N_CAT)
    r = {"cls": cls, "n_test": int(len(T["sqrts"]))}
    for v, w in WIN.items():
        t, c = np.sort(T[v]), np.sort(C[v])
        lo, hi = t - w, t + w
        cs = (np.searchsorted(c, hi) - np.searchsorted(c, lo)) / len(c)
        ts = (np.searchsorted(t, hi) - np.searchsorted(t, lo)) / len(t)
        g = np.log10(((1 - SHARE) * cs + SHARE * ts) / np.maximum(cs, 0.5 / len(c)))
        r[v] = {"gain_median": float(np.median(g)), "c_zero": float(np.mean(cs == 0)),
                "test_median": float(np.median(t)), "cat_median": float(np.median(c))}
    out[probe] = r
print("EEKIN " + json.dumps(out))
'''
VARS = ("sqrts", "fwd", "coll", "logm2")
OUT_KIN = os.path.join(DATA, "ee_gain_kin.json")


def arm_runs():
    import mixture_ab as mab
    return mab.MIX_HET["default25"], mab.MIX_HET["control"], mab.class_of()


def trained():
    import yaml
    return {p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_default25_train.yaml")))["processes"]}


def probes():
    import mixture_ab as mab
    arm, _, cls = arm_runs()
    tr = trained()
    out = {}
    for ax in ("EE4", "PP4"):
        for p in json.load(open(os.path.join(mab.READS, f"{arm[ax][0]}_{ax}.json")))["mse"]:
            if cls(p) in tr:
                out[p] = (ax, cls(p))
    return out


def pull(kin=False):
    """sqrt(s) gain (pull) or the four kinematic gains (pull_kin) on CC; the probe list goes inside the script."""
    from readers import site_python
    pr = probes()
    pairs = [[p, c] for p, (ax, c) in pr.items()]
    if kin:
        script = PULL_KIN.replace("pairs = json.load(open(sys.argv[3]))", "pairs = " + json.dumps(pairs))
        res = site_python(script, [str(SHARE), str(N_CAT), "-"], "EEKIN", timeout=3000)
        head, out = {"share": SHARE, "n_cat": N_CAT}, OUT_KIN
    else:
        script = PULL.replace("pairs = json.load(open(sys.argv[4]))", "pairs = " + json.dumps(pairs))
        res = site_python(script, [str(W), str(SHARE), str(N_CAT), "-"], "EEGAIN", timeout=1800)
        head, out = {"window_rel": W, "share": SHARE, "n_cat": N_CAT}, OUT
    for p, v in res.items():
        v["axis"] = pr[p][0]
    json.dump(dict(head, probes=res), open(out, "w"), indent=1)
    miss = [p for p, v in res.items() if "missing" in v]
    print(json.dumps({"n": len(res), "missing": len(miss), "first_missing": miss[:5]}))


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return [round(float(r.correlation), 3), float(r.pvalue)]


def read():
    import mixture_ab as mab
    arm, ctl, cls = arm_runs()
    g = json.load(open(OUT))["probes"]
    rows = {}
    for ax in ("EE4", "PP4"):
        a = [json.load(open(os.path.join(mab.READS, f"{r}_{ax}.json")))["mse"] for r in arm[ax]]
        c = [json.load(open(os.path.join(mab.READS, f"{r}_{ax}.json")))["mse"] for r in ctl[ax]]
        by = {}
        for p, v in g.items():
            if v.get("axis") != ax or "missing" in v or not all(p in x for x in a + c):
                continue
            lr = np.mean([np.log(a[s][p] / c[s][p]) for s in range(2)])
            by.setdefault(v["cls"], []).append((lr, v["gain_median"], v["c_zero"]))
        rows[ax] = {k: tuple(np.mean(np.array(v), axis=0)) for k, v in by.items()}
    out = {"design": "plan catalog_coverage step 43(2); runs mixture_ab.MIX_HET default25 vs control", "window_rel": W}
    for ax, r in rows.items():
        lr, gn, cz = (np.array([x[i] for x in r.values()]) for i in range(3))
        out[ax] = {"n_classes": len(r), "R_median": round(float(np.exp(np.median(lr))), 3),
                   "gain_median": round(float(np.median(gn)), 3), "c_zero_median": round(float(np.median(cz)), 3),
                   "spearman_logR_gain": spearman(lr, gn)}
    allr = [x for r in rows.values() for x in r.values()]
    lr = np.array([x[0] for x in allr]); gn = np.array([x[1] for x in allr])
    out["pooled"] = {"n_classes": len(allr), "spearman_logR_gain": spearman(lr, gn)}
    # the EE4-PP4 gap at matched gain: classes binned in the gain at the pooled quartiles
    q = np.quantile(gn, [0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for i in range(4):
        sel = lambda r: [x[0] for x in r.values() if q[i] <= x[1] <= q[i + 1]]
        e, p = sel(rows["EE4"]), sel(rows["PP4"])
        bins.append({"gain": [round(float(q[i]), 3), round(float(q[i + 1]), 3)], "n_EE4": len(e), "n_PP4": len(p),
                     "R_EE4": round(float(np.exp(np.median(e))), 3) if e else None,
                     "R_PP4": round(float(np.exp(np.median(p))), 3) if p else None})
    out["matched_gain_bins"] = bins
    json.dump(out, open(os.path.join(DATA, "ee_gain_read.json"), "w"), indent=1)
    print(json.dumps(out))


def class_log_r():
    """{axis: {class: (mean over the class's probes of the seeds' mean log R, the control's log MSE likewise)}}."""
    import mixture_ab as mab
    arm, ctl, cls = arm_runs()
    tr = trained()
    out = {}
    for ax in ("EE4", "PP4"):
        a = [json.load(open(os.path.join(mab.READS, f"{r}_{ax}.json")))["mse"] for r in arm[ax]]
        c = [json.load(open(os.path.join(mab.READS, f"{r}_{ax}.json")))["mse"] for r in ctl[ax]]
        by = {}
        for p in a[0]:
            if cls(p) not in tr or not all(p in x for x in a + c):
                continue
            by.setdefault(cls(p), []).append((np.mean([np.log(a[s][p] / c[s][p]) for s in range(2)]),
                                               np.mean([np.log(c[s][p]) for s in range(2)]), p))
        out[ax] = by
    return out


def matched(rows, key):
    """Spearman of log R against `key` within each set and pooled, and R per set in the pooled quartiles of `key`."""
    res = {}
    for ax, r in rows.items():
        res[ax] = {"n_classes": len(r), "spearman_logR": spearman([x[0] for x in r.values()], [x[key] for x in r.values()]),
                   "median": round(float(np.median([x[key] for x in r.values()])), 3)}
    allx = np.array([x[key] for r in rows.values() for x in r.values()])
    res["pooled_spearman_logR"] = spearman([x[0] for r in rows.values() for x in r.values()], allx)
    q = np.quantile(allx, [0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for i in range(4):
        sel = lambda r: [x[0] for x in r.values() if q[i] <= x[key] <= q[i + 1]]
        e, p = sel(rows["EE4"]), sel(rows["PP4"])
        bins.append({"range": [round(float(q[i]), 3), round(float(q[i + 1]), 3)], "n_EE4": len(e), "n_PP4": len(p),
                     "R_EE4": round(float(np.exp(np.median(e))), 3) if e else None,
                     "R_PP4": round(float(np.exp(np.median(p))), 3) if p else None})
    res["matched_bins"] = bins
    return res


def read_kin():
    g = json.load(open(OUT_KIN))["probes"]
    lr = class_log_r()
    out = {"design": "plan catalog_coverage step 43(2), kinematic readings; runs mixture_ab.MIX_HET default25 vs control"}
    for v in VARS:
        rows = {}
        for ax, by in lr.items():
            r = {}
            for k, lst in by.items():
                ok = [(x[0], g[x[2]][v]["gain_median"], g[x[2]][v]["c_zero"]) for x in lst
                      if x[2] in g and "missing" not in g[x[2]]]
                if ok:
                    r[k] = tuple(np.mean(np.array(ok), axis=0))
            rows[ax] = r
        out[v] = matched(rows, 1)
        for ax in rows:
            out[v][ax]["c_zero_median"] = round(float(np.median([x[2] for x in rows[ax].values()])), 3)
    json.dump(out, open(os.path.join(DATA, "ee_gain_kin_read.json"), "w"), indent=1)
    print(json.dumps(out))


def read_level():
    lr = class_log_r()
    rows = {ax: {k: tuple(np.mean(np.array([x[:2] for x in lst]), axis=0)) for k, lst in by.items()} for ax, by in lr.items()}
    out = {"design": "plan catalog_coverage step 43(2), target-level reading: log R against the control's ln MSE",
           "R_median": {ax: round(float(np.exp(np.median([x[0] for x in r.values()]))), 3) for ax, r in rows.items()},
           "control_mse_median": {ax: round(float(np.exp(np.median([x[1] for x in r.values()]))), 4) for ax, r in rows.items()}}
    out.update(matched(rows, 1))
    # the EE4 offset at a fixed control level: OLS log R = a + b ln MSE_control + c [EE4], c's bootstrap 90% interval
    x = np.array([(v[1], ax == "EE4") for ax, r in rows.items() for v in r.values()], dtype=float)
    y = np.array([v[0] for r in rows.values() for v in r.values()])
    X = np.column_stack([np.ones(len(y)), x])
    fit = lambda i: np.linalg.lstsq(X[i], y[i], rcond=None)[0]
    b = fit(np.arange(len(y)))
    rng = np.random.default_rng(0)
    cs = [fit(rng.integers(0, len(y), len(y)))[2] for _ in range(2000)]
    out["ols"] = {"slope_per_ln_mse": round(float(b[1]), 3), "ee4_offset_ratio": round(float(np.exp(b[2])), 3),
                  "ee4_offset_ratio_90": [round(float(np.exp(q)), 3) for q in np.quantile(cs, [0.05, 0.95])],
                  "raw_ee4_over_pp4": round(float(np.exp(np.median(y[x[:, 1] == 1]) - np.median(y[x[:, 1] == 0]))), 3)}
    json.dump(out, open(os.path.join(DATA, "ee_gain_level.json"), "w"), indent=1)
    print(json.dumps(out))


def beams():
    import yaml
    import mixture_ab as mab
    R = os.path.join(ROOT, "recipes")
    rec = yaml.safe_load(open(os.path.join(R, "catalog_v4_default25_train.yaml")))["processes"]
    dsh = yaml.safe_load(open(os.path.join(R, "catalog_v4_default_processes.yaml")))["processes"]
    ee4y = yaml.safe_load(open(os.path.join(R, "catalog_v4_ee_processes.yaml")))
    ee4 = {k: {**v, "lhe": {**ee4y.get("lhe", {}), **v.get("lhe", {})}} for k, v in ee4y["processes"].items()}
    sub = 10000                                    # the short arms' ++data.train_subsample (job_catalog_v3_short.sh)
    beam_of = lambda src: "ee" if dsh[src]["reuse"].startswith("ee4_") else "pp"
    out = {"design": "plan catalog_coverage step 62 (D295): the default share's content per beam and R split by beam "
                     "and by whether the class's own default events were trained",
           "recipe": "recipes/catalog_v4_default25_train.yaml", "train_classes": len(rec)}
    by = {"ee": [], "pp": []}
    for e in rec:
        src = (e.get("sampling") or {}).get("default_source")
        if src:
            by[beam_of(src)].append((e, src))
    n_all = len(rec) * sub
    for b, lst in by.items():
        f = sorted({x[0]["sampling"]["f_default"] for x in lst})
        d = {"classes": len(lst), "f_default": f, "default_events_per_class_subsampled": int(f[0] * sub),
             "default_events_total_subsampled": int(f[0] * sub * len(lst)),
             "share_of_all_training_events": round(f[0] * sub * len(lst) / n_all, 4),
             "catalog_sqrts_window_GeV": [min(x[0]["sqrts"][0] for x in lst), max(x[0]["sqrts"][1] for x in lst)]}
        if b == "ee":
            en = [tuple(dsh[s]["lhe"]["energies"]) for _, s in lst]
            same = sum(1 for _, s in lst if dsh[s]["reuse"] in ee4 and all(
                dsh[s]["lhe"].get(k) == ee4[dsh[s]["reuse"]]["lhe"].get(k)
                for k in ("cards_dir", "run_card", "param_card", "model", "energies", "label_alphas"))
                and dsh[s]["mg5_generate"] == ee4[dsh[s]["reuse"]]["mg5_generate"])
            d.update(energies_GeV=sorted({x for t in en for x in t}),
                     energies_per_class={str(k): v for k, v in sorted(
                         {n: sum(1 for t in en if len(t) == n) for n in {len(t) for t in en}}.items())},
                     same_generator_setup_as_its_EE4_pool=same)
        else:
            d["pools_reused"] = {k: sum(1 for _, s in lst if dsh[s]["reuse"].startswith(k + "_")) for k in ("pp13v4", "pp13")}
        out[b] = d
    out["classes_without_share"] = len(rec) - sum(len(v) for v in by.values())
    trained = {x[0]["name"] for v in by.values() for x in v}
    arm, ctl, cls = mab.MIX_HET["default25"], mab.MIX_HET["control"], mab.class_of()
    out["runs"] = {"default25": arm, "control": ctl}
    rs = {}
    for ax in ("EE4", "PP4"):
        a = [mab.read_of(r, ax) for r in arm[ax]]
        c = [mab.read_of(r, ax) for r in ctl[ax]]
        keys = set.intersection(*(set(x) for x in a + c))
        for part, sel in (("trained", lambda p: cls(p) in trained), ("held_out", lambda p: cls(p) not in trained)):
            k = sorted(p for p in keys if sel(p))
            if k:
                rs[f"{ax}_{part}"] = mab.compare(mab.side(a), mab.side(c), c[0], c[1], k, cls)
    out["R_default25_over_control"] = rs
    json.dump(out, open(os.path.join(DATA, "ee_gain_beams.json"), "w"), indent=1)
    print(json.dumps(out))


# step 62's beam ablation (design passed 2026-11-06T04:47): four arms at one commit (5bf3e0f0), run dirs <arm>_tagbeams,
# (seed 42, seed 43); every read from the run's own job (mixture_ab extract of its log)
BEAM_RUNS = {"control": ("FA-ccin2p3-1c6caddf", "FA-ccin2p3-4ea1c1eb"),
             "full": ("FA-ccin2p3-c2e82e97", "FA-ccin2p3-a52c9185"),
             "eeonly": ("FA-ccin2p3-e4e0f158", "FA-ccin2p3-d84270fc"),
             "pponly": ("FA-ccin2p3-2756bece", "FA-ccin2p3-3bafdc58")}


def _default_classes():
    """the training classes with a default source in the share-0.25 recipe, by beam (as beams())"""
    import yaml
    R = os.path.join(ROOT, "recipes")
    rec = yaml.safe_load(open(os.path.join(R, "catalog_v4_default25_train.yaml")))["processes"]
    dsh = yaml.safe_load(open(os.path.join(R, "catalog_v4_default_processes.yaml")))["processes"]
    by = {"ee": set(), "pp": set()}
    for e in rec:
        src = (e.get("sampling") or {}).get("default_source")
        if src:
            by["ee" if dsh[src]["reuse"].startswith("ee4_") else "pp"].add(e["name"])
    return by


def ablation():
    """R of each beam arm over the same-commit share-0 control (mixture_ab.compare: class medians of the seeds' geometric
    mean ratio, bootstrap interval) on EE4 and PP4 split by whether the class's own default events are in the full
    share's training (trained) or not (held_out), and on the held-out axes pooled; per seed alone too (the outcome asks
    'on both seeds'); R_0 = the same-commit control over the step-15 het control (MIX_HET, the rerun spread) and
    s = max(max |ln R_0|, 0.1); the design's outcome; additivity ln R_full against ln R_ee + ln R_pp
    -> data/ee_gain_ablation.json"""
    import mixture_ab as mab
    cls = mab.class_of(); by = _default_classes(); trained = by["ee"] | by["pp"]
    rd = {a: {ax: [mab.read_of(r, ax) for r in runs] for ax in ("EE4", "PP4", "held")} for a, runs in BEAM_RUNS.items()}
    old = {ax: [mab.read_of(r, ax) for r in mab.MIX_HET["control"][ax]] for ax in ("EE4", "PP4", "held")}
    parts = {"EE4_trained": ("EE4", lambda p: cls(p) in trained), "EE4_held_out": ("EE4", lambda p: cls(p) not in trained),
             "PP4_trained": ("PP4", lambda p: cls(p) in trained), "PP4_held_out": ("PP4", lambda p: cls(p) not in trained),
             "held_axes": ("held", lambda p: True)}
    out = {"design": "plan catalog_coverage step 62 (D295), passed 2026-11-06T04:47", "runs": BEAM_RUNS,
           "commit": "5bf3e0f0", "R": {}, "R_per_seed": {}, "R_0": {}}
    def keys(ax, sel, *sides):
        ks = set.intersection(*(set(x) for s in sides for x in s))
        return sorted(k for k in ks if sel(mab.probe_name(k)))
    for part, (ax, sel) in parts.items():
        c = rd["control"][ax]
        k0 = keys(ax, sel, c, old[ax])
        if k0:
            out["R_0"][part] = mab.compare(mab.side(c), mab.side(old[ax]), c[0], c[1], k0, cls)
        for a in ("full", "eeonly", "pponly"):
            k = keys(ax, sel, c, rd[a][ax])
            if not k:
                continue
            out["R"].setdefault(a, {})[part] = mab.compare(mab.side(rd[a][ax]), mab.side(c), c[0], c[1], k, cls)
            out["R_per_seed"].setdefault(a, {})[part] = [
                round(float(np.median(list(mab.class_ratios(rd[a][ax][i], c[i], k, lambda q: cls(mab.probe_name(q))).values()))), 4)
                for i in (0, 1)]
    s = max([abs(np.log(v["R"])) for v in out["R_0"].values()] + [0.1])
    out["s"] = round(float(s), 4)
    lnR = lambda a, part: [float(np.log(x)) for x in out["R_per_seed"][a][part]]
    ee, pp, full = lnR("eeonly", "EE4_trained"), lnR("pponly", "EE4_trained"), lnR("full", "EE4_trained")
    own = all(abs(e - f) <= s for e, f in zip(ee, full)) and all(x > -s for x in pp)
    cross = any(all(x <= -2 * s for x in lnR("pponly", part)) for part in ("EE4_trained", "EE4_held_out"))
    out["outcome_EE4"] = ("E+E- OWN EVENTS" if own and not cross else "CROSS-BEAM" if cross else "UNRESOLVED")
    out["pp_alongside_on_EE4"] = ("neutral (|ln R_pp| <= s)" if all(abs(x) <= s for x in pp) else
                                  "hurts (ln R_pp >= 2s)" if all(x >= 2 * s for x in pp) else "between")
    # PP4 with the roles swapped: is the PP4 gain the pp default events themselves, does e+e- data help pp
    pe, pf, pe_ee = lnR("pponly", "PP4_trained"), lnR("full", "PP4_trained"), lnR("eeonly", "PP4_trained")
    cross_pp = any(all(x <= -2 * s for x in lnR("eeonly", part)) for part in ("PP4_trained", "PP4_held_out"))
    out["outcome_PP4"] = ("PP OWN EVENTS" if all(abs(a - b) <= s for a, b in zip(pe, pf)) and all(x > -s for x in pe_ee)
                          and not cross_pp else "CROSS-BEAM" if cross_pp else "UNRESOLVED")
    out["additivity"] = {part: {"ln_R_full": round(float(np.log(out["R"]["full"][part]["R"])), 4),
                                "ln_R_ee_plus_ln_R_pp": round(float(np.log(out["R"]["eeonly"][part]["R"]) +
                                                                    np.log(out["R"]["pponly"][part]["R"])), 4)}
                         for part in parts if all(part in out["R"][a] for a in ("full", "eeonly", "pponly"))}
    json.dump(out, open(os.path.join(DATA, "ee_gain_ablation.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("s", "outcome_EE4", "pp_alongside_on_EE4", "outcome_PP4", "additivity")}))
    for a, v in out["R"].items():
        for part, r in v.items():
            print(a, part, r, out["R_per_seed"][a][part])
    for part, r in out["R_0"].items():
        print("R_0", part, r)


def fig_ablation():
    """step 62's figure: per test class (the seeds' geometric mean MSE of log|M|^2, probes of one class pooled by their
    geometric mean, as mixture_ab), each arm's zero-shot error against the class's rank in the share-0 control's error,
    on the e+e- default test pools (EE4) and the 13 TeV pp test pools (PP4); below, each arm's error over the control's
    per class (the ratio the R of `ablation` is the median of) -> figs/beam_ablation_<a..d>.{png,pdf} and a legend
    strip figs/beam_ablation_legend"""
    import mixture_ab as mab
    sys.path.insert(0, ROOT)
    import plot_style as ps
    cls = mab.class_of()
    lab = {"control": "no default share", "full": "default share, both beams", "eeonly": "default share, e+e- only",
           "pponly": "default share, pp only"}
    col = {"control": ps.C.grey, "full": ps.C.blue, "eeonly": ps.C.vermillion, "pponly": ps.C.green}
    figs = ps.panels(4)
    for j, ax_name in enumerate(("EE4", "PP4")):
        per = {}
        for a, runs in BEAM_RUNS.items():
            sd = mab.side([mab.read_of(r, ax_name) for r in runs])
            by = {}
            for p, x in sd.items():
                by.setdefault(cls(p), []).append(x)
            per[a] = {c: mab.gmean(v) for c, v in by.items()}
        ks = sorted(set.intersection(*(set(v) for v in per.values())), key=lambda c: per["control"][c])
        rank = np.arange(len(ks))
        fig, ax = figs[j]
        for a in ("control", "full", "eeonly", "pponly"):
            ax.plot(rank, [per[a][c] for c in ks], ".", ms=2.5 if ax_name == "PP4" else 4, color=col[a], label=lab[a])
        ax.set_yscale("log"); ax.set_xlabel("class, by the control's error"); ax.set_ylabel(r"MSE of $\log|\mathcal{M}|^2$")
        ps.process_label(ax, "e$^+$e$^-$ test" if ax_name == "EE4" else "pp 13 TeV test", loc="upper left")
        fig, ax = figs[j + 2]
        ax.axhline(1.0, color=ps.C.grey, lw=0.8)
        for a in ("full", "eeonly", "pponly"):
            ax.plot(rank, [per[a][c] / per["control"][c] for c in ks], ".", ms=2.5 if ax_name == "PP4" else 4,
                    color=col[a], label=lab[a])
        ax.set_yscale("log"); ax.set_xlabel("class, by the control's error"); ax.set_ylabel("error over the control's")
        ps.process_label(ax, "e$^+$e$^-$ test" if ax_name == "EE4" else "pp 13 TeV test", loc="upper left")
    base = os.path.join(HERE, "figs", "beam_ablation")
    ps.save_panels(figs, base)
    ps.legend_strip(figs[0][1], base + "_legend", ncol=2)



# step 68 (Joaquin 2026-11-06 evening: why the e+e- default events cut the e+e- test error): per test event of every
# EE4 probe, every beam arm's prediction (each run's own --save-dir .npz, the same scored rows in every run), the
# event's kinematics and how densely each arm's training put events around it. The control trains the class on its
# catalog pool alone; the share arms on 0.75 of it plus 0.25 default events, generated as the EE4 test pool (same
# MadEvent setup, cuts and energies: ee_gain_beams.json same_generator_setup_as_its_EE4_pool 115/115, D227), so the
# test pool itself stands for the default events' density (self excluded).
EV_DIRS = {"control": ("v4_het_control_tagbeams", "v4_het_control_seed2_tagbeams"),
           "full": ("v4_het_default25_tagbeams", "v4_het_default25_seed2_tagbeams"),
           "eeonly": ("v4_het_default25_eeonly_tagbeams", "v4_het_default25_eeonly_seed2_tagbeams"),
           "pponly": ("v4_het_default25_pponly_tagbeams", "v4_het_default25_pponly_seed2_tagbeams")}
EV_VARS = ("logm2", "sqrts", "fwd", "coll", "ymin", "joint4", "joint5")
EV_EDGES = [-9.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 9.0]     # log10(catalog density / test density) at the event
M2_BIN = 0.25                                                # dex, the per-class log10|M|^2 histograms
OUT_EV = os.path.join(DATA, "ee_gain_events.json")

PULL_EV = r"""
import json, os, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "analysis/common")
from readers import find_pool
cfg = json.loads(sys.argv[1])
dirs, pairs, edges, N_CAT, K, BW = cfg["dirs"], cfg["pairs"], np.array(cfg["edges"]), cfg["n_cat"], cfg["k"], cfg["m2_bin"]
WIN = {"logm2": 0.1, "sqrts": float(np.log10(1.02)), "fwd": 0.1, "coll": 0.1, "ymin": 0.1}
S = os.environ["SCRATCH"]
def boost(p, b):
    b2 = np.clip((b ** 2).sum(-1), 1e-300, None)
    g = 1 / np.sqrt(np.clip(1 - b2, 1e-300, None))
    bp = (b[:, None, :] * p[..., 1:]).sum(-1)
    e = g[:, None] * (p[..., 0] - bp)
    v = p[..., 1:] + (((g - 1) / b2)[:, None] * bp - g[:, None] * p[..., 0])[..., None] * b[:, None, :]
    return np.concatenate([e[..., None], v], -1)
def feats(path, n=None):
    a = np.asarray(np.load(path, mmap_mode="r")[:n], dtype=np.float64)
    k = (a.shape[1] - 1) // 5
    p = a[:, :4 * k].reshape(-1, k, 4)
    P = p[:, 0] + p[:, 1]
    q = boost(p, P[:, 1:] / P[:, :1])
    s2 = np.clip(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1), 1e-300, None)
    ax = q[:, 0, 1:] / np.linalg.norm(q[:, 0, 1:], axis=-1, keepdims=True)
    f = q[:, 2:]
    u = f[..., 1:] / np.clip(np.linalg.norm(f[..., 1:], axis=-1, keepdims=True), 1e-300, None)
    c = (u * ax[:, None, :]).sum(-1)
    fwd = np.log10(np.clip(1 - np.abs(c), 1e-12, None)).min(1)
    if k > 3:
        iu = np.triu_indices(k - 2, 1)
        cij = np.einsum("nid,njd->nij", u, u)[:, iu[0], iu[1]]
        pij = f[:, iu[0]] + f[:, iu[1]]
        mij = pij[..., 0] ** 2 - (pij[..., 1:] ** 2).sum(-1)
        coll = np.log10(np.clip(1 - cij, 1e-12, None)).min(1)
        ymin = np.log10(np.clip(mij / s2[:, None], 1e-12, None)).min(1)
    else:
        coll = ymin = np.zeros(len(a))
    return {"logm2": np.log10(np.clip(np.abs(a[:, -1]), 1e-300, None)), "sqrts": 0.5 * np.log10(s2), "fwd": fwd,
            "coll": coll, "ymin": ymin}
def frac(ref, x, w, self_in=False):
    r = np.sort(ref)
    n = np.searchsorted(r, x + w, side="right") - np.searchsorted(r, x - w, side="left")
    return ((n - 1) / (len(r) - 1)) if self_in else n / len(r)
def ks(a, b):
    g = np.sort(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(np.sort(a), g, side="right") / len(a)
                               - np.searchsorted(np.sort(b), g, side="right") / len(b))))
def knn_ratio(C, T, ti, cols, K=K):
    # log10 (catalog density / test density) at the test events ti: among the K nearest of the union (the event's own
    # index removed, exactly), the counts from each, per sample size; features standardized by the test pool's spread
    # (floored at 0.05, the 2->2 classes' constant coll and ymin)
    sd = np.array([max(np.std(T[v]), 0.05) for v in cols])
    Xc = np.stack([C[v] for v in cols], 1) / sd; Xt = np.stack([T[v] for v in cols], 1) / sd
    U = np.concatenate([Xc, Xt]); lab = np.r_[np.zeros(len(Xc)), np.ones(len(Xt))]
    _, nb = cKDTree(U).query(Xt[ti], k=K + 1)
    own = (len(Xc) + np.asarray(ti))[:, None]
    keep = nb != own
    keep[keep.sum(1) > K, -1] = False                  # the event not among its K+1 (a tie): drop the farthest
    nb = nb[keep].reshape(len(ti), K)
    nc = (lab[nb] == 0).sum(1); nt = K - nc
    return np.log10((np.maximum(nc, 0.5) / len(Xc)) / (np.maximum(nt, 0.5) / (len(Xt) - 1)))
out = {}
for probe, cls, same in pairs:
    try:
        zs = {a: [np.load(f"{S}/zz_v4_short_{d}/EE4/{probe}.npz") for d in ds] for a, ds in dirs.items()}
    except Exception as e:
        out[probe] = {"missing": repr(e)[:200]}; continue
    z0 = zs["control"][0]
    pool, rows, t = str(z0["pool"]), z0["pool_row"], z0["t"].astype(np.float64)
    if any(str(z["pool"]) != pool or not np.array_equal(z["pool_row"], rows) or not np.allclose(z["t"], z0["t"])
           for v in zs.values() for z in v):
        out[probe] = {"missing": "scored rows differ between runs"}; continue
    T = feats(pool)
    cp = find_pool(cls, role="val")
    C = feats(cp, N_CAT) if cp else None
    # alignment: the scored truth t (log(|M|^2 F)) less log F against the pool's log10|M|^2 at pool_row
    lm = (t - z0["logF"].astype(np.float64)) / np.log(10) if len(z0["logF"]) == len(t) else None
    r = {"cls": cls, "same_process": same, "n": int(len(t)), "n_test_pool": int(len(T["logm2"])),
         "n_cat": int(len(C["logm2"])) if C else 0, "cat_pool": os.path.basename(cp) if cp else None,
         "align_maxdev": float(np.max(np.abs(lm - T["logm2"][rows]))) if lm is not None else None}
    se = {a: [((z["q"].astype(np.float64) - t) ** 2) for z in v] for a, v in zs.items()}
    r["mse"] = {a: [float(x.mean()) for x in v] for a, v in se.items()}
    # the error split per run: level (mean of q - t, squared) and shape (variance of q - t), correlation and spread
    # ratio std q / std t (as zero_shot_parent's record)
    def split(z):
        q = z["q"].astype(np.float64)
        return {"bias": float(np.mean(q - t)), "shape": float(np.var(q - t)), "corr": float(np.corrcoef(q, t)[0, 1]),
                "spread": float(np.std(q) / np.std(t))}
    r["split"] = {a: [split(z) for z in v] for a, v in zs.items()}
    if C is not None:
        cov = {}
        for v, w in WIN.items():
            x = T[v][rows]
            cs = frac(C[v], x, w); ts = frac(T[v], x, w, self_in=True)
            cov[v] = np.log10(np.maximum(cs, 0.5 / len(C[v])) / np.maximum(ts, 0.5 / (len(T[v]) - 1)))
        cov["joint4"] = knn_ratio(C, T, rows, ("logm2", "fwd", "coll", "ymin"))
        cov["joint5"] = knn_ratio(C, T, rows, ("logm2", "fwd", "coll", "ymin", "sqrts"))
        for kk in (16, 64):                                    # the joint ratio's sensitivity to the neighbour count
            cov[f"joint5_k{kk}"] = knn_ratio(C, T, rows, ("logm2", "fwd", "coll", "ymin", "sqrts"), K=kk)
        b = np.clip(np.digitize(np.stack([cov[v] for v in cov]), edges) - 1, 0, len(edges) - 2)
        r["bins"] = {v: {"n": np.bincount(b[i], minlength=len(edges) - 1).tolist(),
                         "sse": {a: [np.bincount(b[i], x, minlength=len(edges) - 1).round(6).tolist() for x in s]
                                 for a, s in se.items()}}
                     for i, v in enumerate(cov)}
        r["cov_median"] = {v: float(np.median(c)) for v, c in cov.items()}
        r["ks"] = {v: ks(C[v], T[v]) for v in WIN}
        r["median_cat"] = {v: float(np.median(C[v])) for v in WIN}
    r["median_test"] = {v: float(np.median(T[v])) for v in WIN}
    # log10|M|^2 histograms on a common 0.25-dex grid: catalog, the test pool, and the scored events' SSE per run
    lo = np.floor(min(T["logm2"].min(), C["logm2"].min() if C else 1e9) / BW) * BW
    hi = np.ceil(max(T["logm2"].max(), C["logm2"].max() if C else -1e9) / BW) * BW + BW
    e = np.arange(lo, hi + BW / 2, BW)
    xs = T["logm2"][rows]
    r["m2"] = {"lo": float(lo), "bw": BW, "test": np.histogram(T["logm2"], e)[0].tolist(),
               "cat": np.histogram(C["logm2"], e)[0].tolist() if C else None,
               "scored": np.histogram(xs, e)[0].tolist(),
               "sse": {a: [np.histogram(xs, e, weights=x)[0].round(6).tolist() for x in s] for a, s in se.items()}}
    # the sqrt(s) values of the test pool (the fixed e+e- energies) and the catalog's spread in log10 sqrt(s)
    r["energies"] = sorted({round(float(10 ** x), 1) for x in T["sqrts"]})[:12]
    out[probe] = r
print("EEEV " + json.dumps(out))
"""


def pull_events():
    """step 68 on CC (one read-only site run): ee_gain_events.json, per EE4 probe: the eight runs' MSE from their saved
    predictions, the test events binned by log10(catalog density / test density) at the event per variable (EV_EDGES;
    1D windows as pull_kin, plus a k=32 nearest-neighbour ratio in (log|M|^2, fwd, coll, ymin) and with log sqrt(s))
    with each run's sum of squared errors per bin, the class's catalog and test log10|M|^2 histograms with each run's
    SSE per |M|^2 bin, KS distances catalog vs test per variable"""
    import mixture_ab as mab
    from readers import site_python
    cls = mab.class_of()
    probes = sorted(json.load(open(os.path.join(mab.READS, f"{BEAM_RUNS['control'][0]}_EE4.json")))["mse"])
    pairs = [[p, cls(p), cls(p) == "v4_" + p.split("_", 1)[1]] for p in probes]
    cfg = {"dirs": EV_DIRS, "pairs": pairs, "edges": EV_EDGES, "n_cat": N_CAT, "k": 32, "m2_bin": M2_BIN}
    script = PULL_EV.replace("cfg = json.loads(sys.argv[1])", "cfg = json.loads(" + repr(json.dumps(cfg)) + ")")
    res = site_python(script, ["-"], "EEEV", timeout=3000)
    by = _default_classes()
    for p, v in res.items():
        v["trained_ee"] = cls(p) in by["ee"]
    json.dump({"design": "catalog_coverage step 68 (re-read of step 62's saved predictions, no new run)",
               "runs": BEAM_RUNS, "dirs": EV_DIRS, "edges": EV_EDGES, "n_cat": N_CAT, "probes": res},
              open(OUT_EV, "w"))
    miss = {p: v["missing"] for p, v in res.items() if "missing" in v}
    print(json.dumps({"n": len(res), "missing": len(miss), "first_missing": list(miss.items())[:5],
                      "align_maxdev_max": max((v.get("align_maxdev") or 0) for v in res.values() if "missing" not in v)}))


def _ev():
    d = json.load(open(OUT_EV))["probes"]
    return d, sorted(p for p, v in d.items() if v.get("trained_ee") and "bins" in v), \
        sorted(p for p, v in d.items() if not v.get("trained_ee") and "bins" in v)


def _bin_ratios(d, probes, v, min_n=20):
    """per coverage bin of variable v: events, pooled MSE per arm (seeds' mean SSE over events), and the median over
    classes (probes with >= min_n events in the bin) of the arm's MSE over the control's in that bin"""
    nb = len(EV_EDGES) - 1
    n = np.zeros(nb); S = {a: np.zeros(nb) for a in EV_DIRS}; cr = {a: [[] for _ in range(nb)] for a in EV_DIRS}
    for p in probes:
        b = d[p]["bins"][v]; n += b["n"]
        ss = {a: np.mean(b["sse"][a], 0) for a in EV_DIRS}
        for a in EV_DIRS:
            S[a] += ss[a]
            for i in range(nb):
                if b["n"][i] >= min_n and ss["control"][i] > 0:
                    cr[a][i].append(ss[a][i] / ss["control"][i])
    return [{"bin": EV_EDGES[i:i + 2], "n_events": int(n[i]), "share": round(float(n[i] / n.sum()), 4),
             "mse_pooled": {a: round(float(S[a][i] / n[i]), 4) if n[i] else None for a in EV_DIRS},
             "R_class_median": {a: round(float(np.median(cr[a][i])), 4) if cr[a][i] else None
                                for a in ("full", "eeonly", "pponly")},
             "n_classes": len(cr["full"][i])} for i in range(nb)]


def read_events():
    """step 68's reading -> data/ee_gain_mechanism.json: (1) where the EE4 test events sit against the class's
    catalog events (per class: median log10|M|^2 and kinematics of the test pool minus the catalog val pool's, KS
    distance per variable; the test share in |M|^2 bins where the catalog density is below a tenth of the test
    density, above and below the catalog median); (2) per variable, the test events binned by log10(catalog density /
    test density) at the event, each arm's error there (_bin_ratios), trained e+e- classes and held-out ones apart;
    (3) the error split per arm (class medians over the seeds' means): bias^2 / MSE, shape = var(q - t), correlation,
    spread ratio, from the saved predictions; (4) sensitivity: the joint ratio at k 16 and 64, the class floor min_n 10
    and 50; (5) the per-probe MSE from the predictions against the mixture_ab reads (largest absolute difference)"""
    d, tr, ho = _ev()
    out = {"design": "catalog_coverage step 68: re-read of step 62's saved EE4 predictions (runs BEAM_RUNS @5bf3e0f0)",
           "n_trained": len(tr), "n_held_out": len(ho), "edges": EV_EDGES,
           "density": "1D: share of the class's catalog val pool (1e4) within +-w of the event (w 0.1, log10 sqrt(s) "
                      "+-log10 1.02) over the test pool's share (self excluded); joint4/joint5: k=32 nearest neighbours in "
                      "(log10|M|^2, fwd, coll, ymin[, log10 sqrt(s)]) standardized by the test pool's spread"}
    sh = {v: np.array([d[p]["median_test"][v] - d[p]["median_cat"][v] for p in tr]) for v in d[tr[0]]["median_test"]}
    out["test_minus_catalog_median"] = {v: {"median": round(float(np.median(x)), 3),
                                            "q10_q90": [round(float(q), 3) for q in np.quantile(x, [0.1, 0.9])],
                                            "frac_positive": round(float(np.mean(x > 0)), 3)} for v, x in sh.items()}
    out["ks_median"] = {v: round(float(np.median([d[p]["ks"][v] for p in tr])), 3) for v in d[tr[0]]["ks"]}
    ab, be = [], []
    for p in tr:
        m = d[p]["m2"]; t = np.array(m["test"], float); c = np.array(m["cat"], float)
        td, cd = t / t.sum(), c / c.sum(); low = cd < 0.1 * td
        x = m["lo"] + m["bw"] * (np.arange(len(t)) + 0.5); med = d[p]["median_cat"]["logm2"]
        ab.append(td[low & (x > med)].sum()); be.append(td[low & (x <= med)].sum())
    out["test_share_where_catalog_below_tenth"] = {"above_catalog_median": round(float(np.median(ab)), 3),
                                                   "below_catalog_median": round(float(np.median(be)), 3)}
    out["trained"] = {v: _bin_ratios(d, tr, v) for v in EV_VARS}
    out["held_out"] = {v: _bin_ratios(d, ho, v) for v in ("logm2", "joint5")}
    import mixture_ab as mab
    dev = 0.0
    for a, runs in BEAM_RUNS.items():
        for i, run in enumerate(runs):
            rd = mab.read_of(run, "EE4")
            dev = max([dev] + [abs(v["mse"][a][i] - rd[p]) for p, v in d.items() if p in rd and "mse" in v])
    out["mse_check_max_abs_dev"] = float(f"{dev:.2g}")
    sp = {}
    for part, ps_ in (("trained", tr), ("held_out", ho)):
        sp[part] = {}
        for a in EV_DIRS:
            f = lambda key: [np.mean([x[key] for x in d[p]["split"][a]]) for p in ps_]
            mse = [np.mean(d[p]["mse"][a]) for p in ps_]
            sp[part][a] = {"mse": round(float(np.median(mse)), 3),
                           "bias2_over_mse": round(float(np.median([b ** 2 / m for b, m in zip(f("bias"), mse)])), 3),
                           "shape": round(float(np.median(f("shape"))), 3), "corr": round(float(np.median(f("corr"))), 3),
                           "spread": round(float(np.median(f("spread"))), 3)}
    out["error_split"] = sp
    out["sensitivity"] = {"joint5_k16": _bin_ratios(d, tr, "joint5_k16"), "joint5_k64": _bin_ratios(d, tr, "joint5_k64"),
                          "logm2_min_n10": _bin_ratios(d, tr, "logm2", min_n=10),
                          "logm2_min_n50": _bin_ratios(d, tr, "logm2", min_n=50)}
    json.dump(out, open(os.path.join(DATA, "ee_gain_mechanism.json"), "w"), indent=1)
    for part in ("trained", "held_out"):
        for v in ("logm2", "joint5"):
            for r in out[part][v]:
                print(part, v, r["bin"], r["n_events"], r["share"], r["R_class_median"], r["n_classes"], r["mse_pooled"])
    for k, rows in out["sensitivity"].items():
        print("sensitivity", k, [r["R_class_median"]["full"] for r in rows])
    g = _offset_table(d, tr)
    summ = {k: out[k] for k in ("test_minus_catalog_median", "ks_median", "test_share_where_catalog_below_tenth",
                                "mse_check_max_abs_dev", "error_split")}
    summ["R"] = {part: {v: [r["R_class_median"] for r in out[part][v]] for v in out[part]} for part in ("trained", "held_out")}
    summ["by_offset"] = g
    # the full arm's class-median R per bin at the other neighbour counts and class floors (the choices K=32, min_n=20)
    summ["sensitivity"] = {k: [r["R_class_median"]["full"] for r in rows] for k, rows in out["sensitivity"].items()}
    # the ablation's R (class medians, data/ee_gain_ablation.json from `ablation`), beside the mechanism
    ab = json.load(open(os.path.join(DATA, "ee_gain_ablation.json")))["R"]
    summ["ablation_R"] = {a: {pt: v[pt]["R"] for pt in ("EE4_trained", "EE4_held_out", "PP4_trained")} for a, v in ab.items()}
    print(json.dumps(summ))


def _offset_grid(d, tr):
    """x = log10|M|^2 minus the class's catalog median on a 0.25-dex grid: per bin the class-averaged densities of the
    catalog, the share mixture (0.75 catalog + 0.25 test) and the test pool, the scored events and each arm's SSE"""
    BW = M2_BIN; X = np.arange(-6.0, 12.0 + BW / 2, BW)
    dens = {k: np.zeros(len(X) - 1) for k in ("cat", "share", "test")}
    n_ev = np.zeros(len(X) - 1); sse = {a: np.zeros(len(X) - 1) for a in EV_DIRS}
    for p in tr:
        m = d[p]["m2"]; t = np.array(m["test"], float); c = np.array(m["cat"], float)
        j0 = int(round((m["lo"] - d[p]["median_cat"]["logm2"]) / BW)) - int(round(X[0] / BW))
        idx = j0 + np.arange(len(t)); ok = (idx >= 0) & (idx < len(X) - 1)
        for k, h in (("cat", c / c.sum()), ("test", t / t.sum()), ("share", 0.75 * c / c.sum() + 0.25 * t / t.sum())):
            np.add.at(dens[k], idx[ok], h[ok] / BW / len(tr))
        np.add.at(n_ev, idx[ok], np.array(m["scored"], float)[ok])
        for a in EV_DIRS:
            np.add.at(sse[a], idx[ok], np.mean(m["sse"][a], 0)[ok])
    return 0.5 * (X[1:] + X[:-1]), dens, n_ev, sse


def _offset_table(d, tr):
    """the offset grid in 2-dex ranges: test and catalog share, pooled MSE per arm"""
    xc, dens, n_ev, sse = _offset_grid(d, tr)
    out = []
    for lo_, hi_ in ((-6, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 12)):
        w = (xc >= lo_) & (xc < hi_)
        out.append({"x": [lo_, hi_], "events": int(n_ev[w].sum()), "test_share": round(float(dens["test"][w].sum() * M2_BIN), 3),
                    "catalog_share": round(float(dens["cat"][w].sum() * M2_BIN), 4),
                    "mse": {a: round(float(sse[a][w].sum() / n_ev[w].sum()), 3) for a in EV_DIRS}})
    return out


def fig_mechanism():
    """the mechanism figure, trained e+e- classes (115), x = log10|M|^2 minus the class's catalog median: (a) the
    densities, averaged over classes, of the control's training events (catalog), the e+e- share arms' (0.75 catalog
    + 0.25 default-sampled, the test pool standing for the default events) and the EE4 test events; (b) each arm's MSE
    of log|M|^2 per bin of the same x (events of every class pooled, seeds' mean; bins with >= 200 events, 0.1% of the
    230000 scored); (c) per bin of log10(catalog density / test density) at the test event in log10|M|^2, the class
    median of each arm's MSE over the control's; (d) the same in the joint (log|M|^2, angles, ymin, sqrt(s)) density
    -> figs/ee_mechanism_<a..d> + _legend"""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    d, tr, _ = _ev()
    lab = {"control": "no default share", "full": "default share, both beams", "eeonly": "default share, e+e- only",
           "pponly": "default share, pp only"}
    col = {"control": ps.C.grey, "full": ps.C.blue, "eeonly": ps.C.vermillion, "pponly": ps.C.green}
    xc, dens, n_ev, sse = _offset_grid(d, tr)
    figs = ps.panels(4)
    fig, ax = figs[0]
    for k, lb, cl in (("cat", "training, catalog", ps.C.grey), ("share", "training, with share", ps.C.blue),
                      ("test", "test events", ps.C.purple)):
        ax.step(xc, dens[k], where="mid", color=cl, label=lb)
    ax.set_yscale("log"); ax.set_ylim(1e-4, None)
    ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$ $-$ catalog median"); ax.set_ylabel("density per class")
    ps.legend(ax, "upper left")
    fig, ax = figs[1]
    ok = n_ev >= 200
    for a in EV_DIRS:
        ax.plot(xc[ok], sse[a][ok] / n_ev[ok], ".-", color=col[a], label=lab[a])
    ax.set_yscale("log"); ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$ $-$ catalog median")
    ax.set_ylabel(r"MSE of $\log|\mathcal{M}|^2$")
    for row in _offset_table(d, tr):
        print("panel b", row)
    pos = np.arange(len(EV_EDGES) - 1)
    ticks = ["$<\\!-2$"] + [f"${0.5 * (a + b):g}$" for a, b in zip(EV_EDGES[1:-2], EV_EDGES[2:-1])] + ["$>\\!0.5$"]
    for j, v in ((2, "logm2"), (3, "joint5")):
        fig, ax = figs[j]
        rows = _bin_ratios(d, tr, v)
        ax.axhline(1.0, color=ps.C.grey, label="control")
        for a in ("full", "eeonly", "pponly"):
            ax.plot(pos, [r["R_class_median"][a] for r in rows], "o-", color=col[a], label=lab[a])
        ax.set_yscale("log"); ax.set_xticks(pos[::2]); ax.set_xticklabels(ticks[::2])
        ax.set_xlabel(r"$\log_{10}(\rho_{\rm catalog}/\rho_{\rm test})$ at the event" +
                      (r", $|\mathcal{M}|^2$" if v == "logm2" else ", joint"))
        ax.set_ylabel("error over the control's")
    base = os.path.join(HERE, "figs", "ee_mechanism")
    ps.save_panels(figs, base)
    ps.legend_strip(figs[1][1], base + "_legend", ncol=2)

if __name__ == "__main__":
    {"pull": pull, "read": read, "pull_kin": lambda: pull(kin=True), "read_kin": read_kin, "read_level": read_level,
     "beams": beams, "ablation": ablation, "fig_ablation": fig_ablation,
     "pull_events": pull_events, "read_events": read_events, "fig_mechanism": fig_mechanism}[sys.argv[1]]()
