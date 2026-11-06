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


if __name__ == "__main__":
    {"pull": pull, "read": read, "pull_kin": lambda: pull(kin=True), "read_kin": read_kin, "read_level": read_level,
     "beams": beams, "ablation": ablation, "fig_ablation": fig_ablation}[sys.argv[1]]()
