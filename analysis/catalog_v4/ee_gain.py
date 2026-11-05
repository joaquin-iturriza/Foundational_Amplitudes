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
    python3 analysis/catalog_v4/ee_gain.py pull | read
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


def pull():
    from readers import site_python
    pr = probes()
    pairs = [[p, c] for p, (ax, c) in pr.items()]
    # the probe list goes over inside the script (site_python sends only the script)
    script = PULL.replace("pairs = json.load(open(sys.argv[4]))", "pairs = " + json.dumps(pairs))
    res = site_python(script, [str(W), str(SHARE), str(N_CAT), "-"], "EEGAIN", timeout=1800)
    for p, v in res.items():
        v["axis"] = pr[p][0]
    json.dump({"window_rel": W, "share": SHARE, "n_cat": N_CAT, "probes": res}, open(OUT, "w"), indent=1)
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


if __name__ == "__main__":
    {"pull": pull, "read": read}[sys.argv[1]]()
