"""pretrain_data_saving, D408: the original transfer results against today's, side by side, on e+e- > t t~.

Old one loop (draft Fig. 22's t t~ panel, results.tex fig:transfer_ttbar; Jean Zay sweeps/cscan_* and the older
solo/scaling sweeps that build_compute_scan_walltime.py reads): e+e- > t t~ virtual alone (ee_ttbar_nlo_virt_e4),
scratch ("solo") against fine-tunes from the 8-process foundation (tree t t~ in it) and the 25-process one (targets
out); per data.subsample S and horizon t the best trial by validation; its val and test MSE x divisor^2 (std of the
log target over the first S rows), the read the old figure plots. Training events D = 0.7 S (train_test_val
0.7/0.2/0.1). Each (arm, S) point is the minimum over horizons (the most compute it had).

Old tree (the ladder, sec:ladder, catalog_v3): e+e- > t t~ probe, scratch against rung fine-tunes, best trial of
each search, val MSE(ln|M|^2) (analysis/transfer/cells.py, scratch_sweeps.json).

New (stage 2, this entry): stage2_readout.json, best trial per search; val_ln = MSE of the run's log target in ln
units (tree ln|M|^2, one loop ln|V/B|, whose error equals that of ln|V| as B is exact).

    python analysis/data_saving/old_vs_new.py --fetch     # Jean Zay + CC reads -> data/old_vs_new.json
    python analysis/data_saving/old_vs_new.py --configs   # saved configs of one run per setup, the keys that differ
    python analysis/data_saving/old_vs_new.py             # figure figs/old_vs_new.{png,pdf} + printed table
"""
import argparse
import json
import os
import sys

REPO = "/home/joaquin/work/FA"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "transfer"))
from analysis.common.readers import site_python  # noqa: E402

HERE = os.path.join(REPO, "analysis", "data_saving")
OUT = os.path.join(HERE, "data", "old_vs_new.json")

OLD_V = {   # arm -> S -> sweep prefixes (as build_compute_scan_walltime.CURVES)
    "scratch": {"1k": ["cscan_solo_D1k"], "10k": ["solo_nh8_10k_virt", "cscan_solo_D10kext"],
                "100k": ["scaling_solo_nh8_lowt_virt", "scaling_solo_nh8_anchor_virt_002",
                         "scaling_solo_nh8_curve_virt", "scaling_solo_nh8_anchor2_virt"],
                "1M": ["cscan_solo_D1M"]},
    "ft8": {s: [f"cscan_ft8_D{s}"] for s in ("1k", "10k", "100k", "1M")},
    "ft25": {s: [f"cscan_ft25_D{s}", f"cscan_ft25_D{s}_002"] for s in ("1k", "10k", "100k", "1M")},
}
SUB = {"1k": 1000, "10k": 10000, "100k": 100000, "1M": 1000000}

JZ_PROBE = r'''
import glob, json, os, re, sys, base64
import numpy as np, yaml
OLD = json.loads(base64.b64decode(sys.argv[1])); SUB = json.loads(base64.b64decode(sys.argv[2]))
amp = np.load("data/ee_ttbar_nlo_virt_e4.npy")[:, -1]
out = {"neg_frac": float((amp <= 0).mean()), "cells": [], "configs": {}}
for arm, per in OLD.items():
    for s, prefs in per.items():
        a = amp[:SUB[s]]
        v = np.sign(a) * np.log1p(np.abs(a)) if (a <= 0).any() else np.log(a)
        div = float(v.std())
        for pre in prefs:
            for d in glob.glob(f"sweeps/{pre}_eettbarnlovirte4_t*"):
                m = re.match(rf"^{re.escape(pre)}_eettbarnlovirte4_t(\d+)$", os.path.basename(d))
                if not m:
                    continue
                recs = []
                for f in glob.glob(d + "/results/*.json"):
                    try:
                        r = json.load(open(f))
                    except Exception:
                        continue
                    if r.get("test_loss", 0) > 0 and "val_loss" in r:
                        r["file"] = os.path.basename(f); recs.append(r)
                if not recs:
                    continue
                b = min(recs, key=lambda r: r["val_loss"])
                out["cells"].append(dict(arm=arm, S=s, sweep=os.path.basename(d), t=int(m.group(1)), n=len(recs),
                                         div=div, val=b["val_loss"] * div ** 2, test=b["test_loss"] * div ** 2,
                                         file=b["file"], hours=b.get("traintime_hours")))
for arm in ("scratch", "ft8"):
    for s in ("10k", "100k"):
        cs = [c for c in out["cells"] if c["arm"] == arm and c["S"] == s]
        if not cs:
            continue
        c = min(cs, key=lambda c: c["val"])
        hp = int(re.match(r"hp(\d+)_", c["file"]).group(1))
        hits = sorted(glob.glob(f"runs/{c['sweep']}/*{hp:04d}*/config.yaml"))  # the best trial's own config or none
        out["configs"][f"old_V_{arm}_{s}"] = {"path": hits[0] if hits else None, "cell": c["sweep"], "hp": hp,
                                              "cfg": yaml.safe_load(open(hits[0])) if hits else None}
print("OVN " + json.dumps(out))
'''

CC_PROBE = r'''
import glob, json, sys, yaml, base64
want = json.loads(base64.b64decode(sys.argv[1])); out = {}
for k, pat in want.items():
    hits = sorted(glob.glob(pat))
    out[k] = {"path": hits[0] if hits else None, "cfg": yaml.safe_load(open(hits[0])) if hits else None}
print("OVN " + json.dumps(out))
'''


def b64(x):
    import base64
    return base64.b64encode(json.dumps(x).encode()).decode()


def fetch():
    jz = site_python(JZ_PROBE, [b64(OLD_V), b64(SUB)], "OVN", site="jeanzay")
    from cells import best, scratch  # the ladder's reader
    lad = {"scratch": {k: scratch("ee_ttbar", k, steered=False)[0] for k in range(2, 9)}}
    for f in [f"tp3_r{r}fte" for r in range(4, 10)] + ["tp3_finfte", "tp3_ft", "tp3_ftp"]:
        lad[f] = {k: best(f"{f}_ee_ttbar_d{k}")[0] for k in range(2, 9)}
    s6, f6 = scratch("ee_ttbar", 6, steered=False)[1], best("tp3_r9fte_ee_ttbar_d6")[1]
    s2 = json.load(open(os.path.join(HERE, "data", "stage2_readout.json")))["cells"]
    want = {}
    for c in s2:
        if abs(c["D"] - 3162.28) < 1 and c["target"] in ("epem_ttx", "ee_ttbar_nlo"):
            want[f"new_{c['target']}_{c['arm']}_d7"] = f"runs/{c['sweep']}/trial_{c['best_hp']:04d}*/config.yaml"
    cc = site_python(CC_PROBE, [b64(want)], "OVN", site="ccin2p3")
    json.dump({"old_V": jz, "ladder": lad, "cc_configs": cc, "ladder_trials": {"s6": s6, "f6": f6}},
              open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT)


def flat(d, pre=""):
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            o.update(flat(v, f"{pre}{k}."))
        else:
            o[f"{pre}{k}"] = v
    return o


def configs(d):
    src = {**d["old_V"]["configs"], **d["cc_configs"]}
    cf = {k: flat(v["cfg"]) for k, v in src.items() if v.get("cfg")}
    names = list(cf)
    for n in names:
        print(n, src[n]["path"])
    for key in sorted(set().union(*cf.values())):
        vals = [str(cf[n].get(key, "-"))[:26] for n in names]
        if len(set(vals)) > 1:
            print(f"{key:46s} " + " | ".join(vals))


def plot(d):
    sys.path.insert(0, REPO)
    import plot_style as ps  # noqa: F401
    import matplotlib.pyplot as plt
    s2 = json.load(open(os.path.join(HERE, "data", "stage2_readout.json")))["cells"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    tab = {}
    ax = axs[0]
    lad = d["ladder"]
    for f, lab, st in (("scratch", "old scr", dict(color="k", marker="o", ls="--", mfc="none")),
                       ("tp3_r5fte", "old ft, ladder rung 5", dict(color="C0", marker="s", ls="--", mfc="none")),
                       ("tp3_r9fte", "old ft, ladder rung 9", dict(color="C2", marker="^", ls="--", mfc="none"))):
        pts = sorted((10 ** (int(k) / 2), v) for k, v in lad[f].items() if v)
        if pts:
            ax.plot(*zip(*pts), **st, label=lab)
            tab[f"tree old {f}"] = pts
    for arm, st in (("scr", dict(color="k", marker="o")), ("ft", dict(color="C3", marker="D"))):
        pts = sorted((c["D"], c["val_ln"]) for c in s2
                     if c["target"] == "epem_ttx" and c["arm"] == arm and c.get("val_ln") is not None)
        ax.plot(*zip(*pts), **st, label=f"new {arm}")
        tab[f"tree new {arm}"] = pts
    ax.set_ylabel(r"val MSE($\ln|M|^2$)")
    ax = axs[1]
    cells = d["old_V"]["cells"]
    for arm, lab, st in (("scratch", "old scr", dict(color="k", marker="o", ls="--", mfc="none")),
                         ("ft8", "old ft, 8-process parent", dict(color="C0", marker="s", ls="--", mfc="none")),
                         ("ft25", "old ft, 25-process parent", dict(color="C2", marker="^", ls="--", mfc="none"))):
        pts = [(0.7 * SUB[s], min(c["val"] for c in cells if c["arm"] == arm and c["S"] == s))
               for s in SUB if any(c["arm"] == arm and c["S"] == s for c in cells)]
        if pts:
            ax.plot(*zip(*pts), **st, label=lab)
            tab[f"V old {arm}"] = pts
    for arm, st in (("scr", dict(color="k", marker="o")), ("ft", dict(color="C3", marker="D"))):
        pts = sorted((c["D"], c["val_ln"]) for c in s2
                     if c["target"] == "ee_ttbar_nlo" and c["arm"] == arm and c.get("val_ln") is not None)
        ax.plot(*zip(*pts), **st, label=f"new {arm}")
        tab[f"V new {arm}"] = pts
    ax.set_ylabel(r"val MSE($\ln|V|$)")
    for ax in axs:
        ax.set_xlabel(r"$D$ (training events)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"old_vs_new.{ext}"), dpi=110)
    for k, v in tab.items():
        print(f"{k:20s} " + "  ".join(f"{x:.3g}:{y:.1e}" for x, y in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--configs", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        return fetch()
    d = json.load(open(OUT))
    configs(d) if a.configs else plot(d)


if __name__ == "__main__":
    main()
