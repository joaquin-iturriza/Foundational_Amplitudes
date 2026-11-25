"""Plan selective_rescale (D552) stage 1: three event-dependent rescale schemes at each cell's no-rescale best HPs
(Z+4g hp1333; u u~ g the chosen trial of ftmech_d539_uug_<D>_off_b2048), seeds 42-44, runs selres_s1_<proc>_<D>_<scheme>_s<s>
(seed_chunks/selres_stage1_all.txt), against D539's controls at the same HPs: Z+4g off / rescale = d539s.json (hp1333),
u u~ g off = d539u.json (its own optimum, the same HPs), u u~ g rescale at its own optimum (d539u.json, soft reference).

Measure (design 2026-11-25): best val MSE(log|M|^2) = min val mu-MSE x prepd_std^2; r = three-seed geometric mean over
the control, 2-sigma interval (zng4g_d1e3_read.ratio). u u~ g kit deepest-bin log10-MSE (tools/kit_mlp_readout.deepest),
seed medians; soft tolerance = the rescale arm's seed median plus its three-seed spread log10(max/min).

    python3 analysis/selective_rescale/stage1_read.py [--cached] [--site lxplus]   -> data/stage1.json, figs/stage1
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "finetune_mechanism"))
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from tools.kit_mlp_readout import deepest  # noqa: E402
from zng4g_d1e3_read import REMOTE, ratio  # noqa: E402

SITE = sys.argv[sys.argv.index("--site") + 1] if "--site" in sys.argv else "lxplus"
SCHEMES = ["softbelow_x0p1", "softbelow_x0p3", "eventgate_x0p1"]
CELLS = [(p, d) for p in ("zng", "uug") for d in ("1e4", "1e5")]
SEEDS = (42, 43, 44)
FM = os.path.join(ROOT, "analysis", "finetune_mechanism", "data")
KIT = r'''
import json, os, sys
out = {}
for r in sys.argv[1:]:
    p = os.path.join("runs", r, "kit_benchmark", "kit_result.json")
    if os.path.exists(p):
        d = json.load(open(p)); d = d.get("strata", d)
        out[r] = {k: v["logmse"] for k, v in d.items() if isinstance(v, dict) and "logmse" in v}
print("SELRESKIT " + json.dumps(out))
'''


def name(p, d, a, s):
    return f"selres_s1_{p}_{d}_{a}_s{s}"


def main():
    cache = os.path.join(HERE, "data", "stage1_raw.json")
    if "--cached" in sys.argv:
        raw = json.load(open(cache))
    else:
        runs = [name(p, d, a, s) for p, d in CELLS for a in SCHEMES for s in SEEDS]
        val = site_python(REMOTE.replace("ZNG1K", "SELRES"), [f"'{r}#runs/{r}#sweeps/{r}/results/fixed.json'" for r in runs],
                          "SELRES", site=SITE)
        kit = site_python(KIT, [r for r in runs if "_uug_" in r], "SELRESKIT", site=SITE)
        raw = {"val": val, "kit": kit}
        json.dump(raw, open(cache, "w"))
    s539, u539 = (json.load(open(os.path.join(FM, f))) for f in ("d539s.json", "d539u.json"))
    k539 = json.load(open(os.path.join(FM, "d539u_kit.json")))["cells"]
    ctl = {("zng", d): {"off": s539["cells"][f"zng_{d}"]["off"]["best"], "rescale": s539["cells"][f"zng_{d}"]["eps1e-8"]["best"]}
           for d in ("1e4", "1e5")}
    ctl |= {("uug", d): {"off": u539["cells"][f"uug_{d}"]["off"]["best"], "rescale": u539["cells"][f"uug_{d}"]["eps1e-8"]["best"]}
            for d in ("1e4", "1e5")}
    res = {"missing": [], "blowups": [], "cells": {}}
    for p, d in CELLS:
        cell = {k: {"best": v} for k, v in ctl[(p, d)].items()}
        for a in SCHEMES:
            b = []
            for s in SEEDS:
                v = raw["val"].get(name(p, d, a, s), {})
                if not v.get("val_mse_curve") or "std" not in v:
                    res["missing"].append(name(p, d, a, s)); b.append(None); continue
                c = np.array(v["val_mse_curve"]) * v["std"] ** 2
                if np.any(c > 10 * np.minimum.accumulate(c)):
                    res["blowups"].append(name(p, d, a, s))
                b.append(float(c.min()))
            cell[a] = {"best": b}
            if None not in b:
                cell[a]["over_off"] = ratio(np.array(b), np.array(cell["off"]["best"]))
                cell[a]["over_rescale"] = ratio(np.array(b), np.array(cell["rescale"]["best"]))
        if p == "uug":   # kit: seed medians and decades against both controls
            kc = k539[d]
            for a in SCHEMES:
                deep = [deepest(raw["kit"][name(p, d, a, s)]) if name(p, d, a, s) in raw["kit"] else None for s in SEEDS]
                if None in deep:
                    res["missing"] += [f"kit:{name(p, d, a, s)}" for s, x in zip(SEEDS, deep) if x is None]; continue
                fam = {f: [x[f] for x in deep] for f in deep[0]}
                cell[a]["kit"] = fam
                cell[a]["kit_dec_vs_off"] = {f: float(np.log10(np.median(v) / np.median(kc["off"][f]))) for f, v in fam.items()}
                cell[a]["kit_dec_vs_rescale"] = {f: float(np.log10(np.median(v) / np.median(kc["eps1e-8"][f]))) for f, v in fam.items()}
            rs = kc["eps1e-8"]["soft_x"]
            cell["soft_tolerance_dec"] = float(np.log10(max(rs) / min(rs)))
        res["cells"][f"{p}_{d}"] = cell
    for a in SCHEMES:   # the design's YES, per scheme
        bulk = all(res["cells"][f"zng_{d}"].get(a, {}).get("over_off", {}).get("lo", 9) <= 1 and
                   res["cells"][f"zng_{d}"][a]["over_rescale"]["hi"] < 1 for d in ("1e4", "1e5")
                   if "over_off" in res["cells"][f"zng_{d}"].get(a, {}))
        soft = [res["cells"][f"uug_{d}"][a]["kit_dec_vs_rescale"]["soft_x"] <= res["cells"][f"uug_{d}"]["soft_tolerance_dec"]
                for d in ("1e4", "1e5") if "kit_dec_vs_rescale" in res["cells"][f"uug_{d}"].get(a, {})]
        res.setdefault("yes", {})[a] = {"bulk": bulk, "soft": bool(soft) and all(soft), "complete": len(soft) == 2}
    json.dump(res, open(os.path.join(HERE, "data", "stage1.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line


if __name__ == "__main__":
    main()
