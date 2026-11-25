"""Plan selective_rescale (D552) stage 2 (design passed 2026-11-25): each rule at its own optimum, best against best.
Searches (sweeps_local/selres_<cell>_<arm>_b2048, 16 trials): below x=0.3 and the event gate on Z+4g 1e4 and 1e5, the
Z+4g 1e5 no-rescale and full-rescale controls, below x=0.1 on u u~ g 1e4 and 1e5; each arm = its chosen trial (seed 42,
data/stage2_chosen.json) and seeds 43/44 (runs selres_s2_<cell>_<arm>_s4x). Controls: Z+4g 1e4 no rescale at hp1333
(the no-rescale search's choice, d539s.json; also beside it at hp1961, d539.json, the lower of the two) and the full
rescale at hp1961 (its own search's choice, d539.json); Z+4g 1e5 the two new searches; u u~ g d539u.json /
d539u_kit.json (each arm's own search). Measure and r as stage1_read.py.

    python3 analysis/selective_rescale/stage2_read.py [--cached]   -> data/stage2.json, figs/stage2
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from stage1_read import FM, KIT, REMOTE, deepest, plot, ratio, site_python  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEEDS = (42, 43, 44)
LR_WINDOW = (3.1e-3, 4.1e-2)   # training.lr of every sweep_config_selres_*.yaml search space


def collect(chosen):
    args, kit = [], []
    for s, c in chosen.items():
        rp = json.load(open(glob.glob(os.path.join(ROOT, "sweeps_local", c["sweep"], "results",
                                                   f"hp{c['hp_idx']:04d}_t1953.json"))[0]))["result_path"]
        cc = c["run_dir"].split("/runs/")[0]   # the chosen trial's site checkout (all on CC)
        args.append(f"'{s}|42#{c['run_dir']}#{rp}'")
        for sd in SEEDS[1:]:
            n = f"selres_s2_{s}_s{sd}"
            args.append(f"'{s}|{sd}#{cc}/runs/{n}#{cc}/sweeps/{n}/results/fixed.json'")
        if s.startswith("uug"):
            kit += [{42: c["run_dir"].split("/runs/")[1]}.get(sd, f"selres_s2_{s}_s{sd}") for sd in SEEDS]
    val = site_python(REMOTE.replace("ZNG1K", "SELRES2"), args, "SELRES2", site="ccin2p3")
    k = site_python(KIT, kit, "SELRESKIT", site="ccin2p3")
    return {"val": val, "kit": k, "kit_runs": kit}


def main():
    chosen = json.load(open(os.path.join(HERE, "data", "stage2_chosen.json")))
    cache = os.path.join(HERE, "data", "stage2_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect(chosen)
    json.dump(raw, open(cache, "w"))
    s539, d539, u539 = (json.load(open(os.path.join(FM, f)))["cells"] for f in ("d539s.json", "d539.json", "d539u.json"))
    k539 = json.load(open(os.path.join(FM, "d539u_kit.json")))["cells"]
    res = {"missing": [], "blowups": [], "cells": {}, "lr_dec_from_edge": {}}

    def arm(s):   # best MSE(log|M|^2) of the chosen trial and its two seeds
        b = []
        for sd in SEEDS:
            v = raw["val"].get(f"{s}|{sd}", {})
            if not v.get("val_mse_curve") or "std" not in v:
                res["missing"].append(f"{s}|{sd}"); b.append(None); continue
            c = np.array(v["val_mse_curve"]) * v["std"] ** 2
            up = c > 10 * np.minimum.accumulate(c)   # val above 10x its best so far (D398)
            if up.any():
                res["blowups"].append({"run": f"{s}|{sd}", "at_pct": 100 * (int(np.argmax(up)) + 1) / len(c)})
            b.append(float(c.min()))
        lr = raw["val"].get(f"{s}|42", {}).get("lr")
        if lr:
            res["lr_dec_from_edge"][s] = float(min(np.log10(lr / LR_WINDOW[0]), np.log10(LR_WINDOW[1] / lr)))
        return b

    ctl = {"zng_1e4": {"off": s539["zng_1e4"]["off"]["best"], "rescale": d539["zng_1e4"]["eps1e-8"]["best"],
                       "off_hp1961": d539["zng_1e4"]["off"]["best"]},
           "zng_1e5": {"off": arm("zng_1e5_off"), "rescale": arm("zng_1e5_eps1e-8")},
           "uug_1e4": {"off": u539["uug_1e4"]["off"]["best"], "rescale": u539["uug_1e4"]["eps1e-8"]["best"]},
           "uug_1e5": {"off": u539["uug_1e5"]["off"]["best"], "rescale": u539["uug_1e5"]["eps1e-8"]["best"]}}
    schemes = {"zng_1e4": ["softbelow_x0p3", "eventgate_x0p1"], "zng_1e5": ["softbelow_x0p3", "eventgate_x0p1"],
               "uug_1e4": ["softbelow_x0p1"], "uug_1e5": ["softbelow_x0p1"]}
    for cell, arms in schemes.items():
        c = {k: {"best": v} for k, v in ctl[cell].items()}
        for a in arms:
            b = arm(f"{cell}_{a}")
            c[a] = {"best": b}
            if None not in b and None not in c["off"]["best"] + c["rescale"]["best"]:
                for k in [k for k in ctl[cell] if k != a]:
                    c[a][f"over_{k}"] = ratio(np.array(b), np.array(c[k]["best"]))
        if cell.startswith("uug"):
            d, a = cell[4:], "softbelow_x0p1"
            runs = raw["kit_runs"][3 * (cell == "uug_1e5"): 3 * (cell == "uug_1e5") + 3]
            deep = [deepest(raw["kit"][r]) if r in raw["kit"] else None for r in runs]
            if None in deep:
                res["missing"] += [f"kit:{r}" for r, x in zip(runs, deep) if x is None]
            else:
                fam = {f: [x[f] for x in deep] for f in deep[0]}
                c[a]["kit"] = fam
                c[a]["kit_dec_vs_off"] = {f: float(np.log10(np.median(v) / np.median(k539[d]["off"][f]))) for f, v in fam.items()}
                c[a]["kit_dec_vs_rescale"] = {f: float(np.log10(np.median(v) / np.median(k539[d]["eps1e-8"][f])))
                                              for f, v in fam.items()}
            rs = k539[d]["eps1e-8"]["soft_x"]
            c["soft_tolerance_dec"] = float(np.log10(max(rs) / min(rs)))
        res["cells"][cell] = c
    out = {}   # the design's outcomes
    for a in ("softbelow_x0p3", "eventgate_x0p1"):
        zc = [res["cells"][f"zng_{d}"][a] for d in ("1e4", "1e5")]
        out[a] = {"bulk_yes": all("over_off" in c and c["over_off"]["lo"] <= 1 and c["over_rescale"]["hi"] < 1 for c in zc),
                  "resolved_cost": [d for d, c in zip(("1e4", "1e5"), zc) if "over_off" in c and c["over_off"]["lo"] > 1]}
    for d in ("1e4", "1e5"):
        c = res["cells"][f"uug_{d}"]["softbelow_x0p1"]
        if "over_off" in c:
            o = c["over_off"]
            out[f"softbelow_x0p1_uug_{d}"] = {
                "bulk": "cost" if o["lo"] > 1 else "unresolved" if o["hi"] >= 1 else "gain",
                "smallest_resolved_factor": float(np.exp(2 * o["sigma_r"])),
                "soft_yes": c.get("kit_dec_vs_rescale", {}).get("soft_x", np.inf) <= res["cells"][f"uug_{d}"]["soft_tolerance_dec"]}
    res["outcome"] = out
    json.dump(res, open(os.path.join(HERE, "data", "stage2.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line
    plot(res, "stage2")


if __name__ == "__main__":
    main()
