"""Plan finetune_mechanism D539: kit hard-bin scoring of the 12 u u~ g runs (job_kit.sh KIT_SCORE_ONLY on each run dir).
Per arm (off, eps1e-8) and D (1e4, 1e5): the deepest-bin log10-MSE of log10|M|^2 of each hard family and the bulk
(tools/kit_mlp_readout.deepest), per seed, and log10 of the seed-median ratio eps1e-8 / off.
All 12 runs blew up at hp1961 (d539.json) and are scored on their best-checkpoint model (the sigma stage loads it).

    python3 analysis/finetune_mechanism/d539_kit_read.py [--cached]   -> data/d539_kit.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from tools.kit_mlp_readout import deepest  # noqa: E402

SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else ""   # "s" = the step-4 set at hp1333
TAG = f"d539{SET}"
RUNS = [f"ftmech_{TAG}_uug_{d}_{a}_s{s}" for d in ("1e4", "1e5") for a in ("off", "eps1e-8") for s in (42, 43, 44)]
SITE = r'''
import json, os, sys
out = {}
for r in sys.argv[1:]:
    p = os.path.join("runs", r, "kit_benchmark", "kit_result.json")
    if os.path.exists(p):
        d = json.load(open(p)); d = d.get("strata", d)
        out[r] = {k: v["logmse"] for k, v in d.items() if isinstance(v, dict) and "logmse" in v}
print("D539KIT " + json.dumps(out))
'''


def main():
    cache = os.path.join(HERE, "data", f"{TAG}_kit_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else site_python(SITE, RUNS, "D539KIT")   # same marker for both sets
    json.dump(raw, open(cache, "w"))
    deep = {r: deepest(st) for r, st in raw.items()}
    res = {"missing": [r for r in RUNS if r not in raw], "cells": {}}
    for d in ("1e4", "1e5"):
        cell = {a: {f: [deep.get(f"ftmech_{TAG}_uug_{d}_{a}_s{s}", {}).get(f) for s in (42, 43, 44)]
                    for f in sorted(deep[next(iter(deep))])} for a in ("off", "eps1e-8")}
        # logmse is the MSE of log10|M|^2: log10 of the seed-median ratio, eps1e-8 over off, in decades
        cell["delta_dec"] = {f: float(np.log10(np.median([x for x in cell["eps1e-8"][f] if x is not None])
                                               / np.median([x for x in cell["off"][f] if x is not None]))) for f in cell["off"]}
        res["cells"][d] = cell
    json.dump(res, open(os.path.join(HERE, "data", f"{TAG}_kit.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
