#!/usr/bin/env python3
"""Readout of a collected KIT figure file (tools/kit_figures.py collect): per process and arm, the bulk_test log-MSE,
each hard family's deepest-bin log-MSE (the stratum farthest into the limit: smallest singular variable, or the
largest for a family whose limit is at large values, sudakov) and the fitted exponent (KIT evaluate.py: pred vs
truth vs exact). Prints one table per process.

    python tools/kit_readout.py analysis/kit_benchmark/data/kit_fig_B.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.kit_figures import family  # noqa: E402

UP = ("sudakov",)       # families whose limit is at the large end of the singular variable


def deepest(strata):
    """{family: (stratum, x, logmse)} at the deepest stratum of each hard family"""
    out = {}
    for k, v in strata.items():
        if not k.startswith("hard_test/"):
            continue
        fam = family(k.split("/")[1])
        up = any(u in fam for u in UP)
        cur = out.get(fam)
        if cur is None or (v["x"] > cur[1] if up else v["x"] < cur[1]):
            out[fam] = (k.split("/")[1], v["x"], v["logmse"])
    return out


def main(path):
    d = json.load(open(path))
    by = {}
    for name, r in sorted(d["runs"].items()):
        by.setdefault((r["file"], r["target"]), []).append((name, r))
    for (stem, tgt), runs in sorted(by.items()):
        print(f"== {stem} {tgt}")
        for name, r in runs:
            bulk = [v["logmse"] for k, v in r["strata"].items() if k.startswith("bulk_test/")]
            deep = deepest(r["strata"])
            sl = r.get("slopes") or {}
            cells = [f"{fam} {v[0].split(fam + '_')[-1]}: {v[2]:.3g}" for fam, v in sorted(deep.items())]
            exps = [f"{k.split('/')[0]} {v['pred']:+.2f} (truth {v['truth']:+.2f})" for k, v in sorted(sl.items())]
            print(f"  {r['mode']:6s} {r['tag'] or '-':8s} n={r['n']} bulk {bulk[0] if bulk else float('nan'):.3g} | "
                  + "; ".join(cells) + (" | exp " + "; ".join(exps) if exps else ""))


if __name__ == "__main__":
    main(sys.argv[1])
