"""Hand-off, "the probes that do not gain": why no pretraining helps ee -> Z gamma (gain_heatmap: -0.1 to -0.3 decades
averaged over D at most rungs). Per probe and D cell (the 8k grid, cells.py): scratch's best, the fine-tunes' median best
over the ten pretrainings, the gain log10(scratch / fine-tune) per rung, and two readings of the searches' validation
curves: the fraction of trials whose best step falls in the first quarter of the run (early stopping), and the median
ratio of the last validation to the best (a curve that turns back up). Also scratch's drop from one half-decade of D
to the next, the step where a learning curve in D is steepest.
    python analysis/transfer/nogain.py      -> data/nogain.json, the table printed
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ARM, S, best, scratch  # noqa: E402
from rung_grid import P, fam  # noqa: E402

K = range(2, 9)


def trials(names):
    return [t for n in names for t in S.get(n, []) + S.get(n + "_002", []) if t.get("val_curve")]


def main():
    out = {}
    for p in P:
        rows = {}
        for k in K:
            s = scratch(p, k, steered=False)[0]
            if not s:
                continue
            ft = {r: best(f"{fam(r)}_{p}_d{k}")[0] for r in range(10)}
            ftt = trials([f"{fam(r)}_{p}_d{k}" for r in range(10)])
            sct = trials([f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"])
            early = lambda L: float(np.mean([t["best_step"] < 0.25 * t["T"] for t in L])) if L else None
            turn = lambda L: float(np.median([t["val_curve"][-1] / min(t["val_curve"]) for t in L])) if L else None
            rows[k] = {"scratch": s, "ft_median": float(np.median([v for v in ft.values() if v])),
                       "gain": {r: (float(np.log10(s / v)) if v else None) for r, v in ft.items()},
                       "early_ft": early(ftt), "early_scr": early(sct), "turn_ft": turn(ftt), "turn_scr": turn(sct)}
        ks = sorted(rows)
        drops = {k: float(np.log10(rows[a]["scratch"] / rows[k]["scratch"])) for a, k in zip(ks, ks[1:])}
        out[p] = {"cells": rows, "scratch_drop": drops}
        steep = max(drops, key=drops.get)
        g = {k: np.nanmedian([x for x in rows[k]["gain"].values() if x is not None]) for k in ks}
        print(f"{p:12s} steepest scratch step into d{steep} ({drops[steep]:.2f} decades); median gain per d: "
              + " ".join(f"d{k} {g[k]:+.2f}" for k in ks))
    json.dump(out, open(os.path.join(HERE, "data", "nogain.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    main()
