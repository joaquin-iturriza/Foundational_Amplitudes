#!/usr/bin/env python3
"""finetune_mechanism restart test readout (design passed on finetune_mechanism, 2026-11-20): on b b~ > Z Z tree level
at D in {100, 316, 3162}, each arm against its one-change pair. Arms (8-trial searches, sweep dsave_<stage>_ft_bbx_zz):
C' s7cosrerun (today's restart, cosine rerun), Oc s7cosstep (its step-3458 live weights, fresh optimizer), O s7optstate
(Oc + the parent's optimizer state), W' s7wsdrerun, Bc s7wsdstep (step-3026 live weights, fresh optimizer), B s7wsdbranch
(Bc + optimizer state, no warmup); S s7wsd scratch (the head-start reference).
Primary: per non-blow-up trial (warmup rise < 10) lag = lost_at - warmup_frac against the S best (step6_readout.lost_at),
the count with lag > 0.25 against the pair's, one-sided Fisher p; HOLDS when the share exceeds the pair's by >= 0.3 at
p < 0.05, NO within 0.2, else UNRESOLVED. Secondary: best val MSE(ln|M|^2) per D, pair ratios pooled (geometric mean)
against 5^(1/sqrt 3). Side: the search best's lag per D.

    python analysis/finetune_mechanism/restart_test_readout.py [--cached]   -> data/restart_test.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "data_saving"))
from step6_readout import BAR, LAG, gm, ln, lost_at, rise  # noqa: E402
from trial_overfit import driven_rows  # noqa: E402

CELLS = {100: (4, 4960), 316: (5, 7840), 3162: (7, 8500)}
STAGE = {"C'": "s7cosrerun_ft", "Oc": "s7cosstep_ft", "O": "s7optstate_ft", "W'": "s7wsdrerun_ft",
         "Bc": "s7wsdstep_ft", "B": "s7wsdbranch_ft", "S": "s7wsd_scr"}
PAIRS = [("O", "Oc"), ("Oc", "C'"), ("B", "Bc"), ("Bc", "W'")]


def name(arm, D):
    k, T = CELLS[D]
    return f"dsave_{STAGE[arm]}_bbx_zz_d{k}_t{T}"


def main(cached):
    cache = os.path.join(HERE, "data", "restart_test_rows.json")
    if cached:
        rows = json.load(open(cache))
    else:
        names = [name(a, D) for a in STAGE for D in CELLS if os.path.isdir(os.path.join(REPO, "sweeps_local", name(a, D)))]
        rows = driven_rows(names)
        json.dump(rows, open(cache, "w"))
    rows = [ln(r) for r in rows if r.get("val") and len(r["val"]) > 3]
    out = {"cells": {}, "pairs": {}}
    trials = {a: [] for a in STAGE}
    for D in CELLS:
        g = {a: [r for r in rows if r["sweep"] == name(a, D)] for a in STAGE}
        best = {a: min(v, key=lambda r: r["val_min"]) for a, v in g.items() if v}
        if "S" not in best:
            continue
        cell = {}
        for a, v in g.items():
            if not v:
                continue
            b = best[a]
            c = {"n": len(v), "best_val": b["val_min"], "best_run": b["run_dir"]}
            if a != "S":
                wf = lambda r: r["hp"].get("warmup_frac") or 0.0   # B: warmup fixed 0
                lost = lost_at(b, best["S"])[0]
                c["best_lag"] = (1.0 if lost is None else lost) - wf(b)
                for r in v:
                    if rise(dict(r, hp=dict(r["hp"], warmup_frac=wf(r)))) < 10:
                        lo = lost_at(r, best["S"])[0]
                        trials[a].append((1.0 if lo is None else lo) - wf(r))
            cell[a] = c
        for x, y in PAIRS:
            if x in cell and y in cell:
                cell[f"{x}/{y}"] = cell[x]["best_val"] / cell[y]["best_val"]
        out["cells"][D] = cell
    for x, y in PAIRS:
        tx, ty = np.array(trials[x]), np.array(trials[y])
        if not len(tx) or not len(ty):
            continue
        kx, ky = int((tx > LAG).sum()), int((ty > LAG).sum())
        sx, sy = kx / len(tx), ky / len(ty)
        p = fisher_exact([[kx, len(tx) - kx], [ky, len(ty) - ky]], alternative="greater")[1]
        hs = "HOLDS" if sx - sy >= 0.3 and p < 0.05 else "NO" if abs(sx - sy) <= 0.2 else "UNRESOLVED"
        v = [c[f"{x}/{y}"] for c in out["cells"].values() if f"{x}/{y}" in c]
        G = gm(v) if v else None
        out["pairs"][f"{x}/{y}"] = {"kept": [kx, len(tx)], "pair_kept": [ky, len(ty)], "share_diff": sx - sy, "fisher_p": p,
                                    "head_start": hs, "median_lag": float(np.median(tx)), "pair_median_lag": float(np.median(ty)),
                                    "val_pooled": G, "val": None if G is None else
                                    "LOWER" if G < 1 / BAR else "HIGHER" if G > BAR else "WITHIN"}
    json.dump(out, open(os.path.join(HERE, "data", "restart_test.json"), "w"), indent=1)
    print(json.dumps(out["pairs"], indent=1))


if __name__ == "__main__":
    main("--cached" in sys.argv)
