#!/usr/bin/env python
"""Design check for sigma-driven generation (plan entry sigma_generation, step 3).

Reads analysis/divergences/steering_eval_summary.json (the July ee->uugg held-out evals) and prints, as one JSON line,
whether the step-3 measures move with the tested change and how far the seeds spread:
  logflat = mean over the populated y_min decades of the per-decade MSE of ln|M|^2 (steering_numbers.py's definition),
  deepest = the MSE in the deepest decade y_min < 1e-6.
Per arm and seed (base, sigma^gamma for gamma 1, 2, 3, 5, 10), and the fixed-fraction uugg_f* evals (single seed).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "steering_eval_summary.json")))["files"]


def dec(t):
    return np.array([np.nan if d[1] is None else d[1] for d in F[t]["dec"]])


def stats(t):
    d = dec(t)
    return dict(logflat=float(np.nanmean(d)), deepest=float(d[0]), n_deepest=int(F[t]["dec"][0][0]))


out = {"arms": {}, "fixed_fraction": {}}
for tag in ("base", "sigma", "g2", "g3", "g5", "g10"):
    per = [stats(f"{tag}_s{s}") for s in (0, 1, 2)]
    lf = np.array([p["logflat"] for p in per]); dp = np.array([p["deepest"] for p in per])
    out["arms"][tag] = dict(logflat=lf.tolist(), deepest=dp.tolist(),
                            logflat_max_over_min=float(lf.max() / lf.min()),
                            deepest_max_over_min=float(dp.max() / dp.min()))
b = out["arms"]["base"]
for tag in ("g3", "g10"):
    a = out["arms"][tag]
    out["arms"][tag]["worst_seed_logflat_over_best_base_seed"] = float(max(a["logflat"]) / min(b["logflat"]))
    out["arms"][tag]["worst_seed_deepest_over_best_base_seed"] = float(max(a["deepest"]) / min(b["deepest"]))
for t in sorted(k for k in F if k.startswith("uugg_f")):
    out["fixed_fraction"][t] = stats(t)
print("DESIGNCHECK " + json.dumps(out))
