"""pretrain_data_saving stage 1: the per-(target, D) training horizon T(D) from the scratch calibration runs (D356).

Reads, on CC, runs/data_saving/dsave_scratch_<target>_d<k>_t<steps>_<tag>/result.json (scripts/job_data_saving.sh):
its val_mse_curve, the validation MSE of the mean (training.heterosc_select=mu_mse) in standardized units, one point
every validate_every steps, times the run's prepd_std^2 (data_stats.json) = MSE(ln|M|^2) of the mean, as sec:ladder.
Per run: s_b and best (the curve's minimum and its step), best_frac = s_b / steps, s_4 = the first validation step
with MSE <= 1e-4 (None if never), the step rate (traintime_hours / steps, the whole main stage) and the horizon rule
of the design recorded 2026-11-10 03:34:
    T(D) = 1.1 s_4 if the curve reaches 1e-4; else 1.1 s_b if best_frac < 0.8 (plateaued);
    else "extend": rerun at min(4 steps, 64000) (still improving at its cap; compute-limited at 64000).
T is rounded up to a multiple of validate_every. A run whose curve holds a non-finite value is listed apart
(blowup, with its first non-finite step).

    python analysis/data_saving/horizons.py [--tag cal2] [--out analysis/data_saving/data/horizons_<tag>.json]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from analysis.common.readers import site_python  # noqa: E402

TARGETS = ["epem_ttx", "epem_ccxg", "epem_bbxgg"]
LEVEL = 1e-4

REMOTE = r'''
import glob, json, os, re, sys
tag = sys.argv[1]
out = []
for d in sorted(glob.glob("runs/data_saving/dsave_scratch_*_d*_t*_" + tag)):
    m = re.match(r"dsave_scratch_(.+)_d(\d+)_t(\d+)_" + re.escape(tag) + "$", os.path.basename(d))
    if not m:
        continue
    r = {"dir": d, "target": m.group(1), "k": int(m.group(2)), "steps": int(m.group(3))}
    try:
        res = json.load(open(d + "/result.json"))
        st = json.load(open(d + "/data_stats.json"))
        r.update(curve=res.get("val_mse_curve"), every=res.get("validate_every"), best_step=res.get("best_step"),
                 hours=res.get("traintime_hours"), prepd_std=st["prepd_std"][0],
                 sigma_done=os.path.isdir(d + "/sigma_stage/models") and bool(os.listdir(d + "/sigma_stage/models")))
    except Exception as e:
        r["error"] = repr(e)
    out.append(r)
print("HZN " + json.dumps(out))
'''


def horizon(r):
    """The readout of one run (module docstring)."""
    y = [v * r["prepd_std"] ** 2 for v in r["curve"]]
    ev, n = int(r["every"]), int(r["steps"])
    bad = next((i for i, v in enumerate(y) if not math.isfinite(v)), None)
    o = {"target": r["target"], "k": r["k"], "D": round(10 ** (r["k"] / 2)), "steps": n, "every": ev,
         "rate_s_per_step": 3600 * r["hours"] / n if r.get("hours") else None, "sigma_stage": r.get("sigma_done")}
    if bad is not None:
        return {**o, "blowup_step": (bad + 1) * ev}
    i = min(range(len(y)), key=y.__getitem__)
    s_b, best = (i + 1) * ev, y[i]
    s_4 = next(((j + 1) * ev for j, v in enumerate(y) if v <= LEVEL), None)
    frac = s_b / n
    up = lambda s: int(ev * math.ceil(1.1 * s / ev))
    if s_4 is not None:
        rule, T = "crossed", up(s_4)
    elif frac < 0.8:
        rule, T = "plateau", up(s_b)
    elif n < 64000:
        rule, T = "extend", min(4 * n, 64000)
    else:
        rule, T = "compute_limited", n
    return {**o, "s_b": s_b, "best": best, "best_frac": frac, "s_4": s_4, "last": y[-1], "rule": rule, "T": T}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="cal2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = site_python(REMOTE, [a.tag], "HZN")
    res = []
    for r in rows:
        if "error" in r or not r.get("curve"):
            res.append({k: r.get(k) for k in ("target", "k", "steps", "error")} | {"missing": True})
            continue
        res.append(horizon(r))
    res.sort(key=lambda r: (TARGETS.index(r["target"]) if r["target"] in TARGETS else 9, r["k"]))
    for r in res:
        if r.get("missing"):
            print(f"{r['target']:11s} k={r['k']:2d}  missing ({r.get('error')})")
        elif "blowup_step" in r:
            print(f"{r['target']:11s} k={r['k']:2d}  BLOWUP at step {r['blowup_step']}")
        else:
            print(f"{r['target']:11s} k={r['k']:2d} D={r['D']:6d}  best {r['best']:.3e} at {r['s_b']:5d}/{r['steps']}"
                  f" ({r['best_frac']:.2f})  s_4 {r['s_4']}  -> {r['rule']:8s} T={r['T']}"
                  f"  {r['rate_s_per_step'] or float('nan'):.3f} s/step")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"horizons_{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
