"""pretrain_data_saving stage 1: the per-(target, D) training horizon T(D) from the scratch calibration runs (D356).

Reads, on CC, runs/data_saving/dsave_scratch_<target>_d<k>_t<steps>_<tag>/result.json (scripts/job_data_saving.sh):
its val_mse_curve, the validation MSE of the mean (training.heterosc_select=mu_mse) in standardized units, one point
every validate_every steps, times the run's prepd_std^2 (data_stats.json) = MSE(ln|M|^2) of the mean, as sec:ladder.
Per run: s_b and best (the curve's minimum and its step), best_frac = s_b / steps, s_4 = the first validation step
with MSE <= 1e-4 (None if never), the step rate (traintime_hours / steps, the whole main stage) and the horizon rule
of the design recorded 2026-11-10 03:34:
    T(D) = 1.1 s_4 if the curve reaches 1e-4; else 1.1 s_b if best_frac < 0.8 (plateaued);
    else "extend": rerun at min(4 steps, 64000) (still improving at its cap; compute-limited at 64000).
One-loop targets (*_nlo, trained on V/B; design passed 2026-11-10 09:15): a run whose data_stats.json is not a V/B run
or whose target transform resolved to signedlog/abslog (V changes sign) is listed apart (not_ln_v: its curve is not
the MSE of ln|V|); and a curve that does not reach 1e-4 but has its best in the last fifth still falling (the last
point the minimum, or below 0.9 of the value at 0.8 of the run), or a best above 5.4e-4 at D=1e4 (hp58's tuned one-loop
b b~), gives rule "to_dyhpo" (T from the stage-2 scratch DyHPO; no extend at a fixed lr).
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

TARGETS = ["epem_ttx", "epem_ccxg", "epem_bbxgg", "bbx_zz_nlo", "uux_epemz_nlo", "bbx_gg_nlo", "dg_uwm_nlo",
           "bbx_avevex_nlo", "db_bdz_nlo", "uc_cuz_nlo"]
LOOP_REF = 5.4e-4   # hp58's tuned one-loop b b~ scratch best at D=1e4 (results.tex, het_stage2.py @0b277b39)
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
                 trafos=st.get("amp_trafos_pp") or [st.get("amp_trafos")], over_born=st.get("target_over_born"),
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
    loop = r["target"].endswith("_nlo")
    if loop:
        # the one-loop arm reads MSE of ln|V/B| = MSE of ln|V| only on a plain log target (a sign-changing V/B
        # resolves to signedlog/abslog: another quantity) of a V/B run
        flat = [str(t) for ts in (r.get("trafos") or []) for t in (ts or [])]
        if not r.get("over_born") or any(t.startswith(("signedlog", "abslog")) for t in flat):
            return {**o, "not_ln_v": True, "trafos": flat, "over_born": r.get("over_born")}
    i = min(range(len(y)), key=y.__getitem__)
    s_b, best = (i + 1) * ev, y[i]
    s_4 = next(((j + 1) * ev for j, v in enumerate(y) if v <= LEVEL), None)
    frac = s_b / n
    up = lambda s: int(ev * math.ceil(1.1 * s / ev))
    falling = i == len(y) - 1 or y[-1] < 0.9 * y[int(0.8 * len(y)) - 1]
    if loop and s_4 is None and ((frac >= 0.8 and falling) or (r["k"] == 8 and best > LOOP_REF)):
        # the one-loop design's too-low-lr branch: a fixed-lr curve still falling at its end, or above hp58's tuned
        # one-loop best at D=1e4, gives no horizon; T(D) comes from the stage-2 scratch DyHPO (no extend at fixed lr)
        rule, T = "to_dyhpo", None
    elif s_4 is not None:
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
        elif r.get("not_ln_v"):
            print(f"{r['target']:11s} k={r['k']:2d}  NOT MSE of ln|V| (trafos {r['trafos']}, over_born {r['over_born']})")
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
