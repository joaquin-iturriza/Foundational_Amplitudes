"""pretrain_data_saving stage 2 costing (D373, D375): per target and D, one 8-trial single-fidelity DyHPO from scratch and
one fine-tuning the heldout parent, every trial at the horizon T(D), at the step rate measured in the calibration.

Horizon from the calibration's best step s_b (val mu MSE) of the same (target, D):
the stage-2 design passed 2026-11-10 (06:51 revision): best_frac < 0.8 (plateau): T = 1.1 s_b; best_frac >= 0.8 (still
falling at its cap): the extension rule, T = min(4 x the calibration's steps, 64000). Phase a (the scratch DyHPO that the
design runs first for e+e- > c c~ g and b b~ g g) is not in these numbers: its recorded estimate is 89.0 GPU-h.
e+e- > b b~ trains on V (D375): no V calibration exists, so its T is b b~ > Z Z's (the other 2->2 one-loop target) at the
same D, at its own measured step rate (same particle count and batch): an estimate.
GPU-h per cell = 2 arms x 8 trials x (T x rate + 0.08 h) (rate = main-stage traintime / steps; 0.08 h per trial for the
sigma stage and evaluation, as the passed design).
Disk: 23 MB per trial run dir (the calibration runs' size). Wall: max(longest trial, total / WIDTH), WIDTH from site pick.

    python analysis/data_saving/cost_stage2.py --width <site pick FA --jobs 816>
"""
import argparse
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, required=True, help="the width site pick FA --jobs <trials> offers at costing time")
a = ap.parse_args()
DAT = os.path.join(REPO, "analysis/data_saving/data")
rows = json.load(open(f"{DAT}/horizons_cal2.json")) + json.load(open(f"{DAT}/horizons_cal_vb_cal_vb2.json"))
# a usable calibration row: has a best and is a plain run (the superseded factor-on u u~ > e+e- Z runs are flagged)
cal = {(r["target"], r["k"]): r for r in rows if "s_b" in r and not r.get("not_vb") and not r.get("phys_factor")}
TRIALS, ARMS, MB, OVER = 8, 2, 23, 0.08
TARGETS = [("epem_ttx", r"$t\bar t$", range(2, 11)), ("epem_ccxg", r"$c\bar cg$", range(2, 11)),
           ("epem_bbxgg", r"$b\bar bgg$", range(2, 11)), ("bbx_zz_nlo", r"$b\bar b\to ZZ$, one-loop", range(2, 10)),
           ("uux_epemz_nlo", r"$u\bar u\to e^+e^-Z$, one-loop", range(2, 10)),
           ("epem_bbx_nlo", r"$e^+e^-\to b\bar b$, one-loop on virtual", range(2, 10))]


def horizon(r):
    if r["best_frac"] >= 0.8:
        return min(4 * r["steps"], 64000)
    return int(math.ceil(1.1 * r["s_b"] / r["every"])) * r["every"]


out, tot, longest, missing = [], 0.0, 0.0, []
for t, lab, ks in TARGETS:
    for k in ks:
        src = cal.get(("bbx_zz_nlo" if t == "epem_bbx_nlo" else t, k))
        rate = cal.get((t, k), {}).get("rate_s_per_step")
        if not src or not rate:
            missing.append((t, k))
            continue
        T = horizon(src)
        gpuh = ARMS * TRIALS * (T * rate / 3600 + OVER)
        tot += gpuh
        longest = max(longest, T * rate / 3600 + OVER)
        out.append({"target": t, "k": k, "D": round(10 ** (k / 2)), "T": T, "rate": rate, "gpuh": gpuh,
                    "best_frac": src["best_frac"], "s_b": src["s_b"]})
assert not missing, missing
n_trials = len(out) * ARMS * TRIALS
print(f"{'target':14s} {'D':>7s} {'T':>6s} {'s/step':>7s} {'GPU-h':>7s}")
for r in out:
    print(f"{r['target']:14s} {r['D']:7d} {r['T']:6d} {r['rate']:7.3f} {r['gpuh']:7.2f}")
per = {t: sum(r["gpuh"] for r in out if r["target"] == t) for t, _, _ in TARGETS}
for t, g in per.items():
    print(f"{t:14s} total {g:6.1f} GPU-h")
print(f"sweeps {len(out) * ARMS}, trials {n_trials}, total {tot:.1f} GPU-h; disk {n_trials * MB / 1024:.1f} GB; "
      f"wall ~{max(longest, tot / a.width):.1f} h at width {a.width} (longest trial {longest:.2f} h)")
json.dump({"cells": out, "per_target": per, "total_gpuh": tot, "trials": n_trials, "disk_gb": n_trials * MB / 1024,
           "width": a.width, "wall_h": max(longest, tot / a.width)}, open(f"{DAT}/cost_stage2.json", "w"), indent=1)

fig, ax = ps.figure()
for (t, lab, _), col in zip(TARGETS, ps.CYCLE):
    c = [r for r in out if r["target"] == t]
    ax.plot([r["D"] for r in c], [r["gpuh"] for r in c], marker="o", color=col, label=lab)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$D$")
ax.set_ylabel("GPU-h per cell")
ps.legend(ax, "upper left", ncol=2)
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/cost_stage2"))
