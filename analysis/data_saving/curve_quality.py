#!/usr/bin/env python3
"""pretrain_data_saving: does a trial's validation loss go down together with its training loss through the run
(Joaquin 2026-11-12: "train and val loss go down together, I don't see that in most of the curves you've shown").

Per trial, on the validation curve v_i and the training curve t_i (MSE on the validation and on the training events at
each check i), each first smoothed by a running median over 3 checks so that one spike does not decide:
  separation   (v/t at the end) / (smallest v/t in the run): how many times further validation sits above training at
               the end than where the two were closest. Good <= 3: validation pulled away from training by no more
               than a factor 3 (half a decade). A curve whose validation flattens while training keeps falling fails
               here, and so does one whose validation turns up.
  end_up       (v at the end) / (smallest v in the run). Good <= 2: validation did not turn up at the end.
Good = both. The bars were set by eye on the curves shown to Joaquin (s4_curves, lr_centre_curves, muonly_curves,
nullparent_curves): every curve whose validation visibly leaves its training curve has separation > 3; those he would
see as together (t t~ V/B D=3162 scratch 1.9, u u~ V D=1e4 fine-tune control 3.0) fall at or under it.
The best validation loss itself is read beside, never ranked by a shape bar.
The earlier reads (v_end/min v with a last-quarter change, 2026-11-12 afternoon; steady/spike/jitter/gap before it)
are withdrawn: they passed curves whose validation flattens while training falls by decades.

    python analysis/data_saving/curve_quality.py                # every cached trial (trial_overfit.json, s4 rows)
    python analysis/data_saving/curve_quality.py --cells <tg>   # the L(D) label per cell's best trial
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
BAR = {"separation": 3.0, "end_up": 2.0}


def smooth(x):
    """log10 of x, each point the median of itself and its two neighbours."""
    x = np.log10(np.asarray(x, float))
    return np.array([np.median(x[max(i - 1, 0):i + 2]) for i in range(len(x))])


def quality(v, t):
    lv, lt = smooth(v), smooth(t)
    n = min(len(lv), len(lt))
    g = lv[:n] - lt[:n]                        # log10 (val / train) at each check
    q = {"separation": float(10 ** (g[-1] - g.min())), "end_up": float(10 ** (lv[-1] - lv.min()))}
    q["well"] = bool(q["separation"] <= BAR["separation"] and q["end_up"] <= BAR["end_up"])
    return q


def trials():
    """Every trial with curves: the stage-2 searches (trial_overfit.json) and the s4 fine-tune searches."""
    from trial_overfit import load
    rows = []
    for r in load(cached=True):
        if r.get("val"):
            rows.append(dict(r, stage="s2"))
    for r in json.load(open(os.path.join(DATA, "s4_readout_rows.json"))):
        if r.get("val") and r["sweep"].startswith("dsave_s4_"):
            parts = r["sweep"].split("_")       # dsave_s4_ft_<target>_d<k>_t<T>
            k, T = int(parts[-2][1:]), int(parts[-1][1:])
            hp = r["hp"]
            rows.append(dict(r, stage="s4", arm="ft", target="_".join(parts[3:-2]), D=10 ** (k / 2), T=T,
                             peak_lr=hp["lr"] * hp["lr_scale"]))
    for r in rows:
        r["batch"] = int(min(16384, r["D"] / 2))         # scripts/job_data_saving.sh: batch = min(16384, D/2)
        r.update(quality(r["val"], r.get("train")))
    return rows


def main():
    rows = trials()
    out = {"bars": BAR, "groups": [], "trials": []}
    print(f"{len(rows)} trials with curves; good = separation <= {BAR['separation']:g} and end_up <= {BAR['end_up']:g}")
    for key in sorted({(r["target"], r["stage"], r["arm"]) for r in rows}):
        g = [r for r in rows if (r["target"], r["stage"], r["arm"]) == key]
        b = [r for r in g if r.get("best")]
        gr = {"target": key[0], "stage": key[1], "arm": key[2], "n": len(g), "good": sum(r["well"] for r in g),
              "best_n": len(b), "best_good": sum(r["well"] for r in b),
              "median": {c: float(np.median([r[c] for r in g])) for c in BAR}}
        out["groups"].append(gr)
        print(f"{key[0]:13s} {key[1]} {key[2]:3s} n={len(g):3d} good {gr['good']:3d}"
              + (f"  search bests good {gr['best_good']} of {gr['best_n']}" if b else "")
              + "  median: " + " ".join(f"{c} {gr['median'][c]:.3g}" for c in BAR))
    for r in rows:
        out["trials"].append({k: r.get(k) for k in ("sweep", "run_dir", "stage", "arm", "target", "D", "T", "peak_lr",
                                                     "best", "val_min_ln", "separation", "end_up", "well")})
    json.dump(out, open(os.path.join(DATA, "curve_quality.json"), "w"), indent=1)
    examples(rows)


TLAB = {"epem_ttx": r"$e^+e^-\to t\bar t$", "ee_ttbar_nlo": r"$e^+e^-\to t\bar t$, $V/B$",
        "ee_uu_nlo": r"$e^+e^-\to u\bar u$, $V$", "bbx_zz_nlo": r"$b\bar b\to ZZ$, $V/B$",
        "ee_uu": r"$e^+e^-\to u\bar u$", "bbx_zz": r"$b\bar b\to ZZ$"}
EX_TARGETS = ("epem_ttx", "ee_ttbar_nlo", "ee_uu", "ee_uu_nlo")  # b b~ > Z Z V/B has no good trial at all


def examples(rows):
    """figs/curve_examples_{good,bad}: per target one stage-2 search best at D >= 1000 that the measure calls good (the
    one with the lowest separation) and one it calls bad (the bad one with the median separation, a typical failure):
    training and validation MSE against step, in the units of the reported loss."""
    from suspects_readout import ps
    pick = {}
    for tg in EX_TARGETS:
        g = [r for r in rows if r["stage"] == "s2" and r["target"] == tg and r.get("best") and r["D"] >= 1000]
        good = sorted((r for r in g if r["well"]), key=lambda r: r["separation"])
        bad = sorted((r for r in g if not r["well"]), key=lambda r: r["separation"])
        pick[tg] = (good[0] if good else None, bad[len(bad) // 2] if bad else None)
    for i, kind in enumerate(("good", "bad")):
        fig, axs = ps.figure(ncols=2, nrows=2)
        for ax, tg in zip(axs.ravel(), EX_TARGETS):
            r = pick[tg][i]
            if r is None:
                print(f"  {kind} {tg}: none at D >= 1000")
                continue
            p = r["prepd_std"] ** 2
            st = np.arange(len(r["val"])) * (r.get("every") or 1)
            ax.plot(st, np.array(r["val"]) * p, color=ps.C.vermillion, label="validation")
            ax.plot(st[:len(r["train"])], np.array(r["train"]) * p, color=ps.C.blue, linestyle="--", label="training")
            ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("MSE")
            arm = "scratch" if r["arm"] == "scr" else "fine-tune"
            ps.process_label(ax, TLAB[tg] + f", {arm}, " + rf"$D={r['D']:.0f}$")
            print(f"  {kind} {tg} {arm} D={r['D']:.0f} {r['run_dir']}: separation {r['separation']:.3g}, "
                  f"end_up {r['end_up']:.3g}, val min {r['val_min_ln']:.2e}")
        ps.legend(axs[0, 0], "lower left")
        ps.save(fig, os.path.join(HERE, "figs", f"curve_examples_{kind}"))


def cells(target):
    """The L(D) label (b b~ > Z Z, design PASS 2026-11-12 15:11, measure per Joaquin 2026-11-12): per cell its search's
    best trial; an arm is well trained when in >= 5 of its 7 cells that trial's validation goes down with training."""
    rows = [r for r in trials() if r["stage"] == "s2" and r["target"] == target and r.get("best")]
    out = {}
    for arm in ("scr", "ft"):
        rr = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["D"])
        for r in rr:
            print(f"{target} {arm:3s} D={r['D']:7.0f}  val min {r['val_min_ln']:.2e}  separation {r['separation']:8.3g}"
                  f"  end_up {r['end_up']:6.3g}  " + ("good" if r["well"] else "BAD"))
        n = sum(r["well"] for r in rr)
        out[arm] = {"cells": len(rr), "good": n, "per_cell": {f"{r['D']:.0f}": r["well"] for r in rr}}
        print(f"{target} {arm}: {n} of {len(rr)} search bests good")
    json.dump(out, open(os.path.join(DATA, f"curve_quality_cells_{target}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    if "--cells" in sys.argv:
        cells(sys.argv[sys.argv.index("--cells") + 1])
    else:
        main()
