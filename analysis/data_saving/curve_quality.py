#!/usr/bin/env python3
"""pretrain_data_saving: does a trial's validation keep falling with its training to the end of the run (Joaquin
2026-11-12: "I just said that by the end they diverge, and that's what we care about, the validation going down with
the training loss. If the val loss is better than any other, why would I care it's jittery?").

Per trial, on the validation mu-MSE curve v_i and the train-eval curve t_i (checks i = 0..n-1):
  end_over_best  v_end / min v: the validation at the end of the run over its best; tracks <= 2 (it did not turn up
                 or diverge at the end)
  late_val       median of the last 3 checks over the median of the 3 checks around 75% of the run: validation's
                 change over the last quarter; tracks <= 1 (still falling, or flat, at the end)
  late_train     the same for training (the val/train pair: val turning up while train falls is the failure)
  stalls         tracks, but val within 10% over the last quarter while training halves (val flat, train falling)
Tracks = both bars. The other read is the best validation loss itself (val_min), shown beside: no shape bar ranks it.
The earlier bars (steady, spike, jitter, gap) are dropped: they judged noise along the way, not the end.

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
BAR = {"end_over_best": 2.0, "late_val": 1.0}


def late(x):
    x = np.asarray(x, float)
    q = int(0.75 * (len(x) - 1))
    return float(np.median(x[-3:]) / np.median(x[max(q - 1, 0):q + 2]))


def quality(v, t=None):
    v = np.asarray(v, float)
    q = {"end_over_best": float(v[-1] / v.min()), "late_val": late(v),
         "late_train": late(t) if t is not None and len(t) == len(v) and min(t) > 0 else None}
    q["well"] = bool(q["end_over_best"] <= BAR["end_over_best"] and q["late_val"] <= BAR["late_val"])
    # tracks by the bars, yet val stalls (within 10% over the last quarter) while training still halves: listed apart
    q["stalls"] = bool(q["well"] and q["late_val"] > 0.9 and q["late_train"] is not None and q["late_train"] < 0.5)
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
    print(f"{len(rows)} trials with curves; tracks = end/best <= 2 and late val <= 1")
    for key in sorted({(r["target"], r["stage"], r["arm"]) for r in rows}):
        g = [r for r in rows if (r["target"], r["stage"], r["arm"]) == key]
        b = [r for r in g if r.get("best")]
        gr = {"target": key[0], "stage": key[1], "arm": key[2], "n": len(g), "tracks": sum(r["well"] for r in g), "stalls": sum(r["stalls"] for r in g),
              "best_n": len(b), "best_tracks": sum(r["well"] for r in b),
              "median": {c: float(np.median([r[c] for r in g])) for c in BAR}}
        out["groups"].append(gr)
        print(f"{key[0]:13s} {key[1]} {key[2]:3s} n={len(g):3d} tracks {gr['tracks']:3d} (stalls {gr['stalls']:2d})"
              + (f"  search bests {gr['best_tracks']} of {gr['best_n']}" if b else "")
              + "  median: " + " ".join(f"{c} {gr['median'][c]:.2f}" for c in BAR))
    for r in rows:
        out["trials"].append({k: r.get(k) for k in ("sweep", "run_dir", "stage", "arm", "target", "D", "T", "peak_lr",
                                                     "best", "val_min_ln", "end_over_best", "late_val", "late_train",
                                                     "well", "stalls")})
    # where the turning up sits: stage-2 trials by arm and D band (the reported loss is read at the best checkpoint)
    out["by_D"] = []
    for arm in ("scr", "ft"):
        for lo, hi in ((0, 200), (200, 2000), (2000, 2e5)):
            g = [r for r in rows if r["stage"] == "s2" and r["arm"] == arm and lo <= r["D"] < hi]
            b = {"arm": arm, "D_lo": lo, "D_hi": hi, "n": len(g), "turn_up": sum(not r["well"] for r in g)}
            out["by_D"].append(b)
            print(f"  stage 2 {arm:3s} D in [{lo}, {hi:.0f}): {b['turn_up']} of {b['n']} trials turn up at the end")
    out["suspects"] = suspects()
    json.dump(out, open(os.path.join(DATA, "curve_quality.json"), "w"), indent=1)
    figure(rows)


def suspects():
    """The one-change lr runs of the suspects batch (x1/3, x1/10 at the fine-tune best's other HPs, old template) and
    their controls, read by the same bars, with each run's validation minimum in ln units beside."""
    read = json.load(open(os.path.join(DATA, "suspects_runs_read.json")))
    meta = {m["run_dir"]: m for m in json.load(open(os.path.join(DATA, "suspects_runs.json")))}
    out = []
    for d, r in sorted(read.items()):
        note = meta.get(d, {}).get("note", "control")
        if "epem_ttx" not in d or not r.get("val_mse_curve") or not ("lr" in note or note == "control"):
            continue
        q = quality(r["val_mse_curve"], r.get("train_eval_mse_curve"))
        q.update(run_dir=d, note=note, val_min_ln=min(r["val_mse_curve"]) * r["prepd_std"] ** 2)
        out.append(q)
        print(f"  {note[:40]:40s} {d[-40:]:40s} min {q['val_min_ln']:.2e} end/best {q['end_over_best']:.2f} "
              f"late val {q['late_val']:.2f} late train {q['late_train'] or float('nan'):.2f} tracks {q['well']}")
    return out


def figure(rows):
    """figs/curve_quality_end_{a,b}: every t t~ LO trial at D >= 316, end/best and late val against its peak lr."""
    from suspects_readout import ps
    g = [r for r in rows if r["target"] == "epem_ttx" and r["D"] >= 316]
    sty = {("s2", "scr"): ("scratch", ps.C.blue, "o"), ("s2", "ft"): ("FT old", "black", "s"),
           ("s4", "ft"): ("FT new", ps.C.vermillion, "^")}
    figs = ps.panels(2)
    for (fig, ax), (c, lab) in zip(figs, (("end_over_best", r"$v_{\rm end}/\min v$"),
                                          ("late_val", r"$v$, last quarter change"))):
        for key, (name, col, m) in sty.items():
            h = [r for r in g if (r["stage"], r["arm"]) == key]
            ax.scatter([r["peak_lr"] for r in h], [r[c] for r in h], color=col, marker=m, s=12, label=name)
        ax.axhline(BAR[c], color="grey", linestyle="--", label="bar")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("peak lr")
        ax.set_ylabel(lab)
        ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "curve_quality_end"))


def cells(target):
    """The L(D) label (b b~ > Z Z, design PASS 2026-11-12 15:11, bars per Joaquin 2026-11-12): per cell its search's
    best trial; an arm TRACKS when in >= 5 of its 7 cells that trial's validation falls with training to the end."""
    rows = [r for r in trials() if r["stage"] == "s2" and r["target"] == target and r.get("best")]
    out = {}
    for arm in ("scr", "ft"):
        rr = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["D"])
        for r in rr:
            print(f"{target} {arm:3s} D={r['D']:7.0f}  val min {r['val_min_ln']:.2e}  end/best {r['end_over_best']:6.2f}"
                  f"  late val {r['late_val']:5.2f}  late train {r['late_train'] or float('nan'):5.2f}  "
                  + ("TURNS UP" if not r["well"] else "tracks, val stalls" if r["stalls"] else "tracks"))
        n = sum(r["well"] for r in rr)
        out[arm] = {"cells": len(rr), "tracks": n, "stalls": sum(r["stalls"] for r in rr),
                    "per_cell": {f"{r['D']:.0f}": r["well"] for r in rr}}
        print(f"{target} {arm}: {n} of {len(rr)} search bests track to the end")
    json.dump(out, open(os.path.join(DATA, f"curve_quality_cells_{target}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    if "--cells" in sys.argv:
        cells(sys.argv[sys.argv.index("--cells") + 1])
    else:
        main()
