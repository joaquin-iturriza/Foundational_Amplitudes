#!/usr/bin/env python3
"""pretrain_data_saving: is a trial well optimized, judged by its own validation curve (Joaquin 2026-11-12: "the tree
runs are not fixed at all"; an arm ratio cannot show a run that is badly trained in both arms).

Per trial, on the validation mu-MSE curve v_i (checks i = 0..n-1, every `every` steps) and the train-eval curve t_i,
after the warm-up (checks past warmup_frac of the run):
  steady   share of the post-warm-up log-fall, ln(runmin_w / min v), reached by the running minimum at 75% of the run;
           well >= 2/3 (the fall is not left to the cosine tail)
  spike    largest v_i / min(v_0..v_{i-1}); well <= 3
  jitter   median |ln(v_i / v_{i-1})|; well <= ln 2 (consecutive checks within 2x)
  gap      train's fall while val is flat: from the first check i* whose running minimum is within 1.5x of the final
           minimum to the end, t_{i*} / t_end; well <= 10 (no val/train gap opening while val is flat)
Well optimized = all four. The bars judge the shape only: a clean curve at a worse minimum is not better, so every
comparison reads the validation minimum beside them. steady fails both for a noise floor at a high lr (spiking, the
fall left to the tail) and for under-training at a low lr (clean, still falling at the end): read it with spike/jitter. Shown beside the trial's peak lr (lr x lr_scale), warm-up, batch size and horizon.

    python analysis/data_saving/curve_quality.py      # reads the cached curves (trial_overfit.json, s4_readout_rows.json)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
BAR = {"steady": 2 / 3, "spike": 3.0, "jitter": np.log(2), "gap": 10.0}


def quality(v, t, warmup_frac):
    v = np.asarray(v, float)
    n = len(v)
    w = min(n - 2, int(np.ceil(warmup_frac * n)))          # first check past the warm-up
    run = np.minimum.accumulate(v)
    fall = np.log(run[w] / run[-1])                      # the fall after the warm-up
    q = {"steady": float(np.log(run[w] / run[int(0.75 * (n - 1))]) / fall) if fall > 0 else 0.0,
         "spike": float(max(v[i] / run[i - 1] for i in range(max(w, 1), n))),
         "jitter": float(np.median(np.abs(np.diff(np.log(v[w:]))))),
         "gap": None}
    if t and len(t) == n and min(t) > 0:
        i = int(np.argmax(run <= 1.5 * run[-1]))
        q["gap"] = float(t[i] / t[-1])
    q["well"] = bool(q["steady"] >= BAR["steady"] and q["spike"] <= BAR["spike"] and q["jitter"] <= BAR["jitter"]
                     and (q["gap"] is None or q["gap"] <= BAR["gap"]))
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
        r.update(quality(r["val"], r.get("train"), r["hp"]["warmup_frac"] or 0.0))
    return rows


def main():
    rows = trials()
    out = {"bars": {k: float(v) for k, v in BAR.items()}, "groups": [], "trials": []}
    print(f"{len(rows)} trials with curves; bars: steady >= 2/3, spike <= 3, jitter <= ln2, gap <= 10")
    keys = sorted({(r["target"], r["stage"], r["arm"]) for r in rows})
    for key in keys:
        g = [r for r in rows if (r["target"], r["stage"], r["arm"]) == key]
        fail = {c: sum(not ok(r, c) for r in g) for c in BAR}
        rho = {c: float(spearmanr([r["peak_lr"] for r in g], [r[c] for r in g])[0]) if len(g) > 2 else None
               for c in ("steady", "spike", "jitter")}
        gr = {"target": key[0], "stage": key[1], "arm": key[2], "n": len(g), "well": sum(r["well"] for r in g),
              "fail": fail, "median": {c: float(np.median([r[c] for r in g if r[c] is not None])) for c in BAR},
              "spearman_peak_lr": rho}
        out["groups"].append(gr)
        print(f"{key[0]:13s} {key[1]} {key[2]:3s} n={len(g):3d} well {gr['well']:3d}  fails: "
              + " ".join(f"{c} {fail[c]}" for c in BAR)
              + "  median: " + " ".join(f"{c} {gr['median'][c]:.2f}" for c in BAR)
              + "  rho(peak lr): " + " ".join(f"{c} {rho[c]:+.2f}" for c in rho if rho[c] is not None))
    for r in rows:
        out["trials"].append({k: r.get(k) for k in ("sweep", "run_dir", "stage", "arm", "target", "D", "T", "batch",
                                                     "peak_lr", "best", "steady", "spike", "jitter", "gap", "well")}
                             | {"warmup_frac": r["hp"]["warmup_frac"], "ema": r["hp"]["ema"]})
    out["suspects"] = suspects()
    out["positive"] = [{k: r.get(k) for k in ("run_dir", "target", "arm", "D", "peak_lr", "steady", "spike", "jitter",
                                              "gap")} for r in rows if r["well"]]
    print(f"positive controls (well by all four): {len(out['positive'])}, e.g. "
          + "; ".join(f"{p['run_dir']} steady {p['steady']:.2f} spike {p['spike']:.1f} jitter {p['jitter']:.2f} "
                      f"gap {p['gap']:.1f}" for p in out["positive"][:3]))
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
        wf = float(meta.get(d, {}).get("hp", {}).get("training.cosanneal_warmup_frac", 0.15))
        q = quality(r["val_mse_curve"], r.get("train_eval_mse_curve"), wf)
        q.update(run_dir=d, note=note, val_min_ln=min(r["val_mse_curve"]) * r["prepd_std"] ** 2)
        out.append(q)
        print(f"  {note[:40]:40s} {d[-40:]:40s} min {q['val_min_ln']:.2e} steady {q['steady']:.2f} "
              f"spike {q['spike']:.1f} jitter {q['jitter']:.2f} gap {q['gap'] or 0:.1f} well {q['well']}")
    return out


def figure(rows):
    """figs/curve_quality_lr_{a,b}: spike and jitter of every t t~ LO trial at D >= 316 against its peak lr."""
    from suspects_readout import ps
    g = [r for r in rows if r["target"] == "epem_ttx" and r["D"] >= 316]
    sty = {("s2", "scr"): ("scratch", ps.C.blue, "o"), ("s2", "ft"): ("FT old", "black", "s"),
           ("s4", "ft"): ("FT new", ps.C.vermillion, "^")}
    figs = ps.panels(2)
    for (fig, ax), (c, lab) in zip(figs, (("spike", r"$\max_i v_i/\min_{j<i} v_j$"),
                                          ("jitter", r"median $|\ln(v_i/v_{i-1})|$"))):
        for key, (name, col, m) in sty.items():
            h = [r for r in g if (r["stage"], r["arm"]) == key]
            ax.scatter([r["peak_lr"] for r in h], [r[c] for r in h], color=col, marker=m, s=12, label=name)
        ax.axhline(BAR[c], color="grey", linestyle="--", label="bar")
        ax.set_xscale("log"); ax.set_xlabel("peak lr")
        if c == "spike":
            ax.set_yscale("log")
        ax.set_ylabel(lab)
        ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "curve_quality_lr"))


def ok(r, c):
    x = r[c]
    if x is None:
        return True
    return x >= BAR[c] if c == "steady" else x <= BAR[c]


if __name__ == "__main__":
    main()
