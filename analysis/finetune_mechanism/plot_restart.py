#!/usr/bin/env python3
"""finetune_mechanism step 1 figures, from existing runs (no jobs).
figs/restart_steps: the first steps after a restart from a pretrained parent.
  (a) pre-clip gradient norm and (b) training loss (het. NLL) against step, b b~ > Z Z tree D=3162, the search best of
      each fine-tune arm of data saving step 6 and its warmup test (R, C, W, L) and the scratch search best;
  (c) pre-clip gradient norm, steps 0-40, of the catalog warm-start restarts (results.tex par. v4_warm_start_restart,
      _optstate, _fixes; seed 42) beside the b b~ > Z Z D=3162 arm C best;
  (d) the legend.
figs/restart_steps_zng: pre-clip gradient norm of the Z+ng fine-tunes (kit A_ft, 12500 steps, trial 1262's held-out-family parent).
figs/restart_headstart: per D in {100, 316, 3162} the validation MSE(ln|M|^2) of each arm's search best and the
  scratch search best against step (the step-6 searches, best = lowest validation).
Reads data/restart_steps.json (restart_steps.py), ../data_saving/data/step6_rows.json, curve_quality.trials (stage 2),
../catalog_v4/data/ws_{restart_study,fixes}_steps.json.

    python analysis/finetune_mechanism/plot_restart.py   -> figs/restart_steps, figs/restart_headstart, data/restart_summary.json
"""
import json
import os
import sys

import numpy as np

HERE = os.environ.get("FTM_HERE") or os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "data_saving"))
import plot_style as ps  # noqa: E402
from curve_quality import trials  # noqa: E402
from step6_readout import ln  # noqa: E402

CELLS = {100: (4, 4960), 316: (5, 7840), 3162: (7, 8500)}
NEW = {"C": "s5cosparent", "W": "s5wsdparent", "L": "s6cosparentlongwu"}
ARM = {"R": ("cosine parent,\ncosine fine-tune", ps.C.blue),
       "C": ("cosine parent,\nWSD fine-tune", ps.C.vermillion),
       "W": ("WSD parent,\nWSD fine-tune", ps.C.green),
       "L": ("cosine parent, WSD\nfine-tune, long warmup", ps.C.purple),
       "scr": ("from scratch", ps.C.grey)}
CAT = os.path.join(REPO, "analysis", "catalog_v4", "data")
WS = [("catalog, cosine parent, restart", "ws_restart_study_steps.json", "FA-ccin2p3-0de6be56", ps.C.orange, "-"),
      ("catalog, WSD parent, restart", "ws_fixes_steps.json", "FA-ccin2p3-9079718e", ps.C.yellow, "-"),
      ("catalog, WSD parent branched\nwith optimizer state", "ws_fixes_steps.json", "FA-ccin2p3-1354daee", ps.C.sky, "-"),
      ("catalog, cosine parent continued\nwith optimizer state", "ws_restart_study_steps.json", "FA-ccin2p3-1c90ae73",
       ps.C.grey, ":")]
KIT = [("$u\\bar q\\to Zg$", "zg", ps.C.blue), ("$u\\bar q\\to Zgg$", "zgg", ps.C.vermillion),
       ("$u\\bar q\\to Zggg$", "zggg", ps.C.green), ("$u\\bar q\\to Zgggg$", "zgggg", ps.C.purple)]


def bests():
    """{D: {arm: row}} the lowest-validation trial of each search, as step6_readout."""
    s2 = [ln(r) for r in trials() if r["stage"] == "s2" and r["target"] == "bbx_zz" and r["val"] and len(r["val"]) > 3]
    new = [ln(r) for r in json.load(open(os.path.join(REPO, "analysis", "data_saving", "data", "step6_rows.json")))["curves"]
           if r.get("val") and len(r["val"]) > 3]
    out = {}
    for D, (k, T) in CELLS.items():
        g = {"scr": [r for r in s2 if r["arm"] == "scr" and round(r["D"]) == D],
             "R": [r for r in s2 if r["arm"] == "ft" and round(r["D"]) == D],
             **{a: [r for r in new if r["sweep"] == f"dsave_{s}_ft_bbx_zz_d{k}_t{T}"] for a, s in NEW.items()}}
        out[D] = {a: min(v, key=lambda r: r["val_min"]) for a, v in g.items() if v}
    return out


def main():
    st = json.load(open(os.path.join(HERE, "data", "restart_steps.json")))
    steps = {r["run_dir"]: r for r in st["search"]}
    B = bests()
    summ = {"bbzz_d3162": {}, "catalog": {}, "kit": {}}
    fig, axs = ps.figure(2, 2)
    (a, b), (c, d) = axs
    for arm, r in B[3162].items():
        s = steps[r["run_dir"]]
        lab, col = ARM[arm]
        a.plot(s["step"], s["grad_norm"], color=col, label=lab)
        b.plot(s["step"], s["loss"], color=col, label=lab)
        g = np.array(s["grad_norm"])
        summ["bbzz_d3162"][arm] = {"run": r["run_dir"], "gn_step0": g[0], "gn_max_0_40": float(g[:41].max()),
                                   "gn_median_100_400": float(np.median(g[100:400])), "loss_step0": s["loss"][0],
                                   "loss_step50": s["loss"][50], "warmup": s["warmup"], "lr_step100": s["lr"][100]}
    for ax in (a, c):
        ax.set_yscale("log")
        ax.set_ylabel("gradient norm before clipping")
    b.set_yscale("symlog", linthresh=1.0)
    b.set_ylabel("training loss (het. NLL)")
    for ax in (a, b, c):
        ax.set_xlabel("step after the restart")
    ps.process_label(a, r"$b\bar b\to ZZ$")
    ps.process_label(b, r"$b\bar b\to ZZ$")
    for lab, f, run, col, ls in WS:
        g = json.load(open(os.path.join(CAT, f)))[run]["grad_norm"]
        c.plot(range(len(g)), g, color=col, ls=ls, label=lab)
        summ["catalog"][run] = {"label": lab, "gn_step0": g[0], "gn_max_0_40": max(g)}
    gC = steps[B[3162]["C"]["run_dir"]]["grad_norm"][:41]
    c.plot(range(len(gC)), gC, color=ps.C.vermillion, ls="--", label=ARM["C"][0])   # arm C of panel (a), steps 0-40
    ps.legend_cell(d, [a, c], fontsize=ps.LEGEND_MIN_PT)   # 9 entries: the allowed legend minimum
    ps.save(fig, os.path.join(HERE, "figs", "restart_steps"))

    fig, d = ps.figure(1, 1)
    for lab, p, col in KIT:
        r = next(x for x in st["kit"] if x["run_dir"].endswith(f"kit_A_ft_uq_{p}_me2_n500000_t12500"))
        d.plot(r["step"], r["grad_norm"], color=col, label=lab)
        summ["kit"][p] = {"gn_step0": r["grad_norm"][0], "loss_step0": r["loss"][0],
                          "gn_median_100_400": float(np.median(r["grad_norm"][100:400]))}
    d.set_yscale("log")
    d.set_ylabel("gradient norm before clipping")
    d.set_xlabel("step after the restart")
    ps.legend(d, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "restart_steps_zng"))

    fig, axs = ps.figure(2, 2)
    panels = [axs[0][0], axs[0][1], axs[1][0]]
    for ax, D in zip(panels, CELLS):
        for arm in ("scr", "R", "C", "W", "L"):
            r = B[D].get(arm)
            if r is None:
                continue
            x = (np.arange(len(r["val_ln"])) + 1) * r["every"]   # val[i] is at step (i+1)*every
            ax.plot(x, r["val_ln"], color=ARM[arm][1], label=ARM[arm][0])
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, rf"$D_{{\rm train}}={D}$")
    ps.legend_cell(axs[1][1], axs[0][0])
    ps.save(fig, os.path.join(HERE, "figs", "restart_headstart"))
    json.dump(summ, open(os.path.join(HERE, "data", "restart_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
