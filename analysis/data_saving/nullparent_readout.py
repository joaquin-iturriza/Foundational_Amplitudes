#!/usr/bin/env python3
"""pretrain_data_saving step 2 readout: the null-parent 2x2 (design 2026-11-12 08:59). At each cell (t t~ tree and V/B,
D=316..1e4) both arms go through the fine-tune configs at the stage-2 scratch best HPs: null = every weight a fresh init
(fine_tune.shrink_perturb [0,1]), pre = the parent's weights; against each other and the stage-2 scratch best.
Measure as suspects_readout.py (min of val_mse_curve x prepd_std^2; worst-val-event-removed and test from the sigma
stage's preds). Per target: Gn = gmean null/scratch, Gp = pre/null, Gs = pre/scratch, each with cells above 1; the
design's partition A-E, a branch holding only on both val and worst-removed val.

    python analysis/data_saving/nullparent_readout.py [--cached]   # data/nullparent_readout.json + figs/nullparent_*
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from suspects_readout import FIGS, K, REMOTE, TARGETS, TLAB, YLAB, DATA, dirs as sus_dirs, load as sus_load, gmean, ps, site_python  # noqa: E402

STYLE = {"null": ("fresh init", ps.C.vermillion, "v"), "pre": ("parent weights", ps.C.green, "D"),
         "scr": ("scratch", ps.C.blue, "s"), "ctl": ("fine-tune control", "black", "o")}


def dirs():
    out = {k: v for k, v in sus_dirs().items() if k[0] in ("scr", "ctl")}
    for r in json.load(open(os.path.join(DATA, "nullparent_runs.json"))):
        _, arm, tg, dk = r["note"].split()[:4]
        out[(arm, tg, int(dk[1:]))] = r["run_dir"]
    return out


def load(cached):
    return sus_load(cached, dirs, "nullparent_runs_read.json")


def branch(G, n):
    """The design's partition A-E from the three geomeans and their cells-above-1 counts (n of m)."""
    (Gn, nn, m_n), (Gp, np_, m_p), (Gs, _, _) = G["null/scr"], G["pre/null"], G["pre/scr"]
    if Gn <= 1 / 3 and m_n - nn >= m_n - 1:
        return "A SCRATCH SUSPECT"
    if Gn >= 3 and nn >= m_n - 1:
        return "B PATH"
    if Gp >= 3 and np_ >= m_p - 1:
        return "C WEIGHTS"
    if Gs < 3 and Gp < 3:
        return "D SEARCH"
    return "E NOT RESOLVED"


def main():
    R = load("--cached" in sys.argv)
    res = {"cells": [], "targets": {}}
    for tg in TARGETS:
        reads = {}
        for read, key in (("val", "val_min"), ("val_x", "val_mse_x")):
            G = {}
            for name, a, b in (("null/scr", "null", "scr"), ("pre/null", "pre", "null"), ("pre/scr", "pre", "scr")):
                rr = [R[(a, tg, k)][key] / R[(b, tg, k)][key] for k in K
                      if R[(a, tg, k)].get(key) and R[(b, tg, k)].get(key)]
                G[name] = (gmean(rr), sum(x > 1 for x in rr), len(rr))
            reads[read] = {"G": G, "branch": branch(G, None)}
        b = reads["val"]["branch"]
        res["targets"][tg] = dict(reads, verdict=b if reads["val_x"]["branch"] == b else "E NOT RESOLVED")
        for k in K:
            c = {"target": tg, "D": round(10 ** (k / 2))}
            for a in ("null", "pre", "scr", "ctl"):
                c[a] = {q: R[(a, tg, k)].get(q) for q in ("val_min", "val_mse_x", "test_mse")}
                c[a]["error"] = R[(a, tg, k)].get("error")
            res["cells"].append(c)
            f = lambda a: c[a]["val_min"] or float("nan")  # noqa: E731
            print(f"{tg:13s} D={c['D']:6d} null {f('null'):.2e} pre {f('pre'):.2e} scr {f('scr'):.2e} "
                  f"ctl {f('ctl'):.2e}  null/scr {f('null') / f('scr'):6.2f} pre/null {f('pre') / f('null'):6.2f} "
                  f"pre/scr {f('pre') / f('scr'):6.2f}")
        for read in ("val", "val_x"):
            G = reads[read]["G"]
            print(f"  {tg} {read:5s} " + "  ".join(f"{n} {g:.2f} ({a}/{m} >1)" for n, (g, a, m) in G.items())
                  + f"  -> {reads[read]['branch']}")
        print(f"  {tg} verdict: {res['targets'][tg]['verdict']}")
    json.dump(res, open(os.path.join(DATA, "nullparent_readout.json"), "w"), indent=1)
    figures(R)


def figures(R):
    """nullparent_LD: val mu-MSE against D per target; nullparent_curves_<target>: val (solid) and train (dashed)
    mu-MSE against step, one panel per D."""
    x = [10 ** (k / 2) for k in K]
    fig, axs = ps.figure(ncols=2)
    for ax, tg in zip(axs, TARGETS):
        for a, (lab, col, m) in STYLE.items():
            ax.plot(x, [R[(a, tg, k)].get("val_min", np.nan) for k in K], color=col, marker=m, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$D$"); ax.set_ylabel(YLAB[tg])
        ps.process_label(ax, TLAB[tg])
    ps.legend(axs[0], "upper left")
    ps.save(fig, FIGS + "nullparent_LD")
    for tg in TARGETS:
        fig, axs = ps.figure(ncols=2, nrows=2)
        for i, (ax, k) in enumerate(zip(axs.ravel(), K)):
            for a, (lab, col, _) in STYLE.items():
                r = R[(a, tg, k)]
                if not r.get("val_mse_curve"):
                    continue
                s = np.arange(len(r["val_mse_curve"])) * (r.get("every") or 1)
                ax.plot(s, np.array(r["val_mse_curve"], float) * r["prepd_std"] ** 2, color=col, label=f"{lab}, val")
                if r.get("train_eval_mse_curve"):
                    ax.plot(s[:len(r["train_eval_mse_curve"])], np.array(r["train_eval_mse_curve"], float)
                            * r["prepd_std"] ** 2, color=col, linestyle="--", label=f"{lab}, train")
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(YLAB[tg])
            ps.process_label(ax, rf"$D={round(10 ** (k / 2))}$")
        ps.shared_legend(fig, axs[0, 0], ncol=4)
        ps.save(fig, FIGS + f"nullparent_curves_{tg}")


if __name__ == "__main__":
    main()
