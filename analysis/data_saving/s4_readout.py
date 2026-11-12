#!/usr/bin/env python3
"""pretrain_data_saving step 3 test readout (design 2026-11-12, the fine-tune template with scratch's space): per cell
the best val mu-MSE (ln units) of the new search (dsave_s4_ft_*: layer_decay 1, lr_scale [1/3,3]) against the stage-2
fine-tune search (old template) and the stage-2 scratch search; r_old = new/old, r_scr = new/scratch, per target the
gmean and the design's branch. Figure figs/s4_curves_<target>: the three bests' val (solid) and train (dashed) curves.

    python analysis/data_saving/s4_readout.py [--cached]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from analysis.common.readers import site_python  # noqa: E402
from trial_overfit import REMOTE  # noqa: E402
from suspects_readout import TLAB, YLAB, gmean, ps  # noqa: E402

CELLS = {"epem_ttx": (5, 7), "ee_ttbar_nlo": (5, 6, 7)}
DATA = os.path.join(HERE, "data")


def branch(r):
    g, below = gmean(r), sum(x < 1 for x in r)
    if g <= 1 / 3 and below == len(r):
        return g, "FIXED"
    if g < 1 and below >= len(r) - 1:
        return g, "IMPROVED, NOT RESOLVED"
    return g, "WORSE" if g >= 3 else "NO CHANGE"


def best_rows(cached):
    """{(arm, target, k): the search's best trial row (curves in ln units)}; arms new (s4 ft), old (s2 ft), scr (s2)."""
    sw = {(a, t, k): f"dsave_{'s4' if a == 'new' else 's2'}_{'scr' if a == 'scr' else 'ft'}_{t}_d{k}_t8500"
          for t, ks in CELLS.items() for k in ks for a in ("new", "old", "scr")}
    cache = os.path.join(DATA, "s4_readout_rows.json")
    if cached:
        rows = json.load(open(cache))
    else:
        rows = site_python(REMOTE, sorted(set(sw.values())), "TOVF", timeout=1800)
        json.dump(rows, open(cache, "w"))
    out = {}
    for key, name in sw.items():
        q = [r for r in rows if r["sweep"] == name and r.get("val")]
        for r in q:
            p = r["prepd_std"] ** 2
            r["val_ln"] = [x * p for x in r["val"]]
            r["train_ln"] = [x * p for x in (r.get("train") or [])]
            r["val_min"] = min(r["val_ln"])
        out[key] = (min(q, key=lambda r: r["val_min"]) if q else None, len(q))
    return out


def main():
    B = best_rows("--cached" in sys.argv)
    res = {"cells": [], "targets": {}}
    for t, ks in CELLS.items():
        ro = []
        for k in ks:
            (n, nn), (o, no), (s, ns) = B[("new", t, k)], B[("old", t, k)], B[("scr", t, k)]
            c = {"target": t, "D": round(10 ** (k / 2)), "n_trials": {"new": nn, "old": no, "scr": ns},
                 "new": n and n["val_min"], "old": o and o["val_min"], "scr": s and s["val_min"],
                 "new_best_hp": n and n["hp"]}
            c["r_old"] = c["new"] / c["old"] if n and o else None
            c["r_scr"] = c["new"] / c["scr"] if n and s else None
            res["cells"].append(c)
            ro += [c["r_old"]] if c["r_old"] else []
            print(f"{t:13s} D={c['D']:6d} new {c['new'] or np.nan:.2e} ({nn}) old {c['old'] or np.nan:.2e} ({no}) "
                  f"scr {c['scr'] or np.nan:.2e} ({ns})  r_old {c['r_old'] or np.nan:.2f} r_scr {c['r_scr'] or np.nan:.2f}")
        if ro:
            g, b = branch(ro)
            res["targets"][t] = {"gmean_r_old": g, "n_cells": len(ro), "branch": b}
            print(f"  {t}: gmean r_old {g:.2f} over {len(ro)} cells -> {b}")
    json.dump(res, open(os.path.join(DATA, "s4_readout.json"), "w"), indent=1)
    STY = {"new": ("FT new", ps.C.vermillion), "old": ("FT old", "black"),
           "scr": ("scratch", ps.C.blue)}
    for t, ks in CELLS.items():
        figs = ps.panels(len(ks))
        for (fig, ax), k in zip(figs, ks):
            for a, (lab, col) in STY.items():
                r = B[(a, t, k)][0]
                if r is None:
                    continue
                st = np.arange(len(r["val_ln"])) * (r.get("every") or 1)
                ax.plot(st, r["val_ln"], color=col, label=f"{lab}, val")
                if r["train_ln"]:
                    ax.plot(st[:len(r["train_ln"])], r["train_ln"], color=col, linestyle="--", label=f"{lab}, train")
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(YLAB[t])
            ps.process_label(ax, rf"$D={round(10 ** (k / 2))}$")
            ps.legend(ax, "upper left")
        ps.save_panels(figs, os.path.join(HERE, "figs", f"s4_curves_{t}"))


if __name__ == "__main__":
    main()
