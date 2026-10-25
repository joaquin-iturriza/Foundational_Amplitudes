"""The small-D fine-tuning study, second round (D120; sweep/gen_lowd_dyhpo.py): each method's own single-fidelity DyHPO
search (5 trials per cell, the grid's fine-tune space plus L2-SP's lambda or the frozen depth) against the grid's own
5-trial search of the default fine-tune and against scratch, best against best per cell (CLAUDE.md A/B protocol step 4),
on the first round's 60 cells (analysis/transfer/data/lowd_ft_spec.json). Each value is the best trial's best-checkpoint
MSE of log|M|^2 (val_loss_no_reg times prepd_std^2, as analysis/transfer/cells.py reports a cell). Per D: the median
over its 15 cells of log10(default / method) (a gain in decades) and of log10(scratch / method), and the number of cells
the method wins; every cell without 5 finished trials is listed, never dropped.
Input: data/lowd_dyhpo_sweeps.json, written on CC by
    python analysis/transfer/collect_sweeps.py tp3_lowd_ > analysis/transfer/data/lowd_dyhpo_sweeps.json
    python3 analysis/transfer/lowd_dyhpo.py    -> data/lowd_dyhpo.json, figs/lowd_dyhpo.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
METHODS = {"lpft": "linear probe, then fine-tune", "l2sp": "pulled toward the start", "freeze": "lower blocks frozen"}


def main():
    spec = json.load(open(os.path.join(HERE, "data", "lowd_ft_spec.json")))
    S = json.load(open(os.path.join(HERE, "data", "lowd_dyhpo_sweeps.json")))
    rows, incomplete = [], []
    for c in spec:
        r = {k: c[k] for k in ("probe", "rung", "k", "D", "recorded_ft", "scratch")}
        for m in METHODS:
            name = f"tp3_lowd_{m}_r{c['rung']}_{c['probe']}_d{c['k']}"
            tr = [t for t in S.get(name, []) if t.get("val_loss") is not None and t.get("prepd_std")]
            if len(tr) < 5:
                incomplete.append(f"{name}: {len(tr)}/5")
            if tr:
                b = min(tr, key=lambda t: t["val_loss"] * t["prepd_std"] ** 2)
                r[m] = b["val_loss"] * b["prepd_std"] ** 2
                r[m + "_hp"] = b.get("hp")
                r[m + "_n"] = len(tr)
        rows.append(r)
    summ = {}
    for m in METHODS:
        for k in (2, 3, 4, 5):
            cs = [r for r in rows if r["k"] == k and r.get(m) is not None]
            if not cs:
                continue
            g = np.array([np.log10(r["recorded_ft"] / r[m]) for r in cs])
            gs = np.array([np.log10(r["scratch"] / r[m]) for r in cs])
            summ.setdefault(m, {})[str(k)] = {"n": len(cs), "gain_over_default": float(np.median(g)),
                                              "wins_over_default": int((g > 0).sum()),
                                              "gain_over_scratch": float(np.median(gs)),
                                              "wins_over_scratch": int((gs > 0).sum())}
    dflt = {str(k): float(np.median([np.log10(r["scratch"] / r["recorded_ft"]) for r in rows if r["k"] == k]))
            for k in (2, 3, 4, 5)}
    out = {"summary": summ, "default_over_scratch": dflt, "incomplete": incomplete, "cells": rows}
    json.dump(out, open(os.path.join(HERE, "data", "lowd_dyhpo.json"), "w"), indent=1)
    for m, v in summ.items():
        print(m, json.dumps(v))
    print("incomplete:", len(incomplete), incomplete[:10])
    plot(summ, dflt)


def plot(summ, dflt):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    cols = {"lpft": ps.C.vermillion, "l2sp": ps.C.green, "freeze": ps.C.orange}
    mk = {"lpft": "s", "l2sp": "^", "freeze": "v"}
    x = [1.0, 1.5, 2.0, 2.5]
    a.plot(x, [dflt[str(k)] for k in (2, 3, 4, 5)], marker="o", color=ps.C.blue, label="default fine-tune")
    for m, v in summ.items():
        ks = sorted(int(k) for k in v)
        a.plot([k / 2 for k in ks], [v[str(k)]["gain_over_scratch"] for k in ks], marker=mk[m], color=cols[m],
               label=METHODS[m])
        b.plot([k / 2 for k in ks], [v[str(k)]["gain_over_default"] for k in ks], marker=mk[m], color=cols[m],
               label=METHODS[m])
    for ax, yl in ((a, "gain over scratch (decades)"), (b, "gain over the default (decades)")):
        ax.axhline(0, ls="--", color=ps.C.grey, label="equal")
        ax.set_xlabel(r"$\log_{10} D$"); ax.set_ylabel(yl)
    ps.legend(a, "upper left")
    for ax in (a, b):
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "lowd_dyhpo"))


if __name__ == "__main__":
    main()
