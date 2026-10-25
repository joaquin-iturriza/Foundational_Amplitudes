"""The small-D fine-tuning study, second round (D123, the design D120; sweep/gen_lowd_dyhpo.py): each method's own single-fidelity DyHPO
search (5 trials per cell, the grid's fine-tune space plus L2-SP's lambda or the frozen depth) against the grid's own
5-trial search of the default fine-tune and against scratch, best against best per cell (CLAUDE.md A/B protocol step 4),
on the first round's 60 cells (analysis/transfer/data/lowd_ft_spec.json). Each value is the best trial's best-checkpoint
MSE of log|M|^2 (val_loss_no_reg times prepd_std^2, as analysis/transfer/cells.py reports a cell). Per D: the median
over its 15 cells of log10(default / method) (a gain in decades) and of log10(scratch / method), and the number of cells
the method wins; every cell without 5 finished trials is listed, never dropped.
Input: data/lowd_dyhpo_sweeps.json, written on CC by
    python analysis/transfer/collect_sweeps.py tp3_lowd_ > analysis/transfer/data/lowd_dyhpo_sweeps.json
    python3 analysis/transfer/lowd_dyhpo.py    -> data/lowd_dyhpo.json, figs/lowd_dyhpo.{png,pdf}
An interim read before all trials are in: --complete counts a method's cell only once its sweep has all 5 trials
(-> data/lowd_dyhpo_interim.json, figs/lowd_dyhpo_interim.{png,pdf}; the final files are left alone).
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
METHODS = {"lpft": "linear probe, then fine-tune", "l2sp": "pulled toward the start", "freeze": "lower blocks frozen"}


def main():
    complete = "--complete" in sys.argv[1:]
    tag = "_interim" if complete else ""
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
            if tr and (len(tr) >= 5 or not complete):
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
                                              "wins_over_scratch": int((gs > 0).sum()),
                                              "factor_over_default": float(10 ** np.median(g)),
                                              "factor_over_scratch": float(10 ** np.median(gs))}
    # over all 60 cells: wins and the median, and per probe the median over its three rungs at each D
    pooled = {m: {"n": len(g), "wins_over_default": int((g > 0).sum()), "gain_over_default": float(np.median(g))}
              for m in METHODS
              for g in [np.array([np.log10(r["recorded_ft"] / r[m]) for r in rows if r.get(m) is not None])] if len(g)}
    by_probe = {m: {p: {str(k): float(np.median([np.log10(r["recorded_ft"] / r[m]) for r in rows
                                                if r["probe"] == p and r["k"] == k and r.get(m) is not None]))
                        for k in (2, 3, 4, 5)} for p in sorted({r["probe"] for r in rows})} for m in summ}
    # a sweep with more results than distinct candidates (a candidate run twice): its best is over the distinct ones
    repeats = {f"tp3_lowd_{m}_r{r['rung']}_{r['probe']}_d{r['k']}": r[m + "_n"] for r in rows for m in METHODS
               if r.get(m + "_n", 0) > 5}
    # the single-run noise (data/lowd_ft.json, par:lowd_ft): the default rerun at its recorded trial's HPs against the
    # grid's record of that trial, per cell, and the median of that log ratio over each D's 15 cells, which is what a
    # method no different from the default would show on this comparison
    first = json.load(open(os.path.join(HERE, "data", "lowd_ft.json")))["cells"].values()
    lr_ = np.array([np.log10(c["recorded_ft"] / c["default"]) for c in first])
    noise = {"q90_abs": float(np.quantile(np.abs(lr_), 0.9)), "median_abs": float(np.median(np.abs(lr_))),
             "per_D_median": {str(k): float(np.median([np.log10(c["recorded_ft"] / c["default"]) for c in first
                                                       if c["D"] == D_])) for k, D_ in
                              sorted({(r["k"], r["D"]) for r in rows})}}
    noise["per_D_median_max_abs"] = max(abs(v) for v in noise["per_D_median"].values())
    dflt = {str(k): float(np.median([np.log10(r["scratch"] / r["recorded_ft"]) for r in rows if r["k"] == k]))
            for k in (2, 3, 4, 5)}
    out = {"summary": summ, "default_over_scratch": dflt, "pooled": pooled, "by_probe": by_probe, "repeats": repeats,
           "noise": noise, "incomplete": incomplete, "cells": rows}
    json.dump(out, open(os.path.join(HERE, "data", f"lowd_dyhpo{tag}.json"), "w"), indent=1)
    for m, v in summ.items():
        print(m, json.dumps(v))
    print("incomplete:", len(incomplete), incomplete[:10])
    plot(summ, dflt, tag)
    # last line: the object the results.tex claims key into (tools/verify_claims.py), e.g. summary.l2sp.4.wins_over_default
    print(json.dumps({"summary": summ, "default_over_scratch": dflt, "pooled": pooled, "by_probe": by_probe,
                      "repeats": repeats, "noise": noise, "n_cells": len(rows),
                      "n_incomplete": len(incomplete)}))


def plot(summ, dflt, tag=""):
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
    # symmetric about equal, wide enough that a median within the cells' spread does not read as a trend
    lim = max(0.3, 1.2 * max(abs(v[str(k)]["gain_over_default"]) for v in summ.values() for k in v))
    b.set_ylim(-lim, lim)
    ps.legend(a, "upper left")
    for ax in (a, b):
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", f"lowd_dyhpo{tag}"))


if __name__ == "__main__":
    main()
