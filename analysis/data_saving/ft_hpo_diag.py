#!/usr/bin/env python3
"""pretrain_data_saving step 3 (Joaquin 2026-11-12: 'I don't understand why the hpo has such a hard time with these
finetunes'): every trial of the stage-2 searches (data/trial_overfit.json, read by trial_overfit.py) at T=8500, per
search: the spread of the trials' best val mu-MSE (median and max over best), whether the objective picked the trial
with the lowest val mu-MSE, and per trial the train mu-MSE at the val minimum (ln units, train_eval_mse_curve x
prepd_std^2), so a fine-tune that fits its training events but not the val events is told from one that fits neither.
Output data/ft_hpo_diag.json; figure figs/ft_hpo_diag_<target> (train against val mu-MSE at the val minimum, every
trial, scratch and fine-tune, one panel per D).

    python analysis/data_saving/ft_hpo_diag.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from trial_overfit import load  # noqa: E402
from stage2_readout import run_of, searches  # noqa: E402

TARGETS = {"epem_ttx": r"$e^+e^-\to t\bar t$", "ee_ttbar_nlo": r"$e^+e^-\to t\bar t$, $V/B$",
           "ee_uu": r"$e^+e^-\to u\bar u$", "ee_uu_nlo": r"$e^+e^-\to u\bar u$, $V$"}
DS = (316, 1000, 3162, 10000)


def rows():
    """Trials with curves, each with val_min and train_at_min (ln units) and its search's objective value."""
    S = searches()
    obj = {(sw, run_of(sw, t)): t.get("objective_value") for sw, s in S.items() for t in s["trials"]
           if t.get("status") == "observed"}
    out = []
    for r in load(cached=True):
        if not r.get("val") or not r.get("train"):
            continue
        v, t, p = np.array(r["val"]), np.array(r["train"]), r["prepd_std"] ** 2
        i = int(v.argmin())
        out.append({"sweep": r["sweep"], "arm": r["arm"], "target": r["target"], "D": round(r["D"]), "best": r["best"],
                    "hp": r["hp"], "val_min": float(v[i] * p), "train_at_min": float(t[i] * p),
                    "objective_over_valmin": (obj.get((r["sweep"], os.path.basename(r["run_dir"]))) or np.nan) / v[i]})
    return out


def summary(R):
    cells = []
    for tg in TARGETS:
        for D in sorted({r["D"] for r in R if r["target"] == tg}):
            c = {"target": tg, "D": D}
            for a in ("scr", "ft"):
                q = [r for r in R if r["target"] == tg and r["D"] == D and r["arm"] == a]
                if not q:
                    continue
                v = np.array([r["val_min"] for r in q]); tr = np.array([r["train_at_min"] for r in q])
                c[a] = {"n": len(q), "best_val": v.min(), "median_over_best": float(np.median(v) / v.min()),
                        "max_over_best": float(v.max() / v.min()), "median_train": float(np.median(tr)),
                        "median_val_over_train": float(np.median(v / tr)),
                        "objective_picks_lowest_val": bool(np.allclose([r["objective_over_valmin"] for r in q], 1,
                                                                        rtol=1e-6)),
                        "objective_missing": int(sum(np.isnan(r["objective_over_valmin"]) for r in q))}
            if "scr" in c and "ft" in c:
                c["ft_best_over_scr_best"] = c["ft"]["best_val"] / c["scr"]["best_val"]
                c["ft_train_over_scr_train"] = c["ft"]["median_train"] / c["scr"]["median_train"]
            cells.append(c)
            if D in DS and "ft_best_over_scr_best" in c:
                s, f = c.get("scr"), c.get("ft")
                print(f"{tg:13s} D={D:>6d}  ft/scr best val {c['ft_best_over_scr_best']:6.1f}  median train ft/scr "
                      f"{c['ft_train_over_scr_train']:5.2f}  median/best scr {s['median_over_best']:5.1f} ft "
                      f"{f['median_over_best']:5.1f}  val/train scr {s['median_val_over_train']:5.0f} ft "
                      f"{f['median_val_over_train']:5.0f}")
    print("objective = the trial's val minimum in every search:",
          all(c[a]["objective_picks_lowest_val"] for c in cells for a in ("scr", "ft") if a in c))
    json.dump(cells, open(os.path.join(HERE, "data", "ft_hpo_diag.json"), "w"), indent=1)


def figure(R):
    import plot_style as ps
    ARM = {"scr": ("scratch", ps.C.blue, "o"), "ft": ("fine-tune", ps.C.vermillion, "s")}
    for tg, lab in TARGETS.items():
        fig, axes = ps.figure(ncols=2, nrows=2)
        for k, (ax, D) in enumerate(zip(np.ravel(axes), DS)):
            q = [r for r in R if r["target"] == tg and r["D"] == D]
            for a, (name, col, m) in ARM.items():
                qa = [r for r in q if r["arm"] == a]
                ax.scatter([r["train_at_min"] for r in qa], [r["val_min"] for r in qa], c=col, marker=m, s=14,
                           alpha=0.7, label=name)
                b = [r for r in qa if r["best"]]
                ax.scatter([r["train_at_min"] for r in b], [r["val_min"] for r in b], facecolors="none",
                           edgecolors="k", marker=m, s=40, label=f"{name}, search best")
            lo = min(min(r["train_at_min"], r["val_min"]) for r in q); hi = max(r["val_min"] for r in q)
            ax.plot([lo, hi], [lo, hi], color="0.5", ls=":", label="val = train")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"train MSE$(\ln|\mathcal{M}|^2)$")
            ax.set_ylabel(r"val MSE$(\ln|\mathcal{M}|^2)$")
            ps.process_label(ax, rf"$D={D}$")
            if k == 0:
                ps.legend(ax, "upper left")
        ps.save(fig, os.path.join(HERE, "figs", f"ft_hpo_diag_{tg}"))


if __name__ == "__main__":
    R = rows()
    summary(R)
    figure(R)
