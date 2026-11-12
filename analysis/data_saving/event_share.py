#!/usr/bin/env python3
"""pretrain_data_saving: why a t t~ V cell's validation and test reads disagree. Per run, from its sigma-stage
predictions (sigma_stage/preds_{val,test}.npz, the final model, D276) on CC: MSE of ln(V/B) on val and test, the
share of each that its worst event carries, the MSE without that event, the worst event's pool row, and the
validation mu-MSE that the predictions give (MSE / prepd_std^2, comparable to the run's RESULT_JSON val_loss, which
is read at the training stage's selected checkpoint). Writes data/event_share.json and, for t t~ V cells,
figs/event_share.{png,pdf}: per arm val and test MSE(ln V/B) against D, with all events and without each run's worst.

    python analysis/data_saving/event_share.py <run glob on CC> ...
    e.g. 'runs/dsave_diag_*_ft_ee_ttbar_nlo_d8_t8500/trial_*' 'runs/dsave_s2_ft_ee_ttbar_nlo_d8_t8500/trial_0406'
"""
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402
import plot_style as ps  # noqa: E402

REMOTE = r'''
import glob, json, os, sys
import numpy as np
out = []
for g in sys.argv[1:]:
    for d in sorted(glob.glob(g)):
        r = {"run_dir": d}
        try:
            s = json.load(open(d + "/data_stats.json"))
            r["prepd_std"] = float(s["prepd_std"][0])
            # ln(V/B) and mu = ln / prepd_std^2 hold only for a log transform of a V/B (or tree) target
            assert s["amp_trafos_pp"][0][0] == "log", s["amp_trafos_pp"][0][0]
            assert "_nlo" not in d or s.get("target_over_born"), "one loop trained on V, not V/B"
            for role in ("val", "test"):
                z = np.load(d + f"/sigma_stage/preds_{role}.npz")
                p, t = z["raw_pred"].astype(float).ravel(), z["raw_truth"].astype(float).ravel()
                e2 = (np.log(np.abs(p)) - np.log(np.abs(t))) ** 2
                i = int(e2.argmax())
                assert "pool_row" in z.files and (p * t > 0).all(), "no pool_row, or a sign mismatch"
                rows = z["pool_row"].astype(int)
                o = np.sort(e2)[::-1]
                r[role] = {"n": int(len(e2)), "mse_ln": float(e2.mean()), "mu_mse": float(e2.mean()) / r["prepd_std"] ** 2,
                           "top1_share": float(o[0] / o.sum()), "top5_share": float(o[:5].sum() / o.sum()),
                           "mse_ln_wo_top1": float(o[1:].mean()), "worst_row": int(rows[i]),
                           "worst_truth": float(t[i]), "worst_pred": float(p[i]),
                           "median_truth": float(np.median(t)), "worst_truth_q": float((t < t[i]).mean()),
                           "mse_vb": float(np.mean((p - t) ** 2))}
        except Exception as e:
            r["error"] = repr(e)[:200]
        out.append(r)
print("EVSHARE " + json.dumps(out))
'''


def main():
    rows = site_python(REMOTE, [f"'{g}'" for g in sys.argv[1:]], "EVSHARE")
    json.dump(rows, open(os.path.join(REPO, "analysis/data_saving/data/event_share.json"), "w"), indent=1)
    for r in rows:
        if "error" in r:
            print(r["run_dir"], r["error"])
            continue
        print(os.path.relpath(r["run_dir"], os.path.dirname(os.path.dirname(os.path.dirname(r["run_dir"])))))
        for role in ("val", "test"):
            q = r[role]
            print(f"  {role:4s} mse_ln {q['mse_ln']:.3e} mu {q['mu_mse']:.3e} top1 {q['top1_share']:.2f} "
                  f"top5 {q['top5_share']:.2f} w/o top1 {q['mse_ln_wo_top1']:.3e} worst row {q['worst_row']} "
                  f"V/B truth {q['worst_truth']:.4g} pred {q['worst_pred']:.4g} (median {q['median_truth']:.4g}, quantile {q['worst_truth_q']:.4f})")

    arms = {"tops": ("top parent", ps.C.vermillion, "s"), "parscale": ("stage-2 parent", ps.C.blue, "o"),
            "s2": ("stage-2 best", ps.C.grey, "^"), "offown": ("own off-shell scale", ps.C.green, "D")}
    pts = {}
    for r in rows:
        m = re.search(r"dsave_(?:diag_([a-z]+)|(s2))_ft_ee_ttbar_nlo_d(\d)_t", r["run_dir"])
        if m and "error" not in r:
            pts.setdefault(m.group(1) or m.group(2), []).append((10 ** (int(m.group(3)) / 2), r))
    for a, cc in pts.items():
        assert len({d for d, _ in cc}) == len(cc), f"several runs of arm {a} at one D: pass one run dir per cell"
    fig, axs = ps.figure(ncols=2)
    for n, (ax, role) in enumerate(zip(axs, ("val", "test"))):
        for a, cc in pts.items():
            name, col, mk = arms[a]
            cc.sort(key=lambda x: x[0])
            D = [d for d, _ in cc]
            ax.plot(D, [r[role]["mse_ln"] for _, r in cc], marker=mk, color=col, linestyle="-", label=name)
            ax.plot(D, [r[role]["mse_ln_wo_top1"] for _, r in cc], marker=mk, color=col, linestyle=":",
                    mfc="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(role + r" MSE$(\ln V/B)$")
        ps.process_label(ax, r"$e^+e^-\to t\bar t$, $V$")
        if n == 0:
            ps.legend(ax, "lower left")
        else:
            ax.plot([], [], color="k", linestyle="-", label="all events")
            ax.plot([], [], color="k", linestyle=":", label="w/o worst")
            ps.legend(ax, "lower left")
    ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/event_share"))

    # top parent over stage-2 parent (tops over parscale) per cell, with all events and without each run's worst
    by = {a: {round(d): r for d, r in cc} for a, cc in pts.items()}
    summ = {"cells": {}, "worst_val_rows": sorted({r["val"]["worst_row"] for cc in pts.values() for _, r in cc})}
    if "tops" in by and "parscale" in by:
        for d in sorted(by["tops"]):
            t, q = by["tops"][d], by["parscale"][d]
            summ["cells"][str(d)] = {f"{role}_{m}": t[role][f] / q[role][f] for role in ("val", "test")
                                     for m, f in (("all", "mse_ln"), ("wo1", "mse_ln_wo_top1"))}
            summ["cells"][str(d)]["tops_val_top1_share"] = t["val"]["top1_share"]
            summ["cells"][str(d)]["tops_val_worst_row"] = t["val"]["worst_row"]
        for k in ("val_all", "val_wo1", "test_all", "test_wo1"):
            v = [c[k] for c in summ["cells"].values()]
            summ[f"gmean_{k}"] = float(np.exp(np.mean(np.log(v))))
            summ[f"n_below_{k}"] = int(sum(x < 1 for x in v))
        summ["n_runs_val_worst_4523"] = sum(r["val"]["worst_row"] == 4523 for cc in pts.values() for _, r in cc)
        summ["n_runs"] = sum(len(cc) for cc in pts.values())
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
