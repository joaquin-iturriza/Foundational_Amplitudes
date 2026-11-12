#!/usr/bin/env python3
"""pretrain_data_saving, D408 suspects batch (design PASS 2026-11-12 03:42): one change of the fine-tune recipe at a
time on e+e- > t t~ tree (epem_ttx) and one-loop V/B (ee_ttbar_nlo), D=316, 1e3, 3162, 1e4 (k=5..8), T=8500, seed 42,
at each cell's stage-2 best HPs. Arms (submit_suspects.py, data/suspects_runs.json): fine-tune lr_scale x1/3 (lr3),
x1/10 (lr10), WSD scheduler (wsd), AdamW (adamw), mu-only MSE (mse, training.heterosc_mu_only); scratch wsd and mse.
Controls (same config, HPs, seed, no override): tree fine-tune = the head-map runs (dsave_diag_hmap_ft_epem_ttx),
one-loop fine-tune = the parscale reruns (dsave_diag_parscale_ft_ee_ttbar_nlo; the stage-2 originals beside as the
same-HP pair), scratch = the stage-2 scratch best trial.
Measure per run: min of result.json's val_mse_curve x prepd_std^2 (val mu-MSE in ln units at the best-mu checkpoint);
beside it, from the final model's preds (sigma_stage/preds_*.npz, mu at the best-mu checkpoint), val MSE with and
without the worst val event and test MSE. R = arm / control per cell; per arm and target G = geometric mean of R over
the 4 cells, n< = cells with R < 1, and G_x the same without each run's worst val event. Rule: FIXES G <= 1/3 and
n< >= 3; HURTS G >= 3 and n< <= 1; else NONE; the two reads must agree, else NOT RESOLVED.

    python analysis/data_saving/suspects_readout.py [--cached]   # data/suspects_readout.json + figs/suspects_*
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs/")
K = (5, 6, 7, 8)
TARGETS = ("epem_ttx", "ee_ttbar_nlo")
FT_HP = {"epem_ttx": (1647, 507, 1817, 1387), "ee_ttbar_nlo": (340, 1544, 120, 406)}
SCR_HP = {"epem_ttx": (1605, 844, 640, 657), "ee_ttbar_nlo": (1663, 713, 358, 816)}
FT_ARMS = ("lr3", "lr10", "wsd", "adamw", "mse")
SCR_ARMS = ("wsd", "mse")
TLAB = {"epem_ttx": r"$e^+e^-\to t\bar t$", "ee_ttbar_nlo": r"$e^+e^-\to t\bar t$, $V/B$"}
YLAB = {"epem_ttx": r"val MSE$(\ln|\mathcal{M}|^2)$", "ee_ttbar_nlo": r"val MSE$(\ln(V/B))$"}
STYLE = {"ctl": ("control", "black", "o"), "scr": ("scratch", ps.C.blue, "s"), "pair": ("control rerun", ps.C.grey, "x"),
         "lr3": (r"lr $\times 1/3$", ps.C.vermillion, "v"), "lr10": (r"lr $\times 1/10$", ps.C.orange, "<"),
         "wsd": ("WSD", ps.C.green, "D"), "adamw": ("AdamW", ps.C.purple, "^"), "mse": (r"$\mu$-only MSE", ps.C.sky, "P")}

REMOTE = r'''
import json, os, sys, base64
import numpy as np, yaml
out = {}
for d in json.loads(base64.b64decode(sys.argv[1])):
    r = {}
    try:
        c = yaml.safe_load(open(d + "/config.yaml"))
        j = json.load(open(c["training"]["result_path"]))
        r["every"] = j.get("validate_every")
        r["val_mse_curve"], r["train_eval_mse_curve"] = j.get("val_mse_curve"), j.get("train_eval_mse_curve")
        r["prepd_std"] = s = float(np.ravel(json.load(open(d + "/data_stats.json"))["prepd_std"])[0])
        for role in ("val", "test"):
            f = d + f"/sigma_stage/preds_{role}.npz"
            if not os.path.exists(f):
                continue
            q = np.load(f)
            e = (q["pred"].astype(float).ravel() - q["truth"].astype(float).ravel()) ** 2 * s ** 2
            r[role + "_mse"] = float(e.mean())
            r[role + "_mse_x"] = float((e.sum() - e.max()) / (len(e) - 1))
    except Exception as ex:
        r["error"] = repr(ex)[:200]
    out[d] = r
print("SUS " + json.dumps(out))
'''


def dirs():
    """{(role, target, k): run_dir} for every arm and control."""
    out = {}
    for r in json.load(open(os.path.join(DATA, "suspects_runs.json"))):
        kind, arm, tg, dk = r["note"].split()[1:5]
        out[(arm if kind == "ft" else "scr_" + arm, tg, int(dk[1:]))] = r["run_dir"]
    for tg in TARGETS:
        for k, fh, sh in zip(K, FT_HP[tg], SCR_HP[tg]):
            arm = "hmap" if tg == "epem_ttx" else "parscale"
            out[("ctl", tg, k)] = f"runs/dsave_diag_{arm}_ft_{tg}_d{k}_t8500/trial_{fh:04d}"
            out[("pair", tg, k)] = f"runs/dsave_s2_ft_{tg}_d{k}_t8500/trial_{fh:04d}"
            out[("scr", tg, k)] = f"runs/dsave_s2_scr_{tg}_d{k}_t8500/trial_{sh:04d}"
    return out


def load(cached):
    cache = os.path.join(DATA, "suspects_runs_read.json")
    if not cached:
        import base64
        dd = dirs()
        got = site_python(REMOTE, [base64.b64encode(json.dumps(sorted(set(dd.values()))).encode()).decode()], "SUS")
        json.dump(got, open(cache, "w"))
    got = json.load(open(cache))
    R = {}
    for key, d in dirs().items():
        r = dict(got.get(d, {"error": "not read"}), run_dir=d)
        v = r.get("val_mse_curve")
        if v:
            r["val_min"] = float(np.nanmin(v)) * r["prepd_std"] ** 2
        R[key] = r
    return R


def gmean(x):
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def verdict(G, n):
    return "FIXES" if G <= 1 / 3 and n >= 3 else ("HURTS" if G >= 3 and n <= 1 else "NONE")


def main():
    R = load("--cached" in sys.argv)
    for key, r in sorted(R.items()):
        if "val_min" not in r:
            print("MISSING", key, r.get("error"))
    res = {"cells": [], "arms": []}
    for tg in TARGETS:
        for arm in FT_ARMS + tuple("scr_" + a for a in SCR_ARMS):
            ctl = "scr" if arm.startswith("scr_") else "ctl"
            rv, rx, rt = [], [], []
            for k in K:
                a, c, s = R[(arm, tg, k)], R[(ctl, tg, k)], R[("scr", tg, k)]
                if "val_min" not in a or "val_min" not in c:
                    continue
                cell = {"target": tg, "arm": arm, "k": k, "D": round(10 ** (k / 2)), "arm_val": a["val_min"],
                        "ctl_val": c["val_min"], "R": a["val_min"] / c["val_min"], "over_scratch": a["val_min"] / s["val_min"] if "val_min" in s else None}
                if a.get("val_mse_x") and c.get("val_mse_x"):
                    cell["R_x"] = a["val_mse_x"] / c["val_mse_x"]
                if a.get("test_mse") and c.get("test_mse"):
                    cell["R_test"] = a["test_mse"] / c["test_mse"]
                res["cells"].append(cell)
                rv.append(cell["R"])
                rx += [cell["R_x"]] if "R_x" in cell else []
                rt += [cell["R_test"]] if "R_test" in cell else []
            if not rv:
                continue
            G, n = gmean(rv), sum(x < 1 for x in rv)
            v = verdict(G, n)
            Gx, nx = (gmean(rx), sum(x < 1 for x in rx)) if len(rx) == len(rv) else (None, None)
            vx = verdict(Gx, nx) if Gx is not None else None
            res["arms"].append({"target": tg, "arm": arm, "n_cells": len(rv), "G": G, "n_below": n, "G_x": Gx,
                                "n_below_x": nx, "G_test": gmean(rt) if rt else None,
                                "verdict": v if vx in (None, v) else "NOT RESOLVED", "verdict_val": v, "verdict_x": vx})
    for c in res["cells"]:
        print(f"{c['target']:13s} {c['arm']:9s} D={c['D']:6d} arm {c['arm_val']:.2e} ctl {c['ctl_val']:.2e} "
              f"R {c['R']:.2f} R_x {c.get('R_x', float('nan')):.2f} R_test {c.get('R_test', float('nan')):.2f} "
              f"arm/scr {c['over_scratch'] or float('nan'):.2f}")
    for a in res["arms"]:
        print(f"{a['target']:13s} {a['arm']:9s} G {a['G']:.2f} n< {a['n_below']}/{a['n_cells']} "
              f"G_x {a['G_x'] if a['G_x'] is None else round(a['G_x'], 2)} G_test {a['G_test'] if a['G_test'] is None else round(a['G_test'], 2)} -> {a['verdict']}")
    pair = {f"{tg} d{k}": R[("pair", tg, k)].get("val_min", 0) / R[("ctl", tg, k)]["val_min"]
            for tg in TARGETS for k in K if "val_min" in R[("ctl", tg, k)]}
    res["pair_over_ctl"] = pair
    print("same-HP pair / control:", {k: round(v, 2) for k, v in pair.items()})
    json.dump(res, open(os.path.join(DATA, "suspects_readout.json"), "w"), indent=1)
    figures(R)


def figures(R):
    """suspects_LD: per target (columns) val mu-MSE against D, fine-tune arms (top) and scratch arms (bottom) beside
    their control and the scratch best; suspects_curves_<target>_<ft|scr>: val mu-MSE against step, one panel per D."""
    x = [10 ** (k / 2) for k in K]
    fig, axs = ps.figure(ncols=2, nrows=2)
    for j, tg in enumerate(TARGETS):
        for i, (arms, ctl) in enumerate(((FT_ARMS, "ctl"), (SCR_ARMS, "scr"))):
            ax = axs[i, j]
            for a in ((ctl, "scr") if ctl == "ctl" else ("scr",)):
                lab, col, m = STYLE[a]
                ax.plot(x, [R[(a, tg, k)].get("val_min", np.nan) for k in K], color=col, marker=m, label=lab)
            for a in arms:
                lab, col, m = STYLE[a]
                key = a if ctl == "ctl" else "scr_" + a
                ax.plot(x, [R[(key, tg, k)].get("val_min", np.nan) for k in K], color=col, marker=m, linestyle="none",
                        label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$D$")
            ax.set_ylabel(YLAB[tg])
            ps.process_label(ax, TLAB[tg] + (", fine-tune" if ctl == "ctl" else ", scratch"))
    ps.shared_legend(fig, axs[0, 0], ncol=4)
    ps.save(fig, FIGS + "suspects_LD")

    for tg in TARGETS:
        for grp, arms, ctl in (("ft", FT_ARMS, "ctl"), ("scr", SCR_ARMS, "scr")):
            fig, axs = ps.figure(ncols=2, nrows=2)
            for ax, k in zip(axs.ravel(), K):
                for a in (ctl,) + arms:
                    r = R[(a if grp == "ft" or a == "scr" else "scr_" + a, tg, k)]
                    v = r.get("val_mse_curve")
                    if not v:
                        continue
                    lab, col, _ = STYLE[a]
                    ax.plot(np.arange(len(v)) * (r.get("every") or 1), np.array(v, float) * r["prepd_std"] ** 2,
                            color=col, label=lab, linewidth=1.6 if a in ("ctl", "scr") else 1.0)
                ax.set_yscale("log")
                ax.set_xlabel("step")
                ax.set_ylabel(YLAB[tg])
                ps.process_label(ax, rf"$D={round(10 ** (k / 2))}$")
            ps.shared_legend(fig, axs[0, 0], ncol=3 if grp == "ft" else 3)
            ps.save(fig, FIGS + f"suspects_curves_{tg}_{grp}")


if __name__ == "__main__":
    main()
