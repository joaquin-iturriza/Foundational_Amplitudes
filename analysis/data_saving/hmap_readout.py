#!/usr/bin/env python3
"""pretrain_data_saving step 9: the fine-tune start read at fixed HPs on e+e- > t t~ (tree and one-loop V/B).

Arms, each one run per cell at the stage-2 best fine-tune trial's HPs and seed (dsave_diag_<arm>_ft_<target>_d<k>_*):
  s2        the stage-2 best fine-tune trial itself (r4 parent; its offshell_stats 'parent' was then a no-op)
  parscale  its rerun on the parent's off-shellness scale (r4), D=316..1e4: the no-head-map control
  tops      the same from the r5 parent (catalog_v4's physical-mass tops), D=316..1e4: the r5 no-head-map control
  hmap      as parscale, with the head map at load (cec3610c: the parent's readout mapped to the run's target stats,
            step 0 = the parent's function), every D of the study
  hmaptops  as tops, with the head map, D=316..1e4
  pshared   / psharedtops: the parent's target standardization kept (fine_tune.target_stats=parent_shared), tree only
  scr       the stage-2 best scratch trial (reference)
Per run, read on CC: the main stage's validation HETEROSC loss curve (out_0.log, step 0 included), the per-step train
loss, pre-clip gradient norm and lr (train_steps.npz, binned medians), the final model's (D276, sigma_stage/preds_*.npz)
val and test MSE of the run's log target re-based to ln units, (pred - truth)^2 x prepd_std^2 (ft_defect.py's
measure, comparable across target_stats), mean (r/sigma)^2 on val, and the head-map log line.
Writes data/hmap_runs.json (cache) and data/hmap_readout.json (per cell ratios against the controls).

    python analysis/data_saving/hmap_readout.py [--cached]
"""
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs/")
TLAB = {"epem_ttx": (r"$e^+e^-\to t\bar t$", r"MSE$(\ln|\mathcal{M}|^2)$"),
        "ee_ttbar_nlo": (r"$e^+e^-\to t\bar t$, $V$", r"MSE$(V/B)$")}
# arm: legend, colour, marker (the words Joaquin asked for: 'stage-2 best' is the selected search trial, a 'rerun'
# repeats it at the same HPs and seed; 'head map' starts from the parent's function)
STYLE = {"scr": ("scratch, stage-2 best", ps.C.blue, "o"), "s2": ("fine-tune, stage-2 best", ps.C.vermillion, "s"),
         "parscale": ("fine-tune rerun", ps.C.orange, "^"), "hmap": ("fine-tune rerun, head map", ps.C.green, "D"),
         "tops": ("fine-tune rerun, r5", ps.C.purple, "v"), "hmaptops": ("fine-tune rerun, r5, head map", ps.C.sky, "P")}
TARGETS = ("epem_ttx", "ee_ttbar_nlo")
DIAG_ARMS = ("parscale", "tops", "hmap", "hmaptops", "pshared", "psharedtops")

REMOTE = r'''
import glob, json, os, re, sys
import numpy as np, yaml
out = []
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        r = {"run_dir": d}
        try:
            log = open(d + "/out_0.log").read()
            main = log.split("sigma_stage")[0]
            r["val_curve"] = [(int(m.group(2)), float(m.group(1))) for m in re.finditer(r"Val loss: (\S+) \| step (\d+)", main)]
            hm = re.search(r"head mapped to this run's target stats.*", main)
            r["head_map"] = hm.group(0) if hm else None
            r["rid"] = (re.findall(r'"run_id": "([^"]+)"', log) or [None])[0]
            z = np.load(d + "/train_steps.npz")
            st, lo, gn, lr = z["step"], z["loss"], z["grad_norm"], z["lr"]
            k = max(1, len(st) // 200)
            med = lambda a: [float(np.nanmedian(a[i:i + k])) for i in range(0, len(a), k)]
            r["step"], r["loss"], r["grad_norm"], r["lr"] = st[::k].tolist(), med(lo), med(gn), med(lr)
            g = gn[np.isfinite(gn)]
            r["gn_pct"] = [float(np.percentile(g, q)) for q in (10, 50, 90)]
            s = json.load(open(d + "/data_stats.json"))
            r["prepd_std"] = float(np.ravel(s["prepd_std"])[0])
            for role in ("val", "test"):
                q = np.load(d + f"/sigma_stage/preds_{role}.npz")
                res = q["pred"].astype(float).ravel() - q["truth"].astype(float).ravel()
                r[role + "_mse_ln"] = float(np.mean(res ** 2)) * r["prepd_std"] ** 2
                r[role + "_top1"] = float(np.max(res ** 2) / np.sum(res ** 2))
                r[role + "_z2"] = float(np.mean((res / q["sigma"].astype(float).ravel()) ** 2))
        except Exception as e:
            r["error"] = repr(e)[:200]
        out.append(r)
print("HMAP " + json.dumps(out))
'''


def cell_of(run_dir):
    m = re.search(r"dsave_(s2_(ft|scr)|diag_([a-z]+)_ft)_(.+)_d(\d+)_t\d+/", run_dir)
    arm = m.group(3) or ("s2" if m.group(2) == "ft" else "scr")
    return arm, m.group(4), int(m.group(5))


def patterns():
    s2 = json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]
    pats = [f"runs/{c['sweep']}/trial_{c['best_hp']:04d}" for c in s2 if c["target"] in TARGETS]
    pats += [f"'runs/dsave_diag_{a}_ft_{tg}_d*/trial_*'" for a in DIAG_ARMS for tg in TARGETS]
    return pats


def load(cached):
    cache = os.path.join(DATA, "hmap_runs.json")
    if cached:
        return json.load(open(cache))
    rows = site_python(REMOTE, patterns(), "HMAP")
    json.dump(rows, open(cache, "w"))
    return rows


def gmean(x):
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def main():
    rows = load("--cached" in sys.argv)
    R = {}
    for r in rows:
        if "error" in r:
            print("!!", r["run_dir"], r["error"])
            continue
        R[cell_of(r["run_dir"] + "/")] = r
    D = lambda k: 10 ** (k / 2)
    out = {"pairs": {}}
    for arm, ctrl in (("hmap", "parscale"), ("hmap", "s2"), ("hmaptops", "tops"), ("pshared", "parscale"),
                      ("psharedtops", "tops")):
        for tg in TARGETS:
            ks = sorted(k for (a, t, k) in R if a == arm and t == tg and (ctrl, tg, k) in R)
            if not ks:
                continue
            rv = [R[(arm, tg, k)]["val_mse_ln"] / R[(ctrl, tg, k)]["val_mse_ln"] for k in ks]
            rt = [R[(arm, tg, k)]["test_mse_ln"] / R[(ctrl, tg, k)]["test_mse_ln"] for k in ks]
            out["pairs"][f"{arm}/{ctrl}/{tg}"] = {"D": [D(k) for k in ks], "val": rv, "test": rt, "gmean_val": gmean(rv),
                                                  "gmean_test": gmean(rt), "n_below_val": int(sum(x < 1 for x in rv))}
            print(f"{arm:11s}/{ctrl:8s} {tg:13s} D=" + ",".join(f"{D(k):.0f}" for k in ks) + "  val " +
                  " ".join(f"{x:.2f}" for x in rv) + f"  gmean {gmean(rv):.2f}  test " + " ".join(f"{x:.2f}" for x in rt) +
                  f"  gmean {gmean(rt):.2f}")
    # the one-loop late blow-up: validation HETEROSC loss at the end over its minimum, per run
    out["blowup"] = {}
    for (a, tg, k), r in sorted(R.items()):
        v = [x[1] for x in r["val_curve"] if x[0] > 0]
        if v:
            out["blowup"][f"{a}/{tg}/{D(k):.0f}"] = {"min": min(v), "end": v[-1], "rise": v[-1] - min(v),
                                                     "gn_pct": r["gn_pct"], "head_map": r["head_map"],
                                                     "step0": r["val_curve"][0][1] if r["val_curve"] else None}
    json.dump(out, open(os.path.join(DATA, "hmap_readout.json"), "w"), indent=1)
    figures(R)


def figures(R):
    """hmap_LD: test (stage2_LD's units) and val MSE(ln) against D per arm; hmap_curves_<target>_<x>: the validation
    HETEROSC loss against step at each D of the study; hmap_gradnorm: the pre-clip gradient norm's median and 10%
    quantile against D, with the clipping bound."""
    s2 = {(("s2" if c["arm"] == "ft" else "scr"), c["target"], round(2 * np.log10(c["D"]))): c
          for c in json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"] if c["target"] in TARGETS}
    dtest = {}
    for r in json.load(open(os.path.join(DATA, "stage2_diag_test.json"))):
        m = re.match(r"dsave_diag_([a-z]+)_ft_(.+)_d(\d+)_t", r["sweep"])
        if m and "error" not in r:
            dtest[(m.group(1), m.group(2), int(m.group(3)))] = r.get("mse_vb", r.get("mse_ln"))

    def val(key):
        return R[key]["val_mse_ln"] if key in R else (s2[key]["val_ln"] if key in s2 else None)

    def test(key):
        return s2[key]["test"] if key in s2 else dtest.get(key)

    fig, axs = ps.figure(ncols=2, nrows=2)
    for j, tg in enumerate(TARGETS):
        for i, (get, ylab) in enumerate(((test, TLAB[tg][1] + ", test"), (val, r"MSE$(\ln)$, val"))):
            ax = axs[i, j]
            for arm, (lab, col, m) in STYLE.items():
                ks = sorted(k for k in range(2, 11) if get((arm, tg, k)) is not None)
                ax.plot([10 ** (k / 2) for k in ks], [get((arm, tg, k)) for k in ks], marker=m, color=col,
                        linestyle="-" if arm in ("scr", "s2") else "none", label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$D$")
            ax.set_ylabel(ylab)
            ps.process_label(ax, TLAB[tg][0])
    ps.legend(axs[0, 0], "lower left")
    ps.save(fig, FIGS + "hmap_LD")

    for tg in TARGETS:
        ks = sorted({k for (a, t, k) in R if t == tg})
        figs = ps.panels(len(ks))
        for (f, ax), k in zip(figs, ks):
            for arm, (lab, col, m) in STYLE.items():
                if (arm, tg, k) in R:
                    ax.plot(*zip(*R[(arm, tg, k)]["val_curve"][1:]), color=col, label=lab)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("step")
            ax.set_ylabel("val HETEROSC loss")
            ps.process_label(ax, TLAB[tg][0] + rf", $D=10^{{{k / 2:g}}}$")
            ps.legend(ax, "upper left")
        ps.save_panels(figs, FIGS + f"hmap_curves_{tg}")

    fig, axs = ps.figure(ncols=2)
    for ax, tg in zip(axs, TARGETS):
        for arm in ("scr", "s2", "hmap"):
            lab, col, m = STYLE[arm]
            ks = sorted(k for (a, t, k) in R if a == arm and t == tg)
            x = [10 ** (k / 2) for k in ks]
            ax.plot(x, [R[(arm, tg, k)]["gn_pct"][1] for k in ks], marker=m, color=col, label=lab + ", median")
            ax.plot(x, [R[(arm, tg, k)]["gn_pct"][0] for k in ks], marker=m, color=col, linestyle=":",
                    markerfacecolor="none", label=lab + ", 10%")
        ax.axhline(5, color=ps.C.grey, linestyle="--", label="clip bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(r"$\|g\|$ before clipping")
        ps.process_label(ax, TLAB[tg][0])
    ps.legend(axs[0], "lower left")
    ps.save(fig, FIGS + "hmap_gradnorm")


if __name__ == "__main__":
    main()
