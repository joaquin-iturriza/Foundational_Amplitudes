#!/usr/bin/env python3
"""pretrain_data_saving step 9 (D400 defect hunt): why the fine-tunes lose to scratch on e+e- > t t~. Per stage-2 best
trial (both arms, four targets, D=316..1e4; the run list of data/train_val_gap.json), read on CC:
  - the curves: train HETEROSC loss (train_steps.npz) and validation HETEROSC loss (out_0.log, main stage);
  - the final model's (D276, sigma_stage/preds_{train,val}.npz) standardized residual r = pred - truth and sigma
    on the 2000 scored train events and on the val pool: rms(r), median(sigma), mean((r/sigma)^2);
  - the target standardization (own prepd mean/std) and, for a fine-tune, its parent's (lead a: the head mismatch
    sd_parent/sd_own the parent's mu row is read through under fine_tune.target_stats=own).
Writes data/ft_defect.json and three figure pairs in figs/:
  ft_defect_curves train (line) and val (markers) HETEROSC loss vs step at D=3162, fine-tune against scratch, per target;
  ft_sigma      final model's rms(r) on train and val and median(sigma) on train vs D, per target and arm;
  ft_vs_scratch fine-tune over scratch val MSE(ln) per (target, D) against the scratch val MSE(ln) (lead d) and
                against sd_parent/sd_own (lead a).
    python analysis/data_saving/ft_defect.py [--cached]
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

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs")

REMOTE = r'''
import json, os, re, sys
import numpy as np, yaml
out = []
for d in sys.argv[1:]:
    r = {"run_dir": d}
    try:
        z = np.load(d + "/train_steps.npz")
        st, lo, gn = z["step"], z["loss"], z["grad_norm"]
        k = max(1, len(st) // 400)
        r["step"], r["loss"] = st[::k].tolist(), [float(np.median(lo[i:i + k])) for i in range(0, len(lo), k)]
        r["grad_norm_first100"] = float(np.median(gn[:100]))
        r["val_curve"] = [(int(m.group(2)), float(m.group(1))) for m in
                    re.finditer(r"Val loss: (\S+) \| step (\d+)", open(d + "/out_0.log").read().split("sigma_stage")[0])]
        s = json.load(open(d + "/data_stats.json"))
        r["prepd"] = [float(s["prepd_mean"][0]), float(s["prepd_std"][0])]
        c = yaml.safe_load(open(d + "/config.yaml"))
        pp = (c.get("fine_tune") or {}).get("pretrained_path")
        if pp:
            pdir = os.path.dirname(os.path.dirname(pp.replace("${PROJECT_DIR}/", "")))
            p = json.load(open(os.path.join(pdir, "data_stats.json")))
            r["parent_prepd"] = [float(p["prepd_mean"][0]), float(p["prepd_std"][0])]
        for role in ("train", "val"):
            q = np.load(d + f"/sigma_stage/preds_{role}.npz")
            res = q["pred"].astype(float).ravel() - q["truth"].astype(float).ravel()
            sg = q["sigma"].astype(float).ravel()
            r[role] = {"n": int(len(res)), "rms_r": float(np.sqrt(np.mean(res ** 2))), "med_sigma": float(np.median(sg)),
                       "mean_z2": float(np.mean((res / sg) ** 2)), "q99_z2": float(np.quantile((res / sg) ** 2, 0.99))}
    except Exception as e:
        r["error"] = repr(e)[:200]
    out.append(r)
print("FTDEF " + json.dumps(out))
'''

TARGETS = {"epem_ttx": r"$e^+e^-\to t\bar t$ tree", "ee_ttbar_nlo": r"$e^+e^-\to t\bar t$ one-loop/tree",
           "ee_uu": r"$e^+e^-\to u\bar u$ tree", "ee_uu_nlo": r"$e^+e^-\to u\bar u$ one-loop"}
DS = {"d5": 316, "d6": 1000, "d7": 3162, "d8": 10000}


def key(run_dir):
    m = re.search(r"dsave_s2_(ft|scr)_(.+)_(d\d)_t\d+", run_dir)
    return m.group(2), m.group(1), DS[m.group(3)]


def main():
    cache = os.path.join(DATA, "ft_defect.json")
    if "--cached" in sys.argv:
        rows = json.load(open(cache))
    else:
        runs = [r["run_dir"] for r in json.load(open(os.path.join(DATA, "train_val_gap.json")))]
        rows = site_python(REMOTE, runs, "FTDEF")
        json.dump(rows, open(cache, "w"))
    R = {}
    for r in rows:
        if "error" in r:
            print("!!", r["run_dir"], r["error"])
            continue
        R[key(r["run_dir"])] = r

    # --- table
    print(f"{'target':14s} {'D':>6s} arm  rms_r tr   rms_r val  med_sig tr  z2 tr  z2 val  val NLL min@step end  sd_p/sd")
    for (tg, arm, D), r in sorted(R.items()):
        v = r["val_curve"]
        vmin = min(v, key=lambda x: x[1])
        hm = r["parent_prepd"][1] / r["prepd"][1] if "parent_prepd" in r else float("nan")
        print(f"{tg:14s} {D:6d} {arm:3s} {r['train']['rms_r']:.2e} {r['val']['rms_r']:.2e} {r['train']['med_sigma']:.2e} "
              f"{r['train']['mean_z2']:6.2f} {r['val']['mean_z2']:8.1f} {vmin[1]:8.2f}@{vmin[0]:5d} {v[-1][1]:8.2f} {hm:6.1f}")

    # --- figure 1: curves at D=3162
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, tg in zip(np.ravel(axs), TARGETS):
        for arm, col, lab in (("ft", ps.C.vermillion, "ft"), ("scr", ps.C.blue, "scr")):
            r = R.get((tg, arm, 3162))
            if r is None:
                continue
            ax.plot(r["step"], r["loss"], color=col, alpha=0.5, label=f"{lab}, train")
            ax.plot(*zip(*r["val_curve"]), color=col, marker="o", ls="none", label=f"{lab}, val")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("HETEROSC loss")
        ps.process_label(ax, TARGETS[tg])
    ps.legend(np.ravel(axs)[0], "upper left")
    ps.save(fig, os.path.join(FIGS, "ft_defect_curves"))

    # --- figure 2: final model's residual and sigma, train vs val
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, tg in zip(np.ravel(axs), TARGETS):
        for arm, col, lab in (("ft", ps.C.vermillion, "ft"), ("scr", ps.C.blue, "scr")):
            ks = sorted(k for k in R if k[0] == tg and k[1] == arm)
            if not ks:
                continue
            D = [k[2] for k in ks]
            ax.plot(D, [R[k]["val"]["rms_r"] for k in ks], color=col, marker="o", label=f"{lab} rms $r$ val")
            ax.plot(D, [R[k]["train"]["rms_r"] for k in ks], color=col, marker="s", ls="--", label=f"{lab} rms $r$ train")
            ax.plot(D, [R[k]["train"]["med_sigma"] for k in ks], color=col, marker="^", ls=":",
                    label=rf"{lab} median $\sigma$ train")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(r"standardized units")
        ps.process_label(ax, TARGETS[tg])
    ps.legend(np.ravel(axs)[0], "upper left")
    ps.save(fig, os.path.join(FIGS, "ft_sigma"))

    # --- figure 3: ft/scr val MSE vs scratch level (lead d) and vs head mismatch (lead a)
    fig, (axL, axR) = ps.figure(ncols=2)
    cols = dict(zip(TARGETS, (ps.C.vermillion, ps.C.orange, ps.C.blue, ps.C.green)))
    out = []
    for tg in TARGETS:
        pts = []
        for D in DS.values():
            f, s = R.get((tg, "ft", D)), R.get((tg, "scr", D))
            if f and s:   # val MSE of the transformed target (ln|M|^2, ln(V/B) or ln|V|): (rms r x own std)^2
                mf, ms = ((x["val"]["rms_r"] * x["prepd"][1]) ** 2 for x in (f, s))
                pts.append((D, ms, mf / ms, f["parent_prepd"][1] / f["prepd"][1]))
        if not pts:
            continue
        out += [dict(target=tg, D=p[0], scr_val=p[1], ft_over_scr=p[2], head_mismatch=p[3]) for p in pts]
        axL.plot([p[1] for p in pts], [p[2] for p in pts], color=cols[tg], marker="o", ls="none", label=TARGETS[tg])
        axR.plot([p[3] for p in pts], [p[2] for p in pts], color=cols[tg], marker="o", ls="none")
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color=ps.C.grey, ls=":", label="ft = scr" if ax is axL else None)
        ax.set_ylabel(r"MSE$_{\rm ft}$ / MSE$_{\rm scr}$ (val)")
    axL.set_xlabel(r"MSE$_{\rm scr}$ (val, $\ln$ target)")
    axR.set_xlabel(r"$\sigma_{\rm parent} / \sigma_{\rm own}$")
    from matplotlib.ticker import NullFormatter
    axR.xaxis.set_minor_formatter(NullFormatter())
    ps.legend(axL, "upper left")
    ps.save(fig, os.path.join(FIGS, "ft_vs_scratch"))
    json.dump({"runs": rows, "ft_vs_scratch": out}, open(cache.replace(".json", "_summary.json"), "w"))
    for o in out:
        print("FVS", json.dumps(o))


if __name__ == "__main__":
    main()
