#!/usr/bin/env python3
"""plan warm_start step 14 (D319, seed 42 only, D315): the compared quantities of the warm-start mechanisms toward the
C1 change (default-sampled share 0.25, het loss, k=533), as mixture_ab.S14 names the runs: the same-commit default
restart (control), shrink and perturb, share 0.50 for 267 and for 533 steps, the WSD parent branched at step 3731 with
its cooldown on share 0.25 or 0.50, the WSD parent's final model at the default restart.
(a) gradient norm before clipping, restart steps 0-40, log y; (b) training loss (het NLL per batch), a 25-step running
    median over the main stage; (c) validation loss of the main stage (het NLL, data/reads/<run>_val.json);
(d) R(arm/control) on PP4 and EE4 (mixture_ab.py warm_single, data/warm_start_S14.json): median over classes of the
    class geometric mean of the probe ratios, 95% class bootstrap widened by the median paired seed spread;
(e, f) per-class zero-shot MSE of log|M|^2 on the pp (PP4) and e+e- (EE4) probes as an empirical CDF over classes,
    with the share-0.25 change from scratch (B, seed 42); every model after its sigma stage (D276).
Usage: python3 analysis/catalog_v4/plot_ws_fixes.py [--cached]
    -> figs/ws_fixes.{png,pdf}, data/ws_fixes_steps.json (per-step grad norm, loss and lr from CC)"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from analysis.common import site_python  # noqa: E402
from mixture_ab import S14, class_of  # noqa: E402
from plot_warm_start_compare import class_mse, curve  # noqa: E402

# (label, S14 key, run dir under runs/catalog_v4_short, colour, steps of the main stage)
ARMS = [("default restart (control)", "control", "v4_het_wsx_default_rerun", ps.C.blue, 533),
        ("shrink and perturb", "shrinkperturb", "v4_het_wsx_shrinkperturb", ps.C.vermillion, 533),
        ("share 0.50, 267 steps", "share50_k267", "v4_het_wsx_share50_k267", ps.C.sky, 267),
        ("share 0.50, 533 steps", "share50_k533", "v4_het_warm50_k533", ps.C.green, 533),
        ("WSD branch, share 0.25", "wsdbranch25", "v4_het_wsx_wsdbranch25", ps.C.orange, 533),
        ("WSD branch, share 0.50", "wsdbranch50", "v4_het_wsx_wsdbranch50", ps.C.purple, 533),
        ("WSD final, default restart", "wsdfinal_default", "v4_het_wsx_wsdfinal_default", ps.C.grey, 533)]
RUN = {k: (S14["control"] if k == "control" else S14["arms"][k]) for _, k, _, _, _ in ARMS}
DIRS = {RUN[k]: f"runs/catalog_v4_short/{d}" for _, k, d, _, _ in ARMS if RUN[k]}
SCRIPT = r'''
import json, numpy as np
DIRS = %r
out = {}
for r, d in DIRS.items():
    z = np.load(d + "/train_steps.npz")
    out[r] = {"grad_norm": z["grad_norm"][:41].tolist(), "loss": z["loss"].tolist(), "lr": z["lr"].tolist()}
print("FIXSTEPS " + json.dumps(out))
''' % DIRS
CACHE = os.path.join(HERE, "data", "ws_fixes_steps.json")
SMOOTH = 25


def main():   # dup-ok: one run per arm (seed 42 only), per-arm main-stage length (267 or 533), R from warm_single, B drawn in the CDFs; plot_ws_restart.py is the two-seed C1r_* figure
    if "--cached" in sys.argv:
        g = json.load(open(CACHE))
    else:
        g = site_python(SCRIPT, [], "FIXSTEPS")
        json.dump(g, open(CACHE, "w"))
    cls = class_of()
    s14 = json.load(open(os.path.join(HERE, "data", "warm_start_S14.json")))
    fig, axs = ps.figure(ncols=2, nrows=3)
    (aG, aL), (aV, aR), (aP, aE) = axs
    out, short, ticks = {}, {}, []
    for lab, k, _, col, n_main in ARMS:
        r = RUN[k]
        if not r or r not in g:
            continue
        gn = np.array(g[r]["grad_norm"]); lo = np.array(g[r]["loss"])
        lm = np.array([np.median(lo[t - SMOOTH + 1:t + 1]) for t in range(SMOOTH - 1, n_main)])
        aG.plot(np.arange(len(gn)), gn, color=col, label=lab)
        aL.plot(np.arange(SMOOTH - 1, n_main), lm, color=col)
        s, v = curve(r)
        aV.plot(s, v, color=col)
        short[k] = {"run": r, "grad_max_0_19": float(gn[:20].max()), "loss_0": float(lo[0]),
                    "loss_max_0_39": float(lo[:40].max()), "loss_rise_0_39": float(lo[:40].max() - lo[0]),
                    "loss_median_last50": float(np.median(lo[n_main - 50:n_main])),
                    "loss_spikes_above_0": np.where(lo[:n_main] > 0)[0].tolist(),
                    "val_first": float(v[0]), "val_last": float(v[-1]), "val_min": float(v.min()),
                    "lr_0": g[r]["lr"][0], "lr_peak": float(max(g[r]["lr"][:n_main]))}
        for a, axis in ((aP, "PP4"), (aE, "EE4")):
            m, _ = class_mse((r,), axis, cls)
            a.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col)
            out[f"{k} {axis} median class MSE"] = float(np.median(m))
        row = s14["arms"].get(k)
        if k != "control" and row and row.get("PP4"):
            j = len(ticks) + 1
            ticks.append(lab.replace(", ", "\n"))
            for dx, axis, mk in ((-0.12, "PP4", "o"), (0.12, "EE4", "s")):
                rv, (l, h) = row[axis]["R"], row[axis]["widened"]
                aR.errorbar([j + dx], [rv], yerr=[[rv - l], [h - rv]], color=col, marker=mk, ls="none", capsize=2)
    for a, axis in ((aP, "PP4"), (aE, "EE4")):
        m, _ = class_mse((S14["B"],), axis, cls)
        a.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color="k", ls=":")
        out[f"B {axis} median class MSE"] = float(np.median(m))
    aG.plot([], [], "k:", label="share 0.25 from scratch")
    aR.plot([], [], "ko", ls="none", label="pp")
    aR.plot([], [], "ks", ls="none", label=r"$e^+e^-$")
    aR.axhline(1, color=ps.C.grey, lw=0.8)
    aR.set_xticks(range(1, len(ticks) + 1)); aR.set_xticklabels(ticks, fontsize=6)
    aR.set_ylabel("MSE(arm) / MSE(default restart)")
    aG.set_yscale("log"); aG.set_xlabel("step of the restart"); aG.set_ylabel("gradient norm")
    aL.set_xlabel("step of the restart"); aL.set_ylabel("training loss (het NLL)")
    vmin = min(float(curve(RUN[k])[1].min()) for _, k, _, _, _ in ARMS if RUN[k] and RUN[k] in g)
    aV.set_ylim(vmin - 0.1, 0.5)
    aV.set_xlabel("step of the restart"); aV.set_ylabel("validation loss (het NLL)")
    for a, w in ((aP, "pp"), (aE, r"$e^+e^-$")):
        a.set_xscale("log"); a.set_xlabel(w + r" classes: MSE of $\log|M|^2$"); a.set_ylabel("fraction of classes")
        a.set_ylim(0, 1)
    ps.shared_legend(fig, aG, ncol=3)
    ps.legend(aR, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "ws_fixes"))
    print(json.dumps(out, indent=0))
    print(json.dumps(short))


if __name__ == "__main__":
    main()
