#!/usr/bin/env python3
"""plan warm_start step 12 (D307, D308): the restart study's compared quantities. Five k=533 warm starts of the C1
change (default-sampled share 0.25, het loss) from the checkpoint rerun v4_het_control_ckpt[_seed2], seeds 42 and 43,
as mixture_ab.WARM["C1r_*"] names them: the final model at the default restart (control), the step-3198 model at the
default restart, the step-3198 and step-2132 models continuing their own lr (no re-warm), the final model at lr x3.
(a) [top left] gradient norm before clipping, restart steps 0-40, log y; (b) training loss (het NLL per batch), a 25-step running
    median over the main stage; (c) validation loss of the main stage (het NLL, data/reads/<run>_val.json);
(d, e) per-class zero-shot MSE of log|M|^2 on the pp (PP4) and e+e- (EE4) probes as an empirical CDF over classes
    (per probe the two-seed geometric mean, per class the geometric mean over its probes;
    plot_warm_start_compare.class_mse), the model after the sigma stage (D276);
(f) [middle right] R(arm/control) on PP4 and EE4 (mixture_ab.py warm, data/warm_start_C1r_<arm>.json): median over classes of the
    class geometric mean of the probe ratios, 95% class bootstrap widened on each side by |ln R_seed|.
Usage: python3 analysis/catalog_v4/plot_ws_restart.py [--cached]
    -> figs/ws_restart_study.{png,pdf}, data/ws_restart_study_steps.json (per-step grad norm and loss from CC)"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from analysis.common import site_python  # noqa: E402
from mixture_ab import WARM, class_of  # noqa: E402
from plot_warm_start_compare import class_mse, curve  # noqa: E402

ARMS = [("final model, default restart (control)", "final_default", "C1r_ctrl", ps.C.blue),
        ("step-3198 model, default restart", "step3198_default", "C1r_step3198_default", ps.C.sky),
        ("step-3198 model, continue its lr", "step3198_continue", "C1r_step3198_continue", ps.C.green),
        ("step-2132 model, continue its lr", "step2132_continue", "C1r_step2132_continue", ps.C.orange),
        ("final model, lr x3", "final_lr3x", "C1r_final_lr3x", ps.C.vermillion)]
DIRS = {r: f"runs/catalog_v4_short/v4_het_ws_{a}{sfx}"
        for _, a, c, _ in ARMS for r, sfx in zip(WARM[c]["W"][533], ("", "_seed2"))}
SCRIPT = r'''
import json, numpy as np
DIRS = %r
out = {}
for r, d in DIRS.items():
    z = np.load(d + "/train_steps.npz")
    out[r] = {"grad_norm": z["grad_norm"][:41].tolist(), "loss": z["loss"].tolist(), "lr": z["lr"].tolist()}
print("RESTARTSTEPS " + json.dumps(out))
''' % DIRS
CACHE = os.path.join(HERE, "data", "ws_restart_study_steps.json")
SMOOTH = 25


def main():
    if "--cached" in sys.argv:
        g = json.load(open(CACHE))
    else:
        g = site_python(SCRIPT, [], "RESTARTSTEPS")
        json.dump(g, open(CACHE, "w"))
    cls = class_of()
    fig, axs = ps.figure(ncols=2, nrows=3)
    (aG, aL), (aV, aR), (aP, aE) = axs
    out, short = {}, {}
    for j, (lab, arm, c, col) in enumerate(ARMS):
        runs = WARM[c]["W"][533]
        for i, (r, ls) in enumerate(zip(runs, ("-", "--"))):
            gn = np.array(g[r]["grad_norm"]); lo = np.array(g[r]["loss"])
            n_main = 533                                   # the main stage; the sigma stage's steps follow it
            # running median: single-batch het NLL spikes (up to 2e10) would swamp a running mean
            lm = np.array([np.median(lo[t - SMOOTH + 1:t + 1]) for t in range(SMOOTH - 1, n_main)])
            aG.plot(np.arange(len(gn)), gn, color=col, ls=ls, label=lab if i == 0 else None)
            aL.plot(np.arange(SMOOTH - 1, n_main), lm, color=col, ls=ls)
            s, v = curve(r)
            aV.plot(s, v, color=col, ls=ls)
            short[f"{arm}_s{42 + i}"] = {"grad_max_0_19": float(gn[:20].max()), "loss_0": float(lo[0]),
                                        "loss_max_0_39": float(lo[:40].max()),
                                        "loss_median_last50": float(np.median(lo[n_main - 50:n_main])),
                                        "loss_spikes_above_0": np.where(lo[:n_main] > 0)[0].tolist(),
                                        "val_first": float(v[0]), "val_last": float(v[-1]),
                                        "val_min": float(v.min()), "lr_0": g[r]["lr"][0],
                                        "lr_peak": float(max(g[r]["lr"][:n_main]))}
        for a, axis in ((aP, "PP4"), (aE, "EE4")):
            m, _ = class_mse(runs, axis, cls)
            a.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col)
            out[f"{arm} {axis} median class MSE"] = float(np.median(m))
        if j:
            k = json.load(open(os.path.join(HERE, "data", f"warm_start_{c}.json")))
            r12 = k["k"]["533"]["W_over_control"]
            for dx, axis, mk in ((-0.12, "PP4", "o"), (0.12, "EE4", "s")):
                rv, (l, h) = r12[axis]["all"]["R"], r12[axis]["all"]["widened"]
                aR.errorbar([j + dx], [rv], yerr=[[rv - l], [h - rv]], color=col, marker=mk, ls="none", capsize=2)
    aR.plot([], [], "ko", ls="none", label="pp")
    aR.plot([], [], "ks", ls="none", label=r"$e^+e^-$")
    aR.axhline(1, color=ps.C.grey, lw=0.8)
    aR.set_xticks(range(1, len(ARMS))); aR.set_xticklabels(["3198\ndefault", "3198\ncont.", "2132\ncont.", "lr x3"])
    aR.set_ylabel("MSE ratio, arm over control")
    aG.plot([], [], "k-", label="seed 42"); aG.plot([], [], "k--", label="seed 43")
    aG.set_yscale("log"); aG.set_xlabel("step of the restart"); aG.set_ylabel("gradient norm")
    aL.set_xlabel("step of the restart"); aL.set_ylabel("training loss (het NLL)")
    lo = min(float(curve(r)[1].min()) for _, _, c, _ in ARMS for r in WARM[c]["W"][533])
    aV.set_ylim(lo - 0.1, 0.5)
    aV.set_xlabel("step of the restart"); aV.set_ylabel("validation loss (het NLL)")
    for a, w in ((aP, "pp"), (aE, r"$e^+e^-$")):
        a.set_xscale("log"); a.set_xlabel(w + r" classes: MSE of $\log|M|^2$"); a.set_ylabel("fraction of classes")
        a.set_ylim(0, 1)
    ps.shared_legend(fig, aG, ncol=2)
    ps.legend(aR, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "ws_restart_study"))
    print(json.dumps(out, indent=0))
    print(json.dumps(short))


if __name__ == "__main__":
    main()
