#!/usr/bin/env python3
"""plan warm_start step 6 (D303): the moment-fill test's compared quantities, filled optimizer (fine_tune.
moment_fill_steps=100, arm v4_het_warm25f_k533[_seed2], FA-ccin2p3-b2c69c1a / f5245570) against the same-commit
default restart (v4_het_warm25_k533[_seed2]_tagd303, FA-ccin2p3-0202ba7f / 7b2d6c24), seeds 42 and 43, as
mixture_ab.WARM["C1f"] names them.
(a) gradient norm before clipping against the restart's step, steps 0-40 (train_steps.npz on CC, the fill's own lr-0
    batches not counted), log y; (b) validation loss of the main stage (het NLL, arithmetic mean over processes,
    data/reads/<run>_val.json); (c, d) per-class zero-shot MSE of log|M|^2 on the pp (PP4) and e+e- (EE4) probes as an
    empirical CDF over classes (per probe the two-seed geometric mean, per class the geometric mean over its probes;
    plot_warm_start_compare.class_mse), the model after the sigma stage (D276).
--damped: the same four panels for step 8 (D303), Muon's lr damped at the restart's start (fine_tune.muon_warmup_steps=
168, v4_het_warm25u_k533[_seed2], mixture_ab.WARM["C1u"]) against the same default restart pair.
Usage: python3 analysis/catalog_v4/plot_ws_fill.py [--cached] [--damped]
    -> figs/ws_fill.{png,pdf}, data/ws_fill_steps.json (--damped: figs/ws_muon_damped, data/ws_muon_damped_steps.json)"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from analysis.common import site_python  # noqa: E402
from mixture_ab import WARM, class_of  # noqa: E402
from plot_warm_start_compare import class_mse, curve  # noqa: E402

DIRS = {"FA-ccin2p3-b2c69c1a": "runs/catalog_v4_short/v4_het_warm25f_k533",
        "FA-ccin2p3-f5245570": "runs/catalog_v4_short/v4_het_warm25f_k533_seed2",
        "FA-ccin2p3-0202ba7f": "runs/catalog_v4_short/v4_het_warm25_k533_tagd303",
        "FA-ccin2p3-7b2d6c24": "runs/catalog_v4_short/v4_het_warm25_k533_seed2_tagd303"}
DAMPED = "--damped" in sys.argv
if DAMPED:
    DIRS = {WARM["C1u"]["W"][533][0]: "runs/catalog_v4_short/v4_het_warm25u_k533",
            WARM["C1u"]["W"][533][1]: "runs/catalog_v4_short/v4_het_warm25u_k533_seed2",
            **{r: d for r, d in DIRS.items() if r in WARM["C1f"]["ref"]["default"]}}
ARMS = [("Muon lr damped first" if DAMPED else "optimizer filled first",
         WARM["C1u" if DAMPED else "C1f"]["W"][533], ps.C.vermillion),
        ("default restart", WARM["C1f"]["ref"]["default"], ps.C.blue)]
SCRIPT = r'''
import json, numpy as np
DIRS = %r
out = {r: np.load(d + "/train_steps.npz")["grad_norm"][:41].tolist() for r, d in DIRS.items()}
print("FILLSTEPS " + json.dumps(out))
''' % DIRS
CACHE = os.path.join(HERE, "data", "ws_muon_damped_steps.json" if DAMPED else "ws_fill_steps.json")


def main():
    if "--cached" in sys.argv:
        g = json.load(open(CACHE))
    else:
        g = site_python(SCRIPT, [], "FILLSTEPS")
        json.dump(g, open(CACHE, "w"))
    cls = class_of()
    fig, axs = ps.figure(ncols=2, nrows=2)
    (aG, aV), (aP, aE) = axs
    out = {}
    short = {}                       # the same numbers keyed without spaces (docs/results.tex claims): arm_s<seed>
    for (lab, runs, col), tag in zip(ARMS, ("arm", "default")):
        for i, (r, ls) in enumerate(zip(runs, ("-", "--"))):
            sl = f"{lab}, seed {42 + i}"
            aG.plot(np.arange(len(g[r])), g[r], color=col, ls=ls, label=sl)
            s, v = curve(r)
            aV.plot(s, v, color=col, ls=ls, label=sl)
            out[sl] = {"grad_max_0_19": max(g[r][:20]), "grad_median_20_40": float(np.median(g[r][20:41])),
                      "val_max": float(v.max()), "val_max_at": int(s[int(np.argmax(v))]),
                      "steps_off_scale": [int(x) for x in s[v > 0.5]], "val_last": float(v[-1])}
            short[f"{tag}_s{42 + i}"] = out[sl]
        for a, axis in ((aP, "PP4"), (aE, "EE4")):
            m, _ = class_mse(runs, axis, cls)
            a.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col, label=lab)
            out[f"{lab} {axis} median class MSE"] = float(np.median(m))
    aG.set_yscale("log"); aG.set_xlabel("step of the restart"); aG.set_ylabel("gradient norm")
    # the y range of the curves' steps below 0 (a het NLL spike, e.g. the damped seed 43's 1.9e8 at step 300, goes
    # off scale and is listed in the printed out instead of flattening the rest)
    lo = min(min(v[v < 0]) for _, v in (curve(r) for _, rs, _ in ARMS for r in rs))
    aV.set_ylim(lo - 0.1, 0.5)
    aV.set_xlabel("step of the restart"); aV.set_ylabel("validation loss (het NLL)")
    for a, w in ((aP, "pp"), (aE, r"$e^+e^-$")):
        a.set_xscale("log"); a.set_xlabel(w + r" classes: MSE of $\log|M|^2$"); a.set_ylabel("fraction of classes")
        a.set_ylim(0, 1)
    # the one legend, on the gradient-norm panel (log y, room above the jump); colours are the arms in every panel
    ps.legend(aG, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "ws_muon_damped" if DAMPED else "ws_fill"))
    print(json.dumps(out, indent=0))
    print(json.dumps(short))


if __name__ == "__main__":
    main()
