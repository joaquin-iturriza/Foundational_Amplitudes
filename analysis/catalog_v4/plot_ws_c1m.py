#!/usr/bin/env python3
"""plan warm_start step 6 (D291, design passed 2026-11-06T04:48): the k=533 warm starts restarted with the recipe tuned
on the validation MSE of the mean (hp_1333: lr_scale 1.85, WSD cooldown 0.164) against their default restart.
  (no argument)  C1m: the warm start (FA-ccin2p3-4f9e0532 / 8cb6c7b5, mixture_ab.WARM["C1m"]) against the untuned
                 restart (341bf564 / 254abf4f), the NLL-tuned one (hp_1860, 56cca07a / 75d28cb4), the pretrained start
                 A and the share-0.25 change from scratch B -> figs/ws_c1m_{a,b,c}
  --growth       C10wm (D285): the width growth 8->16 heads (GROW_HP=1333, 31359864 / bfed460f) against the same growth
                 with the default restart (e046ac67 / f35b6a78), width 8 from scratch (the parent, A) and width 16
                 from scratch (B) -> figs/ws_c10wm_{a,b,c}
(a) validation loss of the main stage (het NLL, arithmetic mean over processes, share-0.25 validation) against the
    restart's step, seed 42, warm starts only (A and B validate on other data or target stats; the legend names every
    arm of b and c);
(b, c) per-class zero-shot MSE of log|M|^2 on the pp (PP4) and e+e- (EE4) probes as an empirical CDF over classes
    (per probe the two-seed geometric mean, per class the geometric mean over its probes; plot_warm_start_compare).
Prints each arm's median class MSE and each warm curve's first, highest (with its step) and last value."""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from mixture_ab import WARM, class_of  # noqa: E402
from plot_warm_start_compare import class_mse, curve, pair  # noqa: E402

FIGS = {"ws_c1m": ([("pretrained start", WARM["C1"]["A"], False),
                    ("scratch with change", WARM["C1"]["B"], False),
                    ("warm, default lr", WARM["C1m"]["ref"]["untuned"], True),
                    ("warm, NLL-tuned", WARM["C1m"]["ref"]["nll_tuned"], True),
                    ("warm, MSE-tuned", WARM["C1m"]["W"][533], True)], -1.5),
        "ws_c10wm": ([("width 8, scratch", WARM["C10wm"]["A"], False),
                      ("width 16, scratch", WARM["C10wm"]["B"], False),
                      ("grown 8$\\to$16, default lr", WARM["C10wm"]["ref"]["untuned"], True),
                      ("grown 8$\\to$16, MSE-tuned", WARM["C10wm"]["W"][533], True)], -1.0)}


def main():
    name = "ws_c10wm" if "--growth" in sys.argv[1:] else "ws_c1m"
    arms, ytop = FIGS[name]
    cls = class_of()
    figs = ps.panels(3)
    (fa, aV), (fb, aP), (fc, aE) = figs
    lo = []
    for j, (lab, runs, warm) in enumerate(arms):
        col = ps.CYCLE[j % len(ps.CYCLE)]
        if warm:
            s, v = curve(pair(runs, "PP4")[0])
            aV.plot(s, v, color=col, label=lab); lo.append(v.min())
            print(lab, "val NLL s42: first", round(float(v[0]), 3), "max", round(float(v.max()), 3), "at",
                  int(s[int(np.argmax(v))]), "last", round(float(v[-1]), 3))
        else:
            aV.plot([], [], color=col, label=lab)
        for a, axis in ((aP, "PP4"), (aE, "EE4")):
            m, _ = class_mse(pair(runs, axis), axis, cls)
            a.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col, label=lab)
            print(lab, axis, "median class MSE", round(float(np.median(m)), 4))
    aV.set_ylim(min(lo) - 0.1, ytop); aV.set_xlabel("step of the restart"); aV.set_ylabel("validation loss (het NLL)")
    for a, w in ((aP, "pp"), (aE, r"$e^+e^-$")):
        a.set_xscale("log"); a.set_xlabel(w + r" classes: MSE of $\log|M|^2$"); a.set_ylabel("fraction of classes")
        a.set_ylim(0, 1)
    ps.legend(aV, "upper right")
    ps.save_panels([f for f, _ in figs], os.path.join(HERE, "figs", name))


if __name__ == "__main__":
    main()
