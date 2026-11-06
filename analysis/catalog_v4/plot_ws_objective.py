#!/usr/bin/env python3
"""plan warm_start step 11: which logged quantity moves with the zero-shot reads across warm-start restarts.
Reads data/ws_objective_check.json (ws_objective_check.py rank). One point per comparison and seed (arm over its
same-seed reference): x the deciding reads, the mean of ln R on the pp (PP4) and e+e- (EE4) probes (R: median over
classes of the class geometric mean of probe MSE ratios); y the candidate's change, arm minus reference:
  (a) ln of the zero-shot MSE ratio on the default-sampled held-out tree classes HD4 (no PP4/EE4 class)
  (b) the final validation het NLL, arm minus reference
  (c) ln of the validation MSE of the mean ratio (geometric mean over the trained processes' val splits)
Filled: the comparisons the candidate was chosen on (restart recipes: lr tuning, moment fill, damped Muon, growth
tuning); open: the check set (retraining length 533/1066/2132, change against no change). -> figs/ws_objective_{a,b,c}
Prints each panel's Pearson r per set."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

PANELS = [("read_HD4", r"$\ln(\mathrm{MSE}_{\mathrm{arm}}/\mathrm{MSE}_{\mathrm{ref}})$, held-out trees"),
          ("nll_final", r"$\mathrm{NLL}_{\mathrm{arm}}-\mathrm{NLL}_{\mathrm{ref}}$, validation"),
          ("val_mse", r"$\ln(\mathrm{MSE}_{\mathrm{arm}}/\mathrm{MSE}_{\mathrm{ref}})$, validation")]


def main():
    d = json.load(open(os.path.join(HERE, "data", "ws_objective_check.json")))
    figs = ps.panels(3)
    for (fig, ax), (key, ylab) in zip(figs, PANELS):
        for st, col, face in (("select", ps.CYCLE[0], None), ("check", ps.CYCLE[1], "none")):
            x, y = [], []
            for r in d["comparisons"].values():
                if r["set"] != st:
                    continue
                t = np.mean([r["target"]["PP4"]["ln_seed"], r["target"]["EE4"]["ln_seed"]], axis=0)
                x += list(t); y += r["cand"][key]
            ax.scatter(x, y, s=18, edgecolors=col, facecolors=face or col,
                       label="restart recipes" if st == "select" else "length, change")
            print(key, st, "n", len(x), "pearson", round(float(np.corrcoef(x, y)[0, 1]), 3))
        ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlabel(r"$\frac{1}{2}(\ln R_{pp}+\ln R_{ee})$"); ax.set_ylabel(ylab)
        if key == "read_HD4":
            ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "ws_objective"))


if __name__ == "__main__":
    main()
