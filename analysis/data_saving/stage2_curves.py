#!/usr/bin/env python3
"""pretrain_data_saving stage 2, Joaquin's redraws (2026-11-12): from the stage-2 best trials' own logs (both arms, four
targets, every D of the study):
  stage2_curves      validation HETEROSC loss against step (out_0.log, main stage), one panel per target, colour = D,
                     solid scratch, dashed fine-tune;
  stage2_ft_vs_scr   fine-tune over scratch at every D, test in stage2_LD's quantities (MSE(ln|M|^2) tree, MSE(V/B)
                     one-loop) and val MSE(ln) beside (stage2_readout.json).
Curves cached in data/stage2_curves.json (read on CC with hmap_readout.REMOTE; a pruned run keeps its log, so its curve).

    python analysis/data_saving/stage2_curves.py [--cached]
"""
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
from analysis.common.readers import site_python  # noqa: E402
from hmap_readout import REMOTE  # noqa: E402
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs/")
T = [("ee_uu", r"$e^+e^-\to u\bar u$", r"MSE$(\ln|\mathcal{M}|^2)$"),
     ("epem_ttx", r"$e^+e^-\to t\bar t$", r"MSE$(\ln|\mathcal{M}|^2)$"),
     ("ee_uu_nlo", r"$e^+e^-\to u\bar u$, $V$", r"MSE$(V/B)$"),
     ("ee_ttbar_nlo", r"$e^+e^-\to t\bar t$, $V$", r"MSE$(V/B)$")]


def main():
    cells = json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]
    cache = os.path.join(DATA, "stage2_curves.json")
    if "--cached" in sys.argv:
        rows = json.load(open(cache))
    else:
        rows = site_python(REMOTE, [f"runs/{c['sweep']}/trial_{c['best_hp']:04d}" for c in cells], "HMAP")
        json.dump(rows, open(cache, "w"))
    C = {}
    for r in rows:
        m = re.search(r"dsave_s2_(ft|scr)_(.+)_d(\d+)_t\d+/", r["run_dir"] + "/")
        if r.get("val_curve"):
            C[(m.group(1), m.group(2), int(m.group(3)))] = r["val_curve"][1:]

    for tg, lab, _ in T:      # one figure per target, its legend a strip above (18 curves do not fit inside)
        fig, ax = ps.figure()
        ks = sorted({k for (a, t, k) in C if t == tg})
        cols = dict(zip(ks, ps.sequence(len(ks))))
        for (a, t, k), v in sorted(C.items()):
            if t == tg:
                ax.plot(*zip(*v), color=cols[k], linestyle="-" if a == "scr" else "--")
        for k in ks:
            ax.plot([], [], color=cols[k], label=rf"$D=10^{{{k / 2:g}}}$")
        ax.plot([], [], color=ps.C.grey, linestyle="-", label="scratch")
        ax.plot([], [], color=ps.C.grey, linestyle="--", label="fine-tune")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("val HETEROSC loss")
        ps.process_label(ax, lab)
        ps.shared_legend(fig, ax, ncol=4)
        ps.save(fig, FIGS + f"stage2_curves_{tg}")

    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, (tg, lab, ylab) in zip(axs.ravel(), T):
        by = {(c["arm"], c["D"]): c for c in cells if c["target"] == tg}
        Ds = sorted(D for (a, D) in by if a == "ft" and ("scr", D) in by)
        for f, name, col, m in (("test", "test", ps.C.vermillion, "s"),
                                ("val_ln", "val", ps.C.blue, "o")):
            pts = [(D, by[("ft", D)][f] / by[("scr", D)][f]) for D in Ds if by[("ft", D)].get(f) and by[("scr", D)].get(f)]
            ax.plot(*zip(*pts), marker=m, color=col, label=name)
        ax.axhline(1, color=ps.C.grey, linestyle=":", label="equal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(r"MSE$_{\rm fine-tune}$ / MSE$_{\rm scratch}$")
        ps.process_label(ax, lab)
        ps.legend(ax, "upper left")
    ps.save(fig, FIGS + "stage2_ft_vs_scr")


if __name__ == "__main__":
    main()
