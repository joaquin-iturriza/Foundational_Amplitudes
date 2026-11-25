"""llm_compare D566: per family of the 43 new held-out processes, the median level error and the median sampling shift
of four ways to predict log|M|^2 (data/perevent_level.json, perevent_level.py's CC job): one level shared by all
processes, the per-process level from the diagram descriptors (constant per process), the diagrams' propagator factor
per event plus a level from particle content, and the per-event regressor on process, diagrams and the event.
    python analysis/catalog_daniel/perevent_level_fig.py
Output: figs/perevent_level.{png,pdf}.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

SHORT = {"Multi-Higgs production (hh, hhh, Xhh)": "$HH(X)$", "Higgs + tau lepton processes": r"$H\tau$",
         "Triple-Z production (ZZZ)": "$ZZZ$", "Triple-W production (W+W+W-, W+W-W-)": "$WWW$",
         "gg -> W + qq' (W production from gluon fusion)": "$Wqq'$", "gg -> tt~ + boson (ttX via gluon fusion)":
         r"$t\bar tX$", "Single-particle production (2 -> 1 resonances)": r"$2\to1$"}
ARMS = (("shared", "one shared level", ps.C.grey, "o"), ("regressor", "level from process\nand diagrams", ps.C.blue, "s"),
        ("A", "propagator factor\nper event", ps.C.green, "v"), ("B", "per-event map,\nprocess and diagrams",
                                                                 ps.C.vermillion, "D"))


def main():
    d = json.load(open(os.path.join(HERE, "data", "perevent_level.json")))["summary"]
    fams = sorted(d["shared"]["families"], key=lambda g: -d["regressor"]["families"][g]["e"])
    x = np.arange(len(fams))
    fig, (axE, axS) = ps.figure(ncols=2)
    for k, (c, lab, col, mk) in enumerate(ARMS):
        off = (k - 1.5) * 0.15
        axE.plot(x + off, [d[c]["families"][g]["e"] for g in fams], mk, color=col, label=lab)
        axS.plot(x + off, [d[c]["families"][g]["shift_flat"] for g in fams], mk, color=col, label=lab)
    for ax in (axE, axS):
        ax.set_xticks(x, [SHORT.get(g, g) for g in fams])
        ax.set_xlabel("new held-out family")
    axE.set_yscale("log")
    axE.set_ylabel(r"level error $\langle\hat y-y\rangle_p^2/\sigma_p^2$")
    axS.set_yscale("log")
    axS.set_ylabel(r"shift $|\langle\hat y-y\rangle_{p,{\rm flat}}-\langle\hat y-y\rangle_p|/\sigma_p$")
    ps.legend(axE, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "perevent_level"))


if __name__ == "__main__":
    main()
