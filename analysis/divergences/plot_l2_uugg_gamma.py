#!/usr/bin/env python
"""Steering uugg gamma-response: how the sigma-driven gain over uniform base scales with the
concentration exponent gamma in p(x) prop sigma(x)^gamma. Left: per-y_min-decade sigma/base MSE
ratio, one line per gamma (markers on the true decade center, no dodge). Right: the trade-off vs
gamma -- deepest decade, log-flat overall, and shallowest decade. png+pdf."""
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = [r"$<\!-6$", r"$-6$", r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$"]


def load(tag):
    per = []
    for s in (0, 1, 2):
        d = np.load(f"{REPO}/analysis/divergences/heldout_eval_{tag}_s{s}.npz")
        e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2; ym = d["y_min"]
        per.append([e2[(ym >= lo) & (ym < hi)].mean() for lo, hi in DEC])
    return np.array(per)


b = load("base")
gammas = [1, 2, 3, 5, 10]
tags = {1: "sigma", 2: "g2", 3: "g3", 5: "g5", 10: "g10"}
arms = {g: load(tags[g]) for g in gammas}
x = np.arange(len(DEC))
ramp = ps.sequence(len(gammas))

fig, (axL, axR) = ps.figure(ncols=2)

axL.axhline(1.0, color=ps.C.grey, lw=0.8, zorder=1, label="parity")
for g, c in zip(gammas, ramp):
    r = arms[g] / b
    axL.errorbar(x, r.mean(0), yerr=r.std(0), fmt="o-", color=c, label=rf"$\gamma={g}$")
axL.set_xticks(x); axL.set_xticklabels(LBL)
axL.set_xlabel(r"$\log_{10} y_{\min}$")
axL.set_ylabel(r"MSE ratio")
ps.legend(axL, "upper left", ncol=2)
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$", loc="lower right")

# right: three tracks vs gamma
gg = np.array(gammas)
deep = np.array([arms[g][:, 0].mean() / b[:, 0].mean() for g in gammas])       # deepest decade
shal = np.array([arms[g][:, -1].mean() / b[:, -1].mean() for g in gammas])     # shallowest decade
logf = np.array([arms[g].mean() / b.mean() for g in gammas])                   # log-flat overall
axR.axhline(1.0, color=ps.C.grey, lw=0.8, label="parity")
axR.plot(gg, deep, "o-", color=ps.C.blue, label=r"$\log_{10} y_{\min}<-6$")
axR.plot(gg, logf, "s-", color=ps.C.vermillion, label="log-flat over decades")
axR.plot(gg, shal, "^-", color=ps.C.green, label=r"$\log_{10} y_{\min}>-1$")
axR.set_xlabel(r"$\gamma$"); axR.set_ylabel(r"MSE ratio")
axR.set_xticks(gg)
# A corner, not "center right": a legend floating in the middle of the plot with data on
# both sides of it reads as a mistake. make_room opens the top if the curves reach into it.
ps.legend(axR, "upper left")

ps.save(fig, "analysis/divergences/figs/l2_uugg_gamma")
for g in gammas:
    print(f"  gamma={g}: log-flat {arms[g].mean()/b.mean():.1%}, deepest {arms[g][:,0].mean()/b[:,0].mean():.1%}")
