#!/usr/bin/env python
"""Steering uugg result: held-out deep-IR MSE(Δlog|M|^2) per y_min decade, sigma-driven vs base
(uniform) online generation, 3 seeds each. Left: per-decade MSE (mean±spread over seeds). Right:
sigma/base ratio. png + pdf."""
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = [r"$<\!-6$", r"$-6$", r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$"]


def load(arm):
    per = []
    for s in (0, 1, 2):
        d = np.load(f"{REPO}/analysis/divergences/heldout_eval_{arm}_s{s}.npz")
        e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2; ym = d["y_min"]
        per.append([e2[(ym >= lo) & (ym < hi)].mean() for lo, hi in DEC])
    return np.array(per)


b, g = load("base"), load("sigma")
x = np.arange(len(DEC))
fig, (axL, axR) = ps.figure(ncols=2)

# both arms are evaluated on the SAME y_min decades -> markers sit on the true decade center (no
# horizontal dodge, which on a physical x-axis would falsely suggest different y_min).
axL.errorbar(x, b.mean(0), yerr=b.std(0), fmt="o-", color=ps.C.grey, mfc="white",
             label="uniform")
axL.errorbar(x, g.mean(0), yerr=g.std(0), fmt="s-", color=ps.C.vermillion,
             label=r"$\sigma$-driven ($\gamma=1$)")
axL.set_yscale("log"); axL.set_xticks(x); axL.set_xticklabels(LBL)
axL.set_xlabel(r"$\log_{10} y_{\min}$")
axL.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
axL.legend(loc="lower left")
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$", loc="upper right")

ratio = g.mean(0) / b.mean(0)
rr = (g / b)  # per-seed ratio spread (pair seeds)
axR.axhline(1.0, color=ps.C.grey, lw=0.8, label="parity")
axR.errorbar(x, ratio, yerr=rr.std(0), fmt="D-", color=ps.C.vermillion,
             label=r"$\sigma$-driven / uniform")
axR.set_xticks(x); axR.set_xticklabels(LBL)
axR.set_xlabel(r"$\log_{10} y_{\min}$")
axR.set_ylabel(r"MSE ratio")
axR.legend(loc="upper left")

ps.save(fig, "analysis/divergences/figs/l2_uugg_perdecade")
print(f"log-flat overall: base {b.mean():.4e}  sigma {g.mean():.4e}  ratio {g.mean()/b.mean():.1%}")
