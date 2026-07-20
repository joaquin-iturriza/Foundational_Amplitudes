#!/usr/bin/env python
"""L2 uugg result: held-out deep-IR MSE(Δlog|M|^2) per y_min decade, sigma-driven vs base (uniform)
online generation, 3 seeds each. Left: per-decade MSE (mean±spread over seeds). Right: sigma/base
ratio (<1 = sigma better) showing the gain concentrates in the deep IR. png + pdf."""
import os
import numpy as np
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = ["<1e-6", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]


def load(arm):
    per = []
    for s in (0, 1, 2):
        d = np.load(f"{REPO}/analysis/divergences/heldout_eval_{arm}_s{s}.npz")
        e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2; ym = d["y_min"]
        per.append([e2[(ym >= lo) & (ym < hi)].mean() for lo, hi in DEC])
    return np.array(per)


b, g = load("base"), load("sigma")
x = np.arange(len(DEC))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

axL.errorbar(x - 0.06, b.mean(0), yerr=b.std(0), fmt="o-", color="#444", capsize=3, label="base (uniform)")
axL.errorbar(x + 0.06, g.mean(0), yerr=g.std(0), fmt="s-", color="#c1121f", capsize=3, label=r"$\sigma$-driven ($\gamma$=1)")
axL.set_yscale("log"); axL.set_xticks(x); axL.set_xticklabels(LBL)
axL.set_xlabel(r"$y_{\min}$ decade (deep IR $\to$ bulk)"); axL.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
axL.set_title("Held-out deep-IR error per IR decade"); axL.legend(); axL.grid(alpha=0.3)

ratio = g.mean(0) / b.mean(0)
# per-seed ratio spread (pair seeds)
rr = (g / b)
axR.axhline(1.0, color="k", lw=0.8)
axR.errorbar(x, ratio, yerr=rr.std(0), fmt="D-", color="#c1121f", capsize=3)
axR.fill_between(x, 1.0, ratio, where=ratio < 1, color="#c1121f", alpha=0.15)
axR.set_xticks(x); axR.set_xticklabels(LBL)
axR.set_xlabel(r"$y_{\min}$ decade (deep IR $\to$ bulk)"); axR.set_ylabel(r"$\sigma$-driven / base MSE")
axR.set_title(r"$\sigma$ gain concentrates in the deep IR ($<1$ = better)"); axR.grid(alpha=0.3)
for xi, ri in zip(x, ratio):
    axR.annotate(f"{ri:.0%}", (xi, ri), textcoords="offset points", xytext=(0, 6 if ri < 1 else -12),
                 ha="center", fontsize=8)

fig.suptitle(r"L2: coordinate-free $p(x)\propto\sigma(x)^\gamma$ online generation, $ee\to u\bar u gg$ "
             "(3 seeds, matched $\\mu$/budget)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
base = os.path.join(REPO, "analysis/divergences/figs/l2_uugg_perdecade")
os.makedirs(os.path.dirname(base), exist_ok=True)
fig.savefig(base + ".png", dpi=140); fig.savefig(base + ".pdf")
print(f"saved {base}.png / .pdf")
print(f"log-flat overall: base {b.mean():.4e}  sigma {g.mean():.4e}  ratio {g.mean()/b.mean():.1%}")
