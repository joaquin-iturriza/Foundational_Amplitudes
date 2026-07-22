#!/usr/bin/env python
"""L2 uugg gamma-response: how the sigma-driven gain over uniform base scales with the concentration
exponent gamma in p(x) prop sigma(x)^gamma. Left: per-y_min-decade sigma/base MSE ratio, one line per
gamma (markers on the true decade center, no dodge). Right: the trade-off vs gamma -- deepest decade,
log-flat overall, and shallowest decade -- showing deep-IR gain grows while the bulk pays. png+pdf."""
import os
import numpy as np
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = ["<1e-6", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]


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
colors = ["#f6a300", "#e85d04", "#c1121f", "#6a040f", "#370617"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))

axL.axhline(1.0, color="k", lw=0.8, zorder=1)
for g, c in zip(gammas, colors):
    r = arms[g] / b
    axL.errorbar(x, r.mean(0), yerr=r.std(0), fmt="o-", color=c, capsize=2, label=rf"$\gamma={g}$")
axL.set_xticks(x); axL.set_xticklabels(LBL)
axL.set_xlabel(r"$y_{\min}$ decade")
axL.set_ylabel(r"$\sigma$-driven / base MSE")
axL.legend(ncol=2); axL.grid(alpha=0.3)

# right: three tracks vs gamma
gg = np.array(gammas)
deep = np.array([arms[g][:, 0].mean() / b[:, 0].mean() for g in gammas])       # deepest decade
shal = np.array([arms[g][:, -1].mean() / b[:, -1].mean() for g in gammas])      # shallowest decade
logf = np.array([arms[g].mean() / b.mean() for g in gammas])                    # log-flat overall
axR.axhline(1.0, color="k", lw=0.8)
axR.plot(gg, deep, "o-", color="#6a040f", label="deepest decade ($y_{\\min}<10^{-6}$)")
axR.plot(gg, logf, "s-", color="#c1121f", label="log-flat overall")
axR.plot(gg, shal, "^-", color="#8d99ae", label="shallowest decade ($y_{\\min}>0.1$)")
axR.set_xlabel(r"concentration exponent $\gamma$"); axR.set_ylabel(r"$\sigma$/base MSE ratio")
axR.set_xticks(gg); axR.legend(); axR.grid(alpha=0.3)
for xi, yi in zip(gg, deep):
    axR.annotate(f"{yi:.0%}", (xi, yi), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

fig.suptitle(r"$ee\to u\bar u gg$", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
base = os.path.join(REPO, "analysis/divergences/figs/l2_uugg_gamma")
os.makedirs(os.path.dirname(base), exist_ok=True)
fig.savefig(base + ".png", dpi=140); fig.savefig(base + ".pdf")
print(f"saved {base}.png / .pdf")
for g in gammas:
    print(f"  gamma={g}: log-flat {arms[g].mean()/b.mean():.1%}, deepest {arms[g][:,0].mean()/b[:,0].mean():.1%}")
