#!/usr/bin/env python
"""The sigma-steering figures of docs/results.tex sec:div-sampling, drawn from analysis/divergences/steering_eval_summary.json
(per held-out evaluation: event count and MSE of log|M|^2 per y_min decade, per y_min band, per sqrt(s) region; computed
on Jean Zay by steering_numbers.py --fetch, where the heldout_eval_*.npz live). The same figures as plot_l2_uugg.py,
plot_l2_uugg_gamma.py, plot_gamma_response.py, plot_l2_compare.py (uug, uuggg) and plot_l2_bbbarg.py, which need the npz
files; the statistics are the ones those scripts compute (steering_numbers.py REMOTE), so the figures run locally.

    python3 analysis/divergences/plot_steering.py      # writes figs/l2_{uugg_perdecade,uugg_gamma,gamma_response_extended,
                                                        #   uug_multiscale,uuggg_perdecade,bbbarg}.{png,pdf}, prints the plotted ratios
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

F = json.load(open(os.path.join(HERE, "steering_eval_summary.json")))["files"]
OUT = os.path.join(HERE, "figs")
LBL = [r"$<\!-6$", r"$-6$", r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$"]   # lower edge of each y_min decade
X = np.arange(len(LBL))
YL = r"MSE$(\log|\mathcal{M}|^2)$"
YR = r"MSE$_\mathrm{steered}$ / MSE$_\mathrm{uniform}$"   # sigma-steered arm over uniform keep
REG = [("peak", r"$<3$"), ("shoulder", r"$3\!-\!15$"), ("continuum", r"$>15$"), ("peak_ir", r"$<3$, IR")]
printed = {}


def dec(tag, nmin=1):
    """Per-decade MSE of one evaluation; NaN where the decade holds fewer than nmin events."""
    return np.array([d[1] if d[0] >= nmin else np.nan for d in F[tag]["dec"]])


def seeds(tag, nmin=1):
    return np.array([dec(f"{tag}_s{s}", nmin) for s in (0, 1, 2)])


def decade_axes(ax, ylabel=YL):
    ax.set_xticks(X); ax.set_xticklabels(LBL)
    ax.set_xlabel(r"$\log_{10} y_{\min}$"); ax.set_ylabel(ylabel)


# ---- ee -> u ubar gg, gamma = 1 against uniform, three seeds
b, g = seeds("base"), seeds("sigma")
fig, (axL, axR) = ps.figure(ncols=2)
axL.errorbar(X, b.mean(0), yerr=b.std(0), fmt="o-", color=ps.C.grey, mfc="white", label="uniform")
axL.errorbar(X, g.mean(0), yerr=g.std(0), fmt="s-", color=ps.C.vermillion, label="steered")
axL.set_yscale("log"); decade_axes(axL)
ps.legend(axL, "lower left")
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$")
axR.axhline(1.0, color=ps.C.grey, ls=":", label="parity")
axR.errorbar(X, g.mean(0) / b.mean(0), yerr=(g / b).std(0), fmt="D-", color=ps.C.vermillion, label=r"$\gamma=1$")
decade_axes(axR, YR)
ps.legend(axR, "upper left")
ps.save(fig, os.path.join(OUT, "l2_uugg_perdecade"))
printed["uugg_g1_ratio_per_decade"] = (g.mean(0) / b.mean(0)).round(3).tolist()

# ---- ee -> u ubar gg, ratio per decade for each gamma, three seeds
gammas = {1: "sigma", 2: "g2", 3: "g3", 5: "g5", 10: "g10"}
arms = {k: seeds(t) for k, t in gammas.items()}
fig, (axL, axR) = ps.figure(ncols=2)
axL.axhline(1.0, color=ps.C.grey, ls=":", zorder=1, label="parity")
for (k, a), c in zip(arms.items(), ps.sequence(len(arms))):
    axL.errorbar(X, (a / b).mean(0), yerr=(a / b).std(0), fmt="o-", color=c, label=rf"$\gamma={k}$")
decade_axes(axL, YR)
ps.legend(axL, "upper left", ncol=2)
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$", loc="lower right")
gg = np.array(list(arms))
deep = np.array([a[:, 0].mean() / b[:, 0].mean() for a in arms.values()])
shal = np.array([a[:, -1].mean() / b[:, -1].mean() for a in arms.values()])
logf = np.array([a.mean() / b.mean() for a in arms.values()])
axR.axhline(1.0, color=ps.C.grey, ls=":", label="parity")
axR.plot(gg, deep, "o-", color=ps.C.blue, label=r"$\log_{10} y_{\min}<-6$")
axR.plot(gg, logf, "s-", color=ps.C.vermillion, label="log-flat")
axR.plot(gg, shal, "^-", color=ps.C.green, label=r"$\log_{10} y_{\min}>-1$")
axR.set_xlabel(r"$\gamma$"); axR.set_ylabel(YR); axR.set_xticks(gg)
ps.legend(axR, "upper left")
ps.save(fig, os.path.join(OUT, "l2_uugg_gamma"))
printed["uugg_gamma"] = dict(gamma=gg.tolist(), deepest=deep.round(3).tolist(), logflat=logf.round(3).tolist(),
                             shallowest=shal.round(3).tolist())

# ---- gamma response, seed 0: u ubar gg (to 30) and u ubar ggg (to 20), ratio sigma arm over uniform per y_min band
TRACKS = [("lt1e-3", r"$\log_{10} y_{\min}<-3$", ps.C.blue, "o"),
          ("lt1e-6", r"$\log_{10} y_{\min}<-6$", ps.C.vermillion, "s"),
          ("gt1e-1", r"$\log_{10} y_{\min}>-1$", ps.C.green, "^")]
PROCS = [(r"$e^+e^-\to u\bar u gg$", "base_s0", [(1, "sigma_s0"), (2, "g2_s0"), (3, "g3_s0"), (5, "g5_s0"),
                                                 (10, "g10_s0"), (20, "uugg_g20_s0"), (30, "uugg_g30_s0")]),
         (r"$e^+e^-\to u\bar u ggg$", "uuggg_base_s0", [(1, "uuggg_sigma_s0"), (3, "uuggg_g3_s0"),
                                                        (10, "uuggg_g10_s0"), (20, "uuggg_g20_s0")])]
fig, axes = ps.figure(ncols=2, sharey=True)
for ax, (proc, bt, rows) in zip(axes, PROCS):
    ax.axhline(1.0, color=ps.C.grey, ls=":", label="parity")
    gx = np.array([k for k, _ in rows])
    for key, lab, c, m in TRACKS:
        if F[bt]["bands"][key][0] == 0:   # u ubar ggg has no event below 1e-6
            continue
        r = np.array([F[t]["bands"][key][1] / F[bt]["bands"][key][1] for _, t in rows])
        ax.plot(gx, r, m + "-", color=c, label=lab)
        printed[f"gamma_resp {proc} {key}"] = dict(zip(gx.tolist(), r.round(3).tolist()))
    ax.set_xscale("log"); ax.set_xticks(gx); ax.set_xticklabels([str(k) for k in gx]); ax.minorticks_off()
    ax.set_xlabel(r"$\gamma$")
    ps.process_label(ax, proc, loc="upper left")
axes[0].set_ylabel(YR)
ps.legend(axes[1], "lower left", handles=axes[0].get_legend_handles_labels()[0])
ps.save(fig, os.path.join(OUT, "l2_gamma_response_extended"))

# ---- ee -> u ubar g, seed 0: per decade and per sqrt(s) region (|sqrt s - M_Z| in GeV; IR = y_min < 1e-3)
# labels in words (Joaquin 2026-11-17): the arms say they are heteroscedastic steering against a uniform keep, y_min is
# defined on its axis, the regions read in order (the peak's IR subset next to the peak)
STEER = "heteroscedastic steering, kept\n" + r"in proportion to $\sigma^\gamma$, "
UUG = [("proposals kept uniformly", "uug_base_s0", ps.C.grey, "o"), (STEER + r"$\gamma=1$", "uug_sigma_s0", ps.C.sky, "s"),
       (STEER + r"$\gamma=3$", "uug_g3_s0", ps.C.vermillion, "D")]
REG_UUG = [("peak", r"$<3$"), ("peak_ir", r"$<3$, IR"), ("shoulder", r"$3\!-\!15$"),
           ("continuum", r"$>15$")]
fig, (axL, axR) = ps.figure(ncols=2)
xr = np.arange(len(REG_UUG))
# each arm its own marker, fill and line style as well as colour, the points of a window spread apart (Joaquin
# 2026-11-17: the three were hard to tell apart)
LS, FILL = ("-", "--", "-."), ("white", None, None)
for j, (lab, t, c, m) in enumerate(UUG):
    axL.plot(X, dec(t), marker=m, ls=LS[j], color=c, mfc=FILL[j] or c, label=lab)
    axR.plot(xr + (j - 1) * 0.25, [F[t]["reg"][k][1] for k, _ in REG_UUG], m, color=c, mfc=FILL[j] or c, label=lab)
axL.set_yscale("log"); decade_axes(axL)
axL.set_xlabel(r"$\log_{10} y_{\min}$,  $y_{\min}=\min_{i<j} s_{ij}/\hat s$")
ps.legend(axR, "lower left")
ps.process_label(axL, r"$e^+e^-\to u\bar u g$")
axR.set_yscale("log"); axR.set_xticks(xr); axR.set_xticklabels([l for _, l in REG_UUG])
axR.set_xlabel(r"$|\sqrt{\hat s}-M_Z|$ [GeV];  IR: $y_{\min}<10^{-3}$"); axR.set_ylabel(YL)   # one-line ticks (Joaquin 2026-11-17)
ps.save(fig, os.path.join(OUT, "l2_uug_multiscale"))
printed["uug_region_over_base_g3"] = {k: round(F["uug_g3_s0"]["reg"][k][1] / F["uug_base_s0"]["reg"][k][1], 3) for k, _ in REG}
printed["uug_overall"] = [F[t]["mse"] for _, t, _, _ in UUG]

# ---- ee -> u ubar ggg, seed 0, per decade
fig, ax = ps.figure()
for lab, t, c, m in [("uniform", "uuggg_base_s0", ps.C.grey, "o"), (r"$\gamma=1$", "uuggg_sigma_s0", ps.C.sky, "s"),
                     (r"$\gamma=3$", "uuggg_g3_s0", ps.C.vermillion, "D")]:
    ax.plot(X, dec(t), m + "-", color=c, label=lab)
    printed[f"uuggg {t}"] = dict(mse=round(F[t]["mse"], 5), dec=np.round(dec(t), 4).tolist())
ax.set_yscale("log"); decade_axes(ax)
ps.legend(ax, "upper left")
ps.process_label(ax, r"$e^+e^-\to u\bar u ggg$")
ps.save(fig, os.path.join(OUT, "l2_uuggg_perdecade"))

# ---- ee -> b bbar g, gamma = 3 against uniform, three seeds: per decade (>= 20 events) and per sqrt(s) region
bb, bs = seeds("bbbarg_base", 20), seeds("bbbarg_g3", 20)
rb = np.array([[F[f"bbbarg_base_s{s}"]["reg"][k][1] for k, _ in REG[:3]] for s in (0, 1, 2)])
rs = np.array([[F[f"bbbarg_g3_s{s}"]["reg"][k][1] for k, _ in REG[:3]] for s in (0, 1, 2)])
fig, (axL, axR) = ps.figure(ncols=2)
axL.errorbar(X, bb.mean(0), yerr=bb.std(0), fmt="o-", color=ps.C.grey, mfc="white", label="uniform")
axL.errorbar(X, bs.mean(0), yerr=bs.std(0), fmt="D-", color=ps.C.vermillion, label=r"$\gamma=3$")
axL.set_yscale("log"); decade_axes(axL)
axL.set_xlabel(r"$\log_{10} y_{\min}$, $y_{ij}=2p_i\!\cdot\!p_j/s$")
ps.legend(axL, "lower left")
ps.process_label(axL, r"$e^+e^-\to b\bar b g$")
xr = np.arange(3)
axR.errorbar(xr - 0.1, rb.mean(0), yerr=rb.std(0), fmt="o", color=ps.C.grey, mfc="white", label="uniform")
axR.errorbar(xr + 0.1, rs.mean(0), yerr=rs.std(0), fmt="D", color=ps.C.vermillion, label=r"$\gamma=3$")
axR.set_yscale("log"); axR.set_xticks(xr); axR.set_xticklabels([l for _, l in REG[:3]])
axR.set_xlabel(r"$|\sqrt{s}-M_Z|$ [GeV]"); axR.set_ylabel(YL)
ps.save(fig, os.path.join(OUT, "l2_bbbarg"))
printed["bbbarg_region_ratio_seedmean"] = dict(zip([k for k, _ in REG[:3]], (rs.mean(0) / rb.mean(0)).round(3).tolist()))
printed["bbbarg_region_ratio_per_seed"] = (rs / rb).round(3).tolist()
printed["bbbarg_overall_per_seed"] = [round(F[f"bbbarg_g3_s{s}"]["mse"] / F[f"bbbarg_base_s{s}"]["mse"], 3) for s in (0, 1, 2)]

print(json.dumps(printed))
