"""The catalog's coverage check, drawn (analysis/catalog_v3/coverage_check.py; docs/results.tex, catalog_v3, the
catalog's coverage). Three figures from data/coverage_rows.json and the summary data/coverage_check.json:
  coverage_range_{a,b}  per class family (hadronic and e+e- beams, 2->2, 2->3, 2->4), the 0.1% and 99.9% quantiles of
                        log10|M|^2 under the current cuts, the proposed coverage and the wide window, each relative to
                        the class's current-sample median; medians over the family's classes
  coverage_pp_{a,b}     the hadronic classes' 13 TeV pp events: (a) the share inside the current cuts and window, the
                        proposed coverage and window, the proposed coverage up to 6 TeV; (b) the effective sample size
                        per event of the pp events reweighted from each sqrt(s_hat) measure; medians per family
  coverage_fwd_{a..d}   the forward angle u = log(1/(1-|cos theta|)) of the most forward final particle, the classes'
                        normalised histograms averaged per family, for the pp events and the three samples; (d) the
                        e+e- classes (no pp pool)
    python3 analysis/catalog_v3/coverage_check.py && python3 analysis/catalog_v3/coverage_fig.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

rows = json.load(open(os.path.join(HERE, "data", "coverage_rows.json")))
S = json.load(open(os.path.join(HERE, "data", "coverage_check.json")))
FIG = os.path.join(HERE, "figs")
KEYS = [("current", "current cuts", ps.C.blue), ("proposed", "proposed coverage", ps.C.vermillion),
        ("wide", r"proposed, $\sqrt{\hat s}$ to 6 TeV", ps.C.green)]
NF = (2, 3, 4)
U_EDGES = np.linspace(0.0, 10.0, 21)
ok = [r for r in rows.values() if "error" not in r and r.get("type", "tree") == "tree"]   # the tree classes

# ---- log|M|^2 range per family
figs = ps.panels(2)
for (fig, ax), kd, lab in zip(figs, ("hadronic", "ee"), ("hadronic beams", r"$e^+e^-$ beams")):
    ends = []
    for j, (k, name, col) in enumerate(KEYS):
        lo, hi = [], []
        for nf in NF:
            g = [r for r in ok if r["kind"] == kd and r["nfinal"] == nf]
            lo.append(np.median([r[k]["q"][0] - r["current"]["q"][2] for r in g]))
            hi.append(np.median([r[k]["q"][-1] - r["current"]["q"][2] for r in g]))
        x = np.arange(len(NF)) + (j - 1) * 0.26
        ax.bar(x, np.array(hi) - np.array(lo), 0.24, bottom=lo, color=col, label=name)
        ends += lo + hi
    ax.set_ylim(min(ends) - 0.5, max(ends) + 0.5)       # floating bars: autoscale does not reach their bottoms
    ax.set_xticks(np.arange(len(NF)))
    ax.set_xticklabels([rf"$2\to{n}$" for n in NF])
    ax.set_xlabel(lab)
    ax.set_ylabel(r"$\log_{10}|\mathcal{M}|^2$, $0.1$ to $99.9\%$")
ps.legend(figs[1][1], "upper left")
ps.make_room(figs[1][1])
ps.save_panels(figs, os.path.join(FIG, "coverage_range"))

# ---- pp events: coverage shares and sqrt(s_hat) measures
fams = [f"hadronic_2to{n}" for n in NF]
figs = ps.panels(2)
ax = figs[0][1]
for j, (f, name, col) in enumerate([("pp_in_cur_window", "current cuts, window", ps.C.blue),
                                     ("pp_in_new_window", "proposed, window", ps.C.vermillion),
                                     ("pp_in_new_wide", r"proposed, to 6 TeV", ps.C.green)]):
    ax.bar(np.arange(len(NF)) + (j - 1) * 0.26, [S[k][f] for k in fams], 0.24, color=col, label=name)
ax.set_xticks(np.arange(len(NF)))
ax.set_xticklabels([rf"$2\to{n}$" for n in NF])
ax.set_ylabel("share of pp events covered")
ax.set_xlabel("hadronic classes")
ax.set_ylim(0, 1)
ps.legend(ax, "upper left")
ps.make_room(ax)
ax = figs[1][1]
for j, (m, name, col) in enumerate([("uniform_window", "uniform, window", ps.C.blue),
                                     ("uniform_wide", "uniform, to 6 TeV", ps.C.orange),
                                     ("log_wide", "log uniform, to 6 TeV", ps.C.green),
                                     ("lumi_wide", r"$\hat s^{-3/2}$, to 6 TeV", ps.C.purple)]):
    ax.bar(np.arange(len(NF)) + (j - 1.5) * 0.2, [S[k][f"pp_ess_{m}"] for k in fams], 0.19, color=col, label=name)
ax.set_xticks(np.arange(len(NF)))
ax.set_xticklabels([rf"$2\to{n}$" for n in NF])
ax.set_ylabel(r"$(\sum w)^2/(n\sum w^2)$")
ax.set_xlabel("hadronic classes")
ax.set_ylim(0, 1)
ps.legend(ax, "upper left", ncol=2)
ps.make_room(ax)
ps.save_panels(figs, os.path.join(FIG, "coverage_pp"))

# ---- forward angle
c = 0.5 * (U_EDGES[1:] + U_EDGES[:-1])
dens = lambda h: np.asarray(h, float) / max(sum(h), 1) / np.diff(U_EDGES)
figs = ps.panels(4)
sets = [("hadronic", n, rf"hadronic, $2\to{n}$") for n in NF] + [("ee", None, r"$e^+e^-$, all")]
for (fig, ax), (kd, nf, lab) in zip(figs, sets):
    g = [r for r in ok if r["kind"] == kd and (nf is None or r["nfinal"] == nf)]
    if kd == "hadronic":
        ax.step(c, np.mean([dens(r["pp"]["u_hist"]) for r in g], 0), where="mid", color=ps.C.grey, label="13 TeV pp")
    for k, name, col in KEYS:
        ax.step(c, np.mean([dens(r[k]["u_hist"]) for r in g], 0), where="mid", color=col, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log\,1/(1-|\cos\theta|)$, most forward final")
    ax.set_ylabel("density")
    ps.process_label(ax, lab)
h, l = figs[0][1].get_legend_handles_labels()        # the pp line is drawn on the hadronic panels only
ps.legend(figs[3][1], "lower left", handles=h, labels=l)
ps.make_room(figs[3][1])
ps.save_panels(figs, os.path.join(FIG, "coverage_fwd"))
