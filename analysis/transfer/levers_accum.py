"""Transfer study, the conditionings re-judged on fine-tuning (B) and Daniel Schiller's per-process gradient
accumulation (C): each arm is rung 9's pretraining with one change (hp73, 64k steps), fine-tuned on the twelve probes
at D = 10^2 and 10^3 (8k grid) by a 5-trial search that shares its DyHPO seed, hence its candidate pool, with rung 9's
search on the same cell. Per cell: log10(L_rung9 / L_arm), loss = MSE of log|M|^2 at the best checkpoint; > 0 means the
arm fine-tunes better. One panel per arm; a cell is drawn once both searches have their 5 trials.
    python analysis/transfer/levers_accum.py    -> figs/levers_accum_a..e, _legend; per-arm tests on stdout
"""
import io, contextlib, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import LAB, P, cell  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

ARMS = [("tp3_levdiagfte", "Feynman diagrams on"), ("tp3_levoffshfte", "off-shellness off"),
        ("tp3_levcoupfte", "coupling scalars off"), ("tp3_levgenfte", "generation feature off"),
        ("tp3_accumfte", "per-process accumulation")]
DS = [(4, r"$D=10^2$", ps.C.blue, "o"), (6, r"$D=10^3$", ps.C.vermillion, "s")]
base = os.path.join(ROOT, "analysis", "transfer", "figs", "levers_accum")
figs = ps.panels(len(ARMS))
for (fig, ax), (fam, lab) in zip(figs, ARMS):
    allv = []
    for k, dlab, col, m in DS:
        xs, ys = [], []
        for i, p in enumerate(P):
            a, r = cell(fam, p, k), cell("tp3_r9fte", p, k)
            if a and r:
                xs.append(np.log10(r / a)); ys.append(i)
        allv += xs
        ax.scatter(xs, ys, color=col, marker=m, label=dlab)
    ax.axvline(0, color="0.5", ls="--", label="rung 9")
    ax.set_yticks(range(len(P))); ax.set_yticklabels([LAB[p] for p in P]); ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{10}(L_\mathrm{rung\,9}/L_\mathrm{arm})$")
    ps.process_label(ax, lab)
    v = np.array(allv)
    if len(v):
        print(f"{lab:26s} {len(v):2d} cells  median {np.median(v):+.2f} dex  arm better in {np.mean(v > 0):.0%}"
              f"  Wilcoxon p={wilcoxon(v).pvalue:.3g}")
ps.legend_strip(figs[0][1], base + "_legend", ncol=3)
ps.save_panels(figs, base)
