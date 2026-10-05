"""Transfer study, the whole fine-tune grid in one picture: for each of the twelve probes (rows) and each pretraining
(columns: ee->uu as rung 0, ladder rungs 1-9), the fine-tune's gain over scratch, the mean over the probe's D cells of
log10(L_scratch / L_fine-tune), each half-decade of D weighted the same (the rule of rung_focus.py). The 8k grid
(cells.USE_32K off): every pretraining at the same compute in every cell. The first rung that adds the probe's
structure (rung_focus.FOCUS) is outlined.
    python analysis/transfer/gain_heatmap.py      -> figs/gain_heatmap
"""
import io, contextlib, os, sys
import numpy as np
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, best, scratch  # noqa: E402
from rung_grid import LAB, P, fam  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from rung_focus import FOCUS  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

R = list(range(10))
G = np.full((len(P), len(R)), np.nan)
for i, p in enumerate(P):
    K = [k for k in range(2, 9) if scratch(p, k, steered=False)[0]]
    for j, r in enumerate(R):
        v = [best(f"{fam(r)}_{p}_d{k}")[0] for k in K]
        if all(v):
            G[i, j] = np.mean([np.log10(scratch(p, k, steered=False)[0] / x) for k, x in zip(K, v)])

fig, ax = ps.figure()
m = ax.pcolormesh(np.arange(len(R) + 1) - 0.5, np.arange(len(P) + 1) - 0.5, G, cmap="RdBu", vmin=-1.2, vmax=1.2)
for i, p in enumerate(P):
    ax.add_patch(Rectangle((FOCUS[p] - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.5,   # the outline is the marker
                           label="first rung with the probe's structure" if i == 0 else None))
ax.set_xticks(R, [str(r) for r in R])
ax.set_yticks(range(len(P)), [LAB[p] for p in P])
ax.set_ylim(len(P) - 0.5, -0.5)
ax.set_xlabel("pretraining (rung)")
ps.colorbar(ax, m, r"$\langle\log_{10}(L_{\rm scratch}/L_{\rm fine\text{-}tune})\rangle_D$")
ps.shared_legend(fig, ax, ncol=1)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "gain_heatmap"))
for i, p in enumerate(P):
    print(f"{p:12s} " + " ".join(f"{10 ** g:5.1f}" for g in G[i]))
