"""Transfer study, distance from the pretraining: each pretraining's zero-shot loss on each probe, up to an affine map
(tools/zero_shot_ft.py: the parent's weights, before any fine-tune step, on the probe's validation split; its output h
fitted to the probe's standardized target z by the best a + b h, so L0* = min_ab E[(a + b h - z)^2] = 1 - corr(h, z)^2,
0 when the frozen network already predicts the probe up to units, 1 when its output carries no linear information about
it), against the fine-tune's gain over scratch on that probe (the gain_heatmap value: mean over the probe's D cells of
log10(L_scratch / L_fine-tune), 8k grid). One point per (pretraining, probe), coloured by probe. The raw zero-shot loss
(val_loss) is not used: it mixes the parent's output units with the probe's (docs/results.tex sec:ladder).
    python analysis/transfer/zero_shot_gain.py      -> figs/zero_shot_gain
"""
import io, contextlib, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT  # noqa: E402
from rung_grid import LAB, P  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from gain_heatmap import G, R  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

Z = json.load(open(os.path.join(ROOT, "analysis", "transfer", "zero_shot.json")))
par = lambda r: "uu64" if r == 0 else f"r{r}"
X = np.full_like(G, np.nan)
for i, p in enumerate(P):
    for j, r in enumerate(R):
        d = Z.get(f"tp3_{par(r)}fte_{p}_d8")
        if d:
            X[i, j] = d["val_loss_affine"]

fig, ax = ps.figure()
cols = ps.sequence(len(P), "tab20", 0.0, 1.0)
for i, p in enumerate(P):
    ax.scatter(X[i], G[i], color=cols[i], marker="osD^v<>ph*XP"[i], label=LAB[p])   # marker + colour: twelve probes
ax.set_xscale("log")
ax.set_xlabel(r"$\min_{a,b}\,\langle(a+b\,h-z)^2\rangle = 1-\rho(h,z)^2$")
ax.set_ylabel(r"$\langle\log_{10}(L_{\rm scratch}/L_{\rm fine\text{-}tune})\rangle_D$")
base = os.path.join(ROOT, "analysis", "transfer", "figs", "zero_shot_gain")
ps.legend_strip(ax, base + "_legend", ncol=4)
ps.save(fig, base)
ok = np.isfinite(X) & np.isfinite(G)
print("all cells: Spearman", spearmanr(np.log(X[ok]), G[ok]))
for i, p in enumerate(P):
    o = np.isfinite(X[i]) & np.isfinite(G[i])
    rho = spearmanr(X[i][o], G[i][o])[0]
    print(f"{p:12s} zs(rung0)={X[i,0]:.3g} zs range {np.nanmin(X[i]):.3g}-{np.nanmax(X[i]):.3g}  within-probe rho={rho:+.2f}")
