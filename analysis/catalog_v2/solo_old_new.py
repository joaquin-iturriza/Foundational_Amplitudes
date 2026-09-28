"""The old from-scratch solo set (tab:scaling; sweeps/scaling_solo_full on Jean Zay, solo_full_old.json) and
the catalog solo references at batch 1024 (solo_b1k) and 16384 (solo16k.json) on one compute axis, for the
four processes they share: ee->aa, ee->uu, ee->uug, ee->uugg. Each cell at its best trial's val_loss (the
DyHPO result; CLAUDE.md, Reported values); each curve with its floor-aware fit A C^-alpha + L_inf (CLAUDE.md,
Scaling fits). Compute = flops_per_step(heads, n_p, batch) x steps with each set's own heads (4 old, 8 new)
and batch (16384 old, 1024 / 16384 new).
    python analysis/catalog_v2/solo_old_new.py
Writes solo_old_new (2x2, one panel per process, png + pdf)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from solo_b1k import solo_mse
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
OLD = json.load(open(os.path.join(HERE, "solo_full_old.json")))
S16 = json.load(open(os.path.join(HERE, "solo16k.json"))); B1K = solo_mse()
SETS = [  # label, heads, batch, steps, value(p, S), marker, filled, colour
    ("old set (tab:scaling): uniform $\\sqrt{s}$, 70k, 4 heads", 4, 16384, [4, 126, 400, 1265, 4000], lambda p, S: min(OLD[f"{p}|{S}"]), "D", True, ps.C.grey),
    ("catalog pool, bs 1024", 8, 1024, [33, 67, 134, 268, 536, 1072], lambda p, S: B1K[f"{p}|{S}"], "s", False, ps.C.blue),
    ("catalog pool, bs 16384", 8, 16384, [63, 125, 250, 500, 1000, 2000, 4000], lambda p, S: min(S16[f"{p}|{S}"]), "o", True, ps.C.vermillion)]
PROCS = ["ee_aa", "ee_uu", "ee_uug", "ee_uugg"]
fig, axes = ps.figure(ncols=2, nrows=2)
for ax, p in zip(np.ravel(axes), PROCS):
    for lab, h, bs, T, val, mk, filled, col in SETS:
        x = np.array([flops_per_step(h, NP[p], bs) * S for S in T]); y = np.array([val(p, S) for S in T])
        ax.plot(x, y, marker=mk, ls="none", color=col, mfc=col if filled else "none", label=lab)
        f = fit_power_law_with_floor(x, y)
        if f:
            g = np.geomspace(x[0] / 1.3, x[-1] * 1.3, 100)
            ax.plot(g, f[0] * g ** -f[1] + f[2], color=col, ls="--")
            print(f"{p:8s} {lab[:22]:22s} alpha {f[1]:.2f}  L_inf {f[2]:.2g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"compute $C$ [FLOP]"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL.get(p, p), loc="lower left")
H, L = np.ravel(axes)[0].get_legend_handles_labels()
H.append(np.ravel(axes)[0].plot([], [], color="black", ls="--")[0]); L.append(r"fit $A\,C^{-\alpha}+L_\infty$")
ps.shared_legend(fig, np.ravel(axes)[0], ncol=2, handles=H, labels=L)
ps.save(fig, os.path.join(HERE, "solo_old_new"))
