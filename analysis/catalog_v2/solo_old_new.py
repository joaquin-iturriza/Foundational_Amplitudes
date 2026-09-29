"""The old from-scratch solo set (tab:scaling; sweeps/scaling_solo_full on Jean Zay, solo_full_old.json) and
the catalog solo references at batch 1024 (solo_b1k) and 16384 (solo16k.json) on one compute axis, for the
four processes they share: ee->aa, ee->uu, ee->uug, ee->uugg. Each cell at its best trial's val_loss (the
DyHPO result; CLAUDE.md, Reported values); each curve with its floor-aware fit A C^-alpha + L_inf (CLAUDE.md,
Scaling fits). Compute = flops_per_step(heads, n_p, batch) x steps with each set's own heads (4 old, 8 new)
and batch (16384 old, 1024 / 16384 new).
The old set also has a different target and inputs: log|M|^2 as is (no massive propagators divided out,
no t-channel factor, no sign head) and the old feature set (no coupling, internal-mass or off-shell
inputs). Its loss LEVELS are therefore not on the catalog sets' scale (CLAUDE.md: target-side levers
make val_loss_no_reg incomparable); only the exponents and the shapes of the curves compare across sets.
The catalog sets at bs 1024 and 16384 share one target and compare in level too. A fourth series is the
old set's pools with the catalog setup (target, inputs, 8 heads) at the old set's best HPs per step count,
one run per cell (sweep/run_fixed_hp.py, scripts/job_fixed_hp_old.sh; solo16kflatold.json): same target
as the catalog sets, same pool and HPs as the old set. A fifth is the old set's pools with the catalog setup,
swept like the catalog references (solo16kflat.json, gen_solo16k_configs.py --flat): the pool the one difference
from the catalog pool, bs 16384 series.
    python analysis/catalog_v2/solo_old_new.py --collect > analysis/catalog_v2/solo16kflatold.json   (on CC: the
        fixed-HP runs' results/fixed.json val_loss, the best checkpoint)
    python analysis/catalog_v2/solo_old_new.py
Writes solo_old_new (2x2, one panel per process, png + pdf); --no-b1k leaves the bs-1024 series out (solo_old_new_no1k)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from solo_b1k import solo_mse
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
if "--collect" in sys.argv:
    import glob, siteconf
    out = {}
    for f in glob.glob(os.path.join(siteconf.SWEEP_DIR, "solo16kflatold_t*_*", "results", "fixed.json")):
        T, p = os.path.basename(os.path.dirname(os.path.dirname(f)))[len("solo16kflatold_t"):].split("_", 1)
        out[f"{p}|{T}"] = json.load(open(f))["val_loss"]
    print(json.dumps(out)); sys.exit()
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
OLD = json.load(open(os.path.join(HERE, "solo_full_old.json")))
S16 = json.load(open(os.path.join(HERE, "solo16k.json"))); B1K = solo_mse()
FLO = json.load(open(os.path.join(HERE, "solo16kflatold.json")))
FLS = json.load(open(os.path.join(HERE, "solo16kflat.json")))
SETS = [  # label, heads, batch, steps, value(p, S), marker, filled, colour
    ("old set (tab:scaling): other target, uniform $\\sqrt{s}$, 70k, 4 heads", 4, 16384, [4, 126, 400, 1265, 4000], lambda p, S: min(OLD[f"{p}|{S}"]), "D", True, ps.C.grey),
    ("old pool, catalog setup, old best HPs", 8, 16384, [4, 126, 400, 1265, 4000], lambda p, S: FLO[f"{p}|{S}"], "^", True, ps.C.green),
    ("old pool, catalog setup, swept", 8, 16384, [63, 125, 250, 500, 1000, 2000, 4000], lambda p, S: min(FLS[f"{p}|{S}"]), "D", False, ps.C.purple),
    ("catalog pool, bs 1024", 8, 1024, [33, 67, 134, 268, 536, 1072], lambda p, S: B1K[f"{p}|{S}"], "s", False, ps.C.blue),
    ("catalog pool, bs 16384", 8, 16384, [63, 125, 250, 500, 1000, 2000, 4000], lambda p, S: min(S16[f"{p}|{S}"]), "o", True, ps.C.vermillion)]
if "--no-b1k" in sys.argv: SETS = [x for x in SETS if x[2] != 1024]   # the bs-1024 series left out (talk version)
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
ps.save(fig, os.path.join(HERE, "solo_old_new" + ("_no1k" if "--no-b1k" in sys.argv else "")))
