"""Seed spread of the 2->2 solo references (solo16k): each (process, steps) cell rerun at its best trial's HPs with
seeds 1, 2, 3 (scripts/job_seed_2to2.sh; sweeps/solo16kseed_t<T>_<p>_s<S>/results/fixed.json, val_loss = the
best checkpoint), next to the sweep's own best trial (seed 42; solo16k.json).
    python analysis/catalog_v2/solo16k_seeds.py --collect > analysis/catalog_v2/solo16k_seeds.json   (on CC)
    python analysis/catalog_v2/solo16k_seeds.py
The exponent: the floor-aware law A C^-alpha + L_inf (CLAUDE.md, Scaling fits) fitted to each seed's curve (seed 42
is the sweep's best trial, so it is the selected one, not a draw like 1-3); alpha = the mean over the four
fits, the uncertainty their standard deviation. Writes solo16k_seeds_a ... _f: (a-e) one process each, loss against steps for every seed, every run at its
best checkpoint and none left out (the printout lists each cell's four values); (f) alpha per process, each
seed's fit and mean +- sd."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
STEPS = [63, 125, 250, 500, 1000, 2000, 4000]; SEEDS = [1, 2, 3]
PROCS = ["ee_aa", "uubar_uubar", "ee_uu", "ee_ddbar", "ee_bb_nlo"]
JSON = os.path.join(HERE, "solo16k_seeds.json")

if "--collect" in sys.argv:
    import siteconf
    out = {}
    for p in PROCS:
        for T in STEPS:
            for s in SEEDS:
                f = os.path.join(siteconf.SWEEP_DIR, f"solo16kseed_t{T}_{p}_s{s}", "results", "fixed.json")
                if os.path.exists(f): out[f"{p}|{T}|{s}"] = json.load(open(f))["val_loss"]
    print(json.dumps(out)); sys.exit()

import plot_style as ps
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor
D = json.load(open(JSON)); S42 = json.load(open(os.path.join(HERE, "solo16k.json")))
COLS = {42: "black", 1: ps.C.blue, 2: ps.C.vermillion, 3: ps.C.green}
curve = lambda p, s: np.array([min(S42[f"{p}|{T}"]) if s == 42 else D.get(f"{p}|{T}|{s}", np.nan) for T in STEPS], float)
figs = ps.panels(len(PROCS) + 1)
ALPHA = {}
print(f"{'process':17s} alpha per seed (42 = sweep best; 1, 2, 3)      mean +- sd")
for (fig, ax), p in zip(figs, PROCS):
    Y = np.array([curve(p, s) for s in [42] + SEEDS]); med = np.nanmedian(Y, 0)
    al = []
    for k, s in enumerate([42] + SEEDS):
        y = Y[k]; t = np.array(STEPS, float)
        ax.plot(t, y, marker="o", ls="-" if s == 42 else ":", color=COLS[s], label="sweep best (seed 42)" if s == 42 else f"seed {s}")
        f = fit_power_law_with_floor(t, y) if np.isfinite(y).sum() >= 4 else None
        al.append(f[1] if f else np.nan)
    ALPHA[p] = np.array(al)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("optimizer steps (bs 16384)"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL.get(p, p), loc="lower left")
    ps.legend(ax, "upper right")
    print(f"{p:17s} " + " ".join(f"{x:.2f}" for x in ALPHA[p]) + f"   {np.nanmean(ALPHA[p]):.2f} +- {np.nanstd(ALPHA[p]):.2f}   | per step, seeds 42/1/2/3: "
          + "  ".join(f"t{T}:" + "/".join(f"{v:.1g}" for v in Y[:, j]) for j, T in enumerate(STEPS)))
fig, ax = figs[-1]
yy = np.arange(len(PROCS))[::-1]
for y, p in zip(yy, PROCS):
    for k, s in enumerate([42] + SEEDS):
        ax.plot([ALPHA[p][k]], [y], ls="none", marker="o", mfc="none", color=COLS[s])
    ax.errorbar([np.nanmean(ALPHA[p])], [y], xerr=[np.nanstd(ALPHA[p])], fmt="s", color="black", capsize=3)
ax.set_yticks(yy, [LABEL.get(p, p) for p in PROCS]); ax.set_ylim(-0.6, len(PROCS) - 0.4)
ax.set_xlabel(r"$\alpha$ in $A\,C^{-\alpha}+L_\infty$")
H = [ax.plot([], [], ls="none", marker="o", mfc="none", color=COLS[s])[0] for s in [42] + SEEDS] + [ax.errorbar([], [], xerr=[], fmt="s", color="black", capsize=3)]
ps.legend(ax, "lower right", handles=H, labels=["seed 42 (sweep best)", "seed 1", "seed 2", "seed 3", r"mean $\pm$ sd"], ncol=2)
ps.save_panels(figs, os.path.join(HERE, "solo16k_seeds"))
