"""Seed spread of the 2->2 solo references (solo16k): each (process, steps) cell rerun at its best trial's HPs with
seeds 1, 2, 3 (scripts/job_seed_2to2.sh; sweeps/solo16kseed_t<T>_<p>_s<S>/results/fixed.json, val_loss = the
best checkpoint), next to the sweep's own best trial (seed 42; solo16k.json).
    python analysis/catalog_v2/solo16k_seeds.py --collect > analysis/catalog_v2/solo16k_seeds.json   (on CC)
    python analysis/catalog_v2/solo16k_seeds.py
The exponent: the floor-aware law A C^-alpha + L_inf (CLAUDE.md, Scaling fits) fitted to each seed's curve (seed 42
is the sweep's best trial, so it is the selected one, not a draw like 1-3); alpha = the mean over the four
fits, the uncertainty their standard deviation. Writes solo16k_seeds (a: loss against steps, every seed, per
process; b: alpha per process, each seed's fit and the mean)."""
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
COL = dict(zip(PROCS, (ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple)))
curve = lambda p, s: np.array([min(S42[f"{p}|{T}"]) if s == 42 else D.get(f"{p}|{T}|{s}", np.nan) for T in STEPS], float)
fig, (a, b) = ps.figure(ncols=2)
print(f"{'process':17s} alpha per seed (42 = sweep best; 1, 2, 3)      mean +- sd   | loss spread over seeds per step count (max/min)")
for i, p in enumerate(PROCS):
    al = []
    for s in [42] + SEEDS:
        y = curve(p, s); ok = np.isfinite(y); t = np.array(STEPS, float)[ok]
        a.plot(t, y[ok], marker="o" if s == 42 else ".", ls="-" if s == 42 else ":", color=COL[p], label=LABEL.get(p, p) if s == 42 else None)
        f = fit_power_law_with_floor(t, y[ok]) if ok.sum() >= 4 else None
        al.append(f[1] if f else np.nan)
    al = np.array(al); m, sd = np.nanmean(al), np.nanstd(al)
    b.plot([i] * 4, al, ls="none", marker="o", mfc="none", color=COL[p])
    b.errorbar([i], [m], yerr=[sd], fmt="s", color=COL[p], capsize=3)
    Y = np.array([curve(p, s) for s in [42] + SEEDS])
    spread = np.nanmax(Y, 0) / np.nanmin(Y, 0)
    print(f"{p:17s} " + " ".join(f"{x:.2f}" for x in al) + f"   {m:.2f} +- {sd:.2f}   | " + " ".join(f"{x:.1f}" for x in spread))
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("optimizer steps (bs 16384)"); a.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
H = [a.plot([], [], color="black", marker="o", ls="-")[0], a.plot([], [], color="black", marker=".", ls=":")[0]]
ps.legend(a, "lower left", handles=a.get_legend_handles_labels()[0] + H, labels=a.get_legend_handles_labels()[1] + ["sweep best (seed 42)", "seeds 1, 2, 3"])
b.set_xticks(range(len(PROCS)), [LABEL.get(p, p) for p in PROCS]); b.set_xlim(-0.6, len(PROCS) - 0.4)
b.set_ylabel(r"$\alpha$ in $A\,C^{-\alpha}+L_\infty$")
Hb = [b.plot([], [], ls="none", marker="o", mfc="none", color="black")[0], b.errorbar([], [], yerr=[], fmt="s", color="black", capsize=3)]
ps.legend(b, "upper right", handles=Hb, labels=["each seed's fit", "mean $\\pm$ sd"])
ps.save(fig, os.path.join(HERE, "solo16k_seeds"))
