"""The batch-16384 solo references (sweeps/solo16k_t<T>_<process>, sweep/gen_solo16k_configs.py): the 12
reference processes of the steps curve alone on their full catalog pools at the canonical batch, 63 ...
4000 steps, 6 DyHPO trials per (steps, process).
    python analysis/catalog_v2/solo16k.py --collect > <site>.json   (on each site holding solo16k sweeps; merge the
                                                                  per-site json into analysis/catalog_v2/solo16k.json)
    python analysis/catalog_v2/solo16k.py                                              (plots from the json)
Values: per sweep, the best trial's val_loss (the DyHPO result: the best checkpoint's val_loss_no_reg;
CLAUDE.md, Reported values), with the number of finished trials. Fit: the floor-aware law
L = A C^-alpha + L_inf over the step counts that have a result (CLAUDE.md, Scaling fits; needs 4),
C the per-process compute flops_per_step(8, n_p, 1) x 16384 x steps. Printed next to the batch-1024 set
(solo_b1k) and the joint exponents (steps_tuned). Writes solo16k_a ... _f: loss against steps per class,
this set (filled) and the batch-1024 set (open) on a common per-process-compute axis."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
STEPS = [63, 125, 250, 500, 1000, 2000, 4000]
REFS = {"tree 2->2": ["ee_aa", "uubar_uubar"], "resonant 2->2": ["ee_uu", "ee_ddbar"],
        "tree 2->3": ["ee_uug", "udbar_WpZZ"], "tree 2->4": ["ee_uugg", "udbar_WpZaa"],
        "positive 1-loop": ["uubar_ZaZ_nlo", "ee_bb_nlo"], "signed 1-loop": ["udbar_Wgg_nlo", "uubar_ddbara_nlo"]}
JSON = os.path.join(HERE, "solo16k.json")

if "--collect" in sys.argv:
    import siteconf
    out = {}
    for p in sum(REFS.values(), []):
        for T in STEPS:
            v = [json.load(open(f))["val_loss"] for f in glob.glob(os.path.join(siteconf.SWEEP_DIR, f"solo16k_t{T}_{p}", "results", "*.json"))]
            out[f"{p}|{T}"] = sorted(float(x) for x in v)
    print(json.dumps(out)); sys.exit()

import plot_style as ps
import census as C
from solo_b1k import solo_mse
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
D = json.load(open(JSON)); B1K = solo_mse()
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
comp = lambda p, bs, S: flops_per_step(8, NP[p], 1) * bs * S

def series(p):
    t = [T for T in STEPS if D.get(f"{p}|{T}")]
    return t, [min(D[f"{p}|{T}"]) for T in t], [len(D[f"{p}|{T}"]) for T in t]

def fit(x, y):
    r = fit_power_law_with_floor(np.array(x, float), np.array(y, float)) if len(x) >= 4 else None
    return r

print(f"{'process':18s} {'trials per step count':>22s} | {'bs 16384 alpha (L_inf)':>24s} | {'bs 1024 alpha':>13s}")
FIT = {}
for c, procs in REFS.items():
    for p in procs:
        t, y, n = series(p)
        r = fit([comp(p, 16384, T) for T in t], y); FIT[p] = r
        S1 = [33, 67, 134, 268, 536, 1072]
        r1 = fit([comp(p, 1024, S) for S in S1], [B1K[f"{p}|{S}"] for S in S1])
        a = f"{r[1]:.2f} ({r[2]:.2g})" if r else f"no fit ({len(t)} step counts)"
        print(f"{p:18s} {' '.join(f'{T}:{k}' for T, k in zip(t, n)):>22s} | {a:>24s} | {r1[1]:13.2f}")

figs = ps.panels(len(REFS))
for (fig, ax), (c, procs) in zip(figs, REFS.items()):
    for p, col in zip(procs, (ps.C.blue, ps.C.vermillion)):
        t, y, _ = series(p)
        ax.plot([comp(p, 16384, T) for T in t], y, marker="o", ls="none", color=col, label=f"{LABEL[p]}, bs 16384")
        if FIT[p]:
            A, a, Li = FIT[p][:3]; g = np.geomspace(comp(p, 16384, t[0]) / 1.3, comp(p, 16384, t[-1]) * 1.3, 100)
            ax.plot(g, A * g ** -a + Li, color=col, ls="--")
        S1 = [33, 67, 134, 268, 536, 1072]
        ax.plot([comp(p, 1024, S) for S in S1], [B1K[f"{p}|{S}"] for S in S1], marker="s", mfc="none", ls="none", color=col, label=f"{LABEL[p]}, bs 1024")
    ax.plot([], [], color="black", ls="--", label=r"fit $A\,C^{-\alpha}+L_\infty$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"per-process compute $C_p$ [FLOP]"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.legend(ax, "upper right")
ps.save_panels(figs, "analysis/catalog_v2/solo16k")
