"""The steps curve with every point tuned at its own horizon (docs/results.tex, steps curve).
Joint: the FULL pools, at the arithmetic-mean DyHPO best per horizon of the 5k-pool sweeps
(sweeps/catalog_steps_t<N>_mean), seeds 1, 2, 42, under the arithmetic mean (runs/steps_full_t<N>_s*)
and the geometric mean (runs/steps_fullgeo_t<N>_s*); both propagator factors, sign head. (The 5k-pool
joint runs, steps_tuned*, are superseded: the joint and solo curves must see the same pools.)
Solo: two reference processes per class trained alone on the FULL pools at bs 1024, DyHPO per step
count (sweeps/solob1k_t<S>_<process>, S = 33 ... 1072). The axis is training compute,
C = flops_per_step(8 heads, mean particles per event, batch) x steps (sweep/analyze_pretraining_scaling):
the joint run at bs 16384 over the catalog's mean multiplicity, a solo run at bs 1024 over its own. (The
bs-34 references, solofull_t* and the 5k-pool solo_t*, are superseded: 34-event gradients made them noise.)
    python analysis/catalog_v2/steps_tuned.py --collect > analysis/catalog_v2/steps_tuned.json   (where the runs are)
    python analysis/catalog_v2/steps_tuned.py                                                    (plots from the json)
Every curve is fitted with the floor-aware law L = A C^-alpha + L_inf (CLAUDE.md, Scaling fits; the
profiled fit of sweep/analyze_pretraining_scaling.py): the joint arms on all their non-diverged runs
pooled over 1000-8000 steps (500 is on the edge of stability; plotted, not fitted), the uncertainty the
spread of leave-one-seed-out refits; each solo process on its six points. Both sides are the LAST
validation of the chosen run: the joint seeds, and the solo DyHPO best trial per step count.
Writes steps_tuned_a ... _f (class median against compute, fits dashed) and steps_tuned_alpha (the fitted
alpha per class); the floors are printed, "unconstrained" where the fit puts them at 0. Diverged runs are
left out."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
STEPS = [500, 1000, 2000, 4000, 8000]
SOLO_STEPS = [33, 67, 134, 268, 536, 1072]
BS_JOINT, BS_SOLO, HEADS = 16384, 1024, 8
REFS = {"tree 2->2": ["ee_aa", "uubar_uubar"], "resonant 2->2": ["ee_uu", "ee_ddbar"],
        "tree 2->3": ["ee_uug", "udbar_WpZZ"], "tree 2->4": ["ee_uugg", "udbar_WpZaa"],
        "positive 1-loop": ["uubar_ZaZ_nlo", "ee_bb_nlo"], "signed 1-loop": ["udbar_Wgg_nlo", "uubar_ddbara_nlo"]}
JSON = os.environ.get("STEPS_TUNED_JSON", os.path.join(HERE, "steps_tuned.json"))   # override: test on a subset

def final(run_dir):
    js = sorted(glob.glob(os.path.join(run_dir, "**", "per_process_metrics.json"), recursive=True))
    if not js: return None
    d = json.load(open(js[-1]))
    return {n: v[-1] for n, v in d["proc_val_losses_no_reg"].items() if v}

if "--collect" in sys.argv:
    out = {"joint": [], "solo": {}}
    for N in STEPS:
        runs = [("arith", S, r) for S in (1, 2, 42) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_full_t{N}_s{S}"))]
        runs += [("geo", S, r) for S in (1, 2, 42) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_fullgeo_t{N}_s{S}"))]
        for arm, S, r in runs:
            f = final(r)
            if f: out["joint"].append({"arm": arm, "steps": N, "seed": S, "final": f})
    for S in SOLO_STEPS:
        for procs in REFS.values():
            for p in procs:
                # the best trial from the per-trial results (full precision: summary.txt keeps six
                # decimals), then that trial's LAST validation, read like the joint runs'
                res = glob.glob(os.path.join(ROOT, "sweeps", f"solob1k_t{S}_{p}", "results", f"hp*_t{S}_*.json"))
                if not res: continue
                vl = {int(re.search(r"hp(\d+)_", os.path.basename(f)).group(1)): json.load(open(f))["val_loss"] for f in res}
                hp = min(vl, key=vl.get)
                f = final(os.path.join(ROOT, "runs", f"solob1k_t{S}_{p}", f"trial_{hp:04d}"))
                out["solo"][f"{p}|{S}"] = f[p] if f and p in f else vl[hp]
    print(json.dumps(out)); sys.exit()

import plot_style as ps
import census as C
D = json.load(open(JSON))
cls = json.load(open(os.path.join(HERE, "steps_agg.json")))["cls"].get
# a run whose median per-process loss ends above 0.5 diverged (at 500 steps the DyHPO-best lr,
# 2.3e-2, sits on the edge of stability: four of its five reseeds blow up); it is left out of the
# curves, and the power laws are fitted over FIT_STEPS, where every run is sound
DIV = [r for r in D["joint"] if np.median(list(r["final"].values())) > 0.5]
D["joint"] = [r for r in D["joint"] if r not in DIV]
print("diverged, left out: " + ", ".join(f"{r['arm']} t{r['steps']} s{r['seed']}" for r in DIV))
FIT_STEPS = [1000, 2000, 4000, 8000]
from solo_datalimit_labels import LABEL
ARMS = [("arith", ps.C.blue, "joint, arithmetic mean"), ("geo", ps.C.vermillion, "joint, geometric mean")]
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
_JN = sorted({n for r in D["joint"] for n in r["final"]})
N_JOINT = float(np.mean([NP[n] for n in _JN if n in NP]))   # mean multiplicity over the trained catalog (uniform sampler)
C_JOINT = lambda steps: flops_per_step(HEADS, N_JOINT, BS_JOINT) * steps
C_SOLO = lambda p, steps: flops_per_step(HEADS, NP[p], BS_SOLO) * steps

def fitf(c, l):
    r = fit_power_law_with_floor(c, l)
    return None if r is None else dict(A=r[0], alpha=r[1], Linf=r[2], chi2r=r[3])

def joint_points(arm, c, steps=None):
    """[(seed, events per process, class median)] over the runs that did not diverge (and, with
    `steps`, only those horizons)."""
    pts = []
    for r in D["joint"]:
        if r["arm"] == arm and (steps is None or r["steps"] in steps):
            v = [x for n, x in r["final"].items() if cls(n) == c]
            if v: pts.append((r["seed"], C_JOINT(r["steps"]), float(np.median(v))))
    return pts

def solo_points(p):
    t = [S for S in SOLO_STEPS if f"{p}|{S}" in D["solo"]]
    return [C_SOLO(p, S) for S in t], [D["solo"][f"{p}|{S}"] for S in t]

FIT = {}
for c, procs in REFS.items():
    for arm, _, _ in ARMS:
        pts = joint_points(arm, c, steps=FIT_STEPS)    # 500 steps is on the edge of stability
        f = fitf([q[1] for q in pts], [q[2] for q in pts])
        loo = [fitf([q[1] for q in pts if q[0] != sd], [q[2] for q in pts if q[0] != sd]) for sd in sorted({q[0] for q in pts})]
        ok = [x for x in loo if x]
        if f:
            for k in ("alpha", "Linf"):
                f[k + "_rng"] = [min(x[k] for x in ok), max(x[k] for x in ok)] if ok else [f[k], f[k]]
        FIT[arm, c] = f
    for p in procs:
        FIT["solo", p] = fitf(*solo_points(p))

figs = ps.panels(len(REFS))
for (fig, ax), (c, procs) in zip(figs, REFS.items()):
    for arm, col, lab in ARMS:
        pts = joint_points(arm, c)
        by = {}
        for sd, e, v in pts: by.setdefault(e, []).append(v)
        e = sorted(by)
        ax.plot(e, [np.exp(np.mean(np.log(by[k]))) for k in e], marker="o", ls="none", color=col, label=lab)
        ax.fill_between(e, [min(by[k]) for k in e], [max(by[k]) for k in e], color=col, alpha=0.2)
        f = FIT[arm, c]
        if f:     # each fit drawn over its own data only (a fit is not an extrapolation)
            g = np.geomspace(min(e) / 1.3, max(e) * 1.3, 100)
            ax.plot(g, f["A"] * g ** -f["alpha"] + f["Linf"], color=col, ls="--")
    for p, mk in zip(procs, ("s", "D")):
        e, y = solo_points(p)
        ax.plot(e, y, marker=mk, ls="none", color=ps.C.grey, label=f"{LABEL[p]} alone")
        f = FIT["solo", p]
        if f:
            g = np.geomspace(min(e) / 1.3, max(e) * 1.3, 100)
            ax.plot(g, f["A"] * g ** -f["alpha"] + f["Linf"], color=ps.C.grey, ls=":")
    ax.plot([], [], color="black", ls="--", label=r"fit $A\,C^{-\alpha}+L_\infty$, joint")
    ax.plot([], [], color=ps.C.grey, ls=":", label=r"fit $A\,C^{-\alpha}+L_\infty$, alone")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training compute $C$ [FLOP]"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.shared_legend(fig, ax, ncol=1)
ps.save_panels(figs, "analysis/catalog_v2/steps_tuned")

lfmt = lambda v: "unconstr." if v <= 0 else f"{v:.2g}"
print(f"{'class':16s} " + " | ".join(f"{k:>30s}" for k in ("joint arith  a [loo] Linf chi2r", "joint geo", "alone (2 processes) a Linf")))
for c, procs in REFS.items():
    cells = []
    for arm, _, _ in ARMS:
        f = FIT[arm, c]
        cells.append(f"{f['alpha']:.2f} [{f['alpha_rng'][0]:.2f},{f['alpha_rng'][1]:.2f}] {lfmt(f['Linf'])} {f['chi2r']:.1f}" if f else "no fit")
    cells.append("  ".join(f"{FIT['solo', p]['alpha']:.2f}/{lfmt(FIT['solo', p]['Linf'])}" if FIT["solo", p] else "no fit" for p in procs))
    print(f"{c:16s} " + " | ".join(f"{x:>30s}" for x in cells))

yy = np.arange(len(REFS))[::-1]
for base, key, xlab, logx in (("steps_tuned_alpha", "alpha", r"exponent $\alpha$ in $A\,C^{-\alpha}+L_\infty$", False),):
    fig, ax = ps.figure()
    for k, (arm, col, lab) in enumerate(ARMS):
        m, lo, hi, y = [], [], [], []
        for i, c in enumerate(REFS):
            f = FIT[arm, c]
            if not f: continue
            rng = f[key + "_rng"]; v = f[key]
            m.append(v); lo.append(max(v - rng[0], 0)); hi.append(max(rng[1] - v, 0)); y.append(yy[i] + (-0.18 if k == 0 else 0.0))
        ax.errorbar(m, y, xerr=[lo, hi], fmt="o", color=col, capsize=2, ls="none", label=lab)
    for j, mk in enumerate(("s", "D")):
        m, y = [], []
        for i, (c, procs) in enumerate(REFS.items()):
            f = FIT["solo", procs[j]]
            if f: m.append(f[key]); y.append(yy[i] + 0.18)
        ax.plot(m, y, marker=mk, ls="none", color=ps.C.grey, label="alone, " + ("first" if j == 0 else "second") + " reference process")
    ax.set_yticks(yy, [C.CLASS_LABEL[c] for c in REFS]); ax.set_ylim(-0.6, len(REFS) - 0.4)
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ps.shared_legend(fig, ax, ncol=1)
    ps.save(fig, f"analysis/catalog_v2/{base}")
