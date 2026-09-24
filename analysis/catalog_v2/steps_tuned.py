"""The steps curve with every point tuned at its own horizon (docs/results.tex, steps curve).
Joint: the arithmetic-mean DyHPO best per horizon (sweeps/catalog_steps_t<N>_mean, its best trial
is seed 42) plus runs/steps_tuned_t<N>_s{1,2} at the same HPs, and the geometric mean at those HPs
(runs/steps_tunedgeo_t<N>_s{1,2,42}); 5k pools, both propagator factors, sign head.
Solo: two reference processes per class trained alone on the FULL pools at bs 1024, DyHPO per step
count (sweeps/solob1k_t<S>_<process>, S = 33 ... 1072). The axis is events seen per process: 16384/478
per step for the joint run, 1024 per step alone. (The bs-34 references, solofull_t* and the 5k-pool
solo_t*, are superseded: 34-event gradients made them noise.)
    python analysis/catalog_v2/steps_tuned.py --collect > analysis/catalog_v2/steps_tuned.json   (where the runs are)
    python analysis/catalog_v2/steps_tuned.py                                                    (plots from the json)
Every curve is fitted with the floor-aware law L = A C^-alpha + L_inf (CLAUDE.md, Scaling fits; the
profiled fit of sweep/analyze_pretraining_scaling.py): the joint arms on all their non-diverged runs
pooled, the uncertainty the spread of leave-one-seed-out refits; each solo process on its six points.
Writes steps_tuned_a ... _f (class median against events per process, fits dashed), steps_tuned_alpha
and steps_tuned_floor (the fitted alpha and L_inf per class). Diverged runs are left out."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
STEPS = [500, 1000, 2000, 4000, 8000]
SOLO_STEPS = [33, 67, 134, 268, 536, 1072]
EV_JOINT, EV_SOLO = 16384 / 478, 1024        # events per process per step
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
        s = open(os.path.join(ROOT, "sweeps", f"catalog_steps_t{N}_mean", "summary.txt")).read()
        best = re.search(r"^\s+(hp_\d+)\s+val_loss", s, re.M).group(1).replace("hp_", "trial_")
        runs = [("arith", 42, os.path.join(ROOT, "runs", f"catalog_steps_t{N}_mean", best))]
        runs += [("arith", S, r) for S in (1, 2) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_tuned_t{N}_s{S}"))]
        runs += [("geo", S, r) for S in (1, 2, 42) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_tunedgeo_t{N}_s{S}"))]
        for arm, S, r in runs:
            f = final(r)
            if f: out["joint"].append({"arm": arm, "steps": N, "seed": S, "final": f})
    for S in SOLO_STEPS:
        for procs in REFS.values():
            for p in procs:
                sw = os.path.join(ROOT, "sweeps", f"solob1k_t{S}_{p}", "summary.txt")
                m = re.search(r"Best val_loss: ([0-9.eE+-]+)", open(sw).read()) if os.path.exists(sw) else None
                if m: out["solo"][f"{p}|{S}"] = float(m.group(1))
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
from analyze_pretraining_scaling import fit_power_law_with_floor

def fitf(c, l):
    r = fit_power_law_with_floor(c, l)
    return None if r is None else dict(A=r[0], alpha=r[1], Linf=r[2], chi2r=r[3])

def joint_points(arm, c):
    """[(seed, events per process, class median)] over the runs that did not diverge."""
    pts = []
    for r in D["joint"]:
        if r["arm"] == arm:
            v = [x for n, x in r["final"].items() if cls(n) == c]
            if v: pts.append((r["seed"], r["steps"] * EV_JOINT, float(np.median(v))))
    return pts

def solo_points(p):
    t = [S for S in SOLO_STEPS if f"{p}|{S}" in D["solo"]]
    return [S * EV_SOLO for S in t], [D["solo"][f"{p}|{S}"] for S in t]

FIT = {}
for c, procs in REFS.items():
    for arm, _, _ in ARMS:
        pts = joint_points(arm, c)
        f = fitf([q[1] for q in pts], [q[2] for q in pts])
        loo = [fitf([q[1] for q in pts if q[0] != sd], [q[2] for q in pts if q[0] != sd]) for sd in sorted({q[0] for q in pts})]
        if f: f["alpha_rng"] = [min(x["alpha"] for x in loo if x), max(x["alpha"] for x in loo if x)]; f["Linf_rng"] = [min(x["Linf"] for x in loo if x), max(x["Linf"] for x in loo if x)]
        FIT[arm, c] = f
    for p in procs:
        FIT["solo", p] = fitf(*solo_points(p))

cgrid = np.geomspace(1e4, 2e6, 200)
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
        if f: ax.plot(cgrid, f["A"] * cgrid ** -f["alpha"] + f["Linf"], color=col, ls="--")
    for p, mk in zip(procs, ("s", "D")):
        e, y = solo_points(p)
        ax.plot(e, y, marker=mk, ls="none", color=ps.C.grey, label=f"{LABEL[p]} alone")
        f = FIT["solo", p]
        if f: ax.plot(cgrid, f["A"] * cgrid ** -f["alpha"] + f["Linf"], color=ps.C.grey, ls=":")
    ax.plot([], [], color="black", ls="--", label=r"fit $A\,C^{-\alpha}+L_\infty$, joint")
    ax.plot([], [], color=ps.C.grey, ls=":", label=r"fit $A\,C^{-\alpha}+L_\infty$, alone")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("events seen per process"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.shared_legend(fig, ax, ncol=1)
ps.save_panels(figs, "analysis/catalog_v2/steps_tuned")

print(f"{'class':16s} " + " | ".join(f"{k:>30s}" for k in ("joint arith  a [loo] Linf chi2r", "joint geo", "alone (2 processes) a Linf")))
for c, procs in REFS.items():
    cells = []
    for arm, _, _ in ARMS:
        f = FIT[arm, c]
        cells.append(f"{f['alpha']:.2f} [{f['alpha_rng'][0]:.2f},{f['alpha_rng'][1]:.2f}] {f['Linf']:.2g} {f['chi2r']:.1f}" if f else "no fit")
    cells.append("  ".join(f"{FIT['solo', p]['alpha']:.2f}/{FIT['solo', p]['Linf']:.2g}" if FIT["solo", p] else "no fit" for p in procs))
    print(f"{c:16s} " + " | ".join(f"{x:>30s}" for x in cells))

yy = np.arange(len(REFS))[::-1]
for base, key, xlab, logx in (("steps_tuned_alpha", "alpha", r"exponent $\alpha$ in $A\,C^{-\alpha}+L_\infty$", False),
                               ("steps_tuned_floor", "Linf", r"floor $L_\infty$, MSE($\log|\mathcal{M}|^2$)", True)):
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
