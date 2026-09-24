"""The steps curve with every point tuned at its own horizon (docs/results.tex, steps curve).
Joint: the FULL pools, at the arithmetic-mean DyHPO best per horizon of the 5k-pool sweeps
(sweeps/catalog_steps_t<N>_mean), seeds 1, 2, 42, under the arithmetic mean (runs/steps_full_t<N>_s*)
and the geometric mean (runs/steps_fullgeo_t<N>_s*); both propagator factors, sign head.
Solo: two reference processes per class trained alone on the FULL pools at bs 1024, DyHPO per step
count (sweeps/solob1k_t<S>_<process>, S = 33 ... 1072; the signed pools from their rerun
solob1kv_t<S>_<process>, solo_b1k.py).
    python analysis/catalog_v2/steps_tuned.py --collect > analysis/catalog_v2/steps_tuned.json   (where the runs are)
    python analysis/catalog_v2/steps_tuned.py                                                    (plots from the json)
Values: every joint run at its best checkpoint (census.at_best: each process at the validation with
the lowest aggregate val_loss_no_reg), every solo sweep at its best trial's best validation
(CLAUDE.md, Reported values). No run is left out; a run whose loss rose after its best checkpoint is
listed with where.
Axis: per-process training compute, C_p = (events of process p seen) x flops_per_step(8 heads, n_p,
1) (sweep/analyze_pretraining_scaling.py), n_p the particles per event of p. In the joint run the
uniform sampler draws events in proportion to pool size, so p sees 16384 N_p / sum_q N_q events per
step (N_p its train pool, read from the run's log: 100k for trees, 50k for loop pools); alone it sees
1024 per step. A class's joint point sits at the median C_p over the class.
Fits: the floor-aware law L = A C^-alpha + L_inf (CLAUDE.md, Scaling fits; the profiled fit of
sweep/analyze_pretraining_scaling.py), the joint arms on every run at every horizon, the uncertainty the
spread of leave-one-seed-out refits; each solo process on its six points.
Writes steps_tuned_a ... _f (class median against per-process compute, fits dashed) and
steps_tuned_alpha (the fitted alpha per class); the floors are printed, "unconstrained" where the fit
puts them at 0."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import census as C
STEPS = [500, 1000, 2000, 4000, 8000]
SOLO_STEPS = [33, 67, 134, 268, 536, 1072]
BS_JOINT, BS_SOLO, HEADS = 16384, 1024, 8
REFS = {"tree 2->2": ["ee_aa", "uubar_uubar"], "resonant 2->2": ["ee_uu", "ee_ddbar"],
        "tree 2->3": ["ee_uug", "udbar_WpZZ"], "tree 2->4": ["ee_uugg", "udbar_WpZaa"],
        "positive 1-loop": ["uubar_ZaZ_nlo", "ee_bb_nlo"], "signed 1-loop": ["udbar_Wgg_nlo", "uubar_ddbara_nlo"]}
JSON = os.environ.get("STEPS_TUNED_JSON", os.path.join(HERE, "steps_tuned.json"))   # override: test on a subset

if "--collect" in sys.argv:
    out = {"joint": [], "pool": {}}
    for N in STEPS:
        runs = [("arith", S, r) for S in (1, 2, 42) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_full_t{N}_s{S}"))]
        runs += [("geo", S, r) for S in (1, 2, 42) for r in glob.glob(os.path.join(ROOT, "runs", f"steps_fullgeo_t{N}_s{S}"))]
        for arm, S, r in runs:
            d = C.metrics(r); ib, comb, proc = C.at_best(d)
            out["joint"].append({"arm": arm, "steps": N, "seed": S, "best_idx": ib,
                                 "best_not_last": C.best_not_last(d), "final": proc})
            if not out["pool"]:   # train pool sizes, from the "[train] <name>: <N> events" lines
                log = sorted(glob.glob(os.path.join(r, "**", "out_0.log"), recursive=True))[-1]
                for m in re.finditer(r"\[train\] (\S+): (\d+) events", open(log, errors="replace").read()):
                    out["pool"][m.group(1)] = int(m.group(2))
    print(json.dumps(out)); sys.exit()

import plot_style as ps
D = json.load(open(JSON))
from solo_b1k import solo_mse
D["solo"] = solo_mse()
cls = json.load(open(os.path.join(HERE, "steps_agg.json")))["cls"].get
rose = [r for r in D["joint"] if r.get("best_not_last")]
print("loss rose after the best checkpoint (reported, kept at their best checkpoint): "
      + (", ".join(f"{r['arm']} t{r['steps']} s{r['seed']} (best at validation {r['best_not_last'][0] + 1} of "
                   f"{r['best_not_last'][1]}, last/best {r['best_not_last'][2]:.3g})" for r in rose) or "none"))
from solo_datalimit_labels import LABEL
ARMS = [("arith", ps.C.blue, "joint, arithmetic"), ("geo", ps.C.vermillion, "joint, geometric")]
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
POOL = D["pool"]; POOL_SUM = float(sum(POOL.values()))
EV_JOINT = lambda p: BS_JOINT * POOL[p] / POOL_SUM             # events of p per joint step (uniform sampler)
C_JOINT = lambda p, steps: flops_per_step(HEADS, NP[p], 1) * EV_JOINT(p) * steps
C_SOLO = lambda p, steps: flops_per_step(HEADS, NP[p], 1) * BS_SOLO * steps

def fitf(c, l):
    r = fit_power_law_with_floor(c, l)
    return None if r is None else dict(A=r[0], alpha=r[1], Linf=r[2], chi2r=r[3])

def joint_points(arm, c):
    """[(seed, class median of the per-process compute, class median loss)], every run of the arm."""
    pts = []
    for r in D["joint"]:
        if r["arm"] == arm:
            names = [n for n in r["final"] if cls(n) == c and n in POOL]
            if names:
                pts.append((r["seed"], float(np.median([C_JOINT(n, r["steps"]) for n in names])),
                            float(np.median([r["final"][n] for n in names]))))
    return pts

def solo_points(p):
    t = [S for S in SOLO_STEPS if f"{p}|{S}" in D["solo"]]
    return [C_SOLO(p, S) for S in t], [D["solo"][f"{p}|{S}"] for S in t]

FIT = {}
for c, procs in REFS.items():
    for arm, _, _ in ARMS:
        pts = joint_points(arm, c)
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
    ax.plot([], [], color="black", ls="--", label=r"fits $A\,C^{-\alpha}+L_\infty$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"per-process training compute $C_p$ [FLOP]"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.legend(ax, "upper right")
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
fig, ax = ps.figure()
for k, (arm, col, lab) in enumerate(ARMS):
    m, lo, hi, y = [], [], [], []
    for i, c in enumerate(REFS):
        f = FIT[arm, c]
        if not f: continue
        rng = f["alpha_rng"]; v = f["alpha"]
        m.append(v); lo.append(max(v - rng[0], 0)); hi.append(max(rng[1] - v, 0)); y.append(yy[i] + (-0.18 if k == 0 else 0.0))
    ax.errorbar(m, y, xerr=[lo, hi], fmt="o", color=col, capsize=2, ls="none", label=lab)
m, y = [], []            # both reference processes of a class, one marker: "alone" is one series
for i, (c, procs) in enumerate(REFS.items()):
    for p_ in procs:
        f = FIT["solo", p_]
        if f: m.append(f["alpha"]); y.append(yy[i] + 0.18)
ax.plot(m, y, marker="s", ls="none", color=ps.C.grey, label="alone (2 processes)")
ax.set_yticks(yy, [C.CLASS_LABEL[c] for c in REFS]); ax.set_ylim(-0.6, len(REFS) - 0.4)
ax.set_xlabel(r"exponent $\alpha$ in $A\,C^{-\alpha}+L_\infty$")
ps.legend(ax, "lower right")
ps.save(fig, "analysis/catalog_v2/steps_tuned_alpha")
