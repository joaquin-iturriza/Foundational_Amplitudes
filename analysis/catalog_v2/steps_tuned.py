"""The steps curve with every point tuned at its own horizon (docs/results.tex, steps curve).
Joint: the arithmetic-mean DyHPO best per horizon (sweeps/catalog_steps_t<N>_mean, its best trial
is seed 42) plus runs/steps_tuned_t<N>_s{1,2} at the same HPs, and the geometric mean at those HPs
(runs/steps_tunedgeo_t<N>_s{1,2,42}); 5k pools, both propagator factors, sign head.
Solo: two reference processes per class trained alone on the FULL pools, DyHPO per horizon
(sweeps/solofull_t<N>_<process>; the 5k-pool references are data-limited, solo_datalimit.py).
    python analysis/catalog_v2/steps_tuned.py --collect > analysis/catalog_v2/steps_tuned.json   (where the runs are)
    python analysis/catalog_v2/steps_tuned.py                                                    (plots from the json)
Writes steps_tuned_a ... _f (class median against steps: both joint arms with the seed band, the two
solo processes), steps_tuned_alpha and steps_tuned_start (power-law fit per class over 1000-8000 steps:
exponent and the fitted loss at 1000 steps, joint arms per seed, solo per process). Diverged runs are left out."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
STEPS = [500, 1000, 2000, 4000, 8000]
REFS = {"tree 2->2": ["ee_aa", "uubar_uubar"], "resonant 2->2": ["ee_uu", "ee_ddbar"],
        "tree 2->3": ["ee_uug", "udbar_WpZZ"], "tree 2->4": ["ee_uugg", "udbar_WpZaa"],
        "positive 1-loop": ["uubar_ZaZ_nlo", "ee_bb_nlo"], "signed 1-loop": ["udbar_Wgg_nlo", "uubar_ddbara_nlo"]}
JSON = os.path.join(HERE, "steps_tuned.json")

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
        for procs in REFS.values():
            for p in procs:
                sw = os.path.join(ROOT, "sweeps", f"solofull_t{N}_{p}", "summary.txt")
                m = re.search(r"Best val_loss: ([0-9.eE+-]+)", open(sw).read()) if os.path.exists(sw) else None
                if m: out["solo"][f"{p}|{N}"] = float(m.group(1))
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
x = np.log(np.array(STEPS) / 1000.0)

def fit(t, y):
    a, b = np.polyfit(np.log(np.array(t) / 1000.0), np.log(y), 1); return -a, np.exp(b)

def joint_by_seed(arm, c):
    """{seed: {steps: class median}} over the runs that did not diverge."""
    by = {}
    for r in D["joint"]:
        if r["arm"] == arm:
            v = [x for n, x in r["final"].items() if cls(n) == c]
            if v: by.setdefault(r["seed"], {})[r["steps"]] = np.median(v)
    return by

def joint_curve(arm, c, steps=FIT_STEPS):
    """{seed: [class median per horizon]} for the seeds present at every horizon of `steps`."""
    return {s: [d[N] for N in steps] for s, d in joint_by_seed(arm, c).items() if all(N in d for N in steps)}

def solo_curve(p):
    t = [N for N in STEPS if f"{p}|{N}" in D["solo"]]
    return t, [D["solo"][f"{p}|{N}"] for N in t]

# (a-f) class median against steps
figs = ps.panels(len(REFS))
for (fig, ax), (c, procs) in zip(figs, REFS.items()):
    for arm, col, lab in ARMS:
        by = joint_by_seed(arm, c)
        t = [N for N in STEPS if any(N in d for d in by.values())]
        v = [[d[N] for d in by.values() if N in d] for N in t]
        n = min(len(x) for x in v)
        ax.plot(t, [np.exp(np.mean(np.log(x))) for x in v], marker="o", color=col,
                label=f"{lab} ({n}-{max(len(x) for x in v)} seeds)")
        ax.fill_between(t, [min(x) for x in v], [max(x) for x in v], color=col, alpha=0.2)
    for p, mk in zip(procs, ("s", "D")):
        t, y = solo_curve(p)
        ax.plot(t, y, marker=mk, color=ps.C.grey, ls="--", label=f"{LABEL[p]} alone, full pool")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(n) for n in STEPS]); ax.minorticks_off()
    ax.set_xlabel("training steps"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.shared_legend(fig, ax, ncol=1)
ps.save_panels(figs, "analysis/catalog_v2/steps_tuned")

# power-law fits: joint per seed (median over seeds, bar = min to max), solo per process
FIT = {}
print(f"fits over {FIT_STEPS}")
print(f"{'class':16s} " + " | ".join(f"{k:>22s}" for k in ("joint arith", "joint geo", "solo (2 processes)")) + "   (alpha, L at 1000)")
for c, procs in REFS.items():
    cells = []
    for arm, _, _ in ARMS:
        cur = joint_curve(arm, c)
        FIT[arm, c] = np.array([fit(FIT_STEPS, y) for y in cur.values()]) if cur else np.zeros((0, 2))
        f = FIT[arm, c]
        cells.append(f"{f[:,0].mean():.2f}±{np.ptp(f[:,0])/2:.2f} {np.exp(np.log(f[:,1]).mean()):.2g}" if len(f) else "")
    FIT["solo", c] = np.array([fit(FIT_STEPS, [D["solo"][f"{p}|{N}"] for N in FIT_STEPS]) for p in procs])
    f = FIT["solo", c]
    cells.append(" ".join(f"{a:.2f}/{b:.2g}" for a, b in f))
    print(f"{c:16s} " + " | ".join(f"{s:>22s}" for s in cells))
yy = np.arange(len(REFS))[::-1]
SERIES = [("arith", ps.C.blue, "joint, arithmetic mean", "o"), ("geo", ps.C.vermillion, "joint, geometric mean", "o"),
          ("solo", ps.C.grey, "alone, full pool (2 processes)", "s")]
off = {"arith": -0.18, "geo": 0.0, "solo": 0.18}
for j, (base, xlab, logx) in enumerate((("steps_tuned_alpha", r"exponent $\alpha$, $L\propto t^{-\alpha}$", False),
                                         ("steps_tuned_start", r"fitted MSE($\log|\mathcal{M}|^2$) at 1000 steps", True))):
    fig, ax = ps.figure()
    for k, col, lab, mk in SERIES:
        pts = [(yy[i], FIT[k, c][:, j]) for i, c in enumerate(REFS) if len(FIT[k, c])]
        m = [np.exp(np.log(v).mean()) if logx else v.mean() for _, v in pts]
        lo = [mi - v.min() for mi, (_, v) in zip(m, pts)]; hi = [v.max() - mi for mi, (_, v) in zip(m, pts)]
        ax.errorbar(m, [p + off[k] for p, _ in pts], xerr=[lo, hi], fmt=mk, color=col, capsize=2, ls="none", label=lab)
    ax.set_yticks(yy, [C.CLASS_LABEL[c] for c in REFS]); ax.set_ylim(-0.6, len(REFS) - 0.4)
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ps.shared_legend(fig, ax, ncol=1)
    ps.save(fig, f"analysis/catalog_v2/{base}")
