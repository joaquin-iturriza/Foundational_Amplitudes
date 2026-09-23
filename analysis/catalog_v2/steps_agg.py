"""The steps curve under three training aggregations (docs/results.tex, tab:steps_agg).
Joint runs runs/steps_t<N>_s* (arithmetic mean), runs/steps_geo_t<N>_s* (geometric mean),
runs/steps_tau1e-2_t<N>_s* (geometric mean floored at tau = 1e-2); solo references
sweeps/ref_t<N>_<process> as in joint_vs_solo.py.
    python analysis/catalog_v2/steps_agg.py --collect > analysis/catalog_v2/steps_agg.json   (where the runs are)
    python analysis/catalog_v2/steps_agg.py                                                (plots from the json)
A run whose final combined loss is above 0.5 is marked diverged and left out of the class curves
and the distribution; the curves figure shows it.
Writes analysis/catalog_v2/steps_agg_a ... _f (class median against steps, one panel per class),
steps_agg_ecdf (per-process loss at the longest horizon), steps_agg_curves (combined validation
loss during training at the longest horizon)."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import census as C
STEPS = [1000, 2000, 4000]
ARMS = [("arith", "steps_", "arithmetic mean"), ("geo", "steps_geo_", "geometric mean"),
        ("tau", "steps_tau1e-2_", r"geometric, $\tau=10^{-2}$")]
REFP = {4: "ee_uu", 5: "ee_uug", 6: "ee_uugg"}
JSON = os.path.join(HERE, "steps_agg.json")
NP = json.load(open(os.path.join(HERE, "n_particles.json")))


if "--collect" in sys.argv:   # the class needs signed_pools.csv, which lives where the pools are
    s27, all50 = C.signed_classes()
    def cls(n):
        if n in all50: return "signed 1-loop"
        if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
        if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
        return f"tree 2->{NP[n]-2}"
    out = {"runs": [], "solo": {}, "cls": {n: cls(n) for n in NP}}
    for key, pre, _ in ARMS:
        for N in STEPS:
            for r in sorted(glob.glob(os.path.join(ROOT, "runs", f"{pre}t{N}_s*"))):
                js = sorted(glob.glob(os.path.join(r, "**", "per_process_metrics.json"), recursive=True))
                if not js: continue
                d = json.load(open(js[-1]))
                out["runs"].append({"arm": key, "steps": N, "seed": int(r.rsplit("_s", 1)[1]),
                                    "every": d["validate_every_n_steps"], "combined": d["val_loss_no_reg"],
                                    "final": {n: v[-1] for n, v in d["proc_val_losses_no_reg"].items() if v and n in NP}})
    for N in STEPS:
        for k, p in REFP.items():
            f = os.path.join(ROOT, "sweeps", f"ref_t{N}_{p}", "summary.txt")
            v = [float(m.group(1)) for m in re.finditer(r"val_loss=([0-9.eE+-]+)", open(f).read())] if os.path.exists(f) else []
            out["solo"][f"{N}_{k}"] = min(v) if v else None
    print(json.dumps(out)); sys.exit()

import plot_style as ps
SOLO_K = {"tree 2->2": 4, "resonant 2->2": 4, "tree 2->3": 5, "tree 2->4": 6}
COL = {"arith": ps.C.blue, "geo": ps.C.vermillion, "tau": ps.C.green}
D = json.load(open(JSON))
cls = D["cls"].get
runs = D["runs"]
for r in runs: r["diverged"] = float(np.median(list(r["final"].values()))) > 0.5
div = [r for r in runs if r["diverged"]]
print("diverged: " + (", ".join(f"{r['arm']} t{r['steps']} s{r['seed']}" for r in div) or "none"))
good = [r for r in runs if not r["diverged"]]
seeds = max(r["seed"] for r in runs)

# (a-f) class median against steps, band = min to max over seeds, solo reference where one exists
figs = ps.panels(len(C.CLASSES))
for (fig, ax), c in zip(figs, C.CLASSES):
    for key, _, label in ARMS:
        m, lo, hi = [], [], []
        for N in STEPS:
            v = [np.median([x for n, x in r["final"].items() if cls(n) == c]) for r in good if r["arm"] == key and r["steps"] == N]
            m.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
        ax.plot(STEPS, m, marker="o", color=COL[key], label=label)
        ax.fill_between(STEPS, lo, hi, color=COL[key], alpha=0.2)
    if c in SOLO_K:
        k = SOLO_K[c]
        ax.plot(STEPS, [D["solo"][f"{N}_{k}"] for N in STEPS], marker="s", color=ps.C.grey, ls="--",
                label=rf"$2\to{k-2}$ alone, same compute")
    ax.fill_between([], [], [], color=ps.C.grey, alpha=0.2, label=f"min to max over {seeds} seeds")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(STEPS, [str(n) for n in STEPS]); ax.minorticks_off()
    ax.set_xlabel("training steps"); ax.set_ylabel(r"median MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.shared_legend(fig, ax, ncol=1)
ps.save_panels(figs, "analysis/catalog_v2/steps_agg")

# distribution of the per-process loss at the longest horizon: seed geometric mean per process
T = STEPS[-1]
fig, ax = ps.figure()
for key, _, label in ARMS:
    rs = [r for r in good if r["arm"] == key and r["steps"] == T]
    names = rs[0]["final"].keys()
    v = np.sort([np.exp(np.mean([np.log(r["final"][n]) for r in rs])) for n in names])
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COL[key], label=f"{label} ({len(rs)} seeds)")
ax.set_xscale("log"); ax.set_xlabel(r"per-process MSE($\log|\mathcal{M}|^2$)")
ax.set_ylabel("fraction of processes")
ps.shared_legend(fig, ax, ncol=1)
ps.save(fig, "analysis/catalog_v2/steps_agg_ecdf")

# combined validation loss during training at the longest horizon, every seed; the diverged run dashed
fig, ax = ps.figure()
for key, _, label in ARMS:
    first = True
    for r in [r for r in runs if r["arm"] == key and r["steps"] == T]:
        x = r["every"] * np.arange(1, len(r["combined"]) + 1)
        if r["diverged"]:
            ax.plot(x, r["combined"], color=COL[key], ls="--", label=f"{label}, diverged seed")
        else:
            ax.plot(x, r["combined"], color=COL[key], label=label if first else None); first = False
ax.set_yscale("log"); ax.set_xlabel("training step")
ax.set_ylabel(r"validation GM$_p$ MSE($\log|\mathcal{M}|^2$)")
ps.shared_legend(fig, ax, ncol=1)
ps.save(fig, "analysis/catalog_v2/steps_agg_curves")
