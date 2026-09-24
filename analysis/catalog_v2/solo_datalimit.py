"""Are the solo references data-limited on the 5k pools? Train against validation MSE of each
reference's best trial per horizon (sweeps/solo_t<N>_<process>, and sweeps/ref_t<N>_<process>
for the three original references at 1000-4000), next to the joint arithmetic-mean runs
(runs/steps_t<N>_s*) on the same pools.
    python analysis/catalog_v2/solo_datalimit.py --collect > analysis/catalog_v2/solo_datalimit.json   (where the runs are)
    python analysis/catalog_v2/solo_datalimit.py                                                      (plots from the json)
The MSE is the post-training evaluation on the train and validation pools (`MSE (prepd)` lines of
out_0.log) of the best checkpoint, which training.es_load_best_model (default true, not overridden in
these sweeps) reloads before evaluation. On the signed pools of the solo_t sweeps that checkpoint was
selected on MSE + sign-head BCE (fixed in 6939d77); their train/validation ratio is unaffected, their
level is not the best-MSE one. Writes analysis/catalog_v2/solo_datalimit_a ... _f (one class per
panel: the two reference processes, validation solid, train dashed) and solo_datalimit_ratio
(validation over train against steps, every reference process and the joint run's class medians)."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
STEPS = [500, 1000, 2000, 4000, 8000]
REFS = {"tree 2->2": ["ee_aa", "uubar_uubar"], "resonant 2->2": ["ee_uu", "ee_ddbar"],
        "tree 2->3": ["ee_uug", "udbar_WpZZ"], "tree 2->4": ["ee_uugg", "udbar_WpZaa"],
        "positive 1-loop": ["uubar_ZaZ_nlo", "ee_bb_nlo"], "signed 1-loop": ["udbar_Wgg_nlo", "uubar_ddbara_nlo"]}
JSON = os.path.join(HERE, "solo_datalimit.json")
MSE = re.compile(r"MSE \(prepd\) (train|val)_?(\S*) (\S+): ([0-9.eE+-]+)")

def mse(log):
    """{process: {"train": x, "val": x}} from the post-training evaluation lines."""
    out = {}
    for m in MSE.finditer(open(log).read()):
        out.setdefault(m.group(3), {})[m.group(1)] = float(m.group(4))
    return out

if "--collect" in sys.argv:
    solo = {}
    for procs in REFS.values():
        for p in procs:
            for N in STEPS:
                sw = next((s for s in (f"solo_t{N}_{p}", f"ref_t{N}_{p}") if os.path.exists(os.path.join(ROOT, "sweeps", s, "summary.txt"))), None)
                if sw is None: continue
                best = re.search(r"^\s+(hp_\d+)\s+val_loss", open(os.path.join(ROOT, "sweeps", sw, "summary.txt")).read(), re.M)
                if not best: continue
                log = os.path.join(ROOT, "runs", sw, best.group(1).replace("hp_", "trial_"), "out_0.log")
                if os.path.exists(log):
                    v = mse(log).get(p)
                    if v: solo[f"{p}|{N}"] = dict(v, sweep=sw)
    joint = {}
    for N in (1000, 2000, 4000):
        for r in sorted(glob.glob(os.path.join(ROOT, "runs", f"steps_t{N}_s*"))):
            logs = sorted(glob.glob(os.path.join(r, "*", "out_0.log")))
            if logs: joint[os.path.basename(r)] = mse(logs[-1])
    print(json.dumps({"solo": solo, "joint": joint})); sys.exit()

import plot_style as ps
import census as C
D = json.load(open(JSON))
cls = json.load(open(os.path.join(HERE, "steps_agg.json")))["cls"].get
PCOL = [ps.C.blue, ps.C.vermillion]
from solo_datalimit_labels import LABEL

def series(p, split):
    t = [N for N in STEPS if f"{p}|{N}" in D["solo"]]
    return t, [D["solo"][f"{p}|{N}"][split] for N in t]

# (a-f) per class: validation (solid) and train (dashed) MSE of each reference process's best trial
figs = ps.panels(len(REFS))
for (fig, ax), (c, procs) in zip(figs, REFS.items()):
    for p, col in zip(procs, PCOL):
        t, v = series(p, "val"); ax.plot(t, v, marker="o", color=col, label=LABEL[p])
        t, v = series(p, "train"); ax.plot(t, v, marker="o", color=col, ls="--", mfc="none")
    ax.plot([], [], color=ps.C.grey, label="validation"); ax.plot([], [], color=ps.C.grey, ls="--", label="train")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(n) for n in STEPS]); ax.minorticks_off()
    ax.set_xlabel("training steps, trained alone"); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, C.CLASS_LABEL[c], loc="lower left")
    ps.legend(ax, "upper right")
ps.save_panels(figs, "analysis/catalog_v2/solo_datalimit")

# validation over train against steps: every reference process, and the joint run's class medians
CCOL = dict(zip(REFS, [ps.C.blue, ps.C.sky, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple]))
fig, ax = ps.figure()
for c, procs in REFS.items():
    for i, p in enumerate(procs):
        t = [N for N in STEPS if f"{p}|{N}" in D["solo"]]
        r = [D["solo"][f"{p}|{N}"]["val"] / D["solo"][f"{p}|{N}"]["train"] for N in t]
        ax.plot(t, r, marker="o", color=CCOL[c], alpha=0.8, label=C.CLASS_LABEL[c] if i == 0 else None)
jt = [1000, 2000, 4000]
jr = []
for N in jt:
    runs = [v for k, v in D["joint"].items() if k.startswith(f"steps_t{N}_s")]
    jr.append(np.mean([np.median([x["val"] / x["train"] for n, x in r.items() if "train" in x and "val" in x and cls(n)]) for r in runs]))
ax.plot(jt, jr, marker="s", color="black", label="joint run")
ax.axhline(1, color=ps.C.grey, ls=":", label="validation = train")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(n) for n in STEPS]); ax.minorticks_off()
ax.set_xlabel("training steps"); ax.set_ylabel(r"validation MSE / train MSE")
ps.legend(ax, "upper left")
ps.save(fig, "analysis/catalog_v2/solo_datalimit_ratio")
