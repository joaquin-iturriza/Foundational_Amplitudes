"""The solo references at bs 1024 on the full pools (sweeps/solob1k_t<S>_<process>), trial by trial.
    python analysis/catalog_v2/solo_b1k_inspect.py --collect > analysis/catalog_v2/solo_b1k_inspect.json   (where the runs are)
    python analysis/catalog_v2/solo_b1k_inspect.py                                                         (plots from the json)
Per trial: its HPs (config.yaml), the DyHPO result (best validation loss, results/hp*_t<S>_*.json),
the validation curve (the "Val loss: ... | step N" lines of out_0.log) and the post-training MSE on
the train and validation pools. Single-process runs write no per_process_metrics.json, so the log
is the record. Writes
  solo_b1k_scaling_a ... _l   every trial's result against steps, coloured by lr; the best per step
                              count joined; the floor-aware fit A C^-alpha + L_inf over the best
  solo_b1k_lr                 the best trial's lr per step count, every process, with the search window
  solo_b1k_curves_a ... _l    the best trial's validation curve at each step count, final train MSE marked"""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
STEPS = [33, 67, 134, 268, 536, 1072]
PROCS = ["ee_aa", "uubar_uubar", "ee_uu", "ee_ddbar", "ee_uug", "udbar_WpZZ", "ee_uugg", "udbar_WpZaa",
         "uubar_ZaZ_nlo", "ee_bb_nlo", "udbar_Wgg_nlo", "uubar_ddbara_nlo"]
KIND = {"ee_aa": "tree", "uubar_uubar": "tree", "ee_uu": "resonant", "ee_ddbar": "resonant", "ee_uug": "tree",
        "udbar_WpZZ": "tree", "ee_uugg": "tree", "udbar_WpZaa": "tree", "uubar_ZaZ_nlo": "positive one-loop",
        "ee_bb_nlo": "positive one-loop", "udbar_Wgg_nlo": "signed one-loop", "uubar_ddbara_nlo": "signed one-loop"}
JSON = os.path.join(HERE, "solo_b1k_inspect.json")
LR_WINDOW = (3e-4, 3e-2)

if "--collect" in sys.argv:
    import yaml
    out = {}
    VAL = re.compile(r"Val loss: ([0-9.eE+-]+) \| step (\d+)")
    MSE = re.compile(r"MSE \(prepd\) (train|val)\S* \S+: ([0-9.eE+-]+)")
    for p in PROCS:
        for S in STEPS:
            sw = f"solob1k_t{S}_{p}"
            for f in glob.glob(os.path.join(ROOT, "sweeps", sw, "results", f"hp*_t{S}_*.json")):
                hp = int(re.search(r"hp(\d+)_", os.path.basename(f)).group(1))
                rd = os.path.join(ROOT, "runs", sw, f"trial_{hp:04d}")
                t = {"p": p, "S": S, "hp": hp, "val_loss": json.load(open(f))["val_loss"]}
                if os.path.exists(os.path.join(rd, "config.yaml")):
                    c = yaml.safe_load(open(os.path.join(rd, "config.yaml"))); tr = c["training"]
                    t.update(lr=tr["lr"], lam=tr["regularization_lambda"], wu=tr["cosanneal_warmup_frac"],
                             eta=tr.get("cosanneal_eta_min"), ema=bool(c.get("ema")), ema_decay=tr.get("ema_decay"))
                log = os.path.join(rd, "out_0.log")
                if os.path.exists(log):
                    txt = open(log).read()
                    t["curve"] = [(int(m.group(2)), float(m.group(1))) for m in VAL.finditer(txt)]
                    for m in MSE.finditer(txt):
                        t.setdefault("final_" + m.group(1), float(m.group(2)))
                out.setdefault(f"{p}|{S}", []).append(t)
    print(json.dumps(out)); sys.exit()

import plot_style as ps
import matplotlib as mpl
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from analyze_pretraining_scaling import fit_power_law_with_floor
from solo_datalimit_labels import LABEL
D = json.load(open(JSON))
best = {k: min(v, key=lambda t: t["val_loss"]) for k, v in D.items()}
print(f"{sum(len(v) for v in D.values())} trials over {len(D)} sweeps")

# (1) scaling per process, every trial, coloured by lr
norm = mpl.colors.LogNorm(*LR_WINDOW); cmap = mpl.cm.viridis
figs = ps.panels(len(PROCS))
print(f"{'process':18s} alpha   L_inf   | best per step count (val loss / lr)")
for (fig, ax), p in zip(figs, PROCS):
    tr = [t for S in STEPS for t in D.get(f"{p}|{S}", []) if "lr" in t]
    sc = ax.scatter([t["S"] for t in tr], [t["val_loss"] for t in tr], c=[t["lr"] for t in tr], cmap=cmap, norm=norm, s=14, zorder=3)
    b = [best[f"{p}|{S}"] for S in STEPS if f"{p}|{S}" in best]
    ax.plot([t["S"] for t in b], [t["val_loss"] for t in b], color="black", zorder=2, label="best trial per step count")
    f = fit_power_law_with_floor([t["S"] for t in b], [t["val_loss"] for t in b])
    if f:
        g = np.geomspace(STEPS[0] / 1.3, STEPS[-1] * 1.3, 100)
        ax.plot(g, f[0] * g ** -f[1] + f[2], color=ps.C.vermillion, ls="--", zorder=1, label=r"fit $A\,S^{-\alpha}+L_\infty$")
    print(f"{p:18s} {f[1] if f else float('nan'):5.2f}  {f[2] if f else float('nan'):7.2g} | "
          + "  ".join(f"{t['S']}:{t['val_loss']:.2g}/{t.get('lr', float('nan')):.1g}" for t in b))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(s) for s in STEPS]); ax.minorticks_off()
    ax.set_xlabel("training steps, bs 1024"); ax.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL[p], loc="lower left")
    ps.legend(ax, "upper right")
    ps.colorbar(ax, sc, "learning rate")
ps.save_panels(figs, "analysis/catalog_v2/solo_b1k_scaling")

# (2) the chosen lr per step count
fig, ax = ps.figure()
KCOL = {"tree": ps.C.blue, "resonant": ps.C.sky, "positive one-loop": ps.C.orange, "signed one-loop": ps.C.purple}
seen = set()
for p in PROCS:
    b = [best[f"{p}|{S}"] for S in STEPS if f"{p}|{S}" in best and "lr" in best[f"{p}|{S}"]]
    k = KIND[p]
    ax.plot([t["S"] for t in b], [t["lr"] for t in b], marker="o", color=KCOL[k], alpha=0.8, label=k if k not in seen else None)
    seen.add(k)
for y in LR_WINDOW:
    ax.axhline(y, color=ps.C.grey, ls=":")
ax.plot([], [], color=ps.C.grey, ls=":", label="search window")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(s) for s in STEPS]); ax.minorticks_off()
ax.set_xlabel("training steps, bs 1024"); ax.set_ylabel("learning rate of the best trial")
ps.legend(ax, "lower left")
ps.save(fig, "analysis/catalog_v2/solo_b1k_lr")

# (3) the best trial's validation curve at every step count, final train MSE marked
cols = ps.sequence(len(STEPS))
figs = ps.panels(len(PROCS))
for (fig, ax), p in zip(figs, PROCS):
    for S, col in zip(STEPS, cols):
        t = best.get(f"{p}|{S}")
        if not t or not t.get("curve"): continue
        x, y = zip(*t["curve"])
        ax.plot(x, y, color=col, label=str(S))
        if "final_train" in t:
            ax.plot([S], [t["final_train"]], marker="o", mfc="none", color=col, ls="none")
    ax.plot([], [], marker="o", mfc="none", color=ps.C.grey, ls="none", label="final train MSE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL[p], loc="lower left")
    ps.legend(ax, "upper right", ncol=2)
ps.save_panels(figs, "analysis/catalog_v2/solo_b1k_curves")
