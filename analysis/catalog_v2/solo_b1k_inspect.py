"""The solo references at bs 1024 on the full pools (sweeps/solob1k_t<S>_<process>; the two signed pools
from their rerun solob1kv_t<S>_<process>, solo_b1k.py), trial by trial.
    python analysis/catalog_v2/solo_b1k_inspect.py --collect > analysis/catalog_v2/solo_b1k_inspect.json   (where the runs are)
    python analysis/catalog_v2/solo_b1k_inspect.py                                                         (plots from the json)
Per trial: its HPs (config.yaml), the DyHPO result (best validation loss, results/hp*_t<S>_*.json),
the validation curve (the "Val loss: ... | step N" lines of out_0.log) and the post-training MSE on
the train and validation pools. Single-process runs write no per_process_metrics.json, so the log
is the record. Writes
  solo_b1k_scaling_{1,2}_a..f every trial's result against steps, coloured by lr; the best per step
                              count joined; the floor-aware fit A C^-alpha + L_inf over the best
  solo_b1k_lr                 the best trial's lr per step count, every process, with the search window
  solo_b1k_curves_{1,2}_a..f  the best trial's validation curve at each step count, final train MSE marked
  solo_b1k_hp_a..e            each HP's effect: partial residuals of log10 MSE over all trials (sweep
                              fixed effects, a quadratic in log lr with its optimum per step count,
                              linear in the others), with the share of within-sweep variance it explains
Every value is the trial's best validation MSE, the DyHPO result (CLAUDE.md, Reported values). The
post-training MSE (train circle, val cross) is the best checkpoint's, reloaded before evaluation
(training.es_load_best_model), on the evaluation subsample."""
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

from solo_b1k import SIGNED, SWEEP
if "--collect" in sys.argv:
    import yaml
    out = {}
    VAL = re.compile(r"Val loss: ([0-9.eE+-]+) \| step (\d+)")
    MSE = re.compile(r"MSE \(prepd\) (train|val)\S* \S+: ([0-9.eE+-]+)")
    for p in PROCS:
        for S in STEPS:
            sw = SWEEP(p, S)
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
for v in D.values():      # a trial's value: its best validation MSE, the DyHPO result (solo_b1k)
    for t in v: t["mse"] = t["val_loss"]
best = {k: min(v, key=lambda t: t["mse"]) for k, v in D.items()}
print(f"{sum(len(v) for v in D.values())} trials over {len(D)} sweeps")

# (1) scaling per process, every trial, coloured by lr
norm = mpl.colors.LogNorm(*LR_WINDOW); cmap = mpl.cm.viridis
figs = ps.panels(len(PROCS))
print(f"{'process':18s} alpha   L_inf   | best per step count (val loss / lr)")
for (fig, ax), p in zip(figs, PROCS):
    tr = [t for S in STEPS for t in D.get(f"{p}|{S}", []) if "lr" in t]
    sc = ax.scatter([t["S"] for t in tr], [t["mse"] for t in tr], c=[t["lr"] for t in tr], cmap=cmap, norm=norm, s=14, zorder=3)
    b = [best[f"{p}|{S}"] for S in STEPS if f"{p}|{S}" in best]
    ax.plot([t["S"] for t in b], [t["mse"] for t in b], color="black", zorder=2, label="best trial per step count")
    f = fit_power_law_with_floor([t["S"] for t in b], [t["mse"] for t in b])
    if f:
        g = np.geomspace(STEPS[0] / 1.3, STEPS[-1] * 1.3, 100)
        ax.plot(g, f[0] * g ** -f[1] + f[2], color=ps.C.vermillion, ls="--", zorder=1, label=r"fit $A\,S^{-\alpha}+L_\infty$")
    print(f"{p:18s} {f[1] if f else float('nan'):5.2f}  {f[2] if f else float('nan'):7.2g} | "
          + "  ".join(f"{t['S']}:{t['mse']:.2g}/{t.get('lr', float('nan')):.1g}" for t in b))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(STEPS, [str(s) for s in STEPS]); ax.minorticks_off()
    ax.set_xlabel("training steps, bs 1024"); ax.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL[p], loc="lower left")
    ps.legend(ax, "upper right")
    ps.colorbar(ax, sc, "learning rate")
for i in (0, 1):   # six panels per file set (save_panels letters a-f)
    ps.save_panels(figs[6 * i:6 * i + 6], f"analysis/catalog_v2/solo_b1k_scaling_{i + 1}")

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
ps.legend(ax, "lower right")
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
        if "final_val" in t:
            ax.plot([S], [t["final_val"]], marker="x", color=col, ls="none")
    ax.plot([], [], marker="o", mfc="none", color=ps.C.grey, ls="none", label="train, end")
    ax.plot([], [], marker="x", color=ps.C.grey, ls="none", label="val, end")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, LABEL[p], loc="lower left")
    ps.legend(ax, "upper right", ncol=2)
for i in (0, 1):
    ps.save_panels(figs[6 * i:6 * i + 6], f"analysis/catalog_v2/solo_b1k_curves_{i + 1}")

# (4) what each HP does: log10 MSE of every trial against sweep fixed effects + a quadratic in log lr
# (shared curvature, optimum per step count) + linear terms in log lambda, warm-up, log eta_min, EMA
# on and log(1 - decay) when on. Partial residual of an HP = the residual plus that HP's own term.
R = [t for v in D.values() for t in v if "lr" in t]
sws = sorted({f"{t['p']}|{t['S']}" for t in R})
y = np.log10([t["mse"] for t in R])
lr = np.log10([t["lr"] for t in R])
TERMS = {"lr": np.column_stack([lr * (np.array([t["S"] for t in R]) == S) for S in STEPS] + [lr ** 2]),
         "lam": np.log10([t["lam"] for t in R])[:, None], "wu": np.array([t["wu"] for t in R])[:, None],
         "eta": np.log10([t["eta"] for t in R])[:, None],
         "ema": np.column_stack([[float(t["ema"]) for t in R], [float(t["ema"]) * np.log10(1 - t["ema_decay"]) for t in R]])}
FE = np.array([[f"{t['p']}|{t['S']}" == s for s in sws] for t in R], float)
def lsq(names):
    X = np.column_stack([FE] + [TERMS[n] for n in names]); c = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, c, ((y - X @ c) ** 2).sum()
X, c, rss = lsq(list(TERMS))
within = sum(((y[[i for i, t in enumerate(R) if f"{t['p']}|{t['S']}" == s]] - y[[i for i, t in enumerate(R) if f"{t['p']}|{t['S']}" == s]].mean()) ** 2).sum() for s in sws)
res = y - X @ c
off = FE.shape[1]; part = {}
for n, M in TERMS.items():
    k = M.shape[1]; part[n] = (M @ c[off:off + k], 100 * (lsq([m for m in TERMS if m != n])[2] - rss) / within); off += k
print(f"within-sweep sd of log10 MSE {np.sqrt(within / len(y)):.2f} dex; unexplained {np.sqrt(rss / len(y)):.2f} dex")
a = c[FE.shape[1] + len(STEPS)]
lopt = {S: -c[FE.shape[1] + i] / (2 * a) for i, S in enumerate(STEPS)}
S_ = np.array([t["S"] for t in R])
XH = {"lr": (lr - np.array([lopt[s] for s in S_]), r"$\log_{10}(\eta/\eta^*_S)$"),
      "lam": (TERMS["lam"][:, 0], r"$\log_{10}\lambda$"), "wu": (TERMS["wu"][:, 0], "warm-up fraction"),
      "eta": (TERMS["eta"][:, 0], r"$\log_{10}\eta_{\min}$"),
      "ema": (np.array([np.log10(1 - t["ema_decay"]) if t["ema"] else 0.5 for t in R]), r"$\log_{10}(1-$EMA decay$)$, off at 0.5")}
part["lr"] = (a * XH["lr"][0] ** 2, part["lr"][1])   # about the optimum; the per-step-count constants sit in the fixed effects
figs = ps.panels(len(XH))
for (fig, ax), (n, (x, lab)) in zip(figs, XH.items()):
    ax.scatter(x, res + part[n][0] - np.mean(part[n][0]), s=8, color=ps.C.blue, alpha=0.5, label="trials")
    o = np.argsort(x)
    ax.plot(x[o], (part[n][0] - np.mean(part[n][0]))[o], color=ps.C.vermillion,
            label=rf"fit (${part[n][1]:.0f}\%$ of var.)")
    ax.set_xlabel(lab); ax.set_ylabel(r"partial residual, $\log_{10}$ MSE")
    ps.legend(ax, "upper right")
    print(f"  {n:4s} explains {part[n][1]:5.1f}% of within-sweep variance")
print("lr optimum per step count (log10):", {S: round(v, 2) for S, v in lopt.items()})
ps.save_panels(figs, "analysis/catalog_v2/solo_b1k_hp")
