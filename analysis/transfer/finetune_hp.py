"""Transfer study, the HP landscape of the fine-tune grids (every *fte cell at the grid's horizons: rungs 1-9, ee_uu at 32k
and 64k, the star arms; the grid's protocol, docs/results.tex sec:ladder: 5 trials, 3 random start-up then 2 DyHPO-guided,
lr searched over [1e-3, 1e-2] with the common space), to judge whether the protocol can run fewer trials or steps.
Data: analysis/transfer/finetune_hp.json (collect_sweeps.py tp3_ on every site, merged; a cell's trials in evaluation
order). Every ratio: val_loss_no_reg at the trial's best checkpoint over the best of its cell (same probe, D, parent).
D = 10^(k/2).
  <base>_landscape  per k, the median ratio in bins of lr, lambda, warm-up fraction, and against EMA (on/off)
  <base>_protocol   (a) the best of the first n trials over the best of five, median and 90th percentile, per k;
                    (b) where the best checkpoint sits in the horizon, and the first validation within 1.1x of it
    python analysis/transfer/finetune_hp.py
"""
import json, os, re, sys
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

F = json.load(open(os.path.join(ROOT, "analysis", "transfer", "finetune_hp.json")))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs", "finetune_hp")
F = {n: v for n, v in F.items() if len(v) >= 5}
kof = lambda n: int(re.search(r"_d(\d+)$", n).group(1))
K = sorted({kof(n) for n in F})
COL = dict(zip(K, ps.sequence(len(K))))
lab = lambda k: rf"$D=10^{{{k / 2:g}}}$"
YL = r"$\mathcal{L}_{\rm val}/\mathcal{L}_{\rm val}^{\rm cell\ best}$, median"
def plain_log(ax, axis, ticks, fmt):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks)); a.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    a.set_minor_locator(NullLocator()); a.set_minor_formatter(NullFormatter())


rows = [(kof(n), t, t["val_loss"] / min(x["val_loss"] for x in v)) for n, v in F.items() for t in v]


def binned(ax, key, edges, logx):
    for k in K:
        x = np.array([t[key] for kk, t, r in rows if kk == k]); y = np.array([r for kk, t, r in rows if kk == k])
        xc, ym = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 10:
                xc.append(np.sqrt(lo * hi) if logx else (lo + hi) / 2); ym.append(np.median(y[m]))
        ax.plot(xc, ym, "o-", color=COL[k], label=lab(k))
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log"); ax.set_ylabel(YL)
    plain_log(ax, "y", [1, 1.5, 2, 3, 4], lambda v: f"{v:g}")


fig, axes = ps.figure(ncols=2, nrows=2)
binned(axes[0, 0], "lr", np.geomspace(1e-3, 1e-2, 9), True); axes[0, 0].set_xlabel("lr")
plain_log(axes[0, 0], "x", [1e-3, 2e-3, 5e-3, 1e-2], lambda v: rf"${v * 1e3:g}\times10^{{-3}}$" if v < 1e-2 else r"$10^{-2}$")
binned(axes[0, 1], "lambda", np.geomspace(1e-10, 1e-6, 9), True); axes[0, 1].set_xlabel(r"$\lambda$")
binned(axes[1, 0], "warmup", np.linspace(0.05, 0.2, 7), False); axes[1, 0].set_xlabel("warm-up fraction")
ax = axes[1, 1]
for e, mk, name in ((True, "o-", "EMA on"), (False, "s--", "EMA off")):
    ax.plot(K, [np.median([r for kk, t, r in rows if kk == k and t["ema"] == e]) for k in K], mk, color="k", label=name)
ax.set_xlabel(r"$k$, $D=10^{k/2}$"); ax.set_yscale("log"); ax.set_ylabel(YL)
plain_log(ax, "y", [1, 1.2, 1.5, 2, 2.5], lambda v: f"{v:g}")
ps.legend(ax, "upper left"); ps.make_room(ax)
ps.shared_legend(fig, axes[0, 0], ncol=4)
ps.save(fig, FIG + "_landscape")

fig, (a, b) = ps.figure(ncols=2)
for k in K:
    reg = np.array([[min(t["val_loss"] for t in v[:m]) / min(t["val_loss"] for t in v) for m in range(1, 6)]
                    for n, v in F.items() if kof(n) == k])
    a.plot(range(1, 6), np.median(reg, 0), "o-", color=COL[k], label=lab(k))
    a.plot(range(1, 6), np.quantile(reg, 0.9, 0), "o:", color=COL[k])
a.plot([], [], "-", color="k", label="median"); a.plot([], [], ":", color="k", label="90th percentile")
a.set_xlabel("trials run, in evaluation order"); a.set_yscale("log")
a.set_ylabel(r"$\mathcal{L}_{\rm val}^{{\rm best\ of\ first}\ n}/\mathcal{L}_{\rm val}^{\rm best\ of\ 5}$")
a.set_xticks(range(1, 6)); plain_log(a, "y", [1, 2, 5, 10], lambda v: f"{v:g}")
for key, mk, name in (("best_step", "o", "best checkpoint"), ("reach110", "s", r"first validation within $1.1\times$ of it")):
    med, lo, hi = [], [], []
    for k in K:
        best = [min(v, key=lambda t: t["val_loss"]) for n, v in F.items() if kof(n) == k]
        f = np.array([(t["best_step"] / t["T"]) if key == "best_step" else t["reach110"] for t in best
                      if t.get(key) is not None])
        med.append(np.median(f)); lo.append(np.quantile(f, 0.1)); hi.append(np.quantile(f, 0.9))
    b.errorbar(np.array(K) + (0.12 if key == "reach110" else -0.12), med,
               yerr=[np.array(med) - lo, np.array(hi) - med], fmt=mk, color="k", mfc="none" if key == "reach110" else "k",
               capsize=2, label=name)
b.set_xlabel(r"$k$, $D=10^{k/2}$"); b.set_ylabel("fraction of the horizon, best trial")
ps.legend(b, "upper left"); ps.make_room(b)
ps.shared_legend(fig, a, ncol=5)
ps.save(fig, FIG + "_protocol")

for k in K:
    v = [v for n, v in F.items() if kof(n) == k]
    print(f"k={k}: {len(v)} cells, GPU-h {sum(t['hours'] or 0 for c in v for t in c):.0f}")
