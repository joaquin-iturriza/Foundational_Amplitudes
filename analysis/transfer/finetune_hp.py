"""Transfer study, the HP landscape of the fine-tune grids (every *fte cell at the grid's horizons, lr_scale and layer_decay fixed at 1 (the fine-tune lr is searched): rungs 1-9, ee_uu (64k), the
star arms; the grid's protocol, docs/results.tex sec:ladder: 5 trials, 3 random start-up then 2 DyHPO-guided,
lr searched over [1e-3, 1e-2] with the common space), to judge whether the protocol can run fewer trials or steps.
Data: analysis/transfer/finetune_hp.json (collect_sweeps.py tp3_ on every site, merged; a cell's trials in evaluation
order). Every ratio: val_loss_no_reg at the trial's best checkpoint over the best of its cell (same probe, D, parent).
D = 10^(k/2).
  <base>_landscape_a..f  per D, the median ratio in bins of lr, lambda, warm-up, eta_min, EMA decay (EMA-on trials), and
                    for EMA off and on; bars: 16th-84th percentile over 200 resamples of the cells
  <base>_protocol   (a) the best of the first n trials over the best of five, median and 90th percentile, per k (bars as above);
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
# the study's final setup only: the 32k-step ee->uu parent (tp3_uufte, replaced by the 64k one) and the two first-design
# star arms (resonance, Sudakov; replaced) are left out
F = {n: v for n, v in F.items() if not n.startswith(("tp3_uufte_", "tp3_sresonancefte_", "tp3_ssudakovfte_"))}
FIG = os.path.join(ROOT, "analysis", "transfer", "figs", "finetune_hp")
# a cell is ONE sweep's first five trials in evaluation order (the protocol: three random start-up, two guided). A cell
# run on two sites (started on one, generated again on the other by a move) holds two independent DyHPO runs: the site
# with the most results is the cell, the other's trials are left out. A cell whose sweep has fewer than five results is
# listed, not used.
def one_sweep(v):
    by = {}
    for t in v:
        by.setdefault(t["site"], []).append(t)
    site = max(by, key=lambda s: len(by[s]))
    return sorted(by[site], key=lambda t: t["order"] if t["order"] is not None else 99), len(by) > 1


SPLIT, SHORT, LONG = [], {}, {}
G = {}
for n, v in F.items():
    w, split = one_sweep(v)
    if split:
        SPLIT.append(n)
    if len(w) < 5:
        SHORT[n] = len(w)
        continue
    if len(w) > 5:
        LONG[n] = len(w)
    G[n] = w[:5]
F = G
# the protocol questions (best of the first n, start-up against guided) need cells that ran it: three start-up trials,
# then two guided. A cell whose start-up trials failed and were replaced (refills after the EOS quota) is not one.
CLEAN = {n: v for n, v in F.items() if [bool(t["startup"]) for t in v] == [True, True, True, False, False]}
print(f"{len(CLEAN)} of {len(F)} cells ran three start-up then two guided trials (the protocol panel uses those)")
print(f"{len(F)} cells; run on two sites, one site's sweep kept: " + ", ".join(sorted(SPLIT)))
print("cut to their first five trials: " + ", ".join(f"{n} ({k})" for n, k in sorted(LONG.items())))
print("not used, fewer than five results in one sweep: " + ", ".join(f"{n} ({k})" for n, k in sorted(SHORT.items())))
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


NB = 200                                                    # bootstrap resamples, over cells
RNG = np.random.default_rng(0)
cells_k = {k: [v for n, v in F.items() if kof(n) == k] for k in K}


def boot(k, stat, cs=None):
    """stat(list of cells) -> array; its value and its 16th/84th percentiles over cell resamples."""
    cs = cells_k[k] if cs is None else cs
    b = np.array([stat([cs[i] for i in RNG.integers(0, len(cs), len(cs))]) for _ in range(NB)])
    return stat(cs), np.nanpercentile(b, 16, 0), np.nanpercentile(b, 84, 0)


def ratios_of(cs, sel=lambda t: True):
    return [(t, t["val_loss"] / min(x["val_loss"] for x in v)) for v in cs for t in v if sel(t)]


def binned(ax, key, edges, logx, sel=lambda t: True):
    for k in K:
        def stat(cs):
            tr = ratios_of(cs, sel)
            x = np.array([t[key] for t, _ in tr]); y = np.array([r for _, r in tr])
            return np.array([np.median(y[(x >= lo) & (x < hi)]) if ((x >= lo) & (x < hi)).sum() >= 10 else np.nan
                             for lo, hi in zip(edges[:-1], edges[1:])])
        m, lo, hi = boot(k, stat)
        xc = np.sqrt(edges[:-1] * edges[1:]) if logx else (edges[:-1] + edges[1:]) / 2
        ok = ~np.isnan(m)
        ax.errorbar(xc[ok], m[ok], yerr=[m[ok] - lo[ok], hi[ok] - m[ok]], fmt="o-", color=COL[k], capsize=2, label=lab(k))
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log"); ax.set_ylabel(YL)
    plain_log(ax, "y", [1, 1.5, 2, 3, 4, 5], lambda v: f"{v:g}")


figs = ps.panels(6)
(_, a0), (_, a1), (_, a2), (_, a3), (_, a4), (_, a5) = figs
binned(a0, "lr", np.geomspace(1e-3, 1e-2, 9), True); a0.set_xlabel("lr (fine-tune)")
plain_log(a0, "x", [1e-3, 2e-3, 5e-3, 1e-2], lambda v: rf"${v * 1e3:g}\times10^{{-3}}$" if v < 1e-2 else r"$10^{-2}$")
binned(a1, "lambda", np.geomspace(1e-10, 1e-6, 9), True); a1.set_xlabel(r"$\lambda$")
binned(a2, "warmup", np.linspace(0.05, 0.2, 7), False); a2.set_xlabel("warm-up fraction")
binned(a3, "eta_min", np.geomspace(1e-10, 1e-7, 7), True); a3.set_xlabel(r"$\eta_{\min}$")
binned(a4, "ema_decay", np.linspace(0.9, 0.999, 7), False, sel=lambda t: t["ema"]); a4.set_xlabel("EMA decay (EMA on)")
for k in K:                                                 # EMA off / on, as the other panels: the HP on x, a line per D
    m, lo, hi = boot(k, lambda cs: np.array([np.median([r for _, r in ratios_of(cs, lambda t: t["ema"] == e)])
                                             for e in (False, True)]))
    a5.errorbar([0, 1], m, yerr=[m - lo, hi - m], fmt="o-", color=COL[k], capsize=2)
a5.set_xticks([0, 1], ["off", "on"]); a5.set_xlim(-0.4, 1.4); a5.set_xlabel("EMA")
a5.set_yscale("log"); a5.set_ylabel(YL); plain_log(a5, "y", [1, 1.5, 2, 3], lambda v: f"{v:g}")
for _, ax in figs:
    ps.make_room(ax)
a0.plot([], [], " ", label=r"bars: $68\%$ bootstrap over cells")
ps.legend_strip(a0, FIG + "_landscape_legend", ncol=4)
ps.save_panels(figs, FIG + "_landscape")

fig, (a, b) = ps.figure(ncols=2)
cells_clean = {k: [v for n, v in CLEAN.items() if kof(n) == k] for k in K}
for k in K:
    reg = lambda cs: np.array([[min(t["val_loss"] for t in v[:m]) / min(t["val_loss"] for t in v) for m in range(1, 6)]
                               for v in cs])
    for q, mk in ((0.5, "o-"), (0.9, "o:")):
        m, lo, hi = boot(k, lambda cs: np.quantile(reg(cs), q, 0), cells_clean[k])
        a.errorbar(np.arange(1, 6) + (0.08 if q == 0.9 else 0), m, yerr=[m - lo, hi - m], fmt=mk, color=COL[k], capsize=2,
                   label=lab(k) if q == 0.5 else None)
a.plot([], [], "-", color="k", label="median"); a.plot([], [], ":", color="k", label="90th percentile")
a.set_xlabel("trials run, in evaluation order"); a.set_yscale("log")
a.set_ylabel(r"$\mathcal{L}_{\rm val}^{{\rm best\ of\ first}\ n}/\mathcal{L}_{\rm val}^{\rm best\ of\ 5}$")
a.set_xticks(range(1, 6)); plain_log(a, "y", [1, 2, 5, 10], lambda v: f"{v:g}")
for key, mk, name in (("best_step", "o", "best checkpoint"), ("reach110", "s", r"first validation within $1.1\times$ of it")):
    med, lo, hi = [], [], []
    for k in K:
        best = [min(v, key=lambda t: t["val_loss"]) for n, v in F.items() if kof(n) == k]   # any one-sweep cell
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
