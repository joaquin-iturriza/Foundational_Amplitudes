#!/usr/bin/env python3
"""Disentangle dataset-size vs iterations on the clean scaling_p 2D grid, and
show that the optimal-lr hump peaks at the convergence horizon (val_loss floor).

Ordered series (dataset size D, step budget t) are keyed by a colourbar rather than a
legend: at 11pt a five-entry legend does not fit a third of \\textwidth, and a colourbar
shows the ordering that a qualitative legend hides."""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

recs = json.load(open(sys.argv[1])); out = sys.argv[2]


def bv(r, k):
    h = r['hp_best'].get(k); return h['val'] if h else None


sp = [r for r in recs if r['family'] == 'scaling_p' and r['converged'] and bv(r, 'training.lr')]
Ds = sorted(set(r['n_train'] for r in sp if r['n_train']))
Ts = sorted(set(r['t_steps'] for r in sp if r['t_steps']))


def make_norm(values):
    """Shared LogNorm for a set of series values."""
    return plt.matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values))


def norm_colors(values, norm):
    """Colours read off the SAME norm the colourbar displays.

    Using ps.sequence() here instead spaces colours by INDEX, so on a non-log-uniform grid
    (good_t = 10,32,100,316,...) a curve is drawn at a colour that reads off the bar as a
    different value entirely. That bar keys the lr*(t,D) surface, so a misread row gives the
    wrong lr search window."""
    return [plt.get_cmap(ps.CMAP)(norm(v)) for v in values]


def ramp_bar(a, norm, label):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=ps.CMAP)
    ps.colorbar(a, sm, label)


# THREE separate panel files. As a 2x2 this left a hole where the fourth panel would go; as a
# 1x3 column it was three plots stacked down a narrow canvas, wasting the width. Separate files
# let results.tex put two on the first line and the third centred underneath. Each panel takes
# its own colourbar, horizontal and under it, which is what keeps a panel at ~3.1in so two fit.
figs = ps.panels(3)
ax = [f[1] for f in figs]
norm_D = make_norm(Ds)
cols_D = norm_colors(Ds, norm_D)

# A: lr vs t_steps, one line per dataset size
a = ax[0]
for i, D in enumerate(Ds):
    pts = collections.defaultdict(list)
    for r in sp:
        if r['n_train'] == D: pts[r['t_steps']].append(math.log10(bv(r, 'training.lr')))
    ts = sorted(pts); ys = [10**np.mean(pts[t]) for t in ts]
    a.plot(ts, ys, 'o-', color=cols_D[i])
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training steps $t$'); a.set_ylabel(r'optimal learning rate')
ramp_bar(a, norm_D, r'$D$')

# B: lr vs dataset size, one line per t_steps (only well-sampled t)
a = ax[1]
good_t = [t for t in Ts if sum(1 for r in sp if r['t_steps'] == t) >= 8]
norm_T = make_norm(good_t)
cols_g = norm_colors(good_t, norm_T)
for i, t in enumerate(good_t):
    pts = collections.defaultdict(list)
    for r in sp:
        if r['t_steps'] == t: pts[r['n_train']].append(math.log10(bv(r, 'training.lr')))
    ds = sorted(pts)
    if len(ds) < 3: continue
    ys = [10**np.mean(pts[d]) for d in ds]
    a.plot(ds, ys, 's-', color=cols_g[i])
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training set size $D$'); a.set_ylabel(r'optimal learning rate')
ramp_bar(a, norm_T, r'$t$')

# C: learning curves — best val loss vs t_steps per D, with the lr-peak t marked
a = ax[2]
for i, D in enumerate(Ds):
    pts = collections.defaultdict(list); lrp = collections.defaultdict(list)
    for r in sp:
        if r['n_train'] == D and r['best_val_loss'] and r['best_val_loss'] > 0:
            pts[r['t_steps']].append(r['best_val_loss'])
            lrp[r['t_steps']].append(math.log10(bv(r, 'training.lr')))
    ts = sorted(pts); ys = [np.median(pts[t]) for t in ts]
    a.plot(ts, ys, 'o-', color=cols_D[i])
    tpk = max(lrp, key=lambda t: np.mean(lrp[t]))       # t where the optimal lr peaks
    if tpk in pts:
        a.scatter([tpk], [np.median(pts[tpk])], s=90, marker='*', color=cols_D[i],
                  zorder=5, label=r'$t$ of peak lr' if i == 0 else None)
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training steps $t$'); a.set_ylabel(r'best validation loss')
a.legend(loc='upper right')
ramp_bar(a, norm_D, r'$D$')

base = out[:-4] if out.lower().endswith(('.png', '.pdf')) else out
ps.save_panels(figs, base)
