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

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

recs = json.load(open(sys.argv[1])); out = sys.argv[2]


def bv(r, k):
    h = r['hp_best'].get(k); return h['val'] if h else None


sp = [r for r in recs if r['family'] == 'scaling_p' and r['converged'] and bv(r, 'training.lr')]
Ds = sorted(set(r['n_train'] for r in sp if r['n_train']))
Ts = sorted(set(r['t_steps'] for r in sp if r['t_steps']))


def ramp_bar(a, values, label):
    """Colourbar keying an ordered set of lines, in place of a legend."""
    norm = plt.matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=ps.CMAP)
    cb = fig.colorbar(sm, ax=a, fraction=0.046, pad=0.03)
    cb.set_label(label)
    return norm


# 2x2 rather than 1x3: three panels each carrying a colourbar do not fit across
# \textwidth at 11pt (the bar lands on the next panel's y-label). The 4th cell is removed.
fig, axg = ps.figure(ncols=2, nrows=2, layout="constrained")
ax = [axg[0, 0], axg[0, 1], axg[1, 0]]
axg[1, 1].remove()
cols_D = ps.sequence(len(Ds))
cols_T = ps.sequence(len(Ts))

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
ramp_bar(a, Ds, r'$D$')

# B: lr vs dataset size, one line per t_steps (only well-sampled t)
a = ax[1]
good_t = [t for t in Ts if sum(1 for r in sp if r['t_steps'] == t) >= 8]
cols_g = ps.sequence(len(good_t))
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
ramp_bar(a, good_t, r'$t$')

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
ramp_bar(a, Ds, r'$D$')

base = out[:-4] if out.lower().endswith(('.png', '.pdf')) else out
ps.save(fig, base)
