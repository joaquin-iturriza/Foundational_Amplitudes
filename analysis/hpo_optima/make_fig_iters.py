#!/usr/bin/env python3
"""Optimal training.lr vs t_steps: geomean per exact t_steps, pooled and by
regime. Makes the inverted-U (that a single slope hides) visible."""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

recs = json.load(open(sys.argv[1])); out = sys.argv[2]
raw = [r for r in recs if r['converged'] and r['lr_key'] == 'training.lr']


def bv(r, k):
    h = r['hp_best'].get(k); return h['val'] if h else None


def geomean_by_t(rows):
    g = collections.defaultdict(list)
    for r in rows:
        lr = bv(r, 'training.lr'); t = r['t_steps']
        if lr and lr > 0 and t: g[t].append(math.log10(lr))
    ts = sorted(g); return ts, [10**np.mean(g[t]) for t in ts]


fig, ax = ps.figure(ncols=2)
a = ax[0]
xs = [r['t_steps'] for r in raw if bv(r, 'training.lr') and r['t_steps']]
ys = [bv(r, 'training.lr') for r in raw if bv(r, 'training.lr') and r['t_steps']]
a.scatter(xs, ys, s=8, alpha=0.25, color=ps.C.grey, label='individual sweep')
ts, gm = geomean_by_t(raw)
a.plot(ts, gm, 'o-', color=ps.C.vermillion, label='geometric mean')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training steps $t$'); a.set_ylabel(r'optimal learning rate')
a.legend(loc='lower center')

a = ax[1]
for nd, col, lab in [(8, ps.C.blue, 'joint, 8 processes'), (1, ps.C.vermillion, 'single process')]:
    ts, gm = geomean_by_t([r for r in raw if r['n_datasets'] == nd])
    a.plot(ts, gm, 'o-', color=col, label=lab)
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training steps $t$'); a.set_ylabel(r'optimal learning rate')
a.legend(loc='lower right')

base = out[:-4] if out.lower().endswith(('.png', '.pdf')) else out
ps.save(fig, base)
