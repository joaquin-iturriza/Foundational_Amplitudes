#!/usr/bin/env python3
"""HPO optima across the converged Bayesian sweeps: the six views behind the search-range rules.

Laid out 2 wide x 3 tall rather than 3 x 2: six panels across \\textwidth would each be
2.2in, which cannot carry 11pt axis labels. What each panel shows, and the rules read off
them, live in the results.tex caption and body text.
"""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

recs = json.load(open(sys.argv[1])); out = sys.argv[2]
raw = [r for r in recs if r['converged'] and r['lr_key'] == 'training.lr']
# NOTE: the finetune lr_scale panel this fed was dropped from the figure; the caption in
# docs/results.tex no longer promises it. Left out rather than left dangling.


def bv(r, k):
    h = r['hp_best'].get(k); return h['val'] if h else None


# SIX separate panel files. As one 2x3 canvas this measured 6.66in against a 6.5in text width
# -- clipped, not merely overfull -- because both columns carry their own y-label and log tick
# labels. Six files of ~3.2in go three lines of two, which is the same arrangement on the page
# and fits. The panels answer six different questions; nothing about them was ever shared.
figs = ps.panels(6)
# Kept as a 3x2 object array so the panel code below still reads ax[row, col].
ax = np.empty((3, 2), dtype=object)
for _i, (_f, _a) in enumerate(figs):
    ax[_i // 2, _i % 2] = _a

# 1. lr vs num_heads (muP)
a = ax[0, 0]
xs = [r['num_heads'] for r in raw if bv(r, 'training.lr') and r['num_heads']]
ys = [bv(r, 'training.lr') for r in raw if bv(r, 'training.lr') and r['num_heads']]
a.scatter(xs, ys, s=10, alpha=0.4, color=ps.C.blue, label='individual sweep')
g = collections.defaultdict(list)
for x, y in zip(xs, ys): g[x].append(math.log10(y))
gx = sorted(g); gy = [10**np.mean(g[k]) for k in gx]
a.plot(gx, gy, 'o-', color=ps.C.vermillion, label='geometric mean')
a.set_xscale('log', base=2); a.set_yscale('log')
a.set_xticks(sorted(set(xs))); a.get_xaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
a.set_xlabel(r'$n_{\rm heads}$'); a.set_ylabel('optimal learning rate')
a.legend(loc='lower left')

# 2. lr vs batchsize
a = ax[0, 1]
xs = [r['batchsize_cfg'] for r in raw if bv(r, 'training.lr') and r['batchsize_cfg']]
ys = [bv(r, 'training.lr') for r in raw if bv(r, 'training.lr') and r['batchsize_cfg']]
a.scatter(xs, ys, s=10, alpha=0.4, color=ps.C.green, label='individual sweep')
g = collections.defaultdict(list)
for x, y in zip(xs, ys): g[x].append(math.log10(y))
gx = sorted(g); gy = [10**np.mean(g[k]) for k in gx]
a.plot(gx, gy, 'o-', color=ps.C.vermillion, label='geometric mean')
a.set_xscale('log', base=2); a.set_yscale('log')
# Every batch size here is a power of two, so label the exponent: "$2^{10}$" is a third the
# width of "1024" and the seven ticks stop colliding. Rotating them was the old fix, and a
# rotated label block is tall -- which is exactly what the tick-label rule forbids.
a.set_xticks(sorted(set(xs))); a.get_xaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda v, _: rf"$2^{{{int(round(math.log2(v)))}}}$"))
a.set_xlabel('batch size'); a.set_ylabel('optimal learning rate')
a.legend(loc='lower left')

# 3. lr vs t_steps: geomean per t_steps by regime
a = ax[1, 0]
xs = [r['t_steps'] for r in raw if bv(r, 'training.lr') and r['t_steps']]
ys = [bv(r, 'training.lr') for r in raw if bv(r, 'training.lr') and r['t_steps']]
a.scatter(xs, ys, s=8, alpha=0.18, color=ps.C.grey, label='individual sweep')
for nd, col, lab in [(8, ps.C.blue, 'joint, 8 processes'), (1, ps.C.vermillion, 'single process')]:
    g = collections.defaultdict(list)
    for r in raw:
        lr = bv(r, 'training.lr'); t = r['t_steps']
        if lr and lr > 0 and t and r['n_datasets'] == nd: g[t].append(math.log10(lr))
    gx = sorted(g); gy = [10**np.mean(g[k]) for k in gx]
    a.plot(gx, gy, 'o-', color=col, label=lab)
a.axvline(3000, color=ps.C.green, ls=':', label=r'$t^\ast=3\times10^3$')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'training steps $t$'); a.set_ylabel('optimal learning rate')
a.legend(loc='upper left')

# 4. lr histogram + recommended band
a = ax[1, 1]
ys = np.log10([bv(r, 'training.lr') for r in raw if bv(r, 'training.lr')])
a.hist(ys, bins=30, color=ps.C.blue, label='converged sweeps')
a.axvspan(math.log10(1e-3), math.log10(1e-2), color=ps.C.green, alpha=0.2,
          label=r'recommended $[10^{-3},10^{-2}]$')
a.axvline(math.log10(3e-3), color=ps.C.vermillion, ls='--', label=r'median $3\times10^{-3}$')
a.set_xlabel(r'$\log_{10}$ optimal learning rate'); a.set_ylabel('sweeps')
a.legend(loc='upper left')

# 5. HP importance bars (sampler knobs excluded: the balanced sampler is no longer used)
a = ax[2, 0]
imp_raw = {r'$\eta$': 0.285, r'$\lambda$': 0.233, 'warmup frac': 0.219,
           'EMA decay': 0.199, r'$\eta_{\min}$': 0.198}
names = list(imp_raw); vals = [imp_raw[n] for n in names]
a.barh(range(len(names)), vals, color=ps.C.blue)
a.set_yticks(range(len(names))); a.set_yticklabels(names); a.invert_yaxis()
a.set_xlabel(r'mean $|\rho_{\rm Spearman}({\rm HP},\,{\rm val\ loss})|$')
a.grid(True, axis='x')

# 6. optima p5-p95 vs declared range
a = ax[2, 1]
items = [(r'$\eta$', 3.2e-5, 3e-1, 3.5e-4, 2.1e-2),
         (r'$\lambda$', 1e-11, 1e-2, 2.5e-11, 9.8e-7),
         (r'$\eta_{\min}$', 1e-11, 1e-6, 2.1e-11, 6.9e-7)]
for i, (nm, dl, dh, ol, oh) in enumerate(items):
    a.plot([dl, dh], [i, i], color=ps.C.grey, lw=6, alpha=0.4, solid_capstyle='butt',
           label='declared range' if i == 0 else None)
    a.plot([ol, oh], [i, i], color=ps.C.vermillion, lw=6, solid_capstyle='butt',
           label='optima, p5-p95' if i == 0 else None)
a.set_yticks(range(len(items))); a.set_yticklabels([x[0] for x in items])
a.set_xscale('log'); a.set_xlabel('value'); a.invert_yaxis()
a.legend(loc='lower right')
a.grid(True, axis='x', which='both')

base = out[:-4] if out.lower().endswith(('.png', '.pdf')) else out
ps.save_panels(figs, base)
