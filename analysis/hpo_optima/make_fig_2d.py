#!/usr/bin/env python3
"""Disentangle dataset-size vs iterations on the clean scaling_p 2D grid, and
show that the optimal-lr hump peaks at the convergence horizon (val_loss floor)."""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

recs=json.load(open(sys.argv[1])); out=sys.argv[2]
def bv(r,k):
    h=r['hp_best'].get(k); return h['val'] if h else None
sp=[r for r in recs if r['family']=='scaling_p' and r['converged'] and bv(r,'training.lr')]
Ds=sorted(set(r['n_train'] for r in sp if r['n_train']))
Ts=sorted(set(r['t_steps'] for r in sp if r['t_steps']))
cmap=plt.cm.viridis

def gm(rows,key='training.lr'):
    return 10**np.mean([math.log10(bv(r,key)) for r in rows])

fig,ax=plt.subplots(1,3,figsize=(19,5.6))

# A: lr vs t_steps, one line per dataset size
a=ax[0]
for i,D in enumerate(Ds):
    pts=collections.defaultdict(list)
    for r in sp:
        if r['n_train']==D: pts[r['t_steps']].append(math.log10(bv(r,'training.lr')))
    ts=sorted(pts); ys=[10**np.mean(pts[t]) for t in ts]
    a.plot(ts,ys,'o-',color=cmap(i/max(len(Ds)-1,1)),lw=1.8,ms=5,label=f'n_train={D:,}')
a.axvspan(1000,4000,color='green',alpha=0.12); a.text(2000,a.get_ylim()[1] if False else 3e-4,'convergence\nhorizon',ha='center',fontsize=8,color='green')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('t_steps (iterations)'); a.set_ylabel('optimal training.lr')
a.set_title('A) lr vs ITERATIONS at fixed dataset size\ninverted-U at every D; peak fixed ~3k steps (D-independent)'); a.legend(fontsize=8); a.grid(alpha=.3,which='both')

# B: lr vs dataset size, one line per t_steps (only well-sampled t)
a=ax[1]
good_t=[t for t in Ts if sum(1 for r in sp if r['t_steps']==t)>=8]
for i,t in enumerate(good_t):
    pts=collections.defaultdict(list)
    for r in sp:
        if r['t_steps']==t: pts[r['n_train']].append(math.log10(bv(r,'training.lr')))
    ds=sorted(pts)
    if len(ds)<3: continue
    ys=[10**np.mean(pts[d]) for d in ds]
    a.plot(ds,ys,'s-',color=cmap(i/max(len(good_t)-1,1)),lw=1.8,ms=5,label=f't={t:,}')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('n_train (dataset size)'); a.set_ylabel('optimal training.lr')
a.set_title('B) lr vs DATASET SIZE at fixed iterations\nweak: lr ∝ D^(+0.15..0.3), saturating'); a.legend(fontsize=8); a.grid(alpha=.3,which='both')

# C: learning curves — best_val_loss vs t_steps per D, with lr-peak t marked
a=ax[2]
for i,D in enumerate(Ds):
    pts=collections.defaultdict(list); lrp=collections.defaultdict(list)
    for r in sp:
        if r['n_train']==D and r['best_val_loss'] and r['best_val_loss']>0:
            pts[r['t_steps']].append(r['best_val_loss']); lrp[r['t_steps']].append(math.log10(bv(r,'training.lr')))
    ts=sorted(pts); ys=[np.median(pts[t]) for t in ts]
    col=cmap(i/max(len(Ds)-1,1))
    a.plot(ts,ys,'o-',color=col,lw=1.8,ms=5,label=f'n_train={D:,}')
    # mark the t where optimal lr is maximal
    tpk=max(lrp, key=lambda t:np.mean(lrp[t]))
    if tpk in pts: a.scatter([tpk],[np.median(pts[tpk])],s=180,marker='*',color=col,edgecolor='k',zorder=5)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('t_steps (iterations)'); a.set_ylabel('best_val_loss (learning curve)')
a.set_title('C) LEARNING CURVES (★ = t of peak lr)\nfloor shifts later with D, but ★ stays ~3k → peak≠convergence'); a.legend(fontsize=8); a.grid(alpha=.3,which='both')

fig.suptitle('scaling_p grid: dataset-size vs iterations disentangled — optimal lr is governed by proximity to convergence',fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(out,dpi=115)
print("saved",out)
