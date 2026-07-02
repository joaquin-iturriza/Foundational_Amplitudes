#!/usr/bin/env python3
"""Optimal training.lr vs t_steps: geomean per exact t_steps, pooled and by
regime. Makes the inverted-U (that a single slope hides) visible."""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

recs=json.load(open(sys.argv[1])); out=sys.argv[2]
raw=[r for r in recs if r['converged'] and r['lr_key']=='training.lr']
def bv(r,k):
    h=r['hp_best'].get(k); return h['val'] if h else None
def geomean_by_t(rows):
    g=collections.defaultdict(list)
    for r in rows:
        lr=bv(r,'training.lr'); t=r['t_steps']
        if lr and lr>0 and t: g[t].append(math.log10(lr))
    ts=sorted(g); return ts,[10**np.mean(g[t]) for t in ts]

fig,ax=plt.subplots(1,2,figsize=(15,6))
a=ax[0]
xs=[r['t_steps'] for r in raw if bv(r,'training.lr') and r['t_steps']]
ys=[bv(r,'training.lr') for r in raw if bv(r,'training.lr') and r['t_steps']]
a.scatter(xs,ys,s=14,alpha=0.25,color='gray')
ts,gm=geomean_by_t(raw)
a.plot(ts,gm,'o-',color='red',lw=2,ms=6,label='geomean per t_steps')
a.set_xscale('log');a.set_yscale('log');a.set_xlabel('t_steps');a.set_ylabel('best training.lr')
a.set_title('POOLED: geomean lr per t_steps');a.legend();a.grid(alpha=.3,which='both')
a=ax[1]
for nd,col,lab in [(8,'#0343DE','joint 8-proc (bs 256)'),(1,'#A52A2A','solo (bs 16384)')]:
    ts,gm=geomean_by_t([r for r in raw if r['n_datasets']==nd])
    a.plot(ts,gm,'o-',color=col,lw=2,ms=6,label=lab)
a.set_xscale('log');a.set_yscale('log');a.set_xlabel('t_steps');a.set_ylabel('geomean best training.lr')
a.set_title('geomean lr per t_steps, by regime');a.legend();a.grid(alpha=.3,which='both')
fig.tight_layout()
base=out[:-4] if out.lower().endswith(('.png','.pdf')) else out
fig.savefig(base+'.png',dpi=115); fig.savefig(base+'.pdf')
print("saved",base+'.png',base+'.pdf')
