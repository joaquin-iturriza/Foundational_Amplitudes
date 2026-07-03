#!/usr/bin/env python3
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

recs=json.load(open(sys.argv[1])); out=sys.argv[2]
raw=[r for r in recs if r['converged'] and r['lr_key']=='training.lr']
ft =[r for r in recs if r['converged'] and r['lr_key']=='fine_tune.lr_scale']
def bv(r,k):
    h=r['hp_best'].get(k); return h['val'] if h else None

fig,ax=plt.subplots(2,3,figsize=(16,9))

# 1. lr vs num_heads (muP)
a=ax[0,0]
xs=[r['num_heads'] for r in raw if bv(r,'training.lr') and r['num_heads']]
ys=[bv(r,'training.lr') for r in raw if bv(r,'training.lr') and r['num_heads']]
a.scatter(xs,ys,s=18,alpha=0.4,color='#0343DE')
g=collections.defaultdict(list)
for x,y in zip(xs,ys): g[x].append(math.log10(y))
gx=sorted(g); gy=[10**np.mean(g[k]) for k in gx]
a.plot(gx,gy,'o-',color='red',lw=2,label='geomean')
a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xlabel('num_heads (μP width)')
a.set_ylabel('best training.lr'); a.set_title('lr vs width — flat ⇒ μP transfers'); a.legend(); a.grid(alpha=.3,which='both')

# 2. lr vs batchsize
a=ax[0,1]
xs=[r['batchsize_cfg'] for r in raw if bv(r,'training.lr') and r['batchsize_cfg']]
ys=[bv(r,'training.lr') for r in raw if bv(r,'training.lr') and r['batchsize_cfg']]
a.scatter(xs,ys,s=18,alpha=0.4,color='#2ca02c')
g=collections.defaultdict(list)
for x,y in zip(xs,ys): g[x].append(math.log10(y))
gx=sorted(g); gy=[10**np.mean(g[k]) for k in gx]
a.plot(gx,gy,'o-',color='red',lw=2,label='geomean')
a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xlabel('configured batchsize')
a.set_ylabel('best training.lr'); a.set_title('lr vs batch (NOTE: batch entangled with t_steps & regime)'); a.legend(); a.grid(alpha=.3,which='both')

# 3. lr vs t_steps: GEOMEAN per t_steps by regime -> clear inverted-U (hump)
a=ax[0,2]
xs=[r['t_steps'] for r in raw if bv(r,'training.lr') and r['t_steps']]
ys=[bv(r,'training.lr') for r in raw if bv(r,'training.lr') and r['t_steps']]
a.scatter(xs,ys,s=12,alpha=0.18,color='gray')
for nd,col,lab in [(8,'#0343DE','joint 8-proc geomean'),(1,'#A52A2A','solo geomean')]:
    g=collections.defaultdict(list)
    for r in raw:
        lr=bv(r,'training.lr'); t=r['t_steps']
        if lr and lr>0 and t and r['n_datasets']==nd: g[t].append(math.log10(lr))
    gx=sorted(g); gy=[10**np.mean(g[k]) for k in gx]
    a.plot(gx,gy,'o-',color=col,lw=2,ms=5,label=lab)
a.axvline(3000,color='green',ls=':',label='peak t*~3e3')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('t_steps (iterations)')
a.set_ylabel('best training.lr')
a.set_title('lr vs iterations — INVERTED-U: lr∝t^+0.5 then t^-0.55'); a.legend(fontsize=7); a.grid(alpha=.3,which='both')

# 4. lr histogram + recommended band
a=ax[1,0]
ys=np.log10([bv(r,'training.lr') for r in raw if bv(r,'training.lr')])
a.hist(ys,bins=30,color='#0343DE',alpha=0.7)
a.axvspan(math.log10(1e-3),math.log10(1e-2),color='green',alpha=0.2,label='recommend [1e-3,1e-2]')
a.axvline(math.log10(3e-3),color='red',ls='--',label='median ~3e-3')
a.set_xlabel('log10 best training.lr'); a.set_ylabel('# sweeps')
a.set_title('lr optima MARGINAL spread (condition on t_steps → panel 3)'); a.legend(fontsize=8); a.grid(alpha=.3)

# 5. HP importance bars (sampler knobs excluded: the balanced sampler is no longer used)
a=ax[1,1]
imp_raw={'lr':0.285,'reg_lambda':0.233,'warmup_frac':0.219,
         'ema_decay':0.199,'eta_min':0.198}
imp_ft ={'lr_scale':0.479,'layer_decay':0.287,'warmup_frac':0.286,'reg_lambda':0.268,'eta_min':0.257}
names=list(imp_raw); vals=[imp_raw[n] for n in names]
a.barh(range(len(names)),vals,color='#0343DE',alpha=0.8)
a.set_yticks(range(len(names))); a.set_yticklabels(names,fontsize=8); a.invert_yaxis()
a.set_xlabel('mean |Spearman(HP, val_loss)|'); a.set_title('HP importance (training.lr sweeps)')
a.axvline(0.15,color='gray',ls=':'); a.grid(alpha=.3,axis='x')

# 6. optima p5-p95 vs declared (lr focus, log axis)
a=ax[1,2]
items=[('training.lr',3.2e-5,3e-1,3.5e-4,2.1e-2),
       ('reg_lambda',1e-11,1e-2,2.5e-11,9.8e-7),
       ('eta_min',1e-11,1e-6,2.1e-11,6.9e-7)]
for i,(nm,dl,dh,ol,oh) in enumerate(items):
    a.plot([dl,dh],[i,i],color='gray',lw=6,alpha=0.4,solid_capstyle='butt')
    a.plot([ol,oh],[i,i],color='#A52A2A',lw=6,solid_capstyle='butt')
a.set_yticks(range(len(items))); a.set_yticklabels([x[0] for x in items])
a.set_xscale('log'); a.set_xlabel('value'); a.invert_yaxis()
a.set_title('grey=declared range, red=optima p5–p95'); a.grid(alpha=.3,axis='x',which='both')

fig.suptitle('HPO optima across 422 converged Bayesian sweeps — empirical search-range rules',fontsize=14)
fig.tight_layout(rect=[0,0,1,0.97])
base=out[:-4] if out.lower().endswith(('.png','.pdf')) else out
fig.savefig(base+'.png',dpi=110); fig.savefig(base+'.pdf')
print("saved",base+'.png',base+'.pdf')
