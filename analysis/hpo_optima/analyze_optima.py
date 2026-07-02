#!/usr/bin/env python3
import json, math, sys, os, collections
import numpy as np

recs = json.load(open(sys.argv[1]))

def g(r, k): return r.get(k)
conv = [r for r in recs if r['converged']]
print(f"TOTAL sweeps: {len(recs)}   converged (lr interior, n_obs>=6): {len(conv)}")

# ---- convergence by family + lr_key ----
print("\n=== convergence by family ===")
fam = collections.defaultdict(lambda: [0,0,0])  # total, has_obs, converged
for r in recs:
    f = r['family']
    fam[f][0]+=1
    if r['n_obs']>=6: fam[f][1]+=1
    if r['converged']: fam[f][2]+=1
for f in sorted(fam, key=lambda x:-fam[x][0]):
    t,o,c = fam[f]
    if t>=3:
        print(f"  {f:52} tot={t:4d}  n_obs>=6={o:4d}  converged={c:4d}  ({100*c/max(o,1):.0f}% of usable)")

# ---- split by lr knob ----
raw = [r for r in conv if r['lr_key']=='training.lr']
ft  = [r for r in conv if r['lr_key']=='fine_tune.lr_scale']
print(f"\n=== converged split: training.lr sweeps={len(raw)}   fine_tune.lr_scale sweeps={len(ft)} ===")

def bestval(r, key):
    hb = r['hp_best'].get(key)
    return hb['val'] if hb else None

def summarize(rows, key, logscale=True, label=None):
    vals = [bestval(r,key) for r in rows]
    vals = [v for v in vals if v is not None and (not logscale or v>0)]
    if not vals:
        print(f"  {label or key:34} (no data)"); return None
    a = np.array(vals, float)
    if logscale:
        la = np.log10(a)
        gm = 10**la.mean()
        p = [10**np.percentile(la,q) for q in (5,25,50,75,95)]
        print(f"  {label or key:34} n={len(a):3d}  geomean={gm:.2e}  "
              f"p5={p[0]:.2e} p25={p[1]:.2e} med={p[2]:.2e} p75={p[3]:.2e} p95={p[4]:.2e}")
    else:
        p = [np.percentile(a,q) for q in (5,25,50,75,95)]
        print(f"  {label or key:34} n={len(a):3d}  mean={a.mean():.3f}  "
              f"p5={p[0]:.3f} p25={p[1]:.3f} med={p[2]:.3f} p75={p[3]:.3f} p95={p[4]:.3f}")
    return a

print("\n=== [training.lr sweeps] optimal-HP distributions across converged sweeps ===")
summarize(raw,'training.lr', True, 'training.lr')
summarize(raw,'training.regularization_lambda', True, 'regularization_lambda')
summarize(raw,'training.cosanneal_warmup_frac', False,'cosanneal_warmup_frac')
summarize(raw,'training.cosanneal_eta_min', True, 'cosanneal_eta_min')
summarize(raw,'training.ema_decay', False,'ema_decay')
summarize(raw,'training.sampler_alpha_ema', False,'sampler_alpha_ema')
summarize(raw,'training.sampler_min_alpha_frac', False,'sampler_min_alpha_frac')

print("\n=== [fine_tune.lr_scale sweeps] optimal-HP distributions ===")
summarize(ft,'fine_tune.lr_scale', True, 'fine_tune.lr_scale')
summarize(ft,'fine_tune.layer_decay', False,'fine_tune.layer_decay')
summarize(ft,'training.regularization_lambda', True, 'regularization_lambda')
summarize(ft,'training.cosanneal_warmup_frac', False,'cosanneal_warmup_frac')
summarize(ft,'training.cosanneal_eta_min', True, 'cosanneal_eta_min')

# ---- how does best lr move with context? correlations in log space ----
def corr_report(rows, lrkey, title):
    print(f"\n=== {title}: does best {lrkey} move with context? (log10 lr vs axis) ===")
    data=[]
    for r in rows:
        lr = bestval(r,lrkey)
        if not lr or lr<=0: continue
        data.append(r|{'_lr':lr})
    def col(rows,f):
        xs=[];ys=[]
        for r in rows:
            v=f(r)
            if v is None or (isinstance(v,float) and not math.isfinite(v)) or v<=0: continue
            xs.append(v); ys.append(r['_lr'])
        return np.array(xs,float), np.array(ys,float)
    for name,f,islog in [('num_heads (muP width)', lambda r:r['num_heads'], False),
                         ('n_train (data size)',   lambda r:r['n_train'], True),
                         ('t_steps (iterations)',  lambda r:r['t_steps'], True),
                         ('eff_bs (eff batch)',    lambda r:r['eff_bs'], True)]:
        x,y = col(data,f)
        if len(x)<5:
            print(f"  {name:26} n={len(x):3d}  (too few)"); continue
        lx = np.log10(x) if islog else x
        ly = np.log10(y)
        r_p = np.corrcoef(lx,ly)[0,1]
        # slope of log10 lr vs (log10 x or x)
        slope = np.polyfit(lx,ly,1)[0]
        uniq = sorted(set(np.round(x,3)))
        print(f"  {name:26} n={len(x):3d}  pearson(r)={r_p:+.2f}  slope(dlog10lr/d{'log10' if islog else ''}x)={slope:+.3f}  #distinct_x={len(uniq)}")

corr_report(raw,'training.lr','[training.lr sweeps]')
corr_report(ft,'fine_tune.lr_scale','[fine_tune.lr_scale sweeps]')

# ---- muP width transfer check: group best lr by num_heads ----
def by_width(rows, lrkey, title):
    print(f"\n=== {title}: best {lrkey} grouped by num_heads (muP transfer check) ===")
    grp=collections.defaultdict(list)
    for r in rows:
        lr=bestval(r,lrkey); nh=r['num_heads']
        if lr and lr>0 and nh: grp[nh].append(lr)
    for nh in sorted(grp):
        a=np.log10(grp[nh])
        print(f"  nh={nh:3d}  n={len(a):3d}  geomean_lr={10**a.mean():.2e}  [p25={10**np.percentile(a,25):.2e}, p75={10**np.percentile(a,75):.2e}]")

by_width(raw,'training.lr','[training.lr sweeps]')

# ---- range-shrink recommendation per HP ----
def shrink(rows, key, logscale, title):
    hb=[r['hp_best'][key] for r in rows if key in r['hp_best'] and r['hp_best'][key]['val'] is not None]
    hb=[h for h in hb if h['val'] is not None and (not logscale or h['val']>0)]
    if not hb: return
    lows=[h['low'] for h in hb if h['low'] is not None]
    highs=[h['high'] for h in hb if h['high'] is not None]
    vals=np.array([h['val'] for h in hb],float)
    dlo=min(lows) if lows else None; dhi=max(highs) if highs else None
    if logscale:
        lv=np.log10(vals); p5,p95=10**np.percentile(lv,5),10**np.percentile(lv,95)
        span_decades_declared = math.log10(dhi/dlo) if dlo and dhi and dlo>0 else None
        span_decades_used = math.log10(p95/p5) if p5>0 else None
        print(f"  {title:30} declared[{dlo:.1e},{dhi:.1e}] ({span_decades_declared:.1f} dec) "
              f"-> optima p5-p95 [{p5:.1e},{p95:.1e}] ({span_decades_used:.1f} dec)")
    else:
        p5,p95=np.percentile(vals,5),np.percentile(vals,95)
        print(f"  {title:30} declared[{dlo:.3g},{dhi:.3g}] -> optima p5-p95 [{p5:.3g},{p95:.3g}]")

print("\n=== RANGE-SHRINK: declared range vs where optima actually land (converged) ===")
print(" [training.lr sweeps]")
shrink(raw,'training.lr',True,'training.lr')
shrink(raw,'training.regularization_lambda',True,'regularization_lambda')
shrink(raw,'training.cosanneal_warmup_frac',False,'cosanneal_warmup_frac')
shrink(raw,'training.cosanneal_eta_min',True,'cosanneal_eta_min')
shrink(raw,'training.ema_decay',False,'ema_decay')
shrink(raw,'training.sampler_alpha_ema',False,'sampler_alpha_ema')
shrink(raw,'training.sampler_min_alpha_frac',False,'sampler_min_alpha_frac')
print(" [fine_tune.lr_scale sweeps]")
shrink(ft,'fine_tune.lr_scale',True,'fine_tune.lr_scale')
shrink(ft,'fine_tune.layer_decay',False,'fine_tune.layer_decay')
shrink(ft,'training.regularization_lambda',True,'regularization_lambda')
shrink(ft,'training.cosanneal_warmup_frac',False,'cosanneal_warmup_frac')
shrink(ft,'training.cosanneal_eta_min',True,'cosanneal_eta_min')
