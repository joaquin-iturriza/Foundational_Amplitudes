#!/usr/bin/env python3
"""Per-sweep HP importance: within each converged sweep, how strongly does each
swept HP relate to val_loss? Uses Spearman |rho| over that sweep's observations,
plus a top-vs-bottom quartile separation. Aggregated across sweeps this tells us
which knobs actually matter (worth sweeping) vs which are near-flat (fixable)."""
import json, os, sys, glob, pickle, math, collections
import numpy as np

recs = {r['sweep']: r for r in json.load(open(sys.argv[1]))}
LOG = {'training.lr','training.regularization_lambda','training.cosanneal_eta_min',
       'fine_tune.lr_scale'}

def spearman(x, y):
    if len(x) < 5: return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx, ry)[0,1])

# gather per-HP importance across converged sweeps, split by lr knob family
agg = collections.defaultdict(lambda: {'rho':[], 'n':0})
for r in recs.values():
    if not r['converged']: continue
    st = pickle.load(open(os.path.join(r['path'],'dyhpo_state.pkl'),'rb'))
    cands = st['candidates_raw']; vh = st.get('val_loss_history',{})
    swept = [e['name'] for e in st['hp_space']]
    # one row per observation (best fidelity if several)
    rows=[]
    for hp_idx, obs in vh.items():
        vl = min(obs.values())
        try: vl=float(vl)
        except: continue
        if not math.isfinite(vl): continue
        rows.append((cands[hp_idx], vl))
    if len(rows) < 8: continue
    y = np.array([math.log(v) for _,v in rows])
    fam = 'FT' if r['lr_key']=='fine_tune.lr_scale' else 'RAW'
    for nm in swept:
        xs=[]
        for c,_ in rows:
            v = c.get(nm)
            if v is None: xs.append(np.nan); continue
            xs.append(math.log(v) if (nm in LOG and isinstance(v,(int,float)) and v>0) else
                      (float(v) if isinstance(v,(int,float)) else np.nan))
        xs=np.array(xs)
        m = np.isfinite(xs)
        if m.sum()<8: continue
        rho = spearman(xs[m], y[m])
        if rho is None: continue
        key=(fam, nm.split('.')[-1])
        agg[key]['rho'].append(abs(rho)); agg[key]['n']+=1

print("Per-sweep HP importance = mean |Spearman(HP, log val_loss)| across converged sweeps.")
print("Higher = HP more strongly controls the loss (worth sweeping). ~0.1 = near-flat (fixable).\n")
for fam in ('RAW','FT'):
    print(f"=== {('training.lr' if fam=='RAW' else 'fine_tune.lr_scale')} sweeps ===")
    items=[(k,v) for k,v in agg.items() if k[0]==fam]
    items.sort(key=lambda kv:-np.mean(kv[1]['rho']))
    for (f,nm),v in items:
        a=np.array(v['rho'])
        print(f"  {nm:26} mean|rho|={a.mean():.3f}  median={np.median(a):.3f}  (n_sweeps={v['n']})")
    print()
