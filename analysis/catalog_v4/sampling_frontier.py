#!/usr/bin/env python3
"""Fill against cost of every flat-sampling policy already read on the five step-57 pp classes (plan catalog_coverage, D317),
and the fill of the union with 20000 unused default-share MadEvent events (the probe pools' in-support histogram scaled
to 20000; offline estimate). Cost = median train-pool replay seconds of the policy (seed 42 file) over the old
generation-log train seconds (data/sampling_cost.json). Fill: kept / quota per 0.25-dex bin over in-range bins (lower
edge from the range table, up to the default in-support max), quota = 34992 / bins of the range. Prints the table."""
import json, glob, os, numpy as np, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
import multichannel_check as mc
fr=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../recipes/catalog_v4_flat_range.json'))); sc=json.load(open('data/sampling_cost.json'))['test_classes']
E=np.linspace(-20,8,113)
CL=["v4_uux_epemg","v4_uux_epemvevex","v4_uu_bbxuu","v4_us_ddxsu","v4_udx_csxvevex"]
for c in CL:
    runs={}
    for f in glob.glob(f'data/multichannel_check_reads/{c}*.json'):
        d=json.load(open(f))
        for k,r in d['runs'].items(): runs.setdefault(k,[]).append(r)
    base=json.load(open(f'data/multichannel_check_reads/{c}.json'))
    pp=np.array(base['pp']['hist_in_support'],float)
    lo,hi=np.array(fr[c])/np.log(10); ppmax=E[np.nonzero(pp)[0].max()+1]
    nb=int(np.ceil((hi-lo)/0.25)); q=34992/nb
    inr=(E[:-1]>=lo-1e-9)&(E[:-1]<=ppmax-0.25+1e-9)
    old=sc[c]['old_train_s']
    print(c, f"old train {old}s, judged bins {inr.sum()}, quota/bin {q:.0f}")
    for pol in ["current","multichannel","quota","quota_ladder_res2","quota_ladder_both_res2","quota_ladder_res2_budget500","quota_ladder_both_res2_budget750"]:
        for s in (42,):
            k=f"{pol}_{s}"
            if k not in runs: continue
            r=runs[k][0]
            t=[sum(x['chunk_seconds']) for x in runs[k] if x.get('chunk_seconds')]
            rel=np.median(t)/old if t else float('nan')
            kept=np.array(r['hist_kept'],float)
            fill=np.minimum(1,kept/q)[inr]
            u=np.minimum(q, kept+pp*20000/pp.sum()); uf=(u/q)[inr]
            top=E[:-1][inr][fill<0.95]
            print(f"  {pol:33s} cost x{rel:6.1f} bins>=0.95 {np.mean(fill>=0.95):.2f} short {1-fill.mean():.3f} first-short {top.min() if len(top) else None} | +20k default: bins>=0.95 {np.mean(uf>=0.95):.2f} short {1-uf.mean():.3f}")
