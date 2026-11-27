"""Do the 448-process runs share one per-dataset standardization, and what is it? (draft_rewrite step 4: Fig. 14 right
on MSE(log|M|^2) rather than the standardized loss.) On Jean Zay (the runs' only site), read every arm's
data_stats.json (compare_models/_bigrun_arm_*) and every trial's of the baseline sweep (sweeps/bigrun/bigrun_baseline*):
prepd_std, one per dataset in recipe order (data.preprocess_per_dataset=true). Prints per run the geometric mean over
datasets of prepd_std^2 and the largest relative difference of its prepd_std from the first run's. With one shared
standardization, GM_p(MSE_p of log|M|^2) = GM_p(L_p) x GM_p(sigma_p^2) exactly, L_p the standardized loss.
    python3 analysis/bigrun/std_check.py      -> last line: JSON {run: {gm_var, max_rel_diff, n}}
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.common.readers import site_python

REMOTE = r'''
import glob, json, os
import numpy as np
fs = sorted(glob.glob("compare_models/_bigrun_arm_*/data_stats.json"))
fs += sorted(glob.glob("sweeps/bigrun/bigrun_baseline*/**/data_stats.json", recursive=True))  # sweep_dir of the config
ref, out = None, {}
for f in fs:
    s = np.asarray(json.load(open(f))["prepd_std"], float)
    ref = s if ref is None else ref
    out[f] = {"n": len(s), "gm_var": float(np.exp(np.mean(np.log(s ** 2)))),
              "max_rel_diff": float(np.max(np.abs(s / ref - 1))) if len(s) == len(ref) else None}
print("STDCHECK " + json.dumps(out))
'''

if __name__ == "__main__":
    r = site_python(REMOTE, [], "STDCHECK", site="jeanzay")
    for f, v in r.items():
        print(f"{f:90s} n={v['n']} GM(sigma^2)={v['gm_var']:.5g} max|sigma/sigma_0-1|={v['max_rel_diff']}")
    print(json.dumps(r))
