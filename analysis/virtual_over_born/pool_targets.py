"""Step 0 of the current-defaults V/B test (plan virtual_over_born, D366): the two one-loop pools' targets as trained.

Per pool (e+e- -> t t~ ee_ttbar_nlo_362-1000GeV, e+e- -> u u~ ee_uu_nlo_25-1000GeV; train from the train cache, val and
test from the frozen dir, all rows), read on CC's login node: the stored target V (last column) and its Born sidecar B
(<pool>.born.npy, written by scripts/job_vob_born.sh; its identity record <pool>.born.npy.json must exist), then
std(log|V|), std(log|V/B|), their ratio std(log|V/B|) / std(log|V|) (design thresholds: below 0.5 narrower as in June,
above 1 wider), the share of negative V, and the relative std of V/B (std / |mean|).

    python3 analysis/virtual_over_born/pool_targets.py [--cached]   -> data/pool_targets_site.json
The last line printed is the JSON of the numbers quoted.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from analysis.common.readers import site_python

POOLS = ("ee_ttbar_nlo_362-1000GeV", "ee_uu_nlo_25-1000GeV")
CACHE = os.path.join(HERE, "data", "pool_targets_site.json")

REMOTE = r'''
import json, os, sys, numpy as np
sys.path.insert(0, os.getcwd())   # the probe file sits in $SCRATCH; the checkout is the cwd
import datagen
out = {}
for name in sys.argv[1:]:
    for role in ("train", "val", "test"):
        d = datagen.train_cache_dir() if role == "train" else datagen.frozen_dir()
        pool = os.path.join(d, "%s_%s_amplitudes.npy" % (name, role))
        bp = pool[:-4] + ".born.npy"
        v = np.asarray(np.load(pool, mmap_mode="r")[:, -1], dtype=np.float64)
        b = np.asarray(np.load(bp, mmap_mode="r"), dtype=np.float64).reshape(-1)[:v.size]
        r = v / b
        lv, lr = np.log(np.abs(v)), np.log(np.abs(r))
        out["%s_%s" % (name, role)] = {
            "n": int(v.size), "n_born": int(b.size), "identity_record": os.path.exists(bp + ".json"),
            "frac_V_neg": float((v < 0).mean()), "std_log_V": float(lv.std()), "std_log_VB": float(lr.std()),
            "ratio_std": float(lr.std() / lv.std()), "VB_mean": float(r.mean()),
            "VB_rel_std": float(r.std() / abs(r.mean())), "VB_min": float(r.min()), "VB_max": float(r.max())}
print("POOL_TARGETS " + json.dumps(out))
'''


def main():
    if "--cached" in sys.argv and os.path.exists(CACHE):
        st = json.load(open(CACHE))
    else:
        st = site_python(REMOTE, list(POOLS), "POOL_TARGETS", site="ccin2p3")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(st, open(CACHE, "w"), indent=1)
    for k, v in st.items():
        print("%-40s n %6d  std log|V| %.4g  std log|V/B| %.4g  ratio %.3g  V/B rel std %.3g  V<0 %.3g  record %s"
              % (k, v["n"], v["std_log_V"], v["std_log_VB"], v["ratio_std"], v["VB_rel_std"], v["frac_V_neg"],
                 v["identity_record"]))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
