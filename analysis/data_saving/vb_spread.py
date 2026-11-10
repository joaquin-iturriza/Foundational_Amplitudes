"""pretrain_data_saving step 6: the spread of the one-loop target over its Born, ln|V| - ln B, on each candidate
one-loop target's train pool (5e4 rows) and its Born sidecar <pool>.born.npy (scripts/job_vob_born.sh), read on CC.

Per target: std, p1, p99 of ln|V/B|, the share of V < 0, the signedlog scale s (1% quantile of |V/B|) and the share
of rows with |V/B| > 10 s (where signedlog is within 0.1 of ln(|V/B|/s)), plus a histogram of ln|V/B| - its median (for the
figure). The rule (revised 2026-11-10 by the main session's correction): only a constant V/B is out (std < 1e-6: nothing
to learn, no gap in D); a sign-changing V/B stays in, trained and read through the signedlog transform in both arms.

    python analysis/data_saving/vb_spread.py [--out analysis/data_saving/data/vb_spread.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from analysis.common.readers import site_python  # noqa: E402

POOLS = ["ee_uu_nlo_25-1000GeV", "ee_ttbar_nlo_362-1000GeV", "v4_epem_bbx_nlo_20-6000GeV",
         "v4_bbx_zz_nlo_183-6000GeV", "v4_uux_epemz_nlo_112-6000GeV", "v4_bg_ab_nlo_20-6000GeV",
         "v4_epem_ddxz_nlo_112-6000GeV", "v4_bbx_gg_nlo_20-6000GeV", "v4_dg_uwm_nlo_92-6000GeV",
         "v4_bbx_avevex_nlo_20-6000GeV", "v4_db_bdz_nlo_112-6000GeV", "v4_uc_cuz_nlo_112-6000GeV",
         "v4_bb_bb_nlo_20-6000GeV", "v4_bbx_hz_nlo_217-6000GeV", "v4_bbx_ccxh_nlo_145-6000GeV", "v4_bg_bgg_nlo_30-6000GeV",
         "v4_gg_bbxg_nlo_30-6000GeV", "v4_ucx_cxhu_nlo_145-6000GeV", "v4_udx_ccxwp_nlo_101-6000GeV"]
KEEP_STD = 1e-6   # only a constant V/B (no gap in D possible) is out; sigma is reported beside each target

REMOTE = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import datagen
edges = np.linspace(-3, 3, 121)
out = {}
for stem in sys.argv[1:]:
    f = os.path.join(datagen.train_cache_dir(), stem + "_train_amplitudes.npy")
    b = f[:-4] + ".born.npy"
    if not (os.path.exists(f) and os.path.exists(b)):
        out[stem] = None
        continue
    v = np.load(f, mmap_mode="r")[:, -1].astype(float)
    x = v / np.load(b)
    r = np.log(np.abs(x))
    sq = float(np.quantile(np.abs(x), 0.01))   # data.signedlog_quantile 0.01: the signedlog scale of a signed pool
    h, _ = np.histogram(r - np.median(r), bins=edges)
    out[stem] = dict(n=int(len(r)), std=float(r.std()), median=float(np.median(r)), p1=float(np.percentile(r, 1)),
                     p99=float(np.percentile(r, 99)), neg_frac=float((v < 0).mean()), signedlog_s=sq,
                     share_above_10s=float((np.abs(x) > 10 * sq).mean()), hist=h.tolist(),
                     edges=edges.tolist())
print("VBSPREAD " + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vb_spread.json"))
    a = ap.parse_args()
    res = site_python(REMOTE, POOLS, "VBSPREAD")
    for s, r in res.items():
        if r:
            r["kept"] = r["std"] >= KEEP_STD
            print(f"{s:32s} std {r['std']:.3g}  p1 {r['p1']:.3f}  p99 {r['p99']:.3f}  V<0 {r['neg_frac']:.3f}"
                  f"  s {r['signedlog_s']:.3g}  >10s {r['share_above_10s']:.3f}  {'kept' if r['kept'] else 'dropped'}")
        else:
            print(f"{s:32s} no pool or sidecar")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
