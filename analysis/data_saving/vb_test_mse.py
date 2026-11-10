"""pretrain_data_saving, one-loop arm: the common measure of every one-loop target (Joaquin, 2026-11-10, D371), read on
the test pool of each run trained on V/B (data.target_over_born):

    MSE_VB  = mean(((V/B)_pred - (V/B)_true)^2)         any sign of V; the error of the NLO correction
                                                        (sigma_NLO uses B (1 + V/B)), whatever transform trained it
    MSE_lnV = mean((ln|(V/B)_pred| - ln|(V/B)_true|)^2) one-signed targets only (B exact: the MSE of ln|V|)

from <run>/sigma_stage/preds_test.npz (raw_pred, raw_truth: the target's raw values, i.e. V/B, of the final model after
the sigma stage, D276), for the runs runs/data_saving/dsave_<mode>_<target>_d<k>_t<steps>_<tag> on CC.

    python analysis/data_saving/vb_test_mse.py [--tag cal_vb] [--glob 'runs/vob_tt_VB_*'] [--out ...json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from analysis.common.readers import site_python  # noqa: E402

REMOTE = r'''
import glob, json, os, re, sys
import numpy as np
tag = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "runs/data_saving/dsave_*_*_nlo_d*_t*_" + tag
out = []
for d in sorted(glob.glob(pat)):
    m = re.match(r"dsave_(scratch|ft)_(.+)_d(\d+)_t(\d+)_" + re.escape(tag) + "$", os.path.basename(d))
    r = ({"mode": m.group(1), "target": m.group(2), "k": int(m.group(3)), "steps": int(m.group(4))} if m else
         {"mode": "-", "target": os.path.basename(d), "k": 0, "steps": 0})
    f = d + "/sigma_stage/preds_test.npz"
    try:
        st = json.load(open(d + "/data_stats.json"))
        r["over_born"] = st.get("target_over_born")
        r["phys_factor"] = st.get("target_physics_factor")   # must be off for a V/B run (job_data_saving.sh)
        z = np.load(f)
        p, t = z["raw_pred"].astype(float).ravel(), z["raw_truth"].astype(float).ravel()
        r["n"] = int(len(t))
        r["mse_vb"] = float(np.mean((p - t) ** 2))
        r["var_vb"] = float(np.var(t))
        r["one_signed"] = bool((t > 0).all() or (t < 0).all())
        if r["one_signed"]:
            ok = p * t > 0
            r["mse_lnv"] = float(np.mean((np.log(np.abs(p[ok])) - np.log(np.abs(t[ok]))) ** 2))
            r["wrong_sign"] = int((~ok).sum())
    except Exception as e:
        r["error"] = repr(e)[:200]
    out.append(r)
print("VBT " + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", nargs="+", default=["cal_vb", "cal_vb2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--glob", help="read these run dirs instead (e.g. the virtual_over_born runs 'runs/vob_tt_VB_*')")
    a = ap.parse_args()
    rows = [r for t in a.tag for r in site_python(REMOTE, [t] + ([f"'{a.glob}'"] if a.glob else []), "VBT")]
    rows.sort(key=lambda r: (r["target"], r["mode"], r["k"]))
    for r in rows:
        if "error" in r:
            print(f"{r['mode']:7s} {r['target']:16s} k={r['k']}  {r['error']}")
            continue
        if r.get("phys_factor") and r["target"] not in ("bbx_zz_nlo", "epem_bbx_nlo"):   # 2->2: the factor is a constant
            print(f"{r['mode']:7s} {r['target']:16s} k={r['k']}  physics factor on: not a plain V/B run, not read")
            continue
        ln = f"  MSE ln|V| {r['mse_lnv']:.3e} (wrong sign {r['wrong_sign']})" if r.get("one_signed") else ""
        print(f"{r['mode']:7s} {r['target']:16s} k={r['k']}  MSE V/B {r['mse_vb']:.3e} (var {r['var_vb']:.3g}){ln}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"vb_test_mse_{'_'.join(a.tag)}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
