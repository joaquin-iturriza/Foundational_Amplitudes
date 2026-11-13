"""pretrain_data_saving stage-2 readout (design passed 2026-11-11 12:12): per (target, D, arm) the best trial of its
8-trial single-fidelity DyHPO search (lowest validation mu-MSE, DyHPO's objective, from sweeps_local/<sweep>/
driver_state.json), read on the test pool after the sigma stage (D276, <run>/sigma_stage/preds_test.npz):

    trees (ee_uu, epem_ttx)          MSE(ln|M|^2) = mean((ln p - ln t)^2), p, t the raw |M|^2
    one loops (ee_uu_nlo, ee_ttbar_nlo)  MSE(V/B) = mean(((V/B)_p - (V/B)_t)^2) (D371), MSE(ln|V|) beside it;
                                     u u~ V is trained on V (D375) and divided by the exact Born of its test row
                                     (pool_row of the predictions; the stored V is checked against the pool's)

then per target the joint shared-exponent fit of dstar_fit.py, D*(level) per arm, S = D*_scr / D*_ft with its 16-84%
cell-bootstrap band, at 1e-4, or (D373) at the lowest level both arms reach when one does not reach 1e-4. Beside it:
each arm's own exponent, and per search the trials that diverged against their sampled lr (scratch: lr / c, c the
window centre sqrt(low high); fine-tune: lr_scale) and where the best trial sits.

    python analysis/data_saving/stage2_readout.py --fetch     # read every trial's test predictions on CC (one call)
    python analysis/data_saving/stage2_readout.py             # the readout from the cached fetch
    python analysis/data_saving/stage2_readout.py --diag      # test MSE of the diagnostic runs (dsave_diag_*)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.common.readers import site_python  # noqa: E402
from dstar_fit import LEVEL, dstar, fit, saving  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
TEST = os.path.join(DATA, "stage2_test.json")
OUT = os.path.join(DATA, "stage2_readout.json")
NAME = re.compile(r"dsave_s2_(scr|ft)_(.+)_d(\d+)_t(\d+)$")
TARGETS = ["ee_uu", "epem_ttx", "ee_uu_nlo", "ee_ttbar_nlo", "bbx_zz", "bbx_zz_nlo"]   # + D413's second example

REMOTE = r'''
import datetime, glob, json, os, re, sys
import numpy as np, yaml
sys.path.insert(0, ".")
from tools.nlo_virtual_pipeline import born_path
out, pools = [], {}
for d in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/dsave_s2_*/trial_*")):
    sw = os.path.basename(os.path.dirname(d))
    m = re.match(r"dsave_(?:s2|s5[a-z]+|diag_[a-z]+)_(scr|ft)_(.+)_d(\d+)_t(\d+)$", sw)
    r = {"sweep": sw, "run": os.path.basename(d), "hp_idx": int(re.match(r"trial_(\d+)", os.path.basename(d)).group(1))}
    f = d + "/sigma_stage/preds_test.npz"
    lg = d + "/out_0.log"
    if os.path.exists(lg):
        L = open(lg).read().splitlines()
        try:
            t0 = datetime.datetime.strptime(L[0][1:20], "%Y-%m-%d %H:%M:%S").timestamp()
            r["gpu_h"] = ((os.path.getmtime(f) if os.path.exists(f) else os.path.getmtime(lg)) - t0) / 3600
        except (IndexError, ValueError):
            pass
    if not m or not os.path.exists(f):
        r["error"] = "no sigma_stage/preds_test.npz"
        L = L if os.path.exists(lg) else []
        dv = [l for l in L if "Training diverged" in l]
        er = [l for l in L if " E] " in l or "Error" in l]
        r["fail"] = ("diverged at step " + re.search(r"at step (\d+)", dv[-1]).group(1) if dv else
                     (er[-1][:200] if er else "no error in the log (still running or killed)"))
        out.append(r)
        continue
    tg = m.group(2)
    try:
        ob = json.load(open(d + "/data_stats.json")).get("target_over_born")
        z = np.load(f)
        p, t = z["raw_pred"].astype(float).ravel(), z["raw_truth"].astype(float).ravel()
        row = z["pool_row"].astype(int)
        if tg not in pools:
            # the test pool the run read: its recipe's entry, the candidate of that window whose target matches the
            # stored truth row by row (several shapings of one window can sit side by side)
            e = yaml.safe_load(open(yaml.safe_load(open(d + "/config.yaml"))["data"]["processes_file"]))["processes"][0]
            lo, hi = (int(float(x)) for x in e["sqrts"])
            best = None
            for pf in sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{e['name']}_{lo}-{hi}GeV_test_*amplitudes.npy"))):
                v = np.load(pf, mmap_mode="r")[:, -1].astype(float)
                b = np.load(born_path(pf)).astype(float).ravel() if tg.endswith("_nlo") else None
                if len(v) <= row.max():
                    continue
                st = v[row] / b[row] if ob else v[row]
                c = float(np.max(np.abs(t - st) / np.abs(st)))
                if best is None or c < best[0]:
                    best = (c, pf, v, b)
            pools[tg] = best[1:]
        pf, v, b = pools[tg]
        stored = v[row] / b[row] if ob else v[row]          # the pool's own target at the predicted rows
        r["row_check"] = float(np.max(np.abs(t - stored) / np.abs(stored)))
        r["pool"] = os.path.basename(pf)
        r["n"], r["n_pool"] = int(len(t)), int(len(v))
        if tg.endswith("_nlo") and not ob:                    # trained on V: V/B with the exact Born
            p, t = p / b[row], t / b[row]
        ok = p * t > 0
        e2 = (np.log(np.abs(p[ok])) - np.log(np.abs(t[ok]))) ** 2
        r["mse_ln"] = float(np.mean(e2))
        r["top1_share_ln"] = float(e2.max() / e2.sum())   # the share of the MSE that its worst test event carries
        r["prepd_std"] = float(json.load(open(d + "/data_stats.json"))["prepd_std"][0])
        r["trafo"] = json.load(open(d + "/data_stats.json"))["amp_trafos_pp"][0][0]   # log, or signedlog:<s>
        r["wrong_sign"] = int((~ok).sum())
        if tg.endswith("_nlo"):
            r["mse_vb"] = float(np.mean((p - t) ** 2))
    except Exception as e:
        r["error"] = repr(e)[:200]
    out.append(r)
print("S2T " + json.dumps(out))
'''


def searches():
    """{sweep: {arm, target, k, D, T, lo, hi, trials}} from sweeps_local and the committed sweep configs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(REPO, "sweeps_local/dsave_s2_*/driver_state.json"))):
        sw = os.path.basename(os.path.dirname(f))
        arm, tg, k, T = NAME.match(sw).groups()
        k = int(k)
        cfg = os.path.join(REPO, "sweep", "sweep_config_" + sw + ".yaml")
        ss = {s["name"]: s for s in yaml.safe_load(open(cfg))["search_space"]}
        w = ss["training.lr" if arm == "scr" else "fine_tune.lr_scale"]
        out[sw] = {"arm": arm, "target": tg, "k": k, "D": 10 ** (k / 2), "T": int(T), "lo": w["low"], "hi": w["high"],
                   "trials": list(json.load(open(f))["trials"].values())}
    return out


def run_of(sw, t):
    """The run dir basename of a trial, from its result record (a requeued trial runs in trial_<hp>_r<n>)."""
    f = os.path.join(REPO, "sweeps_local", sw, "results", "hp%04d_t%d.json" % (t["hp_idx"], t["t_steps"]))
    return os.path.basename(json.load(open(f))["run_dir"]) if os.path.exists(f) else None


def x_lr(s, hp):
    """The sampled lr in window units: scratch lr / c (c = sqrt(low high)), fine-tune lr_scale."""
    return hp["training.lr"] / np.sqrt(s["lo"] * s["hi"]) if s["arm"] == "scr" else hp["fine_tune.lr_scale"]


def readout():
    S = searches()
    test = {(r["sweep"], r["run"]): r for r in json.load(open(TEST))}
    rows = json.load(open(TEST))
    gpu = {}
    for r in rows:
        gpu[r["sweep"]] = gpu.get(r["sweep"], 0) + (r.get("gpu_h") or 0)
    # a failed trial's cause, from the run dirs of its hp without predictions (a requeued trial left one per attempt)
    fail = {}
    for r in rows:
        if r.get("fail"):
            fail.setdefault((r["sweep"], r["hp_idx"]), []).append(r["fail"])
    cells, trials = [], []
    for sw, s in S.items():
        obs = [t for t in s["trials"] if t.get("status") == "observed"]
        bad = [t for t in s["trials"] if t.get("status") in ("failed", "lost")]
        for t in s["trials"]:
            te = test.get((sw, run_of(sw, t)), {})
            trials.append({"sweep": sw, "arm": s["arm"], "target": s["target"], "k": s["k"], "D": s["D"],
                           "hp_idx": t["hp_idx"], "rid": t["rid"], "status": t.get("status"), "x_lr": x_lr(s, t["hp"]),
                           "val": t.get("val_loss"), "test": te.get("mse_vb", te.get("mse_ln")),
                           "gpu_h": te.get("gpu_h"),
                           "fail": "; ".join(fail.get((sw, t["hp_idx"]), [])) if t.get("status") in ("failed", "lost")
                           else None,
                           "diverged": t.get("status") in ("failed", "lost")
                           and any(x.startswith("diverged") for x in fail.get((sw, t["hp_idx"]), []))})
        c = {k: s[k] for k in ("arm", "target", "k", "D", "T")}
        c.update(sweep=sw, n_trials=len(s["trials"]), n_observed=len(obs), n_failed=len(bad))
        if obs:
            b = min(obs, key=lambda t: t["val_loss"])
            te = test.get((sw, run_of(sw, b)), {})
            c.update(best_rid=b["rid"], best_hp=b["hp_idx"], val=b["val_loss"], x_lr_best=x_lr(s, b["hp"]),
                     test=te.get("mse_vb", te.get("mse_ln")), test_ln=te.get("mse_ln"), row_check=te.get("row_check"),
                     # val x prepd_std^2 is the MSE of ln|.| only under the log transform
                     val_ln=b["val_loss"] * te["prepd_std"] ** 2 if te.get("prepd_std") and te.get("trafo") == "log"
                     else None,
                     top1=te.get("top1_share_ln"), wrong_sign=te.get("wrong_sign"), pool=te.get("pool"), test_error=te.get("error"),
                     search_gpu_h=gpu.get(sw, 0))   # every run dir of the search: failed and requeued attempts too
        cells.append(c)
    # the readout (test: MSE ln|M|^2 for the trees, MSE V/B for the loops) and beside it the selection measure, the
    # best validation mu-MSE in ln units (val x prepd_std^2: MSE of ln|M|^2, of ln|V/B| for t t~ V, of ln|V| for u u~ V)
    # test_ln: MSE(ln|M|^2) for the trees (= test), MSE(ln|V/B|) for the loops, the entry's unit for the V/B targets
    res = {"test": fits(cells, "test"), "val_ln": fits(cells, "val_ln"), "test_ln": fits(cells, "test_ln")}
    print("\ncells (best of each search by val, test measure):")
    for c in sorted(cells, key=lambda c: (c["target"], c["arm"], c["k"])):
        print(f"  {c['sweep']:36s} obs {c['n_observed']}/{c['n_trials']} failed {c['n_failed']}  val "
              f"{c.get('val', float('nan')):.3e}  test {c.get('test') or float('nan'):.3e} (worst event {c.get('top1') or 0:.2f})  best x_lr "
              f"{c.get('x_lr_best', float('nan')):.2f}  search {c.get('search_gpu_h', 0):.2f} GPU-h"
              + (f"  row_check {c['row_check']:.1e} {c.get('pool')}" if c.get("row_check") is not None else "")
              + (f"  ERROR {c['test_error']}" if c.get("test_error") else ""))
    print("\nfailed or lost trials, sampled lr in window units (scratch lr/c, fine-tune lr_scale), with the cause read "
          "in the run's log:")
    for t in sorted((t for t in trials if t["status"] in ("failed", "lost")), key=lambda t: (t["sweep"], t["x_lr"])):
        xo = [u["x_lr"] for u in trials if u["sweep"] == t["sweep"] and u["status"] == "observed"]
        print(f"  {t['sweep']:36s} {t['rid']}  at {t['x_lr']:.2f} (trained up to {max(xo, default=0):.2f}): "
              f"{t['fail'] or 'no run dir read'}")
    json.dump({"fits": res, "cells": cells, "trials": trials}, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    # the claims line (tools/verify_claims.py reads the last {...} line)
    print(json.dumps({m: {tg: {"S": 10 ** f["log10_S"], "lo": 10 ** f["band"][0], "hi": 10 ** f["band"][2],
                               "level": f["level"], "alpha": f["fit"]["alpha"]} for tg, f in r.items()}
                      for m, r in res.items()} | {"n_searches": len(cells)}, default=float))


def fits(cells, key):
    print(f"\n== joint fits on {key}")
    res = {}
    for tg in TARGETS:
        arms = {}
        for a in ("scr", "ft"):
            cc = sorted((c for c in cells if c["target"] == tg and c["arm"] == a and c.get(key)), key=lambda c: c["D"])
            arms[a] = (np.array([c["D"] for c in cc]), np.array([c[key] for c in cc]))
        if min(len(v[0]) for v in arms.values()) < 3:
            print(f"{tg}: under 3 cells in an arm, not fitted")
            continue
        # D373: 1e-4, or the lowest level both arms reach when one does not reach it
        level = max(LEVEL, max(v[1].min() for v in arms.values()))
        sv = saving(arms, level)
        own = {a: fit({a: arms[a]})["alpha"] for a in arms}
        f = sv["fit"]
        res[tg] = {"level": level, **sv, "alpha_own": own, "n_cells": {a: len(v[0]) for a, v in arms.items()},
                   "log10_Dstar": {a: float(np.log10(dstar(*f[a], f["alpha"], level))) for a in arms}}
        b = sv["band"]
        print(f"{tg:13s} level {level:.1e}  alpha {f['alpha']:.2f} (own scr {own['scr']:.2f}, ft {own['ft']:.2f})  "
              f"rms {f['rms']:.2f} dec  log10 D* scr {res[tg]['log10_Dstar']['scr']:.2f} ft "
              f"{res[tg]['log10_Dstar']['ft']:.2f}  S = {10 ** sv['log10_S']:.3g} band [{10 ** b[0]:.3g}, "
              f"{10 ** b[2]:.3g}]  S_A {10 ** sv['log10_S_A']:.3g}  boot {sv['boot']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--diag", action="store_true", help="read the diagnostic runs runs/dsave_diag_*/trial_* into "
                    "data/stage2_diag_test.json (test MSE per run; their val comes from each run's RESULT_JSON)")
    a = ap.parse_args()
    if a.diag:
        rows = site_python(REMOTE, ["'runs/dsave_diag_*/trial_*'"], "S2T")
        json.dump(rows, open(os.path.join(DATA, "stage2_diag_test.json"), "w"), indent=1)
        sys.exit(f"wrote {len(rows)} diagnostic runs")
    if a.fetch:
        rows = site_python(REMOTE, [], "S2T")
        json.dump(rows, open(TEST, "w"), indent=1)
        print("wrote", TEST, len(rows), "trials,", sum("error" in r for r in rows), "without test predictions")
    readout()
