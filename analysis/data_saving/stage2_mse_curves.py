#!/usr/bin/env python3
"""pretrain_data_saving stage 2 (Joaquin's review 2026-11-12): the mu-MSE curves each stage-2 best trial logged itself
(its full result JSON on CC: val_mse_curve, train_eval_mse_curve at every validation, validate_every; the same numbers
as the run's own plots_0/mse.png), re-based to ln units by the run's prepd_std (data_stats.json, else
data/stage2_test.json), with the HETEROSC validation loss curve beside, the lr schedule (train_steps.npz) and the
target standardization of the run and of its parent. Cache data/stage2_mse_curves.json.

    python analysis/data_saving/stage2_mse_curves.py [--cached]       # fetch (or read the cache) and print a summary
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")

REMOTE = r'''
import json, os, re, sys
import numpy as np, yaml
out = []
for d in sys.argv[1:]:
    r = {"run_dir": d}
    try:
        c = yaml.safe_load(open(d + "/config.yaml"))
        rp = c["training"]["result_path"]
        j = json.load(open(rp))
        r["every"] = j.get("validate_every")
        for k in ("val_mse_curve", "train_eval_mse_curve", "val_curve"):
            r[k] = j.get(k)
        for f, key in ((d + "/data_stats.json", "prepd_std"),):
            if os.path.exists(f):
                r[key] = float(np.ravel(json.load(open(f))["prepd_std"])[0])
        pp = (c.get("fine_tune") or {}).get("pretrained_path")
        if pp:
            pdir = os.path.dirname(os.path.dirname(pp))
            r["parent_prepd_std"] = float(np.ravel(json.load(open(pdir + "/data_stats.json"))["prepd_std"])[0])
        if os.path.exists(d + "/train_steps.npz"):
            z = np.load(d + "/train_steps.npz")
            k = max(1, len(z["step"]) // 200)
            r["lr_step"], r["lr"] = z["step"][::k].tolist(), z["lr"][::k].tolist()
        r["hp"] = {k: c["training"].get(k.split(".")[-1]) for k in ("training.lr", "training.cosanneal_warmup_frac")}
        r["lr_scale"] = (c.get("fine_tune") or {}).get("lr_scale")
    except Exception as e:
        r["error"] = repr(e)[:200]
    out.append(r)
print("S2MSE " + json.dumps(out))
'''


def load(cached=False):
    cache = os.path.join(DATA, "stage2_mse_curves.json")
    if cached:
        return json.load(open(cache))
    cells = json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]
    rows = site_python(REMOTE, [f"runs/{c['sweep']}/trial_{c['best_hp']:04d}" for c in cells], "S2MSE")
    test = {(t["sweep"], t["hp_idx"]): t for t in json.load(open(os.path.join(DATA, "stage2_test.json")))}
    for r, c in zip(rows, cells):
        r.update(arm=c["arm"], target=c["target"], D=c["D"], T=c["T"], rid=c["best_rid"])
        if "prepd_std" not in r and test.get((c["sweep"], c["best_hp"]), {}).get("prepd_std"):
            r["prepd_std"] = test[(c["sweep"], c["best_hp"])]["prepd_std"]
    json.dump(rows, open(cache, "w"))
    return rows


if __name__ == "__main__":
    for r in load("--cached" in sys.argv):
        v = r.get("val_mse_curve") or []
        print(r.get("target"), r.get("arm"), r.get("D"), r.get("error", ""), len(v), r.get("prepd_std"), r.get("parent_prepd_std"))
