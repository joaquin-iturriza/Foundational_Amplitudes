"""Per-process target statistics of the arXiv:2606.23791 catalog (D80, the process-to-statistics network): the mean
and standard deviation of log|M|^2 (natural log, the experiment's "log" transform) over each process's train pool,
for the 1352 training and the 139 held-out processes, plus hp58's own prepd_mean/prepd_std (data_stats.json, in its
recipe order) as a check on the training ones. Login node, CPU only; prints JSON after a __B64__ line (base64 of
gzip), brought home as analysis/catalog_daniel/data/pool_stats.json.
    python analysis/catalog_daniel/pool_stats.py [--rows 100000]
"""
import argparse, base64, glob, gzip, json, os

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=100000, help="train events read per process (the pool holds 1e5)")
a = ap.parse_args()
cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
out = {"rows": a.rows, "train": {}, "heldout": {}, "missing": []}
for split, f in (("train", "recipes/catalog_daniel_train.yaml"), ("heldout", "recipes/catalog_daniel_heldout.yaml")):
    for p in yaml.safe_load(open(f))["processes"]:
        g = sorted(glob.glob(os.path.join(cache, p["name"] + "_*_train_*amplitudes.npy")))
        if not g:
            out["missing"].append(p["name"]); continue
        x = np.load(g[0], mmap_mode="r")
        y = np.log(np.asarray(x[: a.rows, -1], dtype=np.float64))
        out[split][p["name"]] = {"mu": float(y.mean()), "sd": float(y.std()), "n": int(len(y)),
                                 "finite": bool(np.isfinite(y).all())}
ds = json.load(open("runs/dancat_het_pre/trial_0058/data_stats.json"))
names = [p["name"] for p in yaml.safe_load(open("recipes/catalog_daniel_train.yaml"))["processes"]]
out["hp58"] = {n: [m, s] for n, m, s in zip(names, ds["prepd_mean"], ds["prepd_std"])}
print("__B64__"); print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
