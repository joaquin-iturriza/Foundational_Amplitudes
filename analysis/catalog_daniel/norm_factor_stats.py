"""D82, the CPU check of normalization A: how much of the spread of log|M|^2 across the catalog's processes the
per-event factor F = s_hat^(4-N) C_p takes out (s_hat = (p1 + p2)^2 of the two incoming momenta, N the particle count,
C_p the leading diagram's coupling product, diagram_couplings.py). Per process, over its train pool: the mean and
variance of y = log|M|^2 and of L = log s_hat (GeV^2) and their covariance, so every variant (y, y - log C_p,
y - (4-N)(L - log s0), both; any reference s0) follows at home (norm_factor_check.py); and the largest relative
violation of momentum conservation, which checks that the first two particles are the incoming ones. Login node, CPU
only; prints JSON after a __B64__ line (base64 of gzip), brought home as data/norm_factor_stats.json.
    python analysis/catalog_daniel/norm_factor_stats.py
"""
import base64, glob, gzip, json, os

import numpy as np
import yaml

cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
out = {"train": {}, "heldout": {}, "missing": []}
for split, f in (("train", "recipes/catalog_daniel_train.yaml"), ("heldout", "recipes/catalog_daniel_heldout.yaml")):
    for p in yaml.safe_load(open(f))["processes"]:
        g = sorted(glob.glob(os.path.join(cache, p["name"] + "_*_train_*amplitudes.npy")))
        if not g:
            out["missing"].append(p["name"]); continue
        x = np.asarray(np.load(g[0], mmap_mode="r"), dtype=np.float64)
        n = (x.shape[1] - 1) // 5                      # n*4 momenta, n PDG ids, the amplitude
        P = x[:, : 4 * n].reshape(len(x), n, 4)        # (E, px, py, pz)
        pin = P[:, 0] + P[:, 1]
        s = pin[:, 0] ** 2 - (pin[:, 1:] ** 2).sum(1)
        cons = np.abs(pin - P[:, 2:].sum(1)).max(1) / pin[:, 0]
        y, L = np.log(x[:, -1]), np.log(s)
        out[split][p["name"]] = {"n": int(len(y)), "N": int(n), "my": float(y.mean()), "vy": float(y.var()),
                                 "mL": float(L.mean()), "vL": float(L.var()),
                                 "cyL": float(((y - y.mean()) * (L - L.mean())).mean()),
                                 "sqrt_s_range": [float(np.sqrt(s.min())), float(np.sqrt(s.max()))],
                                 "cons_max": float(cons.max())}
print("__B64__"); print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
