"""D83, the CPU exploration of the catalog normalization: per process, the moments of the per-event quantities a
data-free normalization could use, so every combination of them is evaluated at home (norm_explore.py):
  y   log|M|^2
  L   log s_hat, s_hat = (p1 + p2)^2 (GeV^2; the mass dimension 8 - 2N of |M|^2)
  E   sum over final massless partons and photons k of log(s_hat / min_j s_kj), s_kj = (p_k + p_j)^2 over every other
      external leg j (incoming ones as -2 p.k): how close the event sits to a soft or collinear limit
Stored: the mean vector and covariance of (y, L, E) over the pool, and the same mean vector and count in 20 bins of L
(equal width between the pool's 1st and 99th percentile), from which a reweighting of the pool in s_hat follows.
Login node, CPU only, detached: python analysis/catalog_daniel/norm_moments.py <out.json> (progress on stderr).
"""
import json, os, sys, time

import numpy as np
import yaml

MASSLESS = {1, 2, 3, 4, 21, 22}            # partons and the photon with zero generator mass (b: 4.7 GeV, kept out)
cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
files = sorted(os.listdir(cache))
out = {"train": {}, "heldout": {}, "missing": [], "nbins": 20}
t0 = time.time()


def mdot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


for split, f in (("train", "recipes/catalog_daniel_train.yaml"), ("heldout", "recipes/catalog_daniel_heldout.yaml")):
    for p in yaml.safe_load(open(f))["processes"]:
        g = [os.path.join(cache, x) for x in files if x.startswith(p["name"] + "_") and "_train_" in x
             and x.endswith("amplitudes.npy")]
        if not g:
            out["missing"].append(p["name"]); continue
        x = np.asarray(np.load(g[0], mmap_mode="r"), dtype=np.float64)
        n = (x.shape[1] - 1) // 5
        P = x[:, : 4 * n].reshape(len(x), n, 4)
        pdg = np.abs(x[0, 4 * n: 5 * n]).astype(int)
        pin = P[:, 0] + P[:, 1]
        s = mdot(pin, pin)
        E = np.zeros(len(x))
        for k in range(2, n):
            if pdg[k] not in MASSLESS:
                continue
            inv = []
            for j in range(n):
                if j == k:
                    continue
                if j < 2:
                    inv.append(2 * mdot(P[:, j], P[:, k]))          # |t|-like with an incoming leg
                else:
                    q = P[:, j] + P[:, k]
                    inv.append(mdot(q, q))
            m = np.min(np.abs(np.stack(inv, 1)), 1)
            E += np.log(s / np.maximum(m, 1e-12 * s))
        y, L = np.log(x[:, -1]), np.log(s)
        V = np.stack([y, L, E], 1)
        lo, hi = np.percentile(L, [1, 99])
        b = np.clip(((L - lo) / max(hi - lo, 1e-9) * out["nbins"]).astype(int), 0, out["nbins"] - 1)
        cnt = np.bincount(b, minlength=out["nbins"])
        mb = np.stack([np.bincount(b, V[:, c], minlength=out["nbins"]) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        out[split][p["name"]] = {"n": int(len(y)), "N": int(n), "pdg": pdg.tolist(), "mean": V.mean(0).tolist(),
                                 "cov": np.cov(V.T).tolist(), "bin_count": cnt.tolist(), "bin_mean": mb.tolist()}
        k = len(out["train"]) + len(out["heldout"])
        if k % 100 == 0:
            print(f"{k} processes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
json.dump(out, open(sys.argv[1], "w"))
print(f"wrote {sys.argv[1]}: {len(out['train'])} + {len(out['heldout'])} processes in {time.time() - t0:.0f} s",
      file=sys.stderr, flush=True)
