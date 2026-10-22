"""D83 exploration, second pass: per-event scales that see where an event sits relative to its singular regions (the
first pass found log|M|^2 rising with s_hat inside a process, slope +1.1 at 2->2 and +0.6 at 2->3, against the 0 and -1
of the mass dimension: cuts fixed in GeV let a higher-s_hat event reach closer to its soft and collinear limits).
Per event, with s_ij = |(p_i + p_j)^2| between final legs and 2|p_i.p_j| between an incoming and a final leg:
  y     log|M|^2
  L     log s_hat
  Lg    the mean of log s_ij over all pairs (the log of their geometric mean)
  Lmin  log of the smallest s_ij
  Lnn   the mean over final legs of log of the smallest s_ij of that leg
  E     as norm_moments.py: sum over final massless partons and photons of log(s_hat / that leg's smallest s_ij)
Stored per process: the mean vector and covariance of (y, L, Lg, Lmin, Lnn, E), and the mean vector and count in 20
bins of L (between the pool's 1st and 99th percentile). Login node, CPU only, detached:
    python analysis/catalog_daniel/norm_moments2.py <out.json>
"""
import json, os, sys, time

import numpy as np
import yaml

MASSLESS = {1, 2, 3, 4, 21, 22}
VARS = ["y", "L", "Lg", "Lmin", "Lnn", "E"]
cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
files = sorted(os.listdir(cache))
out = {"train": {}, "heldout": {}, "missing": [], "nbins": 20, "vars": VARS}
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
        S = np.full((len(x), n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                if i < 2 and j < 2:
                    continue
                if i < 2:
                    v = 2 * np.abs(mdot(P[:, i], P[:, j]))
                else:
                    q = P[:, i] + P[:, j]
                    v = np.abs(mdot(q, q))
                S[:, i, j] = S[:, j, i] = np.maximum(v, 1e-12 * s)
        lS = np.log(S)
        iu = [(i, j) for i in range(n) for j in range(i + 1, n) if not (i < 2 and j < 2)]
        pairs = np.stack([lS[:, i, j] for i, j in iu], 1)
        Lg, Lmin = pairs.mean(1), pairs.min(1)
        nn = np.stack([np.nanmin(lS[:, k, :], 1) for k in range(2, n)], 1)
        Lnn = nn.mean(1)
        E = np.zeros(len(x))
        for kk, k in enumerate(range(2, n)):
            if pdg[k] in MASSLESS:
                E += np.log(s) - nn[:, kk]
        y, L = np.log(x[:, -1]), np.log(s)
        V = np.stack([y, L, Lg, Lmin, Lnn, E], 1)
        lo, hi = np.percentile(L, [1, 99])
        b = np.clip(((L - lo) / max(hi - lo, 1e-9) * out["nbins"]).astype(int), 0, out["nbins"] - 1)
        cnt = np.bincount(b, minlength=out["nbins"])
        mb = np.stack([np.bincount(b, V[:, c], minlength=out["nbins"]) for c in range(V.shape[1])], 1) \
            / np.maximum(cnt, 1)[:, None]
        out[split][p["name"]] = {"n": int(len(y)), "N": int(n), "mean": V.mean(0).tolist(), "cov": np.cov(V.T).tolist(),
                                 "bin_count": cnt.tolist(), "bin_mean": mb.tolist()}
        k = len(out["train"]) + len(out["heldout"])
        if k % 100 == 0:
            print(f"{k} processes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
json.dump(out, open(sys.argv[1], "w"))
print(f"wrote {sys.argv[1]}: {len(out['train'])} + {len(out['heldout'])} processes in {time.time() - t0:.0f} s",
      file=sys.stderr, flush=True)
