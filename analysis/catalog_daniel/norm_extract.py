"""D83 exploration: per-event features for the CPU shape-transfer proxy (norm_explore6.py). For every catalog process
(train and held-out recipes) the first --n events of its train pool ($SCRATCH/amp_data_cache, 1e5 per process): y =
log|M|^2, L = log s_hat, the logs of every pair invariant over s_hat sorted ascending and padded to 9 (s_ij as
norm_moments2.py: |(p_i+p_j)^2| between final legs, 2|p_i.p_j| between an incoming and a final leg), Lg, Lmin, Lnn, E
of norm_moments2.py. Login node, CPU only:
    python analysis/catalog_daniel/norm_extract.py <out.npz> [--n 400]
"""
import argparse, os, sys, time

import numpy as np
import yaml

MASSLESS = {1, 2, 3, 4, 21, 22}
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
files = sorted(os.listdir(cache))
t0 = time.time()


def mdot(x, y):
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)


names, split, Y, F, NP = [], [], [], [], []
for sp, f in (("train", "recipes/catalog_daniel_train.yaml"), ("heldout", "recipes/catalog_daniel_heldout.yaml")):
    for p in yaml.safe_load(open(f))["processes"]:
        g = [x for x in files if x.startswith(p["name"] + "_") and "_train_" in x and x.endswith("amplitudes.npy")]
        if not g:
            continue
        x = np.asarray(np.load(os.path.join(cache, g[0]), mmap_mode="r")[: a.n], dtype=np.float64)
        n = (x.shape[1] - 1) // 5
        P = x[:, : 4 * n].reshape(len(x), n, 4)
        pdg = np.abs(x[0, 4 * n: 5 * n]).astype(int)
        s = mdot(P[:, 0] + P[:, 1], P[:, 0] + P[:, 1])
        S = np.full((len(x), n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                if i < 2 and j < 2:
                    continue
                v = 2 * np.abs(mdot(P[:, i], P[:, j])) if i < 2 else np.abs(mdot(P[:, i] + P[:, j], P[:, i] + P[:, j]))
                S[:, i, j] = S[:, j, i] = np.maximum(v, 1e-12 * s)
        lS = np.log(S)
        iu = [(i, j) for i in range(n) for j in range(i + 1, n) if not (i < 2 and j < 2)]
        pairs = np.stack([lS[:, i, j] for i, j in iu], 1)
        nn = np.stack([np.nanmin(lS[:, k, :], 1) for k in range(2, n)], 1)
        E = sum(np.log(s) - nn[:, kk] for kk, k in enumerate(range(2, n)) if pdg[k] in MASSLESS) if any(
            pdg[k] in MASSLESS for k in range(2, n)) else np.zeros(len(x))
        L = np.log(s)
        rel = np.sort(pairs - L[:, None], 1)
        rel = np.concatenate([rel, np.zeros((len(x), 9 - rel.shape[1]))], 1)
        F.append(np.concatenate([L[:, None], rel, pairs.mean(1)[:, None], pairs.min(1)[:, None], nn.mean(1)[:, None],
                                 np.asarray(E)[:, None]], 1).astype(np.float32))
        Y.append(np.log(x[:, -1]).astype(np.float32))
        names.append(p["name"]); split.append(sp); NP.append(n)
        if len(names) % 200 == 0:
            print(f"{len(names)} processes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
np.savez_compressed(a.out, names=np.array(names), split=np.array(split), npart=np.array(NP),
                    y=np.stack(Y), feats=np.stack(F))
print(f"wrote {a.out}: {len(names)} processes x {a.n} events in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
