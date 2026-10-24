"""D83 (CPU iteration on the catalog normalization), step 1 of the 2026-10-23 plan: the per-event scales every candidate
target is built from, on the pools themselves, no training. Per process, over up to --n events of its pool, the mean and
covariance of
    y      log|M|^2
    Ls     log s_hat, s_hat = (p1 + p2)^2                   (the dimensionless target with Lambda^2 = s_hat)
    Lp     the mean over pairs of external particles of log 2 p_i.p_j       (Lambda^2 = the pair scale)
    Lk     the sum over the N - 4 extra particles of the log of their k_T clustering scale (below), log s_hat for the
           steps with nothing left to cluster, (N - 4) log s_hat for N <= 4    (Lambda^2 = the clustering history)
    Lkg    the same clustering gluons and photons only
    Lkm    the same clustering every final particle with m_T^2 = p_T^2 + m^2 (generic: no particle list)
    Lq     the Lorentz-invariant emission scales of physics_scales.qperp_logs (what a training run can use: its stored
           momenta are randomly Lorentz transformed)
    Lq0    the same with every massive particle at s_hat (no soft or collinear enhancement)
    Lqs    the same with each step's scale saturating at s_hat, q^2 s_hat / (q^2 + s_hat) (no particle singled out)
so any target y + a.(Ls, Lp, Lk) + const has its per-process mean and variance at home (norm_targets.py).
The clustering: the massless coloured particles and photons of the final state (|PDG| <= 5, 21, 22), the inclusive k_T
measure with R = 1: d_iB = p_T,i^2, d_ij = min(p_T,i^2, p_T,j^2) Delta R_ij^2; at each of the N - 4 steps the smallest d
is the scale of that step, a pair merges (four-momenta added), a beam clustering removes the particle. Each extra massless
emission multiplies |M|^2 by about g^2 / d of its own scale, which s_hat does not see; massive final particles are
not clustered (no soft or collinear enhancement), their steps carry s_hat.
Sets: the catalog's training and held-out processes (train pools, $SCRATCH cache), the 2->4 probes at both cuts and the
positive one-loop probes with their trees (val pools, $DATA_DIR). CPU job (scripts/job_norm_targets_stats.sh); writes
$SCRATCH/zz_norm_targets/norm_targets_stats.json.
    python analysis/catalog_daniel/norm_targets_stats.py [--n 20000] [--workers 16]
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from physics_scales import qperp_logs  # noqa: E402

SETS = {"train": ("recipes/catalog_daniel_train.yaml", "train"), "heldout": ("recipes/catalog_daniel_heldout.yaml", "train"),
        "2to4": ("recipes/catalog_2to4_heldout.yaml", "val"), "2to4c30": ("recipes/catalog_2to4c30_heldout.yaml", "val"),
        "loop": ("recipes/catalog_loop_probes.yaml", "val")}
CLUSTER = {1, 2, 3, 4, 5, 21, 22}
NMAX = 20000


def pool(name, sq, role):
    d = os.path.join(os.environ["SCRATCH"], "amp_data_cache") if role == "train" else os.environ["DATA_DIR"]
    pat = f"{name}_{sq[0]}-{sq[1]}GeV_{role}_*amplitudes.npy" if sq else f"{name}_*_{role}_*amplitudes.npy"
    g = sorted(glob.glob(os.path.join(d, pat))) or sorted(glob.glob(os.path.join(d, f"{name}_*_{role}_*amplitudes.npy")))
    return g[0] if g else None


def pt2_y_phi(p):
    pt2 = p[..., 1] ** 2 + p[..., 2] ** 2
    y = 0.5 * np.log(np.clip(p[..., 0] + p[..., 3], 1e-300, None) / np.clip(p[..., 0] - p[..., 3], 1e-300, None))
    return pt2, y, np.arctan2(p[..., 2], p[..., 1])


def cluster_logs(P, pdg, Ls, cluster=None, mt=False):
    """sum of log scales of the N - 4 clustering steps, per event (cluster: the PDG codes that can be clustered; "all"
    with mt=True: every final particle, the transverse mass m_T^2 = p_T^2 + m^2 in place of p_T^2, so a massive
    particle's scale is at least its mass and no particle type is singled out)"""
    cluster = CLUSTER if cluster is None else cluster
    n, N, _ = P.shape
    m = N - 4
    if m <= 0:
        return m * Ls
    P = P[:, 2:].copy()
    alive = np.broadcast_to(np.array([cluster == "all" or abs(int(c)) in cluster for c in pdg[2:]]), (n, N - 2)).copy()
    out = np.zeros(n)
    for _ in range(m):
        pt2, y, ph = pt2_y_phi(P)
        if mt:
            pt2 = pt2 + np.clip(P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1), 0, None)
        dB = np.where(alive, pt2, np.inf)
        dphi = np.abs(ph[:, :, None] - ph[:, None, :]); dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dR2 = (y[:, :, None] - y[:, None, :]) ** 2 + dphi ** 2
        dij = np.minimum(pt2[:, :, None], pt2[:, None, :]) * dR2
        ok = alive[:, :, None] & alive[:, None, :] & ~np.eye(N - 2, dtype=bool)[None]
        dij = np.where(ok, dij, np.inf)
        flat = np.concatenate([dB, dij.reshape(n, -1)], 1)
        k = flat.argmin(1); dmin = flat[np.arange(n), k]
        none = ~np.isfinite(dmin)
        out += np.where(none, Ls, np.log(np.where(none, 1.0, dmin)))
        beam = (k < N - 2) & ~none
        rows = np.where(beam)[0]; alive[rows, k[rows]] = False
        pr = np.where(~beam & ~none)[0]
        kk = k[pr] - (N - 2); i, j = kk // (N - 2), kk % (N - 2)
        P[pr, i] += P[pr, j]; alive[pr, j] = False
    return out


def one(args):
    setname, name, sq, role = args
    key = f"{name}_{sq[0]}-{sq[1]}GeV" if (setname == "loop" and sq) else name       # one-loop probes: one per window
    f = pool(name, sq, role)
    if f is None:
        return setname, key, None
    x = np.load(f, mmap_mode="r")
    x = np.asarray(x[: min(len(x), NMAX)], dtype=np.float64)
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); pdg = x[0, 4 * N: 5 * N]
    amp = x[:, -1]
    if np.any(amp <= 0):
        return setname, key, {"signed": True, "N": int(N), "pdg": [int(c) for c in pdg]}
    q = P[:, 0] + P[:, 1]
    Ls = np.log(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))
    G = np.einsum("npc,nqc->npq", P * np.array([1.0, -1.0, -1.0, -1.0]), P)
    iu = np.triu_indices(N, 1)
    Lp = np.log(np.clip(2 * np.abs(G[:, iu[0], iu[1]]), 1e-12, None)).mean(1)
    Lk = cluster_logs(P, pdg, Ls)
    Lkg = cluster_logs(P, pdg, Ls, {21, 22})          # gluons and photons only: a quark of the core is not an emission
    Lkm = cluster_logs(P, pdg, Ls, "all", mt=True)    # every final particle, m_T: no particle type singled out
    Lq = qperp_logs(P)                                 # the Lorentz-invariant emission scales (physics_scales.py)
    Lq0 = qperp_logs(P, massive_shat=True)              # the same, massive particles at s_hat
    Lqs = qperp_logs(P, saturate=True)                  # the same, each step's scale saturating at s_hat (norm_blend.py)
    V = np.stack([np.log(amp), Ls, Lp, Lk, Lkg, Lkm, Lq, Lq0, Lqs], 1)
    return setname, key, {"N": int(N), "pdg": [int(c) for c in pdg], "n": int(len(V)), "file": os.path.basename(f),
                           "mean": V.mean(0).tolist(), "cov": np.cov(V.T).tolist()}


def main():
    global NMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=NMAX)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    NMAX = a.n
    jobs = []
    for s, (rec, role) in SETS.items():
        for p in yaml.safe_load(open(rec))["processes"]:
            jobs.append((s, p["name"], p.get("sqrts"), role))
    out = {s: {} for s in SETS}; missing = []
    with Pool(a.workers) as pl:
        for s, n, r in pl.imap_unordered(one, jobs, chunksize=4):
            if r is None:
                missing.append([s, n])
            else:
                out[s][n] = r
    out["_missing"] = missing
    od = os.path.join(os.environ["SCRATCH"], "zz_norm_targets"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "norm_targets_stats.json"), "w"))
    print(f"NORM_TARGETS_DONE {sum(len(v) for k, v in out.items() if not k.startswith('_'))} processes, "
          f"{len(missing)} missing: {missing[:10]}")


if __name__ == "__main__":
    main()
