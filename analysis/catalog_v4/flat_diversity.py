"""How spread the kept events of a flat-in-log|M|^2 part are within each log|M|^2 bin (plan catalog_coverage step 69,
the design reviewer's and repo-reviewer's FIX of 2026-11-07: fill per bin counts kept events and cannot tell a bin filled
by near-copies of one Markov chain's state from one filled by independent draws).

Per event the features are the natural logs of the invariants: s_ij / s for every pair of finals, |t_{a i}| / s for the
first beam a and every final i, and sqrt(s) (GeV). Per bin: the kept count, the share of kept events with another kept
event of the same bin within EPS in every feature (a near-duplicate: every invariant within 1%), the median
nearest-neighbour (Euclidean) distance, the number of distinct sources (a Markov chain counts once, every independently
drawn event counts as its own source) and the largest share of the bin from one source. Against a reference sampler of
independent draws (the quota sampler, same class, seed and bins): per bin, both sides subsampled to the same count, the
median nearest-neighbour distance of the tested sampler over the reference's, and the Kolmogorov-Smirnov statistic of
sqrt(s), min pair mass / sqrt(s) and max |eta| of the finals (descriptive: the two samplers' laws within a bin differ by
construction and neither is a target)."""
import numpy as np

EPS = 0.01        # near-duplicate: every log-invariant within 0.01 (1%)
MIN_N = 5         # bins with fewer kept events on either side are not compared


def _m2(p):
    return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)


def features(P, sq):
    """P (N, 2 + n, 4) momenta (beams first), sq (N,) sqrt(s) -> (N, D) log-invariants"""
    P = np.asarray(P, float); s = np.asarray(sq, float) ** 2
    n = P.shape[1] - 2
    cols = []
    for i in range(2, 2 + n):
        for j in range(i + 1, 2 + n):
            cols.append(np.abs(_m2(P[:, i] + P[:, j])) / s)
    for i in range(2, 2 + n):
        cols.append(np.abs(_m2(P[:, 0] - P[:, i])) / s)
    with np.errstate(divide="ignore"):
        F = np.log(np.maximum(np.stack(cols, 1), 1e-300))
    return np.concatenate([F, np.log(np.asarray(sq, float))[:, None]], 1)


def shape_vars(P, sq):
    """(sqrt(s), min pair mass / sqrt(s), max |eta| of the finals) per event"""
    P = np.asarray(P, float); n = P.shape[1] - 2; sq = np.asarray(sq, float)
    mp = np.full(len(P), np.inf)
    for i in range(2, 2 + n):
        for j in range(i + 1, 2 + n):
            mp = np.minimum(mp, np.sqrt(np.abs(_m2(P[:, i] + P[:, j]))))
    pt = np.hypot(P[:, 2:, 1], P[:, 2:, 2]); pz = P[:, 2:, 3]
    eta = np.abs(np.arcsinh(pz / np.maximum(pt, 1e-300)))
    return sq, mp / sq, eta.max(1)


def _nn(F):
    """nearest-neighbour Euclidean and Chebyshev distances within F (each point to its nearest other point)"""
    d = F[:, None, :] - F[None, :, :]
    e = np.sqrt((d ** 2).sum(-1)); c = np.abs(d).max(-1)
    np.fill_diagonal(e, np.inf); np.fill_diagonal(c, np.inf)
    return e.min(1), c.min(1)


def per_bin(F, b, src, nb, cap=2000, rng=None):
    """per bin j < nb: n, near-duplicate share, median NN distance, distinct sources, largest one-source share.
    src: chain id per event, -1 for an independent draw. Bins above `cap` events are read on a random `cap` subsample."""
    rng = rng or np.random.default_rng(0)
    out = []
    for j in range(nb):
        idx = np.nonzero(b == j)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        n = len(idx)
        if n < 2:
            out.append({"n": int(n)}); continue
        e, c = _nn(F[idx])
        s = src[idx]; ind = s < 0
        ch, cnt = np.unique(s[~ind], return_counts=True)
        out.append({"n": int(n), "dup": float(np.mean(c < EPS)), "nn_med": float(np.median(e)),
                    "sources": int(ind.sum() + len(ch)),
                    "top_source_share": float(max(cnt.max() if len(cnt) else 0, 1 if ind.any() else 0) / n)})
    return out


def _ks(a, b):
    x = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), x, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), x, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def compare(Fa, Va, ba, Fr, Vr, br, nb, cap=2000, seed=0):
    """per bin: tested (a) against reference (r) at equal count m = min(n_a, n_r, cap): NN-median ratio a / r and the
    KS statistics of the three shape variables (V = shape_vars tuple)"""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(nb):
        ia, ir = np.nonzero(ba == j)[0], np.nonzero(br == j)[0]
        m = min(len(ia), len(ir), cap)
        if m < MIN_N:
            out.append({"m": int(m)}); continue
        ia = rng.choice(ia, m, replace=False); ir = rng.choice(ir, m, replace=False)
        ea, _ = _nn(Fa[ia]); er, _ = _nn(Fr[ir])
        out.append({"m": int(m), "nn_ratio": float(np.median(ea) / max(np.median(er), 1e-300)),
                    "ks_sqrts": _ks(Va[0][ia], Vr[0][ir]), "ks_mpair": _ks(Va[1][ia], Vr[1][ir]),
                    "ks_maxeta": _ks(Va[2][ia], Vr[2][ir])})
    return out


def bins_of(logm, edges):
    """bin index of natural-log |M|^2 values on `edges`, -1 outside"""
    nb = len(edges) - 1
    ok = np.isfinite(logm) & (logm >= edges[0]) & (logm < edges[-1])
    b = np.full(len(logm), -1)
    b[ok] = np.clip(np.searchsorted(edges, logm[ok], side="right") - 1, 0, nb - 1)
    return b
