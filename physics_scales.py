"""Per-event physical scales for the regression target, from the momenta alone (no particle list, no data of the
process). One definition for the training target (experiment._physics_logF, data.target_physics_dim_ref=qperp) and
the CPU proxy that chose it (analysis/catalog_daniel/norm_targets_stats.py).

qperp_logs: the emission history of an event, Lorentz invariant. Each extra particle beyond a 2->2 core multiplies |M|^2
by about g^2 / q_perp^2 of its own emission scale; s_hat does not see that scale. For every final particle k the scale
is q_k^2 = min over pairs a < b of the other external legs (incoming included) of s_ak s_bk / s_ab, plus m_k^2 (the
dipole transverse momentum of k off the pair, s_xy = 2 |p_x.p_y|; a massive particle's scale is at least its mass). The
particle with the smallest q^2 is removed and the rest re-evaluated, N - 4 times; the sum of the logs of those N - 4
scales is returned, and (N - 4) log s_hat for N <= 4 (the plain dimensionless target there). Every particle type is
treated alike (the user, 2026-10-22: no gluon- or photon-specific prior in the target).
"""
import numpy as np


def minkowski_dots(P):
    """(n, N, N) p_i.p_j"""
    return np.einsum("npc,nqc->npq", P * np.array([1.0, -1.0, -1.0, -1.0]), P)


def qperp_logs(P):
    """P: (n, N, 4) momenta, rows 0 and 1 incoming, any Lorentz frame. Returns (n,) the sum of log q^2 over the N - 4
    emission steps (GeV^2 units)."""
    P = np.asarray(P, dtype=np.float64)
    n, N, _ = P.shape
    G = minkowski_dots(P)
    Ls = np.log(np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1]))
    if N <= 4:
        return (N - 4) * Ls
    S = np.clip(2 * np.abs(G), 1e-300, None)
    m2 = np.clip(np.einsum("nii->ni", G), 0, None)
    alive = np.ones((n, N), bool)
    out = np.zeros(n)
    for _ in range(N - 4):
        best = np.full((n, N), np.inf)
        for k in range(2, N):
            # min over pairs a < b (a, b != k, alive) of s_ak s_bk / s_ab
            num = S[:, :, k][:, :, None] * S[:, :, k][:, None, :]
            with np.errstate(over="ignore", divide="ignore"):
                q = num / S
            ok = alive[:, :, None] & alive[:, None, :]
            ok[:, k, :] = False; ok[:, :, k] = False
            ok &= ~np.eye(N, dtype=bool)[None]
            q = np.where(ok, q, np.inf).reshape(n, -1).min(1) + m2[:, k]
            best[:, k] = np.where(alive[:, k], q, np.inf)
        k = best.argmin(1)
        out += np.log(best[np.arange(n), k])
        alive[np.arange(n), k] = False
    return out
