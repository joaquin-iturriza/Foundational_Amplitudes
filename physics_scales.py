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


def qperp_logs(P, massive_shat=False, saturate=False, soft=None):
    """P: (n, N, 4) momenta, rows 0 and 1 incoming, any Lorentz frame. Returns (n,) the sum of log q^2 over the N - 4
    emission steps (GeV^2 units). massive_shat: a massive particle (m > 1 GeV, from its own momentum) takes s_hat as
    its scale (it has no soft or collinear enhancement), the massless ones their dipole scale. saturate: each removed
    step's scale is q^2 s_hat / (q^2 + s_hat), q^2 while the step is soft (q^2 << s_hat) and s_hat / 2 when it is hard,
    for every particle alike (the order of removal is unchanged). soft = tau (log units): each step's log q^2 is the
    soft minimum -tau log sum exp(-log q^2 / tau) over every candidate particle and pair instead of the minimum, so the
    scale has no kink where the minimising pair changes (the removal order is still the hard minimum's)."""
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
        soft_lq = np.full((n, N), np.inf)
        for k in range(2, N):
            # min over pairs a < b (a, b != k, alive) of s_ak s_bk / s_ab
            num = S[:, :, k][:, :, None] * S[:, :, k][:, None, :]
            with np.errstate(over="ignore", divide="ignore"):
                q = num / S
            ok = alive[:, :, None] & alive[:, None, :]
            ok[:, k, :] = False; ok[:, :, k] = False
            ok &= ~np.eye(N, dtype=bool)[None]
            q = np.where(ok, q, np.inf).reshape(n, -1)
            if soft:
                with np.errstate(divide="ignore"):
                    lq = np.log(np.minimum(q, 1e300))
                lq = np.where(np.isfinite(q), lq, np.inf)
                lo = lq.min(1, keepdims=True)
                sq_k = lo[:, 0] - soft * np.log(np.exp(-(lq - lo) / soft).sum(1))           # soft min of log(dipole)
                soft_lq[:, k] = np.where(alive[:, k], np.logaddexp(sq_k, np.log(np.maximum(m2[:, k], 1e-300))), np.inf)
            q = q.min(1) + m2[:, k]
            if massive_shat:
                q = np.where(m2[:, k] > 1.0, np.exp(Ls), q)
            best[:, k] = np.where(alive[:, k], q, np.inf)
        k = best.argmin(1)
        qk = best[np.arange(n), k]
        if soft:
            lo = soft_lq.min(1, keepdims=True)
            qk = np.exp(lo[:, 0] - soft * np.log(np.exp(-(soft_lq - lo) / soft).sum(1)))
        out += np.log(qk * np.exp(Ls) / (qk + np.exp(Ls)) if saturate else qk)
        alive[np.arange(n), k] = False
    return out


def hybrid_cp(P):
    """data.target_physics_hybrid's constant of one process (docs/results.tex, par. The hybrid reference scale, a
    design; analysis/catalog_v3/prep_level.py): c_p = the mean over its events of Lq - (N - 4) log s_hat, Lq the
    emission scales (qperp_logs). The s_hat target plus c_p keeps the s_hat target's shape within the process and takes
    the emission-scale target's class mean. Momenta only, no amplitude; s0 cancels. P: (n, N, 4) momenta in GeV."""
    pp = np.asarray(P, dtype=np.float64)
    q = pp[:, 0] + pp[:, 1]
    ls = np.log(np.abs(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1)))
    return float(np.mean(qperp_logs(pp) - (pp.shape[1] - 4) * ls))


def hybrid_blend(P, cp, lam):
    """data.target_physics_hybrid_blend's term of one process (the user 2026-10-25, "run those two pretrainings blending
    emission scales"): lam times the per-event emission-scale term Lq - (N - 4) log s_hat minus its process mean c_p
    (hybrid_cp over the process's own training events, or the frozen stats' c_p), added to the hybrid target's log
    factor. lam = 0 is the hybrid, lam = 1 the emission-scale target (c_p cancels there, whichever
    events it was fitted on). P: (n, N, 4) momenta in GeV. Returns (n,)."""
    pp = np.asarray(P, dtype=np.float64)
    if float(lam) == 0.0:
        return np.zeros(pp.shape[0])
    q = pp[:, 0] + pp[:, 1]
    ls = np.log(np.abs(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1)))
    return float(lam) * (qperp_logs(pp) - (pp.shape[1] - 4) * ls - float(cp))


def hybrid_cps(names, events, frozen=None):
    """The constants c_p of a run's processes: a process whose c_p the frozen data stats carry (data_stats.json's
    target_physics_hybrid_cp: a fine-tune's parent, a resumed run's own) keeps it, so a fine-tune onto a process its
    parent trained on reads the target the parent's standardization was fitted on; any other process gets its own,
    hybrid_cp(events(name)) (events is called only for those). Returns ({name: c_p}, [names taken from frozen])."""
    frozen = frozen or {}
    cps, reused = {}, []
    for name in names:
        if name in frozen and frozen[name] is not None:
            cps[name] = float(frozen[name])
            reused.append(name)
        else:
            cps[name] = hybrid_cp(events(name))
    return cps, reused
