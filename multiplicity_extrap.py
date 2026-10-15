"""Data side of the multiplicity extrapolation of the paper draft (Sec. "Extrapolating to more legs", J. Spinner):
the particle order the autoregressive (causal) attention mask reads, and the per-multiplicity standardization
mu_N = a + b N with one shared sigma. Numpy only, so the CPU tests run without torch.

order_identical_particles: within every group of identical OUTGOING particles (same PDG id) the particles are
permuted among the slots that group occupies; every other particle keeps its slot. The two beams (rows 0, 1) never
move. mode "pdotP" sorts each group by p.P, P = the summed outgoing momentum, DESCENDING (the hardest first, the
softest last: dropping the last particle is the soft limit); mode "shuffle" permutes each group at random (the
draft's "shuffled" arm). For u dbar -> W+ n g this is [u, dbar, W+, g_1 .. g_n] with p_1.P >= ... >= p_n.P, the
draft's virtuality-ordered gluons behind the fixed hard particles. p.P is Lorentz invariant (= sqrt(s) E in the
centre-of-mass frame), so the order does not depend on the frame the rows are stored in.

fit_affine_multiplicity: mu_N = a + b N fitted by least squares over all train events (event-weighted, i.e. each
process's mean weighted by its event count), N = the number of particles of the process (beams included; for
q qbar -> Z + n g, N = n + 3, so affine in N is the draft's affine in n), and one sigma = the pooled rms of the
residual, sqrt(sum_p n_p (s_p^2 + (m_p - a - b N_p)^2) / sum_p n_p): the maximum-likelihood fit of
y ~ Normal(a + b N, sigma^2).
"""
import numpy as np

ORDER_MODES = ("none", "pdotP", "shuffle")


def _mdot(p, q):
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(axis=-1)


def order_identical_particles(momenta, pdg_ids, mode, rng=None, n_initial=2):
    """momenta (N_ev, P, 4) or flat (N_ev, 4P); pdg_ids (N_ev, P) or (P,). Returns momenta reordered (same shape).

    The PDG ids are unchanged by construction (only identical particles trade slots)."""
    assert mode in ORDER_MODES, f"particle_order {mode!r} not in {ORDER_MODES}"
    if mode == "none":
        return momenta
    flat = momenta.ndim == 2
    n_ev = momenta.shape[0]
    pid = np.asarray(pdg_ids)
    P = pid.shape[-1]
    mom = np.asarray(momenta).reshape(n_ev, P, 4)
    row = pid[0] if pid.ndim == 2 else pid
    if pid.ndim == 2:
        assert (pid == row[None, :]).all(), "a process has one PDG layout for all its events"
    out = mom.copy()
    groups = {}
    for k in range(n_initial, P):
        groups.setdefault(int(row[k]), []).append(k)
    groups = [np.array(g) for g in groups.values() if len(g) > 1]
    if not groups:
        return momenta
    if mode == "pdotP":
        Ptot = mom[:, n_initial:, :].sum(axis=1)                       # (N_ev, 4)
        key = _mdot(mom, Ptot[:, None, :])                             # (N_ev, P)
        for g in groups:
            # descending p.P; argsort of -key is stable, so ties keep their stored order
            o = np.argsort(-key[:, g], axis=1, kind="stable")          # (N_ev, |g|)
            out[:, g, :] = np.take_along_axis(mom[:, g, :], o[:, :, None], axis=1)
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        for g in groups:
            o = np.argsort(rng.random((n_ev, len(g))), axis=1)
            out[:, g, :] = np.take_along_axis(mom[:, g, :], o[:, :, None], axis=1)
    out = out.astype(np.asarray(momenta).dtype, copy=False)
    return out.reshape(n_ev, P * 4) if flat else out


def fit_affine_multiplicity(n_events, n_particles, means, stds):
    """Per-process train statistics (event counts, particle counts, mean and std of the transformed target) ->
    (a, b, sigma) of mu_N = a + b N with one shared sigma. Needs at least two distinct N."""
    n = np.asarray(n_events, dtype=np.float64)
    N = np.asarray(n_particles, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    assert len(set(N.tolist())) >= 2, "an affine fit in the multiplicity needs processes of two multiplicities"
    w = n / n.sum()
    Nb, mb = (w * N).sum(), (w * m).sum()
    # event-weighted least squares: the within-process spread does not move the line (its x is constant per process)
    b = float((w * (N - Nb) * (m - mb)).sum() / (w * (N - Nb) ** 2).sum())
    a = float(mb - b * Nb)
    sigma = float(np.sqrt((w * (s ** 2 + (m - a - b * N) ** 2)).sum()))
    return a, b, sigma
