"""Decomposed targets (D48): the target columns and the coherent readout, as pure functions.

The parts of |M|^2 (tools/decomp_targets.py) are X_ab = sum_h A_a(h) A_b(h)^*, the helicity-summed Gram matrix of
the colour-ordered partial amplitudes (basis "colour") or of the diagram amplitudes ("diagram"), with
|M|^2 = Re sum_ab K_ab X_ab exactly. The overall phase of each helicity amplitude is a convention and cancels in X.

Net output per event, n parts: a log-magnitude rho_a and a direction u_a in C^n (|u_a| = 1) per part,
    X^pred_ab = e^{rho_a + rho_b} u_a . u_b^*        (every Hermitian positive semidefinite X is reachable)
    |M|^2_pred = Re sum_ab Kt_ab X^pred_ab,  Kt = K / tr K,
in units of e^m / tr K (m the mean of ln|M|^2, which cancels in the standardized target). The loss on
the standardized ln|M|^2 is the usual one; the parts loss compares each part's share ln(X_aa/|M|^2_true), in
units of the target's std, and the normalised interference X_ab / sqrt(X_aa X_bb) of each pair (bounded by 1).
"""
import numpy as np
import torch


def features(X, M):
    """Target columns of one block: [ln(X_aa/|M|^2) (n), Re c_ab, Im c_ab (a<b)], c_ab = X_ab/sqrt(X_aa X_bb).
    A part that vanishes on some event has no finite share: it is refused, not floored (a floor would put a
    share of about -690 into the loss)."""
    M = np.asarray(M, dtype=np.float64).reshape(-1)
    nb = X.shape[1]
    diag = X[:, np.arange(nb), np.arange(nb)].real
    share = diag / M[:, None]
    assert (share > 1e-12).all(), (f"a part is (numerically) zero on some events (min share per part "
                                   f"{share.min(0)}): drop it from the basis before training on it")
    cols = [np.log(share)]
    iu, ju = np.triu_indices(nb, 1)
    if len(iu):
        c = X[:, iu, ju] / np.sqrt(diag[:, iu] * diag[:, ju])
        cols += [c.real, c.imag]
    out = np.concatenate(cols, axis=1)
    assert np.isfinite(out).all(), "non-finite decomposed target"
    return out


def n_out(nb):
    return nb * (1 + 2 * nb)


def readout(raw, Kt, logc, s, process_ids, nb):
    """(B, n(1+2n)) net outputs -> (y_pred (B,1) standardized ln|M|^2, log-share offset (B,n), Gram of the unit
    directions (B,n,n)). Kt (P,n,n) complex128, logc (P,) = ln tr K, s (P,) the target's std per process.
    The coherent sum is taken in double precision: a basis with strong cancellations (diagrams of a gauge
    theory) would otherwise round S to <= 0."""
    B = raw.shape[0]
    rho = raw[:, :nb].double()
    v = raw[:, nb:].double().reshape(B, nb, 2, nb)
    u = torch.complex(v[:, :, 0], v[:, :, 1])
    u = u / torch.sqrt((u.abs() ** 2).sum(-1, keepdim=True) + 1e-12)
    gram = torch.einsum("bar,bcr->bac", u, u.conj())
    mag = torch.exp(rho[:, :, None] + rho[:, None, :])
    S = (Kt[process_ids] * gram * mag).sum((1, 2)).real
    y_pred = (torch.log(S.clamp(min=1e-300)) / s[process_ids])[:, None].to(raw.dtype)
    lhat = 2.0 * rho - logc[process_ids][:, None]
    return y_pred, lhat, gram


def parts_errors(lhat, gram, y, s, process_ids, nb):
    """Per-event errors (B, n + 2k): each part's share (natural log) and Re, Im of each pair's interference."""
    y = y.double()
    sp = s[process_ids][:, None]
    iu, ju = torch.triu_indices(nb, nb, 1, device=lhat.device)
    g, k = gram[:, iu, ju], iu.numel()
    return torch.cat([(lhat - sp * y[:, :1]) - y[:, 1:1 + nb],
                      g.real - y[:, 1 + nb:1 + nb + k],
                      g.imag - y[:, 1 + nb + k:1 + nb + 2 * k]], 1)


def parts_loss(lhat, gram, y, s, process_ids, nb):
    """Per-event parts loss (B,): mean squared share error in units of the target's std, plus the mean squared
    interference error over pairs."""
    e = parts_errors(lhat, gram, y, s, process_ids, nb)
    loss = ((e[:, :nb] / s[process_ids][:, None]) ** 2).mean(1)
    if e.shape[1] > nb:
        k = (e.shape[1] - nb) // 2
        loss = loss + (e[:, nb:nb + k] ** 2 + e[:, nb + k:] ** 2).mean(1)
    return loss


def exact_raw(X, m, logc):
    """Net outputs that represent X exactly (tests): X = Z Z^dagger, rho_a = ln|z_a| + (ln tr K - m)/2,
    u_a = z_a/|z_a|; m is the mean of ln|M|^2 of the standardization."""
    w, V = np.linalg.eigh(X)
    Z = V * np.sqrt(np.clip(w, 0, None))[:, None, :]
    r = np.linalg.norm(Z, axis=2)
    u = Z / r[..., None]
    rho = np.log(r) + 0.5 * (logc - m)
    return np.concatenate([rho, np.stack([u.real, u.imag], axis=2).reshape(X.shape[0], -1)], axis=1)
