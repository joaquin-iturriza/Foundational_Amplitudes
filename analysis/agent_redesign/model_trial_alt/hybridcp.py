def hybrid_cp(P):
    """Alternative fix (model trial): c_p = mean(Lq - (N - 4) log s_hat) over the events."""
    P = np.array(P, dtype=float)
    beams = P[:, 0, :] + P[:, 1, :]
    shat = np.abs(beams[:, 0] ** 2 - np.sum(beams[:, 1:] ** 2, axis=1))
    return float((qperp_logs(P) - (P.shape[1] - 4) * np.log(shat)).mean())


def hybrid_cps(names, events_of, frozen=None):
    """Alternative fix (model trial): frozen values kept, the rest fitted."""
    keep = {k: float(v) for k, v in (frozen or {}).items() if v is not None and k in names}
    out = {n: keep[n] if n in keep else hybrid_cp(events_of(n)) for n in names}
    return out, [n for n in names if n in keep]
