"""Per-validation σ-ranking probe for the two-stage σ-head SPEED study.

The reweighting scheme uses σ only through π(x) ∝ Q(x)·σ(x)^α, then renormalizes,
so the ABSOLUTE scale / calibration of σ is irrelevant — a global scale error on σ
cancels in π's normalization. What decides whether σ is USABLE as a reweighting
signal is the ORDERING of σ over phase space: does a larger σ pick out a genuinely
harder (higher local error) event? So the figure of merit here is the RANK
correlation between σ and the true local error |r|, NOT pull-std / calibration.

`rank_metrics` runs one forward pass over a loader (μ frozen → cheap) and returns:
  spearman_global : Spearman ρ(σ, |r|) over all events (cross- + within-process)
  spearman_proc   : median over processes of within-process Spearman ρ(σ, |r|)
                    — the high-dim signal: does σ order events WITHIN a process,
                    where the cross-process scale can't do the work for it
  slope           : reliability slope log10(RMS |r| per σ-decile) vs log10(σ)
                    (1 = tracks; secondary, calibration-flavoured)
  mu_mse          : μ-MSE (prepd) — constant across a σ-only fit; freeze self-check
  n               : events used

Import-light (numpy only for the stats); the model forward mirrors
experiment._collect_predictions but also keeps process ids.
"""
import numpy as np
import torch


def _rankdata(a):
    """Average-rank of a 1-D array (ties averaged), no scipy."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return ranks


def _spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _reliability_slope(sigma, resid):
    s = np.asarray(sigma, float)
    r = np.asarray(resid, float)
    qs = np.quantile(s, np.linspace(0, 1, 11))
    px, py = [], []
    for i in range(10):
        m = (s >= qs[i]) & (s <= qs[i + 1]) if i == 9 else (s >= qs[i]) & (s < qs[i + 1])
        if m.sum() >= 10:
            px.append(s[m].mean())
            py.append(np.sqrt(np.mean(r[m] ** 2)))
    if len(px) < 3:
        return float("nan")
    px = np.clip(px, 1e-30, None)
    py = np.clip(py, 1e-30, None)
    return float(np.polyfit(np.log10(px), np.log10(py), 1)[0])


@torch.no_grad()
def rank_metrics(exp, loader, max_events=50000, min_proc=200):
    """Collect (pred, truth, σ, process_id) over `loader` and compute rank FoMs.

    exp        : the AmplitudeExperiment (frozen model, self.model in eval)
    max_events : cap collected events (subsample the loader) to keep it cheap
    min_proc   : min events for a process to enter the per-process median
    """
    is_lloca = exp.modelname in (
        "LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim")
    out_shape = (exp.cfg.model.net.get("out_shape")
                 or exp.cfg.model.net.get("out_channels") or 1)
    preds, truths, sigmas, procs = [], [], [], []
    n = 0
    exp.model.eval()
    for data in loader:
        particles, y, tokens, order_labels, ptr, process_ids = data
        particles = particles.to(exp.device)
        tokens = tokens.to(exp.device)
        order_labels = order_labels.to(exp.device)
        ptr = ptr.to(exp.device)
        pid = process_ids.to(exp.device)
        if is_lloca:
            y_pred = exp.model(
                particles, tokens,
                mean=exp.mom_mean[0], std=exp.mom_std[0],
                ptr=ptr, order_labels=order_labels, process_ids=pid,
            )
        else:  # non-lloca legacy path not used here
            raise RuntimeError("sigma_speed_probe supports the LLoCa path only")
        sig = y_pred[..., -out_shape:]
        mu = y_pred[..., :-out_shape]
        preds.append(mu.cpu().float().numpy().reshape(-1))
        truths.append(y.cpu().float().numpy().reshape(-1))
        sigmas.append(sig.cpu().float().numpy().reshape(-1))
        procs.append(process_ids.cpu().numpy().reshape(-1))
        n += len(y)
        if n >= max_events:
            break

    mu = np.concatenate(preds)
    y = np.concatenate(truths)
    s = np.concatenate(sigmas)
    p = np.concatenate(procs)
    r = np.abs(y - mu)

    spear_g = _spearman(s, r)
    per = []
    for pv in np.unique(p):
        m = p == pv
        if m.sum() >= min_proc:
            rho = _spearman(s[m], r[m])
            if np.isfinite(rho):
                per.append(rho)
    spear_p = float(np.median(per)) if per else float("nan")
    return dict(
        spearman_global=spear_g,
        spearman_proc=spear_p,
        n_proc=len(per),
        slope=_reliability_slope(s, r),
        mu_mse=float(np.mean((mu - y) ** 2)),
        n=int(len(y)),
    )
