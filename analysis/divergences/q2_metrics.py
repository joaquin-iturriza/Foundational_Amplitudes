"""Shared metric layer for the sigma-reweighting (Q2) figures.

Recovers what the lost `recompute_q2_logflat.py` did, and is imported by both
`plot_q2_logflat.py` and `plot_q2_rankmag.py` so the two figures cannot disagree about a
number they both show.

THE METRIC, AND WHY THERE ARE TWO OF THEM

Each arm resamples the ee->uug antenna fine-tune pool by pi ~ Q*score^alpha and is then
evaluated on the same fixed deep-IR test set. An arm's quality is reported as the fraction of
the perfect-oracle reweighting gain it captures,

    gain(arm) = (M_base - M_arm) / (M_base - M_oracle),

so the no-reweighting baseline is 0 and the oracle (reweighting by the true |residual|) is 1.

M is where the two metrics differ:

  event-weighted   M = mean over events of d^2
  log-flat         M = mean over y_min DECADES of (mean of d^2 within that decade)

The antenna test set puts ~50% of its events below y_min = 1e-3, so the event-weighted mean is
dominated by the deep IR and flatters any heuristic that merely follows the deep IR. The
log-flat metric gives every decade equal weight and is the honest one; the whole point of
Fig. q2_logflat is that the two disagree, so both are computed here rather than one being
quietly preferred.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: y_min decade edges of the eval. Six decades, [1e-6, 1e0).
EDGES = np.logspace(-6.0, 0.0, 7)

#: Decade centres, for the per-decade profile panel.
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])

BASE, ORACLE = "baseQ", "oracle"

#: Human-readable series names. The npz basenames are internal tags (`deg029`, `rank_synth46`);
#: a figure legend has to say what the arm IS, so the mapping lives here and no plotting script
#: prints a raw tag.
LABEL = {
    "baseQ":        r"no reweighting",
    "oracle":       r"oracle $|r|$",
    "sigma":        r"$\sigma$ head",
    "deg029":       r"degraded $\sigma$, $\rho=0.29$",
    "rank_real":    r"order only, $\sigma$ order",
    "rank_synth30": r"order only, $\rho=0.30$",
    "rank_synth46": r"order only, $\rho=0.46$",
    "rank_synth70": r"order only, $\rho=0.70$",
}

#: Spearman rank correlation with the true |residual| for each order-only arm. The synthetic
#: arms were BUILT to a target rho (hence the tag), and rho=0.46 is the measured within-process
#: ranking quality of the real sigma head on single-process uug -- which is why synth46 exists:
#: it is the synthetic arm matched to the real head.
RHO = {"baseQ": 0.0, "rank_synth30": 0.30, "rank_synth46": 0.46,
       "rank_synth70": 0.70, "rank_real": 0.46, "oracle": 1.0, "sigma": 0.46}


def load(tag: str, eval_dir: str = HERE) -> tuple[np.ndarray, np.ndarray]:
    """(residual, y_min) for one arm, from its eval npz."""
    d = np.load(os.path.join(eval_dir, f"q2rw_eval_{tag}.npz"))
    return d["pred_logamp"] - d["true_logamp"], d["y_min"]


def metrics(tag: str, eval_dir: str = HERE) -> dict:
    """Event-weighted MSE, per-decade MSE profile, and the log-flat mean, for one arm."""
    r, y = load(tag, eval_dir)
    per = np.full(len(CENTRES), np.nan)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = (y >= lo) & (y < hi)
        if m.any():
            per[i] = np.mean(r[m] ** 2)
    return dict(tag=tag, event=float(np.mean(r ** 2)), per_decade=per,
                logflat=float(np.nanmean(per)), n=int(r.size))


def gains(tags, eval_dir: str = HERE) -> dict:
    """`{tag: {"event": g, "logflat": g}}`, the fraction-of-oracle-gain under both metrics.

    `baseQ` and `oracle` are pulled in whether or not they are listed, since they define the
    0 and 1 of the scale.
    """
    want = list(dict.fromkeys(list(tags) + [BASE, ORACLE]))
    M = {t: metrics(t, eval_dir) for t in want}
    out = {}
    for key in ("event", "logflat"):
        b, o = M[BASE][key], M[ORACLE][key]
        for t in want:
            out.setdefault(t, {})[key] = (b - M[t][key]) / (b - o)
    for t in want:
        out[t]["metrics"] = M[t]
    return out
