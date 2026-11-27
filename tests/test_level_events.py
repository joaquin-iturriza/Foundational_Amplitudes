"""data.level_events (llm_compare step 28): experiment._level_events_slope equals the anchored form's per-event part
b.(x - x0)/sx with x computed by analysis/catalog_daniel/perevent_level.py's events(), the definition the coefficient
file was fitted on. The memmap path against the in-memory path: tests/test_memmap_data.py --synthetic with
data.level_events (see its docstring).

    .venv-cpu/bin/python -m pytest -q tests/test_level_events.py
"""
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
from experiment import AmplitudeExperiment  # noqa: E402
from perevent_level import events  # noqa: E402


def _events(n_ev=400, masses=(0.0, 0.0, 172.5, 0.0, 0.0), seed=0):
    """random on-shell momenta of a 2 -> 3 process in its COM frame (not a physical phase-space sampling)"""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_ev, 5, 4))
    for e in range(n_ev):
        p3 = rng.normal(size=(2, 3)) * 300
        p3 = np.vstack([p3, -p3.sum(0)])
        E = np.sqrt((p3 ** 2).sum(1) + np.array(masses[2:]) ** 2)
        rs = E.sum()
        out[e, 0] = [rs / 2, 0, 0, rs / 2]; out[e, 1] = [rs / 2, 0, 0, -rs / 2]
        out[e, 2:, 0] = E; out[e, 2:, 1:] = p3
    return out


def test_slope_matches_perevent_level():
    pp = _events()
    pdg = np.tile([21, 21, 6, 21, 2], (len(pp), 1))
    rows = np.column_stack([pp.reshape(len(pp), -1), pdg, np.full(len(pp), 1.0)])
    _, L, Lmin, E, _, _ = events(rows)
    co = {"b": [0.7, -1.3, 0.4], "x0": [12.0, -3.0, 5.0], "E_legs": [3, 4]}
    sx = [1.1, 2.2, 3.3]
    want = ((np.column_stack([L, Lmin - L, E]) - co["x0"]) / sx) @ co["b"]
    fake = types.SimpleNamespace(_lev_ev={"sx": sx, "coef": {"p": co}})
    got = AmplitudeExperiment._level_events_slope(fake, "p", pp)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12), np.abs(got - want).max()
    # the factor the target carries: _tp_logF subtracts it (no Breit-Wigner propagators)
    fake._level_events_slope = lambda n, p: AmplitudeExperiment._level_events_slope(fake, n, p)
    lf = AmplitudeExperiment._tp_logF(fake, "p", [], pp, {"level_events": "x", "pool_widths": {}, "widths": {}})
    assert np.allclose(lf, -want)


if __name__ == "__main__":
    test_slope_matches_perevent_level()
    print("OK")
