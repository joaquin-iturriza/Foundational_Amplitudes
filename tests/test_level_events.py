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


def test_level_subsets_factor_check(tmp_path):
    """analysis/llm_compare/level_subsets.py: its factor check passes on a CSV whose log_factor is -b.(x-x0)/sx at its
    x columns (as tools/llm_score.py writes them) and refuses a CSV whose rows are shuffled against the factor"""
    import json
    import pytest
    sys.path.insert(0, os.path.join(ROOT, "analysis", "llm_compare"))
    import level_subsets
    pp = _events(n_ev=300)
    co = {"b": [0.7, -1.3, 0.4], "x0": [12.0, -3.0, 5.0], "E_legs": [3, 4]}
    f = {"sx": [1.1, 2.2, 3.3], "coef": {"p": co}}
    fp = tmp_path / "lev.json"; fp.write_text(json.dumps(f))
    x = AmplitudeExperiment._level_events_x(pp, co["E_legs"])
    lf = -((x - co["x0"]) / f["sx"]) @ co["b"]
    rng = np.random.default_rng(1)
    y = rng.normal(size=len(x))
    rows = lambda lf_: "process,true_logm2,pred_logm2,log_factor,log_shat,q,E\n" + "".join(
        "p,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n" % (y[i], y[i] + 0.3, lf_[i], *x[i]) for i in range(len(x)))
    c = tmp_path / "ok.csv"; c.write_text(rows(lf))
    out = level_subsets.read(str(c), ["p"], str(fp), nboot=2)
    assert out["factor_check_p99_abs"] < 1e-3 and abs(out["new"]["all"]["bias2"] - 0.09) < 1e-6
    rl = level_subsets.read(str(c), ["p"], None, nboot=2, relevel=True)   # the level on each subset's own mean log F
    lo = x[:, 0] <= np.median(x[:, 0])
    assert abs(rl["new"]["all"]["bias2"] - 0.09) < 1e-6 and rl["new"]["all"]["dlogF"] == 0
    assert abs(rl["new"]["s_lo"]["dlogF"] - abs(lf[lo].mean() - lf.mean())) < 1e-4
    c.write_text(rows(lf[::-1]))
    with pytest.raises(AssertionError):
        level_subsets.read(str(c), ["p"], str(fp), nboot=2)

def test_mmap_pools_feat_target(tmp_path):
    """perevent_level.mmap_pools with a features dir (step 28): the pool rows' last column is |M|^2 exp(logF) of that
    dir at the segment's own events, the training target of part (c)'s configuration; without it the raw |M|^2"""
    import json
    from perevent_level import mmap_pools
    P, n = 3, [4, 5]
    segs, ev0, p0 = [], 0, 0
    for i, k in enumerate(n):
        segs.append({"name": f"p{i}", "role": "train", "n": k, "P": P, "ev0": ev0, "p0": p0})
        ev0 += k; p0 += k * P
    rng = np.random.default_rng(1)
    np.save(tmp_path / "mom.npy", rng.normal(size=(p0, 4)).astype(np.float32))
    np.save(tmp_path / "pdg.npy", np.full(p0, 21.0, np.float32))
    amp = rng.uniform(1, 2, ev0); np.save(tmp_path / "amp.npy", amp)
    (tmp_path / "meta.json").write_text(json.dumps({"segs": segs}))
    fd = tmp_path / "feat_x"; fd.mkdir()
    lF = rng.normal(size=ev0) * 5; np.save(fd / "logF.npy", lF)
    (fd / "meta.json").write_text(json.dumps({"key_doc": {"phys": [1.0]}}))
    raw, tgt = mmap_pools([str(tmp_path)]), mmap_pools([str(tmp_path)], ["feat_x"])
    assert np.allclose(raw["p1"](9)[:, -1], amp[4:9])
    assert np.allclose(tgt["p1"](9)[:, -1], amp[4:9] * np.exp(lF[4:9]))
    assert np.allclose(tgt["p0"](2)[:, -1], amp[:2] * np.exp(lF[:2]))
    from perevent_level import LOGF_MEAN   # --level-split's known part: the mean log F of the events last read
    assert np.isclose(LOGF_MEAN["p0"], lF[:2].mean()) and np.isclose(LOGF_MEAN["p1"], lF[4:9].mean())


if __name__ == "__main__":
    test_slope_matches_perevent_level()
    print("OK")

