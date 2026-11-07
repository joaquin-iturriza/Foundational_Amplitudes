"""The quota sampler with channel grids (mg5_pipeline_final._flat_quota_part, sampling.flat_grid; plan catalog_coverage
step 72): the grid map stays inside [0, 1) and is monotone; without flat_grid the quota sampler draws exactly what it
drew before (same kept |M|^2 as the module before the change, f414c171, at the same seed); on the toy |M|^2 of
tests/test_flat_mcmc.py through the multichannel proposal, the grids at the same budget fill at least as much of the
quota as the plain quota sampler, with independent draws (near-copy share <= 2%)."""
import importlib.util, os, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mg5_pipeline_final as mg  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "tests"))
from test_flat_mcmc import M, PDG, toy_label, fd  # noqa: E402


def _pol(budget, grid):
    return dict(flat_range=[np.log(1e-8), np.log(1e9)], flat_dex_per_bin=0.25, flat_budget=budget, oversample=6,
                y_lo=1e-6, democratic=0.5, multichannel=True, mc_resonances=[], mc_isc=True, mc_mperi=True,
                mc_mperi_both=True, n_sqrts_bins=200, flat_grid=grid)


def run(module, grid, budget=12, seed=3, n_flat=2000):
    rng = np.random.default_rng(seed)
    uni = lambda k: np.exp(rng.uniform(np.log(50.0), np.log(3000.0), size=k))
    ev, sq, me = module._flat_quota_part(n_flat, 50.0, 3000.0, uni, True, M, PDG, rng, None, _pol(budget, grid),
                                         toy_label, 0.0, None)
    d = module.FLAT_LAST
    F = fd.features(d["kept_P"], d["kept_sq"])
    pb = fd.per_bin(F, fd.bins_of(d["logm"][d["keep"]], d["edges"]), d["kept_chain"], len(d["quota"]))
    dup = sum(x.get("dup", 0.0) * x["n"] for x in pb) / max(sum(x["n"] for x in pb), 1)
    fill = float(np.minimum(d["kept_per_bin"], d["quota"]).sum() / d["quota"].sum())
    short = np.minimum(d["kept_per_bin"], d["quota"]) / np.maximum(d["quota"], 1)
    return fill, int(d["n_labels"]), dup, np.asarray(me), float((short >= 0.95).mean()), d


def test_grid_map():
    g = mg._ChanGrid(20, 0.2)
    rng = np.random.default_rng(0)
    y = rng.random(5000) ** 4                      # mass near 0
    g.fit(y[:, None], np.ones(len(y)))
    u = np.linspace(0, 1 - 1e-12, 1001)
    m = g.map(0, u)
    assert (m >= 0).all() and (m < 1).all() and (np.diff(m) >= 0).all()
    assert np.mean(g.map(0, rng.random(20000)) < 0.1) > 0.3      # the grid follows the weights (uniform: 0.1)


BEFORE = "f414c171"    # trunk before sampling.flat_grid: its quota sampler is the reference the no-grid path must equal


def test_unchanged_without_grid(tmp_path=None):
    import subprocess, tempfile
    src = subprocess.run(["git", "-C", HERE, "show", f"{BEFORE}:mg5_pipeline_final.py"], capture_output=True, text=True)
    if src.returncode:
        import pytest
        pytest.skip(f"git show {BEFORE} failed: {src.stderr.strip()}")
    old = os.path.join(tmp_path or tempfile.mkdtemp(prefix="zz_probe_grid_"), "mg_before.py")
    open(old, "w").write(src.stdout)
    sys.path.insert(0, HERE)       # the old module imports its siblings from the checkout
    spec = importlib.util.spec_from_file_location("mg_before", old); mt = importlib.util.module_from_spec(spec)
    sys.modules["mg_before"] = mt; spec.loader.exec_module(mt)
    a = run(mg, False, n_flat=500); b = run(mt, False, n_flat=500)
    # equal to rounding: the ladder's solve (b3ba8b94 -> regula falsi, plan step 75) returns the halvings' momenta to
    # ~1e-14, so the toy |M|^2 of the same events agree to ~1e-13 (bitwise before that change)
    np.testing.assert_allclose(a[3], b[3], rtol=1e-11)


def test_grid_fills_toy():
    fq, nq, dq, _, sq, _ = run(mg, False)
    fg, ng, dg, _, sg, d = run(mg, True)
    print("quota12 fill", fq, "bins>=0.95", sq, "labels", nq, "| grid12 fill", fg, "bins>=0.95", sg, "labels", ng,
          "dup", dg, "dims", d["grid_dims"], "t_prop/t_label", round(d["t_propose"], 2), round(d["t_label"], 2))
    assert ng <= 12 * 2000 and dg <= 0.02
    assert fg >= fq


def test_channel_bin_record():
    """FLAT_LAST's per channel and quota bin counts (plan step 75's where-read) sum to the in-range candidates and to
    the kept per bin"""
    d = run(mg, True, n_flat=500)[5]
    cc, kc = np.asarray(d["cand_per_channel_bin"]), np.asarray(d["kept_per_channel_bin"])
    assert cc.shape == kc.shape == (len(d["channels"]), len(d["quota"]))
    assert np.array_equal(cc.sum(0), np.asarray(d["cand_per_bin"]))
    assert np.array_equal(kc.sum(0), np.asarray(d["kept_per_bin"]))


if __name__ == "__main__":
    test_grid_map(); test_unchanged_without_grid(); test_grid_fills_toy(); print("ok")


def test_stall_stop():
    """sampling.flat_stall (plan step 75's cost revision): a range whose top no candidate reaches stops once a round
    fills less than the stall share of what was missing, before the budget, recorded as stop 'stall'; without the key
    the same chunk runs to the budget"""
    def go(stall):
        rng = np.random.default_rng(3)
        uni = lambda k: np.exp(rng.uniform(np.log(50.0), np.log(3000.0), size=k))
        pol = dict(_pol(12, True), flat_range=[np.log(1e-8), np.log(1e30)])      # top 21 dex above the toy's reach
        if stall:
            pol["flat_stall"] = stall
        mg._flat_quota_part(500, 50.0, 3000.0, uni, True, M, PDG, rng, None, pol, toy_label, 0.0, None)
        return dict(mg.FLAT_LAST)
    a, b = go(None), go(0.2)
    assert a["stop"] == "budget" and b["stop"] == "stall"
    assert b["n_labels"] < a["n_labels"] and len(b["rounds"]) == 2


def test_solve_tol():
    """sampling.mc_solve_tol: the ladder's momenta within 1e-10 sqrt(s) of the full collapse"""
    rng = np.random.default_rng(1); sq = np.exp(rng.uniform(np.log(40), np.log(6000), 2000))
    P0 = mg.multiperipheral_draw(2000, sq, np.zeros(4), 1e-6, np.random.default_rng(5), edge=True)
    P1 = mg.multiperipheral_draw(2000, sq, np.zeros(4), 1e-6, np.random.default_rng(5), edge=True, solve_tol=1e-12)
    assert (np.abs(P1 - P0).max((1, 2)) / sq).max() < 1e-10
