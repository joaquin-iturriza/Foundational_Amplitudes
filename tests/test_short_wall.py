"""tools/short_wall.py: the short job's time limit counts the zero-shot reads (catalog_coverage, 2026-10-31: the
mixture arms FA-ccin2p3-6c246603 and 1d46b938 were cut by TIMEOUT in their PP4 reads at 2.5 h)."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import short_wall as sw  # noqa: E402


def test_limits():
    assert sw.parse_limit("02:30:00") == 2.5 and sw.parse_limit("1-00:00:00") == 24 and sw.parse_limit("45:00") == 0.75


def test_mixture_arm_needs_more_than_2p5h():
    axes = ["H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c", "XV3", "PP", "PP4", "EE4"]
    p = sw.plan("recipes/catalog_v4_default50_train.yaml", "v4", axes, iters=4264)
    assert p["hours"] > 2.5, p                      # the limit that cut 6c246603 and 1d46b938
    assert p["probes"]["EE4"] == 169, p              # ee4_epem_cgsxwm excluded (D193)
    assert sw.main(["--train", "recipes/catalog_v4_default50_train.yaml", "--axes", *axes, "--iters", "4264",
                    "--limit", p["slurm_time"]]) is None
    try:
        sw.main(["--train", "recipes/catalog_v4_default50_train.yaml", "--axes", *axes, "--iters", "4264",
                 "--limit", "02:30:00"])
        raise AssertionError("2.5 h passed")
    except SystemExit as e:
        assert e.code == 1
    r = sw.plan("recipes/catalog_v4_default50_train.yaml", "v4", ["PP4", "EE4"], eval_only=True)
    assert r["iterations"] == 0 and r["n_probes"] == 1301 + 169, r


def test_het_arm_counts_its_sigma_stage():
    axes = ["H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c", "PP4", "EE4"]
    a = sw.plan("recipes/catalog_v4_train.yaml", "v4", axes, iters=4264)
    b = sw.plan("recipes/catalog_v4_train.yaml", "v4", axes, iters=4264, sigma_stage=True)
    extra = sw.SIGMA_STAGE_MIN + a["n_probes"] * (sw.S_PROBE_HET - sw.S_PROBE) / 60
    assert abs((b["hours"] - a["hours"]) - extra * sw.MARGIN / 60) < 0.02, (a, b)
    e = sw.plan("recipes/catalog_v4_train.yaml", "v4", ["PP4"], eval_only=True, sigma_stage=True)
    assert e["rates"]["sigma_stage_min"] == 0.0, e   # a re-read trains nothing
    assert e["rates"]["s_per_probe"] == sw.S_PROBE_HET and e["rates"]["startup_min"] == sw.STARTUP_EVAL_MIN, e


def test_eval_reread_fits_step20_times():
    """the step-20 het re-reads of PP4+EE4+HD4+HD5 (1612 probes) ran 36-39 min; the MSE ones under that"""
    axes = ["PP4", "EE4", "HD4", "HD5"]
    h = sw.plan("recipes/catalog_v4_train.yaml", "v4", axes, eval_only=True, sigma_stage=True)
    m = sw.plan("recipes/catalog_v4_train.yaml", "v4", axes, eval_only=True)
    assert 39 / 60 < h["hours"] < 2 * 39 / 60, h
    assert m["hours"] < h["hours"], (m, h)


def test_eval_mse_reread_within_2x_of_step23():
    """the step-23 MSE re-reads of EE4+HD4+HD5 (311 probes) took 198 and 218 s: the plan sits above, within 2x"""
    m = sw.plan("recipes/catalog_v4_pp_mixed_train.yaml", "v4", ["EE4", "HD4", "HD5"], eval_only=True)
    assert 218 / 3600 < m["hours"] < 2 * 218 / 3600, m


def test_eval_with_sigma_stage_counts_load_stage_and_cache():
    """v4_het_long_best: an eval job that first runs a sweep trial's sigma-only stage on the whole training set"""
    tr, ax = "recipes/catalog_v4_default25_train.yaml", ["HD5"]
    e = sw.plan(tr, "v4", ax, eval_only=True, sigma_stage=True)
    s = sw.plan(tr, "v4", ax, eval_only=True, sigma_stage=True, eval_stage=True)
    extra = (sw.STARTUP_MIN - sw.STARTUP_EVAL_MIN + sw.SIGMA_STAGE_MIN + sw.CACHE_BUILD_MIN
             + s["composed_at_load"] * sw.S_COMPOSE / 60) * sw.MARGIN / 60
    assert s["iterations"] == 0 and s["composed_at_load"] > 0, s
    assert abs(s["hours"] - e["hours"] - extra) < 0.02, (s, e, extra)


LONG_AXES = ["H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c", "PP4", "EE4", "HD4", "HD5"]


def test_width_rates_from_the_1x_cells():
    """--width: the long cells' measured V100 rates (size_mse.json, 4264 steps): w32 at 8528 steps needs its own 3.45
    s/step, so the width-8 rate (1.01) would have passed a limit 2x too short"""
    tr = "recipes/catalog_v4_default25_train.yaml"
    base = sw.plan(tr, "v4", LONG_AXES, iters=8528, sigma_stage=True)
    assert base["rates"]["s_per_step"] == sw.S_ITER and base["rates"]["width"] is None, base   # unchanged without --width
    for w, rate, train_h in ((8, 0.983, 1.1643), (16, 1.703, 2.0175), (32, 3.450, 4.0862)):
        p = sw.plan(tr, "v4", LONG_AXES, iters=4264, sigma_stage=True, width=w)
        assert p["rates"]["s_per_step"] == rate and abs(4264 * rate / 3600 - train_h) < 0.01, (w, p)
        q = sw.plan(tr, "v4", LONG_AXES, iters=8528, sigma_stage=True, width=w)
        assert abs(q["hours"] - base["hours"] - 8528 * (rate - sw.S_ITER) / 3600 * sw.MARGIN) < 0.01, (w, q, base)
    w32 = sw.plan(tr, "v4", LONG_AXES, iters=8528, sigma_stage=True, width=32)
    assert w32["hours"] > 12.0, w32
    args = ["--train", tr, "--axes", *LONG_AXES, "--iters", "8528", "--het", "--width", "32"]
    try:
        sw.main(args + ["--limit", base["slurm_time"]])          # the width-8 limit refused at width 32
        raise AssertionError("width-8 limit accepted at width 32")
    except SystemExit as e:
        assert e.code == 1, e
    assert sw.main(args + ["--limit", w32["slurm_time"]]) is None
    try:
        sw.plan(tr, "v4", LONG_AXES, iters=8528, width=64)
        raise AssertionError("unmeasured width accepted")
    except SystemExit as e:
        assert "no measured step rate" in str(e), e


if __name__ == "__main__":
    test_limits(); test_mixture_arm_needs_more_than_2p5h(); test_het_arm_counts_its_sigma_stage(); test_eval_reread_fits_step20_times(); test_eval_mse_reread_within_2x_of_step23(); test_eval_with_sigma_stage_counts_load_stage_and_cache(); test_width_rates_from_the_1x_cells(); print("short_wall tests passed")
