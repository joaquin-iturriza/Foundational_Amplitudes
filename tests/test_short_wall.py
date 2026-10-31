"""tools/short_wall.py: the short job's time limit counts the zero-shot reads (catalog_coverage, 2026-10-31: the
mixture arms FA-ccin2p3-6c246603 and 1d46b938 were cut by TIMEOUT in their PP4 reads at 2.5 h)."""
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
    assert abs((b["hours"] - a["hours"]) - sw.SIGMA_STAGE_MIN * sw.MARGIN / 60) < 0.02, (a, b)
    e = sw.plan("recipes/catalog_v4_train.yaml", "v4", ["PP4"], eval_only=True, sigma_stage=True)
    assert e["rates"]["sigma_stage_min"] == 0.0, e   # a re-read trains nothing


if __name__ == "__main__":
    test_limits(); test_mixture_arm_needs_more_than_2p5h(); test_het_arm_counts_its_sigma_stage(); print("short_wall tests passed")
