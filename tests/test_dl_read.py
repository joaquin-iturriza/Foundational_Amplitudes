"""analysis/critical_batch/dl_read.py: the per-step critical-batch fit recovers a known answer."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "critical_batch"))
import dl_read  # noqa: E402

BS = [64, 128, 256, 512, 1024]


def test_fit_known_answer():
    B, inf = dl_read.fit([2 / (1 + 150 / b) for b in BS], BS)
    assert abs(B - 150) < 1e-6 and abs(inf - 2) < 1e-9


def test_fit_listed_apart():
    assert dl_read.fit([-1, 1, 1, 1, 1], BS) is None              # a window with dL <= 0 is not fitted
    assert dl_read.fit([1, 2, 4, 8, 16], BS)[0] == float("inf")   # linear in b: above the measured range


def test_windows():
    recs = [{"kind": "chunk", "step": s, **{f"dL_b{b}": 2 / (1 + 150 / b) for b in BS}} for s in range(0, 1000, 50)]
    r = dl_read.read(recs, 200, [87, 261])
    assert r["summary"]["n_fitted"] == 5 and abs(r["summary"]["B_dL_median"] - 150) < 1e-6


if __name__ == "__main__":
    test_fit_known_answer(); test_fit_listed_apart(); test_windows(); print("ok")
