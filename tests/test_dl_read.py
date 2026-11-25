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


def test_scaled_optimal_step():
    # dL(s) = s A - s^2 Q with A = 2 and Q = 1 + 150/b: dL_opt = A^2/(4Q) = 1/(1 + 150/b), B_opt = 150
    sc = [0.01, 0.03, 0.1, 0.3, 1.0]
    rec = {"kind": "chunk", "step": 0, **{f"dL_b{b}": -1.0 for b in BS}, **{f"all_A_b{b}": 2.0 for b in BS},
           **{f"dL_s{s:g}_b{b}": 2 * s - s * s * (1 + 150 / b) for s in sc for b in BS}}
    w = dl_read.window([rec], BS)
    assert abs(w["B_opt"] - 150) < 1e-6, w["B_opt"]
    full = {**rec, **{f"dL_b{b}": 2 - (1 + 150 / b) for b in BS}}   # the s = 1 read of the same quadratic
    assert abs(dl_read.window([full], BS)["B_q"] - 150) < 1e-6



def test_exploding_A_and_Q_nonpositive():
    # an exploding read in A alone is flagged; a window whose Q = A - dL(1) <= 0 is listed apart, not silently dropped
    rec = lambda s, a: {"kind": "chunk", "step": s, **{f"dL_b{b}": 0.5 for b in BS}, **{f"all_A_b{b}": a for b in BS}}
    w = dl_read.window([rec(s, 1.0) for s in range(0, 450, 50)] + [rec(450, 1e6)], BS)
    assert w["n_exploding_reads"] == 1
    r = dl_read.read([rec(s, 0.2) for s in range(0, 200, 50)], 200, [87, 261])
    assert r["windows"][0]["B_q_status"] == "Q<=0" and r["summary"]["n_windows_Q_nonpositive"] == 1


if __name__ == "__main__":
    test_fit_known_answer(); test_fit_listed_apart(); test_windows(); test_scaled_optimal_step()
    test_exploding_A_and_Q_nonpositive(); print("ok")
