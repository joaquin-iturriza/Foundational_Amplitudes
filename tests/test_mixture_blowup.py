"""The blow-up check of analysis/catalog_v4/mixture_ab.py (blowup_block, blowups) on synthetic curves."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis", "catalog_v4"))
import mixture_ab as m  # noqa: E402


def curve(xs, every=85):
    return [[every * (k + 1), x] for k, x in enumerate(xs)]


def test_mse_clean_with_warmup_spike():
    # a warm-up spike before step 500 is not judged; the curve then decreases
    b = m.blowup_block(curve([0.5, 50.0, 0.4, 0.3, 0.25, 0.2, 0.19, 0.18]), het=False)
    assert b["blew_up_at"] is None and b["ref_step"] == 510 and b["early_max"] == 50.0


def test_mse_blowup_over_10x_reference():
    b = m.blowup_block(curve([0.5, 0.4, 0.3, 0.25, 0.2, 0.2, 0.19, 1.9, 2.1, 0.3]), het=False)
    assert b["blew_up_at"] == 85 * 9 and b["at_value"] == 2.1      # 1.9 < 10 x 0.2, 2.1 > it; best (0.19) before


def test_spike_recovered_is_not_a_blowup():
    b = m.blowup_block(curve([0.5, 0.4, 0.3, 0.25, 0.2, 0.2, 5.0, 0.15, 0.1]), het=False)
    assert b["spike_at"] == 85 * 7 and b["blew_up_at"] is None and b["best_step"] == 85 * 9


def test_nan_anywhere():
    b = m.blowup_block(curve([0.5, float("nan"), 0.3, 0.25, 0.2, 0.2, 0.19]), het=False)
    assert b["spike_at"] == 170 and math.isnan(b["at_value"]) and b["blew_up_at"] is None     # best after it
    b = m.blowup_block(curve([0.5, 0.2, 0.3, float("nan")]), het=False)
    assert b["blew_up_at"] == 340


def test_het_additive_rule():
    # NLL reference -1.0 at step 510; -1.0 + 0.5 ln 10 = 0.151: 0.1 is under, 0.2 over
    xs = [0.1, 1e9, -0.4, -0.7, -0.9, -1.0, -1.1, 0.1, -1.2]
    assert m.blowup_block(curve(xs), het=True)["blew_up_at"] is None
    xs[-1] = 0.2
    assert m.blowup_block(curve(xs), het=True)["blew_up_at"] == 85 * 9
    xs[-1] = -1.3                                   # 2.8e9-like spike then a new best: recovered
    xs[-2] = 2.8e9
    b = m.blowup_block(curve(xs), het=True)
    assert b["spike_at"] == 85 * 8 and b["blew_up_at"] is None


def test_short_block_uses_first_validation():
    b = m.blowup_block(curve([-1.0, -1.01, -0.9], every=10), het=True)
    assert b["ref_step"] == 10 and b["blew_up_at"] is None


def test_blowups_flags_unchecked_and_blown(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "READS", str(tmp_path))
    vals = curve([0.5, 0.4, 0.3, 0.25, 0.2, 0.2, 3.0])
    json.dump({"run": "T", "loss": "MSE", "blocks": [{"iterations": 595, "every": 85, "vals": vals}]},
              open(tmp_path / "T_val.json", "w"))
    monkeypatch.setitem(m.TRAINED_BY, "R", "T")
    out = m.blowups(["R", "U"])
    assert out["runs"]["R"]["blew_up"] == {"block": 0, "step": 595}
    assert "unchecked" in out["runs"]["U"]
    assert len(out["flagged"]) == 2
