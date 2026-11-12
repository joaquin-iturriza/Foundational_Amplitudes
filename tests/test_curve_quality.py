"""curve_quality.quality: a validation curve that turns up at the end while training falls fails; one that keeps
falling, however jittery, tracks (Joaquin 2026-11-12)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "data_saving"))
from curve_quality import quality


def test_end_criterion():
    n = 40
    train = list(np.logspace(0, -4, n))
    jittery = list(np.logspace(0, -3, n) * (1 + 0.8 * (np.arange(n) % 2)))     # 2x jitter, falling to the end
    assert quality(jittery, train)["well"]
    up = list(np.logspace(0, -3, n)[:30]) + list(np.logspace(-3, -2, 10))        # turns up in the last quarter
    q = quality(up, train)
    assert not q["well"] and q["end_over_best"] > 2 and q["late_val"] > 1
    flat = list(np.logspace(0, -3, n)[:28]) + [1e-3 * 0.99] * 12                 # val flat, train still falling
    q = quality(flat, train)
    assert q["well"] and q["stalls"]


if __name__ == "__main__":
    test_end_criterion()
