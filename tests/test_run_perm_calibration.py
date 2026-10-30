"""short_run_bias.run_perm on iid events: with units of unequal size (66 runs of 1000 against 18 of 555) the raw binned
KS made p too small (0.19 of 150 null draws under 0.05 at c. f537f0e0); scaled by sqrt(n_a n_b / (n_a + n_b)) it holds
its level. Run: python tests/test_run_perm_calibration.py (prints the rates)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "catalog_v4"))
import short_run_bias as b  # noqa: E402


def rates(sizes, reps=150, nperm=500, seed=1):
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    pr, pe = [], []
    for _ in range(reps):
        X = [rng.normal(size=sizes[0]) for _ in range(66)]
        Y = [rng.normal(size=sizes[1]) for _ in range(18)]
        pr.append(b.run_perm(X, Y, nperm=nperm)["p_runs"])
        pe.append(ks_2samp(np.concatenate(X), np.concatenate(Y)).pvalue)
    pr, pe = np.array(pr), np.array(pe)
    return {"sizes": sizes, "run_p_lt_05": float((pr < 0.05).mean()), "event_p_lt_05": float((pe < 0.05).mean()),
            "run_p_below_event_p": float((pr < pe).mean())}


def test_unequal_units_hold_level():
    r = rates((1000, 555))
    assert r["run_p_lt_05"] <= 0.10 and r["run_p_below_event_p"] <= 0.75, r


if __name__ == "__main__":
    for s in ((1000, 555), (1000, 1000)):
        print(rates(s))
