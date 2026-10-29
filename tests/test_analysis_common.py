"""analysis/common: each reader, picker and statistic against a hand-computed case, and against the script whose rule
it carries where that script's function still exists.
    .venv-cpu/bin/pytest tests/test_analysis_common.py   (or python3 tests/test_analysis_common.py)"""
import json, os, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import (best_checkpoint, best_trial, curve_stats, find_pool, gmean, ratio_stats,  # noqa: E402
                             read_errors, read_jsonl, run_metrics, split_pool, sweep_trials, trial_mse)


def test_read_jsonl():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "r.jsonl")
    open(p, "w").write('{"a": 1}\n\n{"a": 2}\n')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(os.path.join(d, "absent.jsonl")) == []


def test_run_metrics_and_best_checkpoint():
    d = tempfile.mkdtemp()
    for sub, v in (("a", [9.0]), ("b", [3.0, 1.0, 2.0])):    # the last file under the path is the run's own
        os.makedirs(os.path.join(d, sub))
        json.dump({"val_loss_no_reg": v, "proc_val_losses_no_reg": {"p": [x * 2 for x in v], "short": [5.0]}},
                  open(os.path.join(d, sub, "per_process_metrics.json"), "w"))
    m = run_metrics(d)
    assert m["val_loss_no_reg"] == [3.0, 1.0, 2.0]
    # every process at the step of the lowest aggregate, never its own minimum; a process with fewer validations
    # than the best index is left out
    assert best_checkpoint(m) == (1, 1.0, {"p": 2.0})
    sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v2"))
    import census
    assert census.at_best(m) == best_checkpoint(m) and census.metrics(d) == m


def test_sweep_trials_and_best_trial():
    S = {"s": [{"val_loss": 2.0, "prepd_std": 1.0}, {"val_loss": None, "prepd_std": 1.0}],
         "s_002": [{"val_loss": 1.0, "prepd_std": 2.0}, {"val_loss": 0.4, "prepd_std": 1.0}]}
    tr = sweep_trials(S, "s")
    assert len(tr) == 4 and len(sweep_trials(S, "s", continuation=False)) == 2
    assert trial_mse(tr[1]) is None and trial_mse({"val_loss": 1.0, "prepd_std": 0}) is None
    v, t = best_trial(tr)              # val_loss times prepd_std^2: 2, -, 4, 0.4
    assert v == 0.4 and t is S["s_002"][1]
    assert best_trial([{"val_loss": None}]) == (None, None)
    assert best_trial(tr, value=lambda t: t["val_loss"])[0] == 0.4


def test_best_trial_reports_unconverted(capsys):
    best_trial([{"val_loss": 1.0, "prepd_std": None}, {"val_loss": 2.0, "prepd_std": 1.0}], label="s")
    assert "best_trial s: 1 of 2 trials have a result but no value" in capsys.readouterr().err


def test_best_trial_matches_cells():
    """the rule of analysis/transfer/cells.py best() on its own sweep file"""
    sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
    import cells
    for name in list(cells.S)[:200]:
        if name.endswith("_002"):
            continue
        a, b = cells.best(name), best_trial(sweep_trials(cells.S, name))
        assert a[0] == b[0] and a[1] is b[1], name
    cells.UNCONVERTED.clear()


def test_find_pool():
    d = tempfile.mkdtemp()
    for f in ("ee_uu_50-1000GeV_val_x_amplitudes.npy", "ee_uu_20-500GeV_val_x_amplitudes.npy",
              "ee_uu_20-500GeV_train_x_amplitudes.npy"):
        open(os.path.join(d, f), "w").close()
    assert find_pool("ee_uu", (50, 1000), data_dir=d).endswith("ee_uu_50-1000GeV_val_x_amplitudes.npy")
    assert find_pool("ee_uu", (1, 2), any_window=True, data_dir=d).endswith("ee_uu_20-500GeV_val_x_amplitudes.npy")
    assert find_pool("ee_uu", None, role="train", data_dir=d).endswith("_train_x_amplitudes.npy")
    assert find_pool("ee_dd", None, data_dir=d) is None
    assert find_pool("ee_uu", (1, 2), data_dir=d) is None              # a missing window is not replaced silently
    open(os.path.join(d, "ee_mu_nlo_20-500GeV_val_x_amplitudes.npy"), "w").close()
    open(os.path.join(d, "ee_mu__E10_20-500GeV_val_x_amplitudes.npy"), "w").close()
    assert find_pool("ee_mu", None, data_dir=d) is None                # never another process's pool
    assert find_pool("ee_mu", (20, 500), any_window=True, data_dir=d) is None


def test_split_pool():
    n = 3
    rows = np.arange(2 * (5 * n + 1), dtype=float).reshape(2, 5 * n + 1)
    p, ids, amp = split_pool(rows, n)
    assert p.shape == (2, 3, 4) and ids.shape == (2, 3) and amp.shape == (2,)
    assert p[1, 2, 3] == rows[1, 11] and ids[0, 0] == 12 and amp[1] == rows[1, -1]
    try:
        split_pool(rows, 2)
        assert False, "a wrong particle count must raise"
    except ValueError:
        pass


def test_read_errors():
    q, t = np.array([1.0, 2.0, 4.0]), np.array([0.0, 0.0, 0.0])
    e = read_errors(q, t)
    assert np.isclose(e["mse"], 7.0) and np.isclose(e["level"], (7 / 3) ** 2)
    assert np.isclose(e["shape"], np.var(q)) and np.isclose(e["mse"], e["level"] + e["shape"])


def test_curve_stats():
    s = curve_stats([4.0, 1.0, 2.0, 3.0], every=10, scale=2.0)
    assert s == {"first": 8.0, "first_step": 10, "best": 2.0, "best_frac": 0.5, "last_over_best": 3.0, "steps": 40}
    assert curve_stats([1.0, 2.0], T=100)["best_frac"] == 0.01


def test_gmean_and_ratio_stats():
    assert np.isclose(gmean([1, 100]), 10.0)
    r = ratio_stats([1.0, 10.0, 100.0], [10.0, 10.0, 10.0])
    assert r["n"] == 3 and np.isclose(r["ratio"], 1.0) and r["n_num_lower"] == 1 and np.isclose(r["gap"], 1.0)
    assert np.allclose(r["ratio_q10_q90"], [10 ** -0.8, 10 ** 0.8])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and k != "test_best_trial_reports_unconverted":   # needs pytest's capsys
            f()
            print("ok", k)
