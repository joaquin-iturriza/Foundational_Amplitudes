"""sweep/ordering.py (D396): wave sizes, and one dispatch order for drive.py and sweep_manager.py's default submit."""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from sweep import ordering  # noqa: E402


def test_wave_sizes():
    assert ordering.wave_sizes(8, 3) == [3, 3, 2]
    assert ordering.wave_sizes(8, 3, [4, 4]) == [4, 4]
    for bad in ([4, 3], [0, 8]):
        with pytest.raises(ValueError):
            ordering.wave_sizes(8, 3, bad)
    assert [ordering.wave_of(p, [3, 3, 2]) for p in range(8)] == [0, 0, 0, 1, 1, 1, 2, 2]


def test_dispatch_order_longest_first_round_robin():
    want = {"a": 3, "b": 3, "c": 3, "d": 2}
    hours = {"a": 0.2, "b": 0.2, "c": 1.0, "d": 1.0}
    assert ordering.dispatch_order(want, hours) == ["c", "d", "c", "d", "c", "a", "b", "a", "b", "a", "b"]


def _sweep(tmp, name, minutes, n=8):
    d = tmp / name
    (d / "jobs").mkdir(parents=True)
    (d / "sweep_config.yaml").write_text(
        "n_trials: %d\ndyhpo: {n_startup: 3}\ncluster: {time: '00:%02d:00'}\n" % (n, minutes))
    for i in range(n):
        (d / "jobs" / ("trial_%04d.sh" % i)).write_text("#!/bin/bash\n")
    return str(d)


def test_sweep_manager_dry_run_matches_drive(tmp_path, monkeypatch, capsys):
    import siteconf
    import sweep_manager as sm
    from sweep.drive import dispatch_order

    dirs = [_sweep(tmp_path, "short_a", 10), _sweep(tmp_path, "long_b", 50), _sweep(tmp_path, "long_c", 50)]
    monkeypatch.setattr(siteconf, "work_gate", lambda names: (list(names), []))
    monkeypatch.setattr(sm, "submit_prebuilds", lambda dirs, dry_run=False: {})
    sm.submit_sweeps(dirs, registry=str(tmp_path / "reg.json"), dry_run=True)
    rows = re.findall(r"wave (\d+)\s+rank\s+(\d+)\s+(\S+)\s+trial_(\d+)\s+nice=\d+\s+deps=(\S+)", capsys.readouterr().out)
    assert len(rows) == 24
    # waves 3, 3, 2 per sweep; a later wave waits (afterany) on the sweep's earlier one, the first wave on nothing
    for sw in ("short_a", "long_b", "long_c"):
        w = [int(r[0]) for r in rows if r[2] == sw]
        assert sorted(w) == [0, 0, 0, 1, 1, 1, 2, 2]
        assert all((r[4] == "-") == (r[0] == "0") for r in rows if r[2] == sw)
        assert all("afterany:" in r[4] for r in rows if r[2] == sw and r[0] != "0")

    class Sw:
        def __init__(self, name, hours):
            self.name, self.hours = name, hours
    # the first wave as drive.py would issue it: the same order, by rank
    first = [r[2] for r in sorted((r for r in rows if r[0] == "0"), key=lambda r: int(r[1]))]
    want = {Sw("short_a", 10 / 60): 3, Sw("long_b", 50 / 60): 3, Sw("long_c", 50 / 60): 3}
    assert first == [s.name for s in dispatch_order(want)]
    assert first[:2] == ["long_b", "long_c"]


def test_drive_done_where_prune_looks(tmp_path, monkeypatch):
    import siteconf
    import prune
    from sweep.drive import mark_cmd

    monkeypatch.setattr(siteconf, "SWEEP_DIR", str(tmp_path))
    code = mark_cmd("dsave_x").split('-c "', 1)[1].rsplit('"', 1)[0]
    exec(code, {})                                                   # what the site runs, with its own SWEEP_DIR
    assert prune.finished("dsave_x", {})


def test_resumed_submit_waits_on_live_earlier_jobs(tmp_path, monkeypatch, capsys):
    import json
    import siteconf
    import sweep_manager as sm

    d = _sweep(tmp_path, "s", 10)
    reg = tmp_path / "reg.json"
    # trials 0-4 went out before (an old registry: every wave 0); job 11 has left the queue, 13 and 14 are live
    jobs = {str(10 + i): {"trial_idx": i, "script": "%s/jobs/trial_%04d.sh" % (d, i), "wave": 0} for i in range(5)}
    reg.write_text(json.dumps({"round_gap": 1, "sweeps": {"s": {"dir": d, "weight": 1, "jobs": jobs,
                                                                "submitted_scripts": [j["script"] for j in jobs.values()]}}}))
    monkeypatch.setattr(siteconf, "work_gate", lambda names: (list(names), []))
    monkeypatch.setattr(sm, "submit_prebuilds", lambda dirs, dry_run=False: {})
    monkeypatch.setattr(sm, "squeue_states", lambda: {"13": "RUNNING", "14": "PENDING"})
    monkeypatch.setattr(sm, "_rebalance", lambda *a, **k: None)
    monkeypatch.setattr(sm, "stamp_submitted", lambda *a, **k: None)
    ids = iter(range(100, 200))
    monkeypatch.setattr(sm, "_run", lambda cmd, dry_run=False: str(next(ids)))
    sm.submit_sweeps([d], registry=str(reg), dry_run=False)
    rows = re.findall(r"wave (\d+)\s+rank\s+\d+\s+s\s+trial_(\d+)\s+nice=\d+\s+deps=(\S+)", capsys.readouterr().out)
    # trials 5-7: positions 5 (wave 1) and 6, 7 (wave 2) of 3, 3, 2
    assert [(int(w), int(t)) for w, t, _ in rows] == [(1, 5), (2, 6), (2, 7)]
    assert rows[0][2] == "afterany:13:14"                             # the live earlier jobs only, never 11
    assert rows[1][2] == "afterany:100:13:14"                         # this call's wave 1 and the live earlier ones


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
