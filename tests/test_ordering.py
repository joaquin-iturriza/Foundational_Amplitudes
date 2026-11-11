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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
