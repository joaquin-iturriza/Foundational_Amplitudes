"""drive.py stops with an error when its start-up sync fails, before any submission."""
import sys, os, types, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import drive


def test_failed_sync_exits_before_submit(monkeypatch):
    calls = []

    def fake_site(*a, **k):
        calls.append(a[0])
        if a[0] == "sync":
            raise RuntimeError("site sync failed (1): untracked files")
        return ""

    sw = types.SimpleNamespace(name="t", n_trials=1, n_startup=1, sampler=types.SimpleNamespace(surrogate="s", objective_scale=1),
                               waves=None)
    monkeypatch.setattr(drive, "site", fake_site)
    monkeypatch.setattr(drive, "Sweep", lambda c, a: sw)
    monkeypatch.setattr(sys, "argv", ["drive.py", "--config", "x.yaml", "--item", "i", "--sites", "ccin2p3"])
    with pytest.raises(SystemExit) as e:
        drive.main()
    assert "sync of ccin2p3 failed" in str(e.value)
    assert calls == ["sync"]
