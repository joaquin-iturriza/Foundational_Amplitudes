"""A `site` command that hangs never kills drive.py (E6983): a pick or poll reads as empty output, a log read raises."""
import sys, os, subprocess, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import drive


def _hang(cmd, **k):
    raise subprocess.TimeoutExpired(cmd, k.get("timeout"))


def test_timeout_reads_empty_or_raises(monkeypatch):
    monkeypatch.setattr(drive.subprocess, "run", _hang)
    assert drive.site("pick", "FA", timeout=1, check=False) == ""          # no plan: the next pass picks again
    with pytest.raises(RuntimeError):
        drive.site("submit", "x", timeout=1)                                 # check=True: a failure like any other
    with pytest.raises(RuntimeError):
        drive.site("logs", "x", timeout=1, check=False, timeout_ok=False)   # unread log: never taken as lost
