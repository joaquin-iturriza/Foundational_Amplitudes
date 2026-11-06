"""recent_ended_sweeps lists only a sweep that ended after the entry's last owner activity (E3887, 2026-11-06: two
sweeps read hours before were listed again). Direct call with injected sweep facts, independent of the live plan."""
import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import plan


def case(monkeypatch, handoff_written, ended_at):
    t = datetime.now(timezone.utc)
    e = {"id": "x", "handoff": {"written": (t + timedelta(hours=handoff_written)).isoformat()}, "findings": []}
    st = {"x": {"sweeps": [{"site": "s", "name": "sw", "at": t.isoformat()}]}}
    sw = {"s": {"sweeps": {"sw": {"live": False, "results": 7, "scripts": 8}},
                "ended": {"sw": (t + timedelta(hours=ended_at)).isoformat()}}}
    monkeypatch.setattr(plan, "last_step", lambda *a, **k: None)
    return plan.recent_ended_sweeps(e, st, sweeps=sw)


def test_already_seen_sweep_dropped(monkeypatch):
    assert case(monkeypatch, handoff_written=-1, ended_at=-3) == []


def test_unseen_sweep_listed(monkeypatch):
    out = case(monkeypatch, handoff_written=-3, ended_at=-1)
    assert len(out) == 1 and "s:sw ended (7 of 8 results)" in out[0]
