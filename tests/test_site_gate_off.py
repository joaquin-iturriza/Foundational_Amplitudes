"""site: a disabled Stop gate opens an ALERT every pass after 15 min; ack of its HOOK INCIDENT refuses while disabled."""
import _tmpdir  # noqa: F401
import importlib.machinery, importlib.util, json, os, sqlite3, time, types
import pytest

SITE = os.path.expanduser("~/.local/bin/site")
_ld = importlib.machinery.SourceFileLoader("site_tool", SITE)
S = importlib.util.module_from_spec(importlib.util.spec_from_loader("site_tool", _ld))
_ld.exec_module(S)


def _proj(tmp_path, disabled_at=None):
    h = tmp_path / ".claude" / "hooks"
    h.mkdir(parents=True)
    if disabled_at is not None:
        t = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(disabled_at))
        (h / "disabled.json").write_text(json.dumps({"plan_check.py": {"incident": "I-x", "time": t}}))
    return {"local": str(tmp_path)}


def _db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts, project, item, kind, key, msg, acked, mute_until, ack_note, tell)")
    return c


def test_alert_after_15_min_only(tmp_path):
    now = time.time()
    c = _db()
    (tmp_path / "none").mkdir()
    assert S.gate_off_pass(c, "FA", {"local": str(tmp_path / "none")}, now) == 0          # absent
    assert S.gate_off_pass(c, "FA", _proj(tmp_path / "new", now - 600), now) == 0         # 10 min
    assert S.gate_off_pass(c, "FA", _proj(tmp_path / "old", now - 1200), now) == 1        # 20 min
    assert c.execute("SELECT kind FROM events").fetchall() == [("ALERT",)]
    assert S.gate_off_pass(c, "FA", _proj(tmp_path / "old2", now - 1200), now) == 0       # still open
    c.execute("UPDATE events SET acked=1, mute_until=?", (now - 1,))
    assert S.gate_off_pass(c, "FA", _proj(tmp_path / "old3", now - 1200), now) == 1       # re-open after the mute


def test_ack_refused_until_resolved(tmp_path, monkeypatch):
    p = _proj(tmp_path, time.time() - 5000)
    c = _db()
    c.execute("INSERT INTO events (id,ts,project,kind,key,msg) VALUES (7,1,'FA','ALERT','k',"
              "'HOOK INCIDENT I-x, for a worker: plan_check.py [Stop], disabled')")
    monkeypatch.setattr(S, "ldb", lambda: c)
    monkeypatch.setattr(S, "cfg_projects", lambda: {"FA": p})
    monkeypatch.setattr(S, "project_of", lambda n: p)
    a = types.SimpleNamespace(events=["E7"], note="n", no_writeup=None, needs_user=None, mute_hours=None, override=None)
    with pytest.raises(SystemExit):
        S.cmd_ack(a)
    os.remove(tmp_path / ".claude" / "hooks" / "disabled.json")                              # --resolve
    S.cmd_ack(a)
    assert c.execute("SELECT acked FROM events WHERE id=7").fetchone()[0]
