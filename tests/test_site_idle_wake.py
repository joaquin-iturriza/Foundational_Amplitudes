"""site supervise's idle wake (idle_pass): STALLED after 30 min when `plan next` lists RESUME/LAUNCH, no job live."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import importlib.machinery, importlib.util, sqlite3, os

SITE = os.path.expanduser("~/.local/bin/site")
_ld = importlib.machinery.SourceFileLoader("site_tool", SITE)
S = importlib.util.module_from_spec(importlib.util.spec_from_loader("site_tool", _ld))
_ld.exec_module(S)

RESUME = "RESUME warm_start agent-x: ready step 1 of warm_start has no one working on it: nothing in flight\nplan: 1 owner action(s)\n"
WORKING = "note: warm_start: its owner agent-x is working\nplan: 0 owner action(s)\n"


def _db(live=False):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts, project, item, kind, key, msg, acked, mute_until)")
    c.execute("CREATE TABLE runs (id, project, state)")
    if live:
        c.execute("INSERT INTO runs VALUES ('FA-ccin2p3-1','FA','RUNNING')")
    c.execute("INSERT INTO runs VALUES ('FA-ccin2p3-0','FA','COMPLETED')")
    return c


def _events(c):
    return c.execute("SELECT kind,key FROM events").fetchall()


def test_resume_opens_after_30_min_once():
    c = _db()
    assert S.idle_pass(c, "FA", ".", RESUME, now=1000) == 0
    assert S.idle_pass(c, "FA", ".", RESUME, now=1000 + 29 * 60) == 0
    assert S.idle_pass(c, "FA", ".", RESUME, now=1000 + 31 * 60) == 1
    assert _events(c) == [("STALLED", "idle:warm_start:RESUME:1")]
    c.execute("UPDATE events SET acked=1")
    assert S.idle_pass(c, "FA", ".", RESUME, now=1000 + 90 * 60) == 0     # once per entry per state


def test_running_job_no_event():
    c = _db(live=True)
    S.idle_pass(c, "FA", ".", RESUME, now=1000)
    assert S.idle_pass(c, "FA", ".", RESUME, now=1000 + 3600) == 0
    assert _events(c) == []


def test_owner_working_no_event():
    c = _db()
    S.idle_pass(c, "FA", ".", WORKING, now=1000)
    assert S.idle_pass(c, "FA", ".", WORKING, now=1000 + 3600) == 0
    assert _events(c) == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")
