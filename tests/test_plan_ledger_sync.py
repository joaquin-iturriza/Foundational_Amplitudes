"""plan.py follows the ledger: a hold or release recorded with `site decide` sets the entry's status (ledger_sync), and
`plan done` closes the entry's site item (close_item). Runs on a temporary ledger and a stub `site`; the live ledger
and plan are never touched.

    .venv-cpu/bin/python tests/test_plan_ledger_sync.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sqlite3, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="ledger_sync_")
DB = os.path.join(TMP, "runs.db")
LOG = os.path.join(TMP, "site_calls")
STUB = os.path.join(TMP, "site")
open(STUB, "w").write('#!/bin/sh\necho "$@" >> %s\npython3 - "$@" <<\'EOF\'\nimport sqlite3, sys\n'
                      'a = sys.argv[1:]\nif a[:2] == ["item", "done"]:\n'
                      '    c = sqlite3.connect(%r); c.execute("UPDATE items SET status=\'done\' WHERE project=? AND id=?",'
                      ' (a[2], a[3])); c.commit()\nEOF\n' % (LOG, DB))
os.chmod(STUB, 0o755)
os.environ.update(PLAN_TEST="1", PLAN_ROOT=TMP, PLAN_RUNS_DB=DB, PLAN_SITE=STUB, PLAN_FILE=os.path.join(TMP, "plan.yaml"),
                  PLAN_STATE=os.path.join(TMP, "state.json"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402


def ledger(items, holds):
    if os.path.exists(DB):
        os.remove(DB)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE items (project TEXT, id TEXT, status TEXT)")
    c.execute("CREATE TABLE decisions (id TEXT, project TEXT, kind TEXT, covers TEXT, in_force INTEGER, recorded INTEGER)")
    c.executemany("INSERT INTO items VALUES ('FA',?,?)", items.items())
    c.executemany("INSERT INTO decisions VALUES (?,'FA',?,?,?,?)", holds)
    c.commit()


def test_hold_from_ledger():
    ledger({"a": "held", "b": "active", "c": "held"}, [("D1", "hold", "a,c", 1, 1), ("D2", "hold", "a", 1, 2)])
    d = {"entries": [{"id": "a", "status": "running"}, {"id": "b", "status": "running"},
                     {"id": "c", "status": "waiting_user"}, {"id": "x", "status": "running"}]}
    plan.ledger_sync(d)
    got = {e["id"]: (e["status"], e.get("hold_decision")) for e in d["entries"]}
    assert got == {"a": ("held", "D2"), "b": ("running", None), "c": ("held", "D1"), "x": ("running", None)}, got


def test_release_from_ledger():
    # D1 released (no longer in force), item active again: the entry runs; an entry still under a hold in force stays
    ledger({"a": "active", "b": "held"}, [("D1", "hold", "a", 0, 1), ("D3", "hold", "b", 1, 2)])
    d = {"entries": [{"id": "a", "status": "held", "hold_decision": "D1"},
                     {"id": "b", "status": "held", "hold_decision": "D3"}]}
    plan.ledger_sync(d)
    assert (d["entries"][0]["status"], d["entries"][0].get("hold_decision")) == ("running", None), d
    assert d["entries"][1]["status"] == "held", d


def test_forbid_stays_held():
    # plan answer --kind forbid: entry held under D5 (forbid), the ledger item stays active: not released on read,
    # whether the forbid is in force or (a later release of something else) not
    ledger({"a": "active", "b": "active"}, [("D5", "forbid", "a", 1, 1), ("D6", "forbid", "b", 0, 2)])
    d = {"entries": [{"id": "a", "status": "held", "hold_decision": "D5"},
                     {"id": "b", "status": "held", "hold_decision": "D6"}]}
    plan.ledger_sync(d)
    assert [e["status"] for e in d["entries"]] == ["held", "held"], d


def test_held_item_without_decision_unchanged():
    ledger({"a": "held"}, [])
    d = {"entries": [{"id": "a", "status": "running"}]}
    plan.ledger_sync(d)
    assert d["entries"][0]["status"] == "running"


def test_close_item():
    ledger({"done_entry": "active", "closed": "done"}, [])
    if os.path.exists(LOG):
        os.remove(LOG)
    plan.close_item("done_entry")
    plan.close_item("closed")          # already done: no call
    plan.close_item("no_item")         # no item: no call
    calls = open(LOG).read().split("\n")
    assert calls[:1] == ["item done FA done_entry"] and len([x for x in calls if x]) == 1, calls
    st = sqlite3.connect(DB).execute("SELECT status FROM items WHERE id='done_entry'").fetchone()[0]
    assert st == "done", st


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k)
    print("%d passed" % n)
