#!/usr/bin/env python3
"""The supervisor's RUN_DONE events and the quiet waiter (plan entry agent_redesign, 2026-10-31), on a scratch registry
and event table; the code under test is ~/.local/bin/site_runends.py and its glue in ~/.local/bin/site.

  R1  a 32-job batch recorded on an entry, one step waiting on all of it (plan wait), ends clean in one pass: one
      RUN_DONE event naming the owner to resume, and the same pass again adds none
  R2  the same batch while the entry's owner is working: no event
  W1  an open RUN_DONE of an entry whose owner is working: the waiter does not return
  W2  the owner stops: the waiter returns with that event
  W3  an ALERT returns at once, whoever owns the entry

Run:  python3 tests/test_run_done_waiter.py
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types

BIN = os.path.expanduser("~/.local/bin")
sys.path.insert(0, BIN)


def load_site(tmp):
    loader = importlib.machinery.SourceFileLoader("site_tool", os.path.join(BIN, "site"))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader("site_tool", loader))
    sys.modules["site_tool"] = m
    loader.exec_module(m)
    m.HOME, m.DB = tmp, os.path.join(tmp, "runs.db")
    return m


def world(owner_state):
    runs = ["FA-ccin2p3-%08x" % i for i in range(32)]
    owners = {"catalog_coverage": {"entry": "catalog_coverage", "status": "running", "owner": "a0000000000000001",
                                   "owner_state": owner_state, "runs": runs + ["FA-ccin2p3-ffffffff"],
                                   "waits": [{"step": "1", "runs": runs}]},
              "agent_redesign": {"entry": "agent_redesign", "status": "running", "owner": None, "owner_state": None,
                                 "runs": []}}
    return runs, owners


def test_run_done_one_event_per_entry():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx = m._rx()
        c = m.ldb()
        runs, owners = world("stopped")
        for r in runs:
            c.execute("INSERT INTO runs (id, project, state) VALUES (?,?,?)", (r, "FA", "COMPLETED"))
        n = rx.run_done_events(m, c, "FA", runs, owners, log=lambda *a: None)
        rows = c.execute("SELECT item, kind, msg FROM events WHERE acked IS NULL").fetchall()
        assert n == 1 and len(rows) == 1, rows
        assert rows[0][:2] == ("catalog_coverage", "RUN_DONE") and "32 run(s)" in rows[0][2] \
            and "RESUME a0000000000000001" in rows[0][2], rows
        assert rx.run_done_events(m, c, "FA", runs, owners, log=lambda *a: None) == 0      # same pass again
        print("R1 ok:", rows[0][2][:120])
        runs, owners = world("working")
        c.execute("DELETE FROM events")
        assert rx.run_done_events(m, c, "FA", runs, owners, log=lambda *a: None) == 0
        assert c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        print("R2 ok: owner working, no event")


def run_waiter(m, out):
    a = types.SimpleNamespace(wait=True, project="FA", new_only=False, all=False, limit=20)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.cmd_events(a)
    out.append(buf.getvalue())


def test_quiet_waiter():
    os.environ["SITE_WAIT_POLL_S"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx = m._rx()
        state = {"owners": world("working")[1]}
        rx.read_owners = lambda mm, pr, rules=None: state["owners"]
        c = m.ldb()
        m.add_event(c, "FA", "catalog_coverage", "RUN_DONE", "runs:x", "32 run(s) recorded on catalog_coverage finished")
        c.close()
        out = []
        t = threading.Thread(target=run_waiter, args=(m, out), daemon=True)
        t.start()
        t.join(4)
        assert t.is_alive() and not out, out
        print("W1 ok: owner working, the waiter stays armed")
        state["owners"] = world("stopped")[1]
        t.join(5)
        assert not t.is_alive() and "RUN_DONE" in out[0], out
        print("W2 ok: owner stopped, the waiter returns:", out[0].strip()[:100])
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx = m._rx()
        rx.read_owners = lambda mm, pr, rules=None: world("working")[1]
        c = m.ldb()
        m.add_event(c, "FA", "catalog_coverage", "ALERT", "a", "a batch past 2x its target")
        c.close()
        out = []
        t = threading.Thread(target=run_waiter, args=(m, out), daemon=True)
        t.start()
        t.join(4)
        assert not t.is_alive() and "ALERT" in out[0], out
        print("W3 ok: an ALERT returns whoever owns the entry")


if __name__ == "__main__":
    test_run_done_one_event_per_entry()
    test_quiet_waiter()
