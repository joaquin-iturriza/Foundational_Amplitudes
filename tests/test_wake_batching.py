#!/usr/bin/env python3
"""One wake per batch (Joaquin 2026-10-31: most messages since his last were batches finishing), on a scratch registry
and event table; the code under test is ~/.local/bin/site_runends.py (run_done_events, run_end), tools/plan.py owners'
"waits" and tools/detach.py --each.

  B1  an entry with 2 runs in one plan wait, one ends clean: no RUN_DONE; then the second ends clean: one RUN_DONE
      listing both
  B2  a run in no wait ends clean: one RUN_DONE at once (unchanged)
  B3  a broken run inside a wait whose other run is live: its RUN event opens at once
  B4  detach --each with 3 commands (rc 0, 3, 0): one DONE event, the job's rc 3, each command's rc in the log
  B5  plan owners lists a step's waited runs

Run:  python3 tests/test_wake_batching.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_run_done_waiter import load_site      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A, B, C = "FA-ccin2p3-0000000a", "FA-ccin2p3-0000000b", "FA-ccin2p3-0000000c"
quiet = lambda *a: None


def owners(runs, waits):
    return {"cov": {"entry": "cov", "status": "running", "owner": "a01", "owner_state": "stopped", "runs": runs,
                    "waits": waits}}


def put(c, rid, state):
    c.execute("INSERT OR REPLACE INTO runs (id, project, state, job, remote_path, sched_id) VALUES (?,?,?,?,?,?)",
              (rid, "FA", state, "scripts/job_x.sh", "/x", "1"))
    c.commit()


def events(c, kind):
    return c.execute("SELECT item, msg FROM events WHERE kind=? AND acked IS NULL", (kind,)).fetchall()


def test_wait_group():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx, c = m._rx(), m.ldb()
        o = owners([A, B], [{"step": "3", "runs": [A, B]}])
        put(c, A, "COMPLETED"), put(c, B, "RUNNING")
        assert rx.run_done_events(m, c, "FA", [A], o, log=quiet) == 0 and not events(c, "RUN_DONE")
        put(c, B, "COMPLETED")
        assert rx.run_done_events(m, c, "FA", [B], o, log=quiet) == 1
        ev = events(c, "RUN_DONE")
        assert len(ev) == 1 and A in ev[0][1] and B in ev[0][1] and "2 run(s)" in ev[0][1], ev
        print("B1 ok: one of two waited runs ended: none; both ended: one event:", ev[0][1][:110])


def test_no_wait_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx, c = m._rx(), m.ldb()
        put(c, C, "COMPLETED")
        assert rx.run_done_events(m, c, "FA", [C], owners([C], []), log=quiet) == 1
        print("B2 ok: a run in no wait gives its RUN_DONE at once")


def test_broken_in_wait():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx, c = m._rx(), m.ldb()
        put(c, A, "FAILED"), put(c, B, "RUNNING")
        rx.read_log = lambda *a, **k: {"head": "", "tail": "Traceback (most recent call last):\nRuntimeError: boom\n"}
        rx._collections = lambda *a, **k: {}
        n = rx.run_end(m, c, "FA", {"local": tmp}, A, None, "FAILED", "cov", log=quiet, cache={})
        ev = events(c, "RUN")
        assert n == 1 and len(ev) == 1 and A in ev[0][1], (n, ev)
        print("B3 ok: a broken run inside a live wait opens its RUN event at once:", ev[0][1][:100])


def test_detach_each():
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "site")
        calls = os.path.join(tmp, "calls")
        open(fake, "w").write('#!/bin/bash\necho "$@" >> %s\n' % calls)
        os.chmod(fake, 0o755)
        env = dict(os.environ, DETACH_SITE=fake)
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "detach.py"), "--entry", "zz_test",
                              "--name", "zz_probe_each", "--each", "echo one", "--each", "echo two; exit 3",
                              "--each", "sleep 1; echo three"], env=env, capture_output=True, text=True, check=True)
        log = out.stdout.split()[1]
        for _ in range(100):
            if os.path.exists(calls):
                break
            time.sleep(0.1)
        time.sleep(0.3)
        lines = open(calls).read().splitlines()
        body = open(log).read()
        assert len(lines) == 1 and "--kind DONE" in lines[0] and "3 commands ended, rc=3" in lines[0] \
            and "1:rc=0 2:rc=3 3:rc=0" in lines[0], lines
        assert "[2] rc=3; last line: two" in body and "[3] rc=0; last line: three" in body, body
        for f in glob.glob(log + "*"):
            os.unlink(f)
        print("B4 ok: 3 commands, one DONE:", lines[0][:140])


def test_owners_waits():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "plan"))
        pf = os.path.join(d, "plan", "plan.yaml")
        open(pf, "w").write("entries:\n- id: gen\n  status: running\n  title: gen\n  next:\n  - x\n"
                            "  step_waits:\n    '1': runs %s and %s\n" % (A, B))
        env = dict(os.environ, **{"PLAN_" + "TEST": "1", "PLAN_" + "ROOT": d, "PLAN_" + "FILE": pf})
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plan.py"), "owners"], env=env,
                           capture_output=True, text=True)
        rows = json.loads(r.stdout.strip().splitlines()[-1])
        assert rows[0]["waits"] == [{"step": "1", "runs": [A, B]}], (rows, r.stderr)
        print("B5 ok: plan owners lists the waited runs:", rows[0]["waits"])


if __name__ == "__main__":
    test_wait_group()
    test_no_wait_unchanged()
    test_broken_in_wait()
    test_detach_each()
    test_owners_waits()
    print("all ok")
