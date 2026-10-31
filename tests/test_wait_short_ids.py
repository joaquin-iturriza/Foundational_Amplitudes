#!/usr/bin/env python3
"""Every clean end of a waited run reaches the entry's owner (2026-10-31, catalog_coverage step 18): the step's wait
named six CPU probes as one full id and five bare 8-hex suffixes, plus six arms of another job script. `plan owners`
read only the full id, so the supervisor raised RUN_DONE for that one probe and acked the other five clean ends with
no event; nobody acted on them for five hours. On a scratch registry and event table; the code under test is
tools/plan.py (wait_run_ids, owners' "waits") and ~/.local/bin/site_runends.py (run_done_events).

  S1  replay of step 18 at 07:01: the six probes ended clean, the six arms still running: one RUN_DONE naming all six
  S2  replay at 09:25: two arms ended clean, four running: no event (their batch is open)
  S3  replay at 10:06: the four other arms ended TIMEOUT (broken, their own RUN events): one RUN_DONE naming the two
      clean arms of that batch, steps 15 and 18
  S4  a single run in its own wait ends clean: one RUN_DONE
  S5  a bare suffix matching two runs, or no run (a commit hash), is not read as a run

Run:  python3 tests/test_wait_short_ids.py
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_run_done_waiter import load_site      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quiet = lambda *a: None
PROBES = ["FA-ccin2p3-" + h for h in "fd7cf028 231e0e90 1ffcb240 c7137105 d2fb03c4 2d00ecdf".split()]
ARMS = ["FA-ccin2p3-" + h for h in "87ea24d0 321a3de2 466656b2 9897833d f0156d88 df858d33".split()]
# the wait texts as recorded at plan commit e9e5eaa5 (2026-10-31 06:29)
WAITS = {"15": "FA-ccin2p3-87ea24d0 FA-ccin2p3-321a3de2 FA-ccin2p3-466656b2 FA-ccin2p3-9897833d "
               "FA-ccin2p3-f0156d88 FA-ccin2p3-df858d33",
         "18": "hd_probe FA-ccin2p3-fd7cf028 231e0e90 1ffcb240 c7137105 d2fb03c4 2d00ecdf (1.2 h); het arms 87ea24d0 "
               "321a3de2 466656b2 9897833d f0156d88 df858d33"}


def put(c, rid, state, job):
    c.execute("INSERT OR REPLACE INTO runs (id, project, state, job, remote_path, sched_id) VALUES (?,?,?,?,?,?)",
              (rid, "FA", state, job, "/x", "1"))
    c.commit()


def sorted_clean(c, rid):
    """what run_end writes for a clean end: a routine RUN event whose note says finished cleanly"""
    c.execute("INSERT INTO events (ts,project,item,kind,key,msg,acked,ack_note,tell) VALUES (1,'FA','catalog_coverage',"
              "'RUN',?,?,1,?,'routine')", ("%s:COMPLETED" % rid, "%s ended COMPLETED" % rid,
                                          "%s ended COMPLETED, finished cleanly (sorted by the supervisor)" % rid))
    c.commit()


def owners(tmp, db, waits, runs, extra=""):
    os.makedirs(os.path.join(tmp, "plan"), exist_ok=True)
    pf = os.path.join(tmp, "plan", "plan.yaml")
    with open(pf, "w") as f:
        f.write("entries:\n- id: catalog_coverage\n  status: running\n  title: cov\n  next:\n")
        for k in waits:
            f.write("  - step %s\n" % k)
        f.write("  step_waits:\n")
        for k, w in waits.items():
            f.write("    '%s': %s\n" % (k, json.dumps(w)))
        f.write(extra)
    env = dict(os.environ, **{"PLAN_" + "TEST": "1", "PLAN_" + "ROOT": tmp, "PLAN_" + "FILE": pf, "PLAN_RUNS_DB": db})
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plan.py"), "owners"], env=env,
                       capture_output=True, text=True)
    rows = {x["entry"]: x for x in json.loads(r.stdout.strip().splitlines()[-1])}
    o = rows["catalog_coverage"]
    o.update(owner="ae7bc0ff5c64982f0", owner_state="stopped", runs=runs)
    return rows


def open_done(c):
    return [m for (m,) in c.execute("SELECT msg FROM events WHERE kind='RUN_DONE' AND acked IS NULL ORDER BY id")]


def test_replay_step18():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx, c = m._rx(), m.ldb()
        for r in PROBES:
            put(c, r, "COMPLETED", "scripts/job_lhe_probe.sh")
            sorted_clean(c, r)
        for r in ARMS:
            put(c, r, "RUNNING", "scripts/job_catalog_v3_short.sh")
        o = owners(tmp, m.DB, WAITS, PROBES + ARMS)
        w18 = [w for w in o["catalog_coverage"]["waits"] if w["step"] == "18"][0]["runs"]
        assert sorted(w18) == sorted(PROBES + ARMS), w18
        # 07:01: the six probes sorted clean in this pass
        assert rx.run_done_events(m, c, "FA", PROBES, o, log=quiet, ended_now=PROBES) == 1
        ev = open_done(c)
        assert len(ev) == 1 and all(p in ev[0] for p in PROBES) and "6 run(s)" in ev[0] and "step 18" in ev[0], ev
        print("S1 ok: 07:01 replay, one RUN_DONE naming the six probes:", ev[0][:150])
        c.execute("UPDATE events SET acked=1 WHERE kind='RUN_DONE'")
        # 09:25: two arms clean, four running
        for r in ARMS[:2]:
            put(c, r, "COMPLETED", "scripts/job_catalog_v3_short.sh")
            sorted_clean(c, r)
        assert rx.run_done_events(m, c, "FA", ARMS[:2], o, log=quiet, ended_now=ARMS[:2]) == 0 and not open_done(c)
        print("S2 ok: 09:25 replay, two of six arms ended: no event")
        # 10:06: four arms TIMEOUT (broken: run_end opened their RUN events); no clean end this pass
        for r in ARMS[2:]:
            put(c, r, "TIMEOUT", "scripts/job_catalog_v3_short.sh")
        assert rx.run_done_events(m, c, "FA", [], o, log=quiet, ended_now=ARMS[2:]) == 1
        ev = open_done(c)
        assert len(ev) == 1 and all(a in ev[0] for a in ARMS[:2]) and not any(a in ev[0] for a in ARMS[2:]) \
            and "steps" not in ev[0] and "step 15, 18" in ev[0], ev
        print("S3 ok: 10:06 replay, arms batch closed on broken runs, its clean ones named:", ev[0][:150])


def test_single_run():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        rx, c = m._rx(), m.ldb()
        r = "FA-ccin2p3-0badc0de"
        put(c, r, "COMPLETED", "scripts/job_lhe_probe.sh")
        o = owners(tmp, m.DB, {"3": "the probe 0badc0de (0.5 h)"}, [r])
        assert o["catalog_coverage"]["waits"] == [{"step": "3", "runs": [r]}], o
        assert rx.run_done_events(m, c, "FA", [r], o, log=quiet, ended_now=[r]) == 1
        ev = open_done(c)
        assert len(ev) == 1 and r in ev[0] and "RESUME ae7bc0ff5c64982f0" in ev[0], ev
        print("S4 ok: one waited run named by its suffix ends clean: one RUN_DONE:", ev[0][:120])


def test_ambiguous_suffix():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        c = m.ldb()
        put(c, "FA-ccin2p3-aaaa0001", "COMPLETED", "j")
        put(c, "FA-jeanzay-aaaa0001", "COMPLETED", "j")
        o = owners(tmp, m.DB, {"4": "runs aaaa0001 at commit 11297eec"}, [])
        assert o["catalog_coverage"]["waits"] == [], o
        print("S5 ok: an ambiguous suffix and a commit hash are not read as runs")


if __name__ == "__main__":
    test_replay_step18()
    test_single_run()
    test_ambiguous_suffix()
    print("all ok")
