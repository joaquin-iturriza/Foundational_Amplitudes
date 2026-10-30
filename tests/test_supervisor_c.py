#!/usr/bin/env python3
"""The supervisor's switch-over C (plan entry agent_redesign) on the night's recorded states.

The code under test is the `site` tool's ~/.local/bin/site_runends.py (and its glue in ~/.local/bin/site); the states
are tests/fixtures/supervisor_20261029/run_ends.json (tools/supervisor_fixture.py: the 227 FA events of 29 Oct 20:00 to
30 Oct 03:56, with each run's log as it stands on CC and the hand-written ack note of the main session).

  C1  every run end sorted and compared with its ack note: every FAILED run and every run whose note names a fault is
      broke and never acked; a run sorted clean or limit is one the note called finished or at its planned limit;
      each disagreement listed with the log line that decided it
  C2  approved queues on the replan under live workers: a stale layout or a parked queue gets no workers and the stale
      one an ALERT, the current queue gets workers within its limits, failed tasks reach the owner as events and none
      goes back to todo; the glue run on a scratch registry with the site faked
  C3  collection: the night's cleanly finished catalog_v3_short runs, with a collection recorded, are collected and
      acked; a collection that reports a missing artefact leaves the event open
  C4  the 2x alert on the default-share generation's progress read from the logs, with the target it should have had
      (an assumption: no target was recorded that night); plan target-check for 3- and 10-job batches

Run:  python3 tests/test_supervisor_c.py
"""
import importlib.machinery
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures", "supervisor_20261029", "run_ends.json")
BIN = os.path.expanduser("~/.local/bin")
sys.path.insert(0, BIN)
import site_runends as S  # noqa: E402

RULES = S.rules_of({"run_ends": {"enabled": True, "workers": ["scripts/job_lhe_parts.sh"],
                                 "end_markers": [r"RUN_SUMMARY \S+ rc=0\b", "RESULT_JSON", "RUN_DONE", "V3_SHORT_DONE"]}})
# what an ack note says when the hand-reader found a fault in the run: a failure cause, a short run, an error line
NOTE_FAULT = (r"No space|filled at|stops being written|[1-9]\d* of them short|produced no|Traceback|IndexError|"
              r"nonzero exit|rc=1|failures not yet read|cause not investigated")


def events():
    return json.load(open(FIX))


def note_class(e):
    """The ack note's verdict: broke (FAILED, or a fault named), limit (a TIMEOUT it called planned / ended with tasks
    left), clean (COMPLETED)."""
    if e["state"] not in ("COMPLETED", "TIMEOUT") or re.search(NOTE_FAULT, e["ack_note"]):
        return "broke"
    return "limit" if e["state"] == "TIMEOUT" else "clean"


def test_c1_sorting():
    ev = events()
    assert len(ev) == 227 and all(e["acked"] for e in ev)
    runs = [e for e in ev if e["kind"] == "RUN"]
    assert len(runs) == 224 and sum(e["state"] == "FAILED" for e in runs) == 31
    assert sum(e["state"] == "TIMEOUT" for e in runs) == 159 and sum(e["state"] == "COMPLETED" for e in runs) == 34
    counts, disagree = {}, []
    for e in runs:
        act = S.decide_run_end(e["run"], e["state"], e["job"], e["item"], e["log"], RULES)
        cls, want = act["cls"], note_class(e)
        counts[(want, cls)] = counts.get((want, cls), 0) + 1
        if e["state"] == "FAILED":
            assert cls == "broke" and "event" in act, (e["event"], act)
        if want == "broke":
            assert cls == "broke" and "event" in act, ("a failure would have been acked", e["event"], act)
        if cls != "broke":
            assert "ack" in act and e["run"] in act["ack"] and act["line"] in act["ack"], act   # names run and line
        else:
            assert e["run"] in act["event"]
        if cls != want:
            disagree.append((e["event"], e["state"], want, cls, act["line"], e["ack_note"]))
    # the only disagreements are runs the sorter keeps open where the note closed them: each has an error line in
    # its log that the note did not name
    for eid, st, want, cls, line, note in disagree:
        assert cls == "broke" and want in ("limit", "clean"), (eid, want, cls)
        assert any(re.search(p, line) for p in RULES["error_patterns"]), (eid, line)
    auto = sum(v for (w, c), v in counts.items() if c != "broke")
    print("[C1] 224 run ends: note/sorter %s" % ", ".join("%s/%s %d" % (w, c, v) for (w, c), v in sorted(counts.items())))
    print("[C1] acked by the supervisor (never reaching an agent): %d of 224; open for the owner: %d"
          % (auto, 224 - auto))
    print("[C1] disagreements (%d), all kept open by the sorter where the note acked them:" % len(disagree))
    for eid, st, want, cls, line, note in disagree:
        print("   E%d %s note=%s sorter=%s | %s" % (eid, st, want, cls, line[:140]))
    # the three MAINTAIN events are no run ends: the sorter never sees them (they stay events)
    assert sum(e["kind"] == "MAINTAIN" for e in ev) == 3


def test_c1_cases():
    """Named cases: a disk-full worker (its log stops before the scheduler's time-limit line), a worker at its limit,
    the same at another wall than its registered queue's, a TIMEOUT of a non-worker, a COMPLETED run with no marker."""
    by = {e["event"]: e for e in events() if e["kind"] == "RUN"}
    sort = lambda eid, q=None: S.sort_run_end(by[eid]["state"], by[eid]["job"], by[eid]["log"], RULES, q)
    assert sort(2189)[0] == "broke" and "stops before the time limit" in sort(2189)[1]       # /sps full
    assert sort(2325)[0] == "limit"                                                         # at its 2 h limit
    assert sort(2325, {"wall_h": 2.0})[0] == "limit"
    assert sort(2325, {"wall_h": 4.0})[0] == "broke"                                        # not the queue's wall
    assert sort(2364)[0] == "broke"                                                         # gen_daniel TIMEOUT
    log = dict(by[2278]["log"], tail="all fine\n", matches=[])
    assert S.sort_run_end("COMPLETED", by[2278]["job"], log, RULES)[0] == "broke"            # no end marker
    assert S.sort_run_end("COMPLETED", "x", {}, RULES) == ("broke", "no log found")
    print("[C1 cases] ok")


def load_site():
    loader = importlib.machinery.SourceFileLoader("site_tool", os.path.join(BIN, "site"))
    spec = importlib.util.spec_from_loader("site_tool", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    sys.modules["site_tool"] = m
    return m


def scratch_site(d):
    """The site tool with its registry in d, a decision D1 approving item gen (active), and no real ssh."""
    m = load_site()
    m.HOME, m.DB, m.READS = d, os.path.join(d, "runs.db"), os.path.join(d, "reads.log")
    c = m.ldb()
    c.execute("INSERT INTO decisions (id,project,kind,quote,covers,recorded,in_force) VALUES "
              "('D1','FA','approve','go','gen',0,1)")
    c.execute("INSERT INTO items (project,id,title,decision,status) VALUES ('FA','gen','gen','D1','active')")
    c.commit()
    return m, c


def test_c2_queues():
    now = time.time()
    q = {"queue": "catalog_v4_default", "max_workers": 40, "wall_h": 2.0, "total_hours": 200.0, "submitted_hours": 0}
    # the night's counts (status lines of the logs): before the replan, the old queue under its old workers
    old = {"current": False, "why": "1082 process(es) laid out otherwise in the recipes now, e.g. dsh_bb_bbgg",
           "todo": 2951, "claimed": 166, "done": 1852, "failed": 198}
    n, why, alert = S.queue_plan(q, 10, old, now)
    assert n == 0 and alert and "not current" in alert
    n, why, alert = S.queue_plan(q, 10, {"current": False, "why": "parked", "todo": 0, "failed": 198}, now)
    assert n == 0 and "parked" in alert
    new = {"current": True, "todo": 1466, "claimed": 190, "done": 3295, "failed": 216}
    assert S.queue_plan(q, 10, new, now)[0] == 30                        # up to the worker limit
    assert S.queue_plan(dict(q, submitted_hours=196), 10, new, now)[0] == 2   # what the hours allow
    n, why, alert = S.queue_plan(dict(q, submitted_hours=200), 10, new, now)
    assert n == 0 and "spent" in alert
    assert S.queue_plan(q, 40, new, now)[0] == 0 and S.queue_plan(dict(q, until=now - 1), 0, new, now)[0] == 0
    assert S.queue_plan(q, 0, dict(new, todo=3), now)[0] == 3
    # the glue on a scratch registry: three passes (the old layout stale, its 198 failed tasks the count seen at
    # registration; the new queue current with 216 failed; then 908 more failed)
    with tempfile.TemporaryDirectory() as d:
        m, c = scratch_site(d)
        lays = [dict(old), dict(new, failed=216), dict(new, failed=216 + 908)]
        subs = []

        def run_local(cmd, cwd, timeout):
            if "lhe_parts.py layout" in cmd:
                return 0, json.dumps(lays[0])
            assert cmd.startswith("site submit ccin2p3 FA scripts/job_lhe_parts.sh --no-sync --item gen "
                                  "--env LHE_QUEUE=catalog_v4_default --hdr=--time=02:00:00"), cmd
            subs.append(cmd)
            return 0, "  submitted FA-ccin2p3-%08x" % len(subs)
        m._run_local = run_local
        a = type("A", (), dict(action="add", project="FA", id="catalog_v4_default", decision="D1", item="gen",
                               site="ccin2p3", job="scripts/job_lhe_parts.sh", max_workers=4, wall=2.0,
                               total_hours=6.0, until=None))
        S.cmd_queue(m, a)
        quiet = lambda *x: None
        n0 = S.queue_pass(m, c, "FA", {"local": ROOT}, quiet)
        assert n0 == 1 and not subs                                          # stale: ALERT, nothing submitted
        lays.pop(0)
        S.queue_pass(m, c, "FA", {"local": ROOT}, quiet)
        assert len(subs) == 3                                                # 6 h at 2 h each, under 4 workers
        lays.pop(0)
        S.queue_pass(m, c, "FA", {"local": ROOT}, quiet)
        assert len(subs) == 3                                                # the hours are spent
        ev = c.execute("SELECT kind,key,msg FROM events WHERE project='FA' AND acked IS NULL ORDER BY id").fetchall()
        failed = [e for e in ev if ":failed:" in e[1]]
        assert [f[2].split(" task(s)")[0].split(": ")[-1] for f in failed] == ["18", "908"], failed
        assert all("never puts a failed task back" in f[2] for f in failed)
        assert any("spent" in e[2] for e in ev), ev
        assert S.queues(c, "FA")[0]["submitted_hours"] == 6.0
        # a decision withdrawn: nothing more, an ALERT
        c.execute("UPDATE decisions SET in_force=0"); c.commit()
        lays[0] = dict(new, failed=216 + 908)
        S.queue_pass(m, c, "FA", {"local": ROOT}, quiet)
        assert len(subs) == 3
    print("[C2] stale layout and parked queue: none resubmitted, ALERT; current queue: within its worker and hour "
          "limits; failed tasks: one open event per new count (18, then 908), none put back")


def test_c3_collection():
    by = {e["event"]: e for e in events() if e["kind"] == "RUN"}
    v3 = [e for e in by.values() if e["job"].endswith("job_catalog_v3_short.sh")]
    assert len(v3) == 4 and all(e["state"] == "COMPLETED" for e in v3)
    coll = {"job": "scripts/job_catalog_v3_short.sh", "args": "--axes TR"}
    calls = []
    ok = lambda run, entry, args: (calls.append((run, entry, args)) or (0, "%s: 12 probes read, 0 off" % run))
    for e in v3:
        act = S.decide_run_end(e["run"], e["state"], e["job"], e["item"], e["log"], RULES, None, coll, ok)
        assert act["cls"] == "clean" and "ack" in act and "collected" in act["ack"], act
    assert [c[1] for c in calls] == ["v3_prep_explore"] * 4 and calls[0][2] == "--axes TR"
    off = lambda run, entry, args: (1, "TR: probe dsh_x has no prediction file")
    act = S.decide_run_end(v3[0]["run"], "COMPLETED", v3[0]["job"], v3[0]["item"], v3[0]["log"], RULES, None, coll, off)
    assert "event" in act and "no prediction file" in act["event"] and v3[0]["run"] in act["event"]
    # a run of another job is not collected; a worker at its limit never is
    e = by[2325]
    act = S.decide_run_end(e["run"], e["state"], e["job"], e["item"], e["log"], RULES, None, coll, off)
    assert act["cls"] == "limit" and "ack" in act
    # the glue: one acked RUN event in the registry, named by the run, read recorded
    with tempfile.TemporaryDirectory() as d:
        m, c = scratch_site(d)
        e = v3[0]
        c.execute("INSERT INTO runs (id,project,site,job,sched_id,state,remote_path,item) VALUES (?,?,?,?,?,?,?,?)",
                  (e["run"], "FA", "ccin2p3", e["job"], e["sched_id"], "COMPLETED", "/x", "gen"))
        c.commit()
        saved, S.read_log = S.read_log, (lambda *a: e["log"])
        m._run_local = lambda cmd, cwd, t: ((0, json.dumps([{"entry": "gen", "job": coll["job"], "args": ""}]))
                                            if "collections" in cmd else (0, "12 probes read, 0 off"))
        assert S.run_end(m, c, "FA", {"local": ROOT, "run_ends": RULES}, e["run"], {}, "COMPLETED", "gen",
                         lambda *x: None, {}) == 0
        r = c.execute("SELECT acked, ack_note, tell FROM events WHERE key=?", (e["run"] + ":COMPLETED",)).fetchone()
        assert r[0] and e["run"] in r[1] and "collected" in r[1] and r[2] == "routine", r
        assert m.read_since(e["run"], 0)
        S.read_log = saved
    print("[C3] 4 clean catalog_v3_short runs collected and acked; a missing artefact leaves the event open")


def default_share_series():
    """(ts, done, expected) of the default-share queue (1082 processes) from the logs' status lines: a worker's first
    status line at its start (end time less its wall), and the last one of a worker that ran to its end."""
    pts = []
    for e in events():
        if e["kind"] != "RUN" or not e["job"].endswith("job_lhe_parts.sh"):
            continue
        head, tail = e["log"]["head"].splitlines(), e["log"]["tail"].splitlines()
        q = next((re.search(RULES["queue_line"], l) for l in head if re.search(RULES["queue_line"], l)), None)
        hs = next((json.loads(l) for l in head if l.startswith('{"todo"')), None)
        if not q or not hs or hs["processes"] != 1082:
            continue
        fin = e["finished"] or e["ts"]
        tot = lambda s: s["todo"] + s["claimed"] + s["done"] + s["failed"]
        if e["state"] == "TIMEOUT":
            pts.append((fin - (float(q.group(2)) + 5 / 60.0) * 3600, hs["done"], tot(hs)))
        else:
            ts = [json.loads(l) for l in tail if l.startswith('{"todo"')]
            if ts and "RUN_SUMMARY" in e["log"]["tail"]:
                pts.append((fin, ts[-1]["done"], tot(ts[-1])))
    return sorted(set(pts))


def test_c4_target():
    s = default_share_series()
    fmt = lambda t: time.strftime("%H:%M", time.localtime(t))
    # the first plan (18:11) and the replan's batch (23:46, the 40 workers after the short-run fix); the target each
    # should have had is an assumption: two generations of 2 h workers, 4 h
    t0 = s[0][0]
    hit = S.overrun(s, t0, 4.0)
    assert hit, s
    print("[C4] first plan from %s, target 4 h: alert at %s (%.1f h in, predicted %.1f h)"
          % (fmt(t0), fmt(t0 + hit[0] * 3600), hit[0], hit[1] or float("nan")))
    t1 = next(p[0] for p in s if p[2] > 4300)
    hit = S.overrun(s, t1, 4.0)
    print("[C4] replan from %s, target 4 h: %s" % (fmt(t1), "alert at %s (%.1f h in, predicted %.1f h)" % (
        fmt(t1 + hit[0] * 3600), hit[0], hit[1] or float("nan")) if hit else "no alert"))
    assert hit
    # never before the prediction passes 2x: every earlier point predicts under 8 h
    base = next(p for p in s if p[0] >= t1)
    for p in s:
        if t1 <= p[0] < t1 + hit[0] * 3600 and (p[1] - base[1]) / p[2] >= S.MIN_FRACTION:
            assert (S.predict_end(base, p, t1) or 0) <= 8.0
    assert S.overrun(s, t0, 100.0) is None                                   # a generous target: no alert
    assert S.overrun([(0, 0, 10), (9 * 3600, 0, 10)], 0, 4.0) == (9.0, None)  # past 2x with nothing done


def test_c4_plan_target():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "plan"))
        pf = os.path.join(d, "plan", "plan.yaml")
        open(pf, "w").write("entries:\n- id: gen\n  status: running\n  title: gen\n")
        env = dict(os.environ, PLAN_TEST="1", PLAN_ROOT=d, PLAN_FILE=pf)
        plan = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plan.py")] + list(a), env=env,
                                         capture_output=True, text=True)
        assert plan("target-check", "gen", "--jobs", "3").returncode == 0           # 3 jobs: passes
        r = plan("target-check", "gen", "--jobs", "10")
        assert r.returncode == 1 and "needs a wall-time target" in r.stdout, r     # 10 jobs: refused
        assert plan("target", "gen", "--batch", "catalog_v4_w3", "--wall", "4", "--jobs", "10").returncode == 0
        assert plan("target-check", "gen", "--jobs", "10").returncode == 0
        t = json.loads(plan("targets", "--json").stdout)
        assert t[0]["batch"] == "catalog_v4_w3" and t[0]["wall_h"] == 4.0 and t[0]["entry"] == "gen" and t[0]["start_ts"]
        assert plan("collection", "gen", "--job", "scripts/job_catalog_v3_short.sh").returncode == 0
        assert json.loads(plan("collections", "--json").stdout)[0]["job"] == "scripts/job_catalog_v3_short.sh"
    print("[C4 plan] target-check: 3 jobs pass, 10 refused until `plan target`; targets and collections listed")


if __name__ == "__main__":
    test_c1_sorting()
    test_c1_cases()
    test_c2_queues()
    test_c3_collection()
    test_c4_target()
    test_c4_plan_target()
    print("all ok")
