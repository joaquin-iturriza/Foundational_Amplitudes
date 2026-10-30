"""Scenarios the setup redesign (D121) must get right, most of them failures of the week of 2026-10-19 or findings of
its review. Nothing live is touched: each runs on a temporary plan, state file, registry, fixture transcripts and
subagent transcripts; `site` is a stub for every command that writes (decide, item add), and only site's read-only
rules (_typed_texts, _interactive, TERMINAL) are imported from it. The claims verifier and the git hooks run in
temporary git repositories.
    python3 tests/test_setup_scenarios.py"""
import concurrent.futures as cf, datetime as dt, json, os, shutil, sqlite3, subprocess, sys, tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN, CHECK = os.path.join(ROOT, "tools", "plan.py"), os.path.join(ROOT, ".claude", "hooks", "plan_check.py")
VERIFY, MIGRATE = os.path.join(ROOT, "tools", "verify_claims.py"), os.path.join(ROOT, "tools", "plan_migrate.py")
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
ago = lambda m: (NOW - dt.timedelta(minutes=m)).isoformat()
epoch = lambda m: int((NOW - dt.timedelta(minutes=m)).timestamp())
J = lambda x: json.dumps(x, separators=(",", ":"))

# the stub site: decide records the decision in the test registry (so a rollback is visible), item add records the
# item unless STUB_ITEM_FAIL is set; every call is logged
STUB = r'''#!/usr/bin/env python3
import os, sqlite3, sys
a = sys.argv[1:]
d = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(d, "calls.log"), "a").write(" ".join(a) + "\n")
c = sqlite3.connect(os.path.join(d, "runs.db"))
if a[0] == "decide":
    import datetime as dt, glob, json
    kind, cov, q = a[a.index("--kind") + 1], a[a.index("--covers") + 1], a[a.index("--quote") + 1]
    # said_at as the real site dates it: the last time the user typed the words (here: the fixture transcripts)
    t = None
    for f in glob.glob(os.path.join(os.environ.get("PLAN_TRANSCRIPTS", "/nonexistent"), "*.jsonl")):
        for line in open(f):
            m = json.loads(line)
            if m.get("type") == "user" and q in str((m.get("message") or {}).get("content")):
                t = max(t or m["timestamp"], m["timestamp"])
    t = t or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    if os.environ.get("STUB_DECIDE_FAIL"):
        sys.exit("site: decide refused")
    if kind in ("hold", "release"):      # as the real site: a hold or release names existing items
        hit = [k for k in cov.split(",") if c.execute("SELECT 1 FROM items WHERE id=?", (k,)).fetchone()]
        if not hit:
            sys.exit("site: a %s names the work items it covers; none of %s matches an item" % (kind, cov))
        for k in hit:
            c.execute("UPDATE items SET status=? WHERE id=?", ("held" if kind == "hold" else "active", k))
    did = "D%d" % (900 + c.execute("SELECT COUNT(*) FROM decisions WHERE id LIKE 'D9__'").fetchone()[0])
    c.execute("INSERT INTO decisions VALUES (?,'FA',?,?,?,?,1,'')", (did, kind, q, cov, t))
    if kind in ("hold", "release"):      # as the real site: the older opposite decision stops covering these items
        other = "release" if kind == "hold" else "hold"
        for oid, ocov in c.execute("SELECT id, covers FROM decisions WHERE kind=? AND in_force=1", (other,)).fetchall():
            left = [k for k in (ocov or "").split(",") if k and k not in cov.split(",")]
            c.execute("UPDATE decisions SET covers=?, in_force=? WHERE id=?", (",".join(left), 1 if left else 0, oid))
    c.commit()
    print("  %s  FA/%s  said %s: \"x\"" % (did, kind, t))
elif a[0] == "run":                      # only the read of sweep_manager's state: sweeps_<site>.json, else down
    p = os.path.join(d, "sweeps_%s.json" % a[1])
    if not os.path.exists(p):
        sys.stderr.write("ssh: connect to host %s port 22: Connection refused\n" % a[1])
        sys.exit(255)
    print("  run %s: python sweep/sweep_manager.py state" % a[1])
    print("SWEEP_STATE " + open(p).read().strip())
elif a[0] == "item" and a[1] == "done":
    c.execute("UPDATE items SET status='done' WHERE id=?", (a[3],))
    c.commit()
elif a[0] == "item":
    if os.environ.get("STUB_ITEM_FAIL"):
        sys.exit("site: item refused")
    dec = a[a.index("--decision") + 1]
    k = c.execute("SELECT kind, in_force FROM decisions WHERE id=?", (dec,)).fetchone()
    if not k or not k[1] or k[0] not in ("approve", "scope", "release"):    # as the real site
        sys.exit("site: an item needs a decision in force that approves it")
    c.execute("INSERT INTO items (project, id, status, decision, match) VALUES ('FA', ?, 'active', ?, ?)",
              (a[3], dec, a[a.index("--match") + 1]))
    c.commit()
    print("  item FA/%s active" % a[3])
'''


class World:
    def __init__(self, entries, runs=(), decisions=(), items=(), events="  no open events", transcript=(),
                 state=None, agents=(), supervisor=True):
        self.d = tempfile.mkdtemp(prefix="plan_scen_")
        j = lambda x: os.path.join(self.d, x)
        self.plan, self.state, self.db, self.tr, self.ag = j("plan.yaml"), j(".state.json"), j("runs.db"), j("tr"), j("ag")
        os.makedirs(self.tr); os.makedirs(self.ag); os.makedirs(j("docs"))
        open(j("docs/results.tex"), "w").write("\\paragraph{Readout.}\\label{par:readout} text\n")
        yaml.safe_dump({"entries": entries}, open(self.plan, "w"))
        json.dump(state or {}, open(self.state, "w"))
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE runs (id TEXT, project TEXT, state TEXT, item TEXT, submitted INTEGER, finished INTEGER)")
        c.execute("CREATE TABLE decisions (id TEXT, project TEXT, kind TEXT, quote TEXT, covers TEXT, said_at TEXT, "
                  "in_force INTEGER, note TEXT)")
        c.execute("CREATE TABLE items (project TEXT, id TEXT, check_cmd TEXT, status TEXT, decision TEXT, match TEXT)")
        c.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?)", [(r[0], "FA") + tuple(r[1:]) for r in runs])
        if not any(x[0] == "D1" for x in decisions):          # the approval every running test entry stands on
            decisions = list(decisions) + [("D1", "approve", "*", 1)]
        c.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,'')", [
            (x[0], "FA", x[1], x[5] if len(x) > 5 else "q", x[2], x[4] if len(x) > 4 else ago(600), x[3])
            for x in decisions])
        c.executemany("INSERT INTO items (project, id, check_cmd, status) VALUES ('FA',?,?,'active')", items)
        c.commit()
        with open(os.path.join(self.tr, "s.jsonl"), "w") as f:
            f.write(J({"type": "summary", "entrypoint": "cli"}) + "\n")
            for ts, role, text in transcript:
                m = {"content": [{"type": "text", "text": text}]} if role == "assistant" else {"content": text}
                f.write(J({"type": role, "timestamp": ts, "entrypoint": "cli", "message": m}) + "\n")
        # subagent transcripts: (id, started, entry its prompt names); every step in `state` gets its own, started
        # at the step's time with that entry's prompt, unless its agent id starts with "forged". An optional fourth
        # element lists later records (time, "work" | "handback"): a tool call, or the SubagentHandback call and its
        # result, as a handed-back agent's transcript ends
        agents = [tuple(a) + ((),) * (4 - len(a)) for a in agents] + [
            (x["agent"], x["at"], eid, ()) for eid, v in (state or {}).items() if eid[:1] != "_"
            for x in (v or {}).get("steps") or [] if not x["agent"].startswith("forged")]
        for aid, started, eid, later in agents:
            p = os.path.join(self.ag, "agent-%s.jsonl" % aid)
            if not os.path.exists(p):
                recs = [{"type": "user", "timestamp": started, "message": {
                    "role": "user", "content": "You are a step agent.\n\nEntry: %s (t)\nQuestion: q" % eid}}]
                for ts, kind in later:
                    name, inp = ("SubagentHandback", {"message": "report"}) if kind == "handback" else \
                        ("Bash", {"command": "site poll"})
                    recs.append({"type": "assistant", "timestamp": ts, "message": {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "tu1", "name": name, "input": inp}]}})
                    recs.append({"type": "user", "timestamp": ts, "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}]}})
                open(p, "w").write("".join(J(r) + "\n" for r in recs))
        # a fake /proc for the stop check's supervisor test: a `site supervise --project FA` process unless told not
        os.makedirs(j("proc/1")); open(j("proc/1/cmdline"), "wb").write(b"bash\0-c\0sleep 1\0")
        if supervisor:
            os.makedirs(j("proc/2")); open(j("proc/2/cmdline"), "wb").write(
                b"/usr/bin/python3\0/home/u/.local/bin/site\0supervise\0--project\0FA\0")
        open(j("site"), "w").write(STUB)
        os.chmod(j("site"), 0o755)
        self.env = dict(os.environ, PLAN_TEST="1", PLAN_FILE=self.plan, PLAN_STATE=self.state, PLAN_RUNS_DB=self.db,
                        PLAN_TRANSCRIPTS=self.tr, PLAN_AGENTS=self.ag, PLAN_SITE=j("site"), PLAN_ROOT=self.d,
                        PLAN_SITE_MODULE=shutil.which("site"), PLAN_SWEEP_SITES="", PLAN_PROC=j("proc"),
                        PLAN_EVENTS_CMD="printf '%s\\n'" % events.replace("'", ""))

    def check(self):
        """The stop check's verdict, plus what reaches the main session. A passing check prints nothing (its notes
        never reach the user's screen, 2026-10-26); the notes are read from `plan next`, where the main session sees
        them, and a passing check that printed anything fails here."""
        r = subprocess.run([sys.executable, CHECK], input="{}", env=self.env, capture_output=True, text=True)
        if r.returncode != 0:
            if "note: " in r.stderr + r.stdout:
                return 99, "a blocking stop check carried a note to the user's screen: " + r.stderr + r.stdout
            return r.returncode, r.stderr + r.stdout + "\n" + self.plan_cmd("next")[1]
        if r.stdout.strip() or r.stderr.strip():
            return 99, "a passing stop check printed to the user's screen: " + r.stderr + r.stdout
        return 0, self.plan_cmd("next")[1]

    def plan_cmd(self, *args, env=None):
        r = subprocess.run([sys.executable, PLAN] + list(args), env=dict(self.env, **(env or {})), capture_output=True,
                           text=True)
        return r.returncode, r.stdout + r.stderr

    def entries(self):
        return {e["id"]: e for e in yaml.safe_load(open(self.plan))["entries"]}

    def calls(self):
        p = os.path.join(self.d, "calls.log")
        return open(p).read() if os.path.exists(p) else ""

    def sql(self, q):
        return sqlite3.connect(self.db).execute(q).fetchall()


def E(eid, **k):
    e = {"id": eid, "title": eid, "question": "q", "status": "running", "quotes": [], "decisions": ["D1"], "next": "n",
         "done_when": None, "notes_ref": None, "match": eid + "*",
         "evidence": {"runs": [], "commit": None, "paths": []}, "dead_ends": [], "asks": []}
    e.update(k)
    return e


_N = [0]


def step(m, agent=None):
    _N[0] += 1
    return {"steps": [{"agent": agent or "s%03d" % _N[0], "at": ago(m)}]}


RESULTS = []
# what every set-next / add-next step must carry: the decision it feeds and two branches, each with an action
WHY = ("--feeds", "whether the arm is adopted at catalog scale", "--if", "it wins: adopt it in the template",
       "--if", "it loses: drop it and record a dead end")


def scenario(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "\n      " + str(detail).strip()[:700]))


def liveness():
    w = World([E("lowd")], events="  E1519  10-25 16:04 FA  RUN  lowd  FA-ccin2p3-x ended COMPLETED")
    rc, err = w.check()
    scenario("nothing in flight blocks, and so does an open event",
             rc == 2 and "lowd: nothing in flight" in err and "open event" in err, err)
    w = World([E("lowd")], state={"lowd": step(10)})
    scenario("a step agent started 10 min ago is the entry in flight", w.check()[0] == 0, w.check()[1])
    w = World([E("lowd")], state={"lowd": step(50)})
    scenario("one started 50 min ago is not", w.check()[0] == 2, w.check()[1])
    w = World([E("lowd")], state={"lowd": step(40, "busy1")}, agents=[("busy1", ago(40), "lowd", [(ago(5), "work")])])
    scenario("a step agent started 40 min ago that wrote a record 5 min ago is in flight", w.check()[0] == 0,
             w.check()[1])
    w = World([E("lowd")], state={"lowd": step(40, "back1")},
              agents=[("back1", ago(40), "lowd", [(ago(20), "work"), (ago(5), "handback")])])
    rc, err = w.check()
    scenario("one that handed back 5 min ago is not, and nothing is in flight since its handback",
             rc == 2 and "lowd: nothing in flight" in err, err)
    w2 = World([E("lowd")], state={"lowd": step(40, "back2")},
               agents=[("back2", ago(40), "lowd", [(ago(5), "handback")]), ("next2", ago(2), "lowd")])
    rc2, out2 = w2.plan_cmd("step", "lowd", "--agent", "next2")
    scenario("a step agent started after the handback is accepted", rc2 == 0 and w2.check()[0] == 0, out2)
    w = World([E("lowd")], state={"lowd": step(50, "quiet1")},
              agents=[("quiet1", ago(50), "lowd", [(ago(35), "work")])])
    rc, err = w.check()
    scenario("one whose transcript was last written more than STEP_MIN (30 min) ago is not", rc == 2
             and "lowd: nothing in flight" in err, err)
    w = World([E("pp")], runs=[("FA-ccin2p3-r1", "RUNNING", "pp", epoch(90), None)])
    scenario("a live registry job is the entry in flight", w.check()[0] == 0, w.check()[1])
    w = World([E("sw")], items=[("sw", "echo '{\"units\": {\"tp3_x\": {\"expected\": 8, \"done\": 3, \"active\": 2}}}'")])
    scenario("M1: active units of the item's check (sweep trials not in runs.db) are the entry in flight",
             w.check()[0] == 0, w.check()[1])
    w = World([E("sw")], items=[("sw", "echo '{\"units\": {\"tp3_x\": {\"expected\": 8, \"done\": 8, \"active\": 0}}}'")])
    scenario("M1: a check with no active unit is not", w.check()[0] == 2, w.check()[1])
    # B1: a step is a real subagent, started after the reason arose
    w = World([E("pp")], runs=[("FA-ccin2p3-r1", "COMPLETED", "pp", epoch(90), epoch(5))], state={"pp": step(20)},
              agents=[("a1good", ago(2), "pp"), ("a0old", ago(10), "pp")])
    rc, err = w.check()
    scenario("a job ended after the last step lists a step", rc == 2 and "runs ended: FA-ccin2p3-r1" in err, err)
    rc, out = w.plan_cmd("next", "--prompts")
    scenario("plan next prints the step agent's exact prompt", rc == 1 and "Why you were started: runs ended" in out, out)
    rc, out = w.plan_cmd("step", "pp", "--agent", "made-up-name")
    scenario("B1: plan step with a name that is no subagent is refused", rc != 0 and "no subagent" in out
             and w.check()[0] == 2, out)
    rc, out = w.plan_cmd("step", "pp", "--agent", "a0old")
    scenario("B1: a subagent started before the run ended is refused", rc != 0 and "before the reason" in out, out)
    rc, out = w.plan_cmd("step", "pp", "--agent", "a1good")
    scenario("B1: a subagent started after it is recorded and clears the step", rc == 0 and w.check()[0] == 0, out)
    # a working step agent owns the runs it submitted and the runs that ended before its last record; after its
    # handback they are listed again
    r1 = [("FA-ccin2p3-r1", "COMPLETED", "pp", epoch(90), epoch(5))]
    w = World([E("pp")], runs=r1, state={"pp": step(20, "wk3")}, agents=[("wk3", ago(20), "pp", [(ago(2), "work")])])
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("next")
    scenario("a run that ended while the step agent kept working is its to handle: not listed",
             rc == 0 and "runs ended" not in err and rc2 == 0 and "runs ended" not in out2, err + out2)
    w = World([E("pp")], runs=[("FA-ccin2p3-r1", "COMPLETED", "pp", epoch(15), epoch(5))],
              state={"pp": step(20, "wk4")}, agents=[("wk4", ago(20), "pp", [(ago(8), "work")])])
    rc, err = w.check()
    scenario("a run the working step agent submitted is its to handle, even if it ended after its last record",
             rc == 0 and "runs ended" not in err, err)
    w = World([E("pp")], runs=r1 + [("FA-ccin2p3-r2", "FAILED", "pp", epoch(90), epoch(3))],
              state={"pp": step(20, "wk5")}, agents=[("wk5", ago(20), "pp", [(ago(4), "work")])])
    rc, err = w.check()
    scenario("with a working step agent and one run it does not own, every ended run is listed",
             rc == 2 and "runs ended: FA-ccin2p3-r1, FA-ccin2p3-r2" in err, err)
    # the step started for that reason moves the entry's last step past both runs: the owned one must reach it too,
    # so it is not lost when the working agent hands back
    w = World([E("pp")], runs=r1 + [("FA-ccin2p3-r2", "FAILED", "pp", epoch(90), epoch(3))],
              state={"pp": step(20, "wkA")}, agents=[("wkA", ago(20), "pp", [(ago(4), "work")]), ("wkB", ago(1), "pp")])
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("step", "pp", "--agent", "wkB")
    reason = [x.get("reason") for x in json.load(open(w.state))["pp"]["steps"] if x["agent"] == "wkB"]
    with open(os.path.join(w.ag, "agent-wkA.jsonl"), "a") as f:          # A hands back after B is recorded
        f.write(J({"type": "assistant", "timestamp": ago(0), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu2", "name": "SubagentHandback", "input": {"message": "report"}}]}}) + "\n")
        f.write(J({"type": "user", "timestamp": ago(0), "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu2", "content": "ok"}]}}) + "\n")
    rc3, err3 = w.check()
    scenario("A owns r1, r2 is not: both listed, B's recorded reason names r1, and after A hands back B is in flight",
             rc == 2 and "runs ended: FA-ccin2p3-r1, FA-ccin2p3-r2" in err and rc2 == 0
             and reason == ["runs ended: FA-ccin2p3-r1, FA-ccin2p3-r2"] and rc3 == 0, err + out2 + str(reason) + err3)
    w = World([E("pp")], runs=r1, state={"pp": step(20, "hb3")},
              agents=[("hb3", ago(20), "pp", [(ago(2), "work"), (ago(1), "handback")])])
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("next", "--prompts")
    scenario("once it hands back, the run that ended during its step is listed",
             rc == 2 and "runs ended: FA-ccin2p3-r1" in err and rc2 == 1
             and "Why you were started: runs ended: FA-ccin2p3-r1" in out2, err + out2)
    w = World([E("lowd")], state={"lowd": step(50)}, agents=[("a2early", ago(40), "lowd")])
    rc, out = w.plan_cmd("step", "lowd", "--agent", "a2early")
    scenario("B1: nothing in flight since 20 min ago: a subagent started 40 min ago is refused", rc != 0, out)
    ask = "Should the readout run on every rung now?"
    wait = dict(status="waiting_user", ask=ask, waiting_since=ago(30), asks=[{"ask": ask, "at": ago(30)}])
    w = World([E("ng", **wait)], agents=[("a3", ago(1), "ng")],
              transcript=[(ago(29), "assistant", "Joaquin, should the readout run on every rung now?"),
                          (ago(3), "user", "Yes, run it on every rung please")])
    rc, out = w.plan_cmd("step", "ng", "--agent", "a3")
    rc2, out2 = w.check()
    scenario("B1: a waiting entry takes no step; plan next lists the reply until it is answered",
             rc != 0 and rc2 == 0 and "NEXT ng: the user wrote" in out2, out + out2)
    runs = [("FA-ccin2p3-old", "RUNNING", "pp", 1, None)] + [("FA-ccin2p3-n%02d" % i, "COMPLETED", "pp", 100 + i, 200)
                                                              for i in range(40)]
    w = World([E("pp")], runs=runs, state={"pp": step(1)})
    scenario("a live run older than 40 newer ones is seen", w.check()[0] == 0, w.check()[1])
    for s in ("UNKNOWN", "HELD", "CANCELLED by 123", "OUT_OF_MEMORY"):
        w = World([E("delta")], runs=[("FA-ccin2p3-u", s, "delta", epoch(100), None)])
        scenario("a run in state %s is not live" % s, w.check()[0] == 2, w.check()[1])


def event_ownership():
    ev = lambda eid: "  E1621  10-26 01:07 FA  RUN  %s  FA-ccin2p3-x ended FAILED" % eid
    w = World([E("cov")], state={"cov": step(40, "wk1")}, agents=[("wk1", ago(40), "cov", [(ago(3), "work")])],
              events=ev("cov"))
    rc, err = w.check()
    scenario("an open event on an entry whose step agent is working does not block",
             rc == 0 and "open event" not in err, err)
    w = World([E("cov")], state={"cov": step(40, "hb1")}, agents=[("hb1", ago(40), "cov", [(ago(3), "handback")])],
              events=ev("cov"))
    rc, err = w.check()
    scenario("the event blocks once the step agent handed back", rc == 2 and "open event" in err, err)
    w = World([E("cov")], state={"cov": step(50, "q1")}, agents=[("q1", ago(50), "cov", [(ago(35), "work")])],
              events=ev("cov"))
    rc, err = w.check()
    scenario("the event blocks when the step agent is quiet 35 min", rc == 2 and "open event" in err, err)
    w = World([E("cov")], state={"cov": step(40, "wk4")}, agents=[("wk4", ago(40), "cov", [(ago(-0.2), "work")])],
              events=ev("cov"))
    rc, err = w.check()
    scenario("a step agent writing while the check runs (records later than the check's now) exempts its event",
             rc == 0 and "open event" not in err, err)
    w = World([E("cov", next={1: "score the sign test", 2: "draw the density figure"})],
              state={"cov": {"steps": [{"agent": "wk5", "at": ago(5), "next": 1}]}},
              agents=[("wk5", ago(5), "cov", [(ago(-0.2), "work")])], events=ev("cov"))
    rc, err = w.check()
    scenario("a step agent recorded with plan step --next exempts the entry's event (step 2 is still listed)",
             "open event" not in err and "cov step 2" in err and "cov step 1" not in err, err)
    w = World([E("cov")], state={"cov": step(10, "wk2")}, events=ev("other_item"))
    rc, err = w.check()
    scenario("an event whose item is not a plan entry still blocks", rc == 2 and "open event" in err, err)
    w = World([E("cov")], state={"cov": step(10, "forged9")},
              agents=[("forged9", ago(10), "other", [(ago(2), "work")])], events=ev("cov"))
    rc, err = w.check()
    scenario("a forged step does not exempt the entry's event", rc == 2 and "open event" in err, err)
    w = World([E("cov")], state={"cov": step(10, "wk3")}, events="  E1622  10-26 01:07 FA")
    rc, err = w.check()
    scenario("an event line under 6 columns blocks", rc == 2 and "open event" in err, err)


def files():
    w = World([E("pp")])
    with cf.ThreadPoolExecutor(20) as ex:
        res = list(ex.map(lambda i: w.plan_cmd("run", "pp", "FA-ccin2p3-p%02d" % i), range(20)))
    scenario("20 parallel plan run: none lost, none crashed", all(rc == 0 for rc, _ in res)
             and len(json.load(open(w.state))["pp"]["runs"]) == 20, [o for rc, o in res if rc][:2])
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLAN_")}
    out = subprocess.run([sys.executable, PLAN, "where"], env=env, capture_output=True, text=True).stdout
    common = subprocess.run(["git", "-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True).stdout.strip()
    scenario("the plan file resolves to the main checkout from a worktree",
             "plan file %s/plan/plan.yaml" % os.path.dirname(common) in out, out)
    w = World([E("pp")])
    w.plan_cmd("run", "pp", "FA-ccin2p3-a")
    scenario("runs go to the state file, not plan.yaml",
             "FA-ccin2p3-a" not in open(w.plan).read() and "FA-ccin2p3-a" in open(w.state).read())


def answers():
    ask = "Should the readout run on every rung now?"
    wait = lambda m=30: dict(status="waiting_user", ask=ask, waiting_since=ago(m), asks=[{"ask": ask, "at": ago(m)}])
    say = [(ago(29), "assistant", "Joaquin, should the readout run on every rung now?"),
           (ago(3), "user", "Yes, run it on every rung please")]
    w = World([E("x", **wait())], transcript=say)
    for kv in ("status=done", "status=waiting_user", "done_when=true", "notes_ref=the", "owner=me"):
        before = open(w.plan).read()
        rc, out = w.plan_cmd("answer", "x", "--kind", "scope", "--quote", "Yes, run it on every rung please", "--set", kv)
        scenario("B2/B3: answer --set %s is refused before anything is recorded" % kv,
                 rc != 0 and open(w.plan).read() == before and "decide" not in w.calls(), out)
    # a reply after a question never blocks the stop (a block is shown on the user's screen; the user, 2026-10-26: "Me
    # not answering a question from you it's not you not answering a question from me"); plan next carries it to the
    # main session on every wake until plan answer or plan unanswered, so an answer is never lost
    w = World([E("ng", **wait())], transcript=say)
    r = subprocess.run([sys.executable, CHECK], input="{}", env=w.env, capture_output=True, text=True)
    rc, nxt = w.plan_cmd("next")
    scenario("an answer after the question does not block the stop and prints nothing; plan next lists it",
             r.returncode == 0 and not (r.stdout + r.stderr).strip() and rc == 1
             and "NEXT ng: the user wrote" in nxt and ask in nxt, r.stderr + r.stdout + nxt)
    scenario("plan next keeps listing it at every later read until it reaches the entry",
             "NEXT ng: the user wrote" in w.plan_cmd("next")[1] and "NEXT ng: the user wrote" in w.check()[1])
    rc, out = w.plan_cmd("answer", "ng", "--kind", "scope", "--quote", "Yes, run it on every rung please", "--set",
                         "next=run it")
    e = w.entries()["ng"]
    scenario("plan answer records the words and updates the entry in one call",
             rc == 0 and e["status"] == "running" and e["next"] == "run it" and e["quotes"][0]["decision"] == "D900"
             and e["asks"][0]["answered_at"], out)
    scenario("once answered, plan next no longer lists the reply", "the user wrote" not in w.plan_cmd("next")[1])
    w = World([E("ng", **wait())], transcript=say[:1] + [(ago(3), "user", "Unrelated: how is the pp generation going")])
    r = subprocess.run([sys.executable, CHECK], input="{}", env=w.env, capture_output=True, text=True)
    scenario("a non-answer while a question is open does not block and prints nothing",
             r.returncode == 0 and not (r.stdout + r.stderr).strip(), r.stderr + r.stdout)
    w.plan_cmd("unanswered", "ng")
    rc, nxt = w.plan_cmd("next")
    scenario("plan unanswered leaves it waiting, off plan next", rc == 0 and "the user wrote" not in nxt
             and w.entries()["ng"]["status"] == "waiting_user", nxt)
    w = World([E("ng", **wait()), E("pp")], transcript=say[:1] + [(ago(3), "user", "how is the pp generation going")])
    r = subprocess.run([sys.executable, CHECK], input="{}", env=w.env, capture_output=True, text=True)
    scenario("a block for another reason does not mention the reply on the user's screen",
             r.returncode == 2 and "pp:" in r.stderr and "the user wrote" not in r.stderr + r.stdout
             and "NEXT ng: the user wrote" in w.plan_cmd("next")[1], r.stderr + r.stdout)
    w = World([E("ng", **wait())], transcript=[(ago(120), "user", "Yes, run it on every rung please")])
    rc, out = w.plan_cmd("answer", "ng", "--kind", "scope", "--quote", "Yes, run it on every rung please")
    scenario("words typed before the question do not answer it", rc != 0 and "decide" not in w.calls(), out)
    w = World([E("ng")])
    rc, out = w.plan_cmd("answer", "ng", "--kind", "scope", "--quote", "zebra quantum pineapple never typed by anyone")
    scenario("a quote the user never typed is refused before site is called", rc != 0 and w.calls() == "", out)
    # B2: a running entry needs words newer than its last ask or step
    w = World([E("run1")], state={"run1": step(60)}, transcript=[(ago(600), "user", "change the window to the lower one")])
    rc, out = w.plan_cmd("answer", "run1", "--kind", "scope", "--quote", "change the window to the lower one")
    scenario("B2: an old quote for a running entry (before its last step) is refused", rc != 0 and w.calls() == "", out)
    # B2: a release needs words typed after the hold
    held = E("syn", status="held", hold_decision="D43")
    w = World([held], decisions=[("D43", "hold", "syn", 1, ago(60))], items=[("syn", None)],
              transcript=[(ago(14400), "user", "release the synthetic direction now please")])
    rc, out = w.plan_cmd("answer", "syn", "--kind", "release", "--quote", "release the synthetic direction now please")
    scenario("B2: a release with words older than the hold is refused", rc != 0 and w.calls() == "", out)
    w = World([held], decisions=[("D43", "hold", "syn", 1, ago(60))], items=[("syn", None)],
              transcript=[(ago(2), "user", "release the synthetic direction now please")])
    rc, out = w.plan_cmd("answer", "syn", "--kind", "release", "--quote", "release the synthetic direction now please")
    scenario("a release with newer words takes it off hold", rc == 0 and w.entries()["syn"]["status"] == "running"
             and "hold_decision" not in w.entries()["syn"], out)
    # M4: approve then item failure rolls the decision back
    w = World([E("ng", **wait())], transcript=say)
    rc, out = w.plan_cmd("answer", "ng", "--kind", "approve", "--quote", "Yes, run it on every rung please",
                         env={"STUB_ITEM_FAIL": "1"})
    scenario("M4: a failed item after the decision rolls the decision back; the entry is unchanged",
             rc != 0 and w.sql("SELECT in_force FROM decisions WHERE id='D900'") == [(0,)]
             and w.entries()["ng"]["status"] == "waiting_user", out)
    w = World([E("ng", **wait())], transcript=say)
    rc, out = w.plan_cmd("answer", "ng", "--kind", "approve", "--quote", "Yes, run it on every rung please")
    scenario("M2: an approval creates the site item with the entry's globs", rc == 0 and "--match ng*" in w.calls(), out)


def adding():
    w = World([], decisions=[("D120", "approve", "the small-D study, first round", 1)])
    rc, out = w.plan_cmd("add", "lowd2", "--title", "t", "--question", "q", "--next", "n", "--match", "tp3_lowd*",
                         "--decision", "D120")
    scenario("M3: plan add with an approval whose covers do not name the entry is refused", rc != 0 and w.calls() == "", out)
    w = World([], decisions=[("D120", "approve", "lowd2", 1)])
    rc, out = w.plan_cmd("add", "lowd2", "--title", "t", "--question", "q", "--next", "n", "--match", "tp3_lowd*",
                         "--decision", "D120")
    scenario("M2: plan add creates the site item with its globs", rc == 0 and "--match tp3_lowd*" in w.calls(), out)
    w = World([E("old")], decisions=[("D7", "approve", "old", 1)])
    rc, out = w.plan_cmd("item", "old", "--decision", "D7", "--match", "tp3_old*")
    scenario("M2: plan item attaches a site item to an existing entry", rc == 0 and "item add FA old" in w.calls(), out)
    w = World([E("x")])
    rc, out = w.plan_cmd("check", "x", "--done-when", "true")
    scenario("B2: a done_when that passes in an empty directory is refused", rc != 0 and "checks nothing" in out, out)
    rc, out = w.plan_cmd("check", "x", "--done-when", "test -f analysis/x/fig.png")
    scenario("a done_when that looks at the work is accepted", rc == 0 and w.entries()["x"]["done_when"], out)
    rc, out = w.plan_cmd("ref", "x", "--label", "the")
    scenario("B2: notes_ref that is no \\label is refused", rc != 0, out)
    rc, out = w.plan_cmd("ref", "x", "--label", "par:readout")
    scenario("notes_ref that is a \\label is accepted", rc == 0 and w.entries()["x"]["notes_ref"] == "par:readout", out)


def questions():
    ask = "Should the small study also run the frozen lower blocks at every rung?"
    w = World([E("lowd", status="waiting_user", ask=ask, waiting_since=ago(20))])
    rc, err = w.check()
    scenario("waiting with no question sent blocks", rc == 2 and "no message sent" in err, err)
    w = World([E("lowd", status="waiting_user", ask=ask, waiting_since=ago(20))],
              transcript=[(ago(25), "assistant", "Joaquin, should the small study also run the frozen lower blocks at "
                                                  "every rung?")])
    scenario("a question written just before plan ask counts (15 min margin)", w.check()[0] == 0, w.check()[1])
    w = World([E("lowd", status="waiting_user", ask=ask, waiting_since=ago(20))],
              transcript=[(ago(90), "assistant", "should the small study also run the frozen lower blocks at every rung?")])
    scenario("one sent long before the ask does not", w.check()[0] == 2, w.check()[1])
    w = World([E("lowd")])
    rc, out = w.plan_cmd("ask", "lowd", "--ask", "ok?", "--searched", SEARCHED)
    scenario("an ask with no checkable words is refused", rc != 0, out)
    done = [{"ask": ask, "at": ago(200), "answered_at": ago(100), "decision": "D5"}]
    w = World([E("lowd", asks=done), E("other")])
    rc, out = w.plan_cmd("ask", "other", "--ask", "Should the small study run the frozen lower blocks at every rung?",
                         "--searched", SEARCHED)
    scenario("plan ask refuses a question already answered", rc != 0 and "already answered" in out, out)
    w = World([E("lowd", asks=done)], state={"lowd": step(1)},
              transcript=[(ago(10), "assistant", "Joaquin, should the small study also run the frozen lower blocks "
                                                 "at every rung?")])
    rc, err = w.check()
    scenario("a message asking an answered question again blocks", rc == 2 and "asked again" in err, err)


def holds():
    w = World([E("syn", status="held", hold_decision="D43")], decisions=[("D43", "hold", "syn,other", 1)])
    scenario("held by a hold in force that names it passes", w.check()[0] == 0, w.check()[1])
    for dec, why in ((("D43", "scope", "syn", 1), "a scope"), (("D43", "hold", "other", 1), "a hold of others"),
                     (("D43", "hold", "syn", 0), "a hold no longer in force")):
        w = World([E("syn", status="held", hold_decision="D43")], decisions=[dec])
        scenario("held by %s blocks" % why, w.check()[0] == 2, w.check()[1])


def done():
    repo = tempfile.mkdtemp(prefix="plan_repo_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q"); os.makedirs(os.path.join(repo, "docs")); os.makedirs(os.path.join(repo, "an"))
    tex = os.path.join(repo, "docs", "results.tex")
    open(tex, "w").write("x\n"); g("add", "."); g("commit", "-qm", "a")
    open(tex, "a").write("\\paragraph{The readout on every rung.}\\label{par:readout} done.\n")
    g("commit", "-qam", "b")
    sha = g("rev-parse", "HEAD").stdout.strip()
    ct = int(g("show", "-s", "--format=%ct", sha).stdout)
    open(os.path.join(repo, "an", "fig.png"), "w").write("p")
    good = dict(title="the readout on every rung", notes_ref="par:readout", done_when="test -f an/fig.png",
                evidence={"runs": [], "commit": sha, "paths": ["an/fig.png"]})
    cases = (("done_when fails", dict(done_when="test -f an/none.png"), [], {}, "done_when fails"),
             ("a trivially true done_when", dict(done_when="true"), [], {}, "checks nothing"),
             ("evidence missing", dict(evidence={"runs": [], "commit": sha, "paths": ["nope.png"]}), [], {}, "missing"),
             ("a run still going", {}, [("FA-ccin2p3-z", "RUNNING", "rd", 1, None)], {}, "still going"),
             ("a run stuck UNKNOWN, no override", {}, [("FA-ccin2p3-z", "UNKNOWN", "rd", 1, None)], {}, "override"),
             ("notes_ref no label", dict(notes_ref="par:none"), [], {}, "not a \\label"),
             ("a write-up that does not name it", dict(title="other", notes_ref="par:readout2"), [], {}, "names"),
             ("continuous work the user has not closed", dict(title="the readout on every rung (keep iterating)"),
              [], {}, "continuous work"),
             ("an entry marked continuous", dict(continuous={"quote": "Keep iterating.", "at": "2026-10-24"}),
              [], {}, "continuous work"),
             ("a write-up older than the last step", {}, [], {"rd": {"steps": [{"agent": "s", "at":
                 dt.datetime.fromtimestamp(ct + 60, dt.timezone.utc).isoformat()}]}}, "before the entry's last step"))
    for name, kw, rr, st, why in cases:
        base = dict(good); base.update(kw)
        w = World([E("rd", **base)], runs=rr, state=st)
        w.env["PLAN_ROOT"] = repo
        if "readout2" in str(kw):
            open(tex, "a").write("\\label{par:readout2}\n")
        rc, out = w.plan_cmd("done", "rd", "--writeup", sha)
        scenario("plan done refuses with %s" % name, rc != 0 and why in out and w.entries()["rd"]["status"] == "running",
                 out)
    w = World([E("rd", **good)], runs=[("FA-ccin2p3-z", "UNKNOWN", "rd", 1, None)])
    w.env["PLAN_ROOT"] = repo
    rc, out = w.plan_cmd("done", "rd", "--writeup", sha, "--override-runs", "sacct lost the job; its log shows it ended")
    scenario("plan done with an UNKNOWN run passes only with a stated override", rc == 0
             and w.entries()["rd"]["runs_override"], out)
    w = World([E("rd", **good)], runs=[("FA-ccin2p3-z", "COMPLETED", "rd", 1, 2)])
    w.env["PLAN_ROOT"] = repo
    rc, out = w.plan_cmd("done", "rd", "--writeup", sha)
    scenario("plan done passes when everything holds", rc == 0 and w.entries()["rd"]["status"] == "done", out)
    w = World([E("x")], state={"x": step(1)})
    w.env["PLAN_EVENTS_CMD"] = "false"
    rc, err = w.check()
    scenario("a broken event query blocks (fails closed)", rc == 2 and "plan check failed" in err, err)


def migration():
    out = os.path.join(tempfile.mkdtemp(), "plan.yaml")
    r = subprocess.run([sys.executable, MIGRATE, "--out", out, "--root", ROOT], capture_output=True, text=True)
    ents = yaml.safe_load(open(out))["entries"] if os.path.exists(out) else []
    bad = [e["id"] for e in ents if e["status"] in ("running", "waiting_user") and not e.get("next")]
    reg = dict(sqlite3.connect(os.path.expanduser("~/.local/share/ccorch/runs.db")).execute(
        "SELECT id, match FROM items WHERE project='FA'").fetchall())
    nomatch = [e["id"] for e in ents if e["id"] in reg and (e.get("match") or "") != (reg[e["id"]] or "")]
    scenario("migration: every open entry has a next step; each carries its site item's globs",
             r.returncode == 0 and ents and not bad and not nomatch, r.stdout + r.stderr + str(bad) + str(nomatch))


def repo_with_script():
    repo = tempfile.mkdtemp(prefix="claims_")
    g = lambda *a, **k: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV, **k)
    g("init", "-q")
    for d in ("docs", "analysis", "tools", ".githooks"):
        os.makedirs(os.path.join(repo, d))
    open(os.path.join(repo, "analysis", "s.py"), "w").write(
        'import json\nprint("working")\nprint(json.dumps({"h3": {"shape": 0.7951, "n": 237}, "mse": 5.83e-8}))\n')
    shutil.copy(VERIFY, os.path.join(repo, "tools", "verify_claims.py"))
    for h in ("pre-commit", "pre-merge-commit"):
        shutil.copy(os.path.join(ROOT, ".githooks", h), os.path.join(repo, ".githooks", h))
    tex = os.path.join(repo, "docs", "results.tex")
    open(tex, "w").write("Old paragraph with 1.23 and no source.\n")
    g("add", "."); g("commit", "-qm", "base", "--no-verify")
    return repo, g, tex, g("rev-parse", "HEAD").stdout.strip()


def claims():
    repo, g, tex, base = repo_with_script()

    def stage(par):
        open(tex, "w").write("Old paragraph with 1.23 and no source.\n\n" + par + "\n")
        g("add", ".")
        r = subprocess.run([sys.executable, VERIFY, "--root", repo, "--staged"], capture_output=True, text=True)
        return r.returncode, r.stdout
    src = "%% src: python3 analysis/s.py @%s\n" % base
    ok = (src + "% claim: h3.shape=0.80\n% claim: h3.n=237\n% claim: mse=5.8e-8\n"
          "On the $237$ tops the shape floor is $0.80$ and the error $5.8\\times10^{-8}$ ($2\\to4$, 2026-10-25).")
    rc, out = stage(ok)
    scenario("verify: claims the script reproduces pass; the untouched paragraph is not checked", rc == 0, out)
    rc, out = stage(ok.replace("h3.shape=0.80", "h3.shape=0.70").replace("$0.80$", "$0.70$"))
    scenario("verify: a number the script does not reproduce is refused", rc == 1 and "recomputed" in out, out)
    rc, out = stage(ok.replace("tops", "tops (40 probes)"))
    scenario("B5: one claim no longer exempts other numbers: an unclaimed integer is refused", rc == 1
             and "40 is no claim" in out, out)
    rc, out = stage(ok.replace("tops", "tops (40 probes)").replace("% claim: mse", "% allow: 40 because it is the number of probes, a design count\n% claim: mse"))
    scenario("B5: an allowed token is skipped", rc == 0, out)
    rc, out = stage("%% src: echo '{\"h3\": {\"shape\": 0.80}}' @%s\n%% claim: h3.shape=0.80\nThe floor is $0.80$." % base)
    scenario("B5: a src that is any shell command is refused", rc == 1 and "python3" in out, out)
    open(os.path.join(repo, "analysis", "new.py"), "w").write(open(os.path.join(repo, "analysis", "s.py")).read())
    rc, out = stage("%% src: python3 analysis/new.py @%s\n%% claim: h3.shape=0.80\nThe floor is $0.80$." % base)
    scenario("B5: a src script not present at @commit is refused", rc == 1 and "does not exist at" in out, out)
    os.remove(os.path.join(repo, "analysis", "new.py")); g("rm", "-q", "--cached", "analysis/new.py")
    rc, out = stage("% src: python3 analysis/s.py\n% claim: h3.shape=0.80\nThe floor is $0.80$.")
    scenario("B5: a src with no @commit is refused", rc == 1 and "no @<commit>" in out, out)
    rc, out = stage("\\begin{itemize}\n  \\item " + src + "% claim: h3.shape=0.80\n  The floor is $0.80$.\n"
                    "  \\item The other number is $0.31$.\n\\end{itemize}")
    scenario("B5: itemize items are checked one by one (a claim in one item covers no other)", rc == 1
             and "0.31 is no claim" in out, out)
    # B4: the git hooks read the real index, whatever the way of committing
    repo, g, tex, base = repo_with_script()
    g("config", "core.hooksPath", ".githooks")
    open(tex, "w").write("Old paragraph with 1.23 and no source.\n\nThe floor is $0.80$ against $0.5$.\n")
    r = subprocess.run("git add docs/results.tex && git commit -qm x", shell=True, cwd=repo, capture_output=True,
                       text=True, env=GIT_ENV)
    scenario("B4: git add && git commit of unsourced numbers is refused", r.returncode != 0 and "CLAIM" in r.stdout + r.stderr,
             r.stdout + r.stderr)
    g("reset", "-q")
    r = g("commit", "--all", "-qm", "x")
    scenario("B4: git commit --all is refused", r.returncode != 0, r.stdout + r.stderr)
    wt = tempfile.mkdtemp(prefix="wt_") + "/w"
    g("worktree", "add", "-q", wt)
    open(os.path.join(wt, "docs", "results.tex"), "a").write("\nAnother number $0.42$.\n")
    r = subprocess.run(["git", "-C", wt, "commit", "-qam", "w"], capture_output=True, text=True, env=GIT_ENV)
    scenario("B4: a commit in a worktree is refused", r.returncode != 0 and "CLAIM" in r.stdout + r.stderr,
             r.stdout + r.stderr)
    g("checkout", "-q", "--", "docs/results.tex")
    g("checkout", "-qb", "side")
    open(tex, "a").write("\nA merged number $0.55$.\n")
    g("commit", "-qam", "side", "--no-verify")
    g("checkout", "-q", "-")
    open(os.path.join(repo, "other.txt"), "w").write("o"); g("add", "other.txt"); g("commit", "-qm", "o")
    r = g("merge", "--no-ff", "-q", "-m", "m", "side")
    scenario("B4: a merge bringing unsourced numbers is refused", r.returncode != 0 and "CLAIM" in r.stdout + r.stderr,
             r.stdout + r.stderr)
    g("merge", "--abort")
    open(os.path.join(repo, "other.txt"), "w").write("o2")
    r = g("commit", "-qam", "unrelated")
    scenario("B4: a commit that does not touch docs/results.tex passes", r.returncode == 0, r.stdout + r.stderr)


GUARD_FILES = os.path.join(ROOT, ".claude", "hooks", "plan_files_guard.py")
GUARD_GIT = os.path.join(ROOT, ".claude", "hooks", "git_verify_guard.py")
CTX = os.path.join(ROOT, ".claude", "hooks", "context_guard.py")
MAIN_GUARD = os.path.join(ROOT, ".claude", "hooks", "main_session_guard.py")
SWITCH = os.path.join(ROOT, "tools", "switch_settings.py")


def hook(path, inp, env=None, cwd=None):
    r = subprocess.run([sys.executable, path], input=json.dumps(inp), capture_output=True, text=True,
                       env=env or os.environ, cwd=cwd)
    return r.returncode, r.stderr + r.stdout


def direct_writes():
    """Third review 1: the plan files change only through tools/plan.py; a step or quote forged on disk does not count."""
    for tool, ti, want in (
            ("Edit", {"file_path": "/x/FA/plan/plan.yaml"}, 2), ("Write", {"file_path": "/x/FA/plan/.state.json"}, 2),
            ("Edit", {"file_path": "/x/FA/plan/step_prompt.txt"}, 0),
            ("Bash", {"command": "echo '{}' > plan/.state.json"}, 2),
            ("Bash", {"command": "sed -i s/held/running/ plan/plan.yaml"}, 2),
            ("Bash", {"command": "cp /tmp/x.json plan/.state.json"}, 2),
            ("Bash", {"command": "python3 -c \"open('plan/plan.yaml','w').write('')\""}, 2),
            ("Bash", {"command": "cd plan && tee plan.yaml < /tmp/x"}, 2),
            ("Bash", {"command": "sh -c 'echo x >> plan/plan.yaml'"}, 2),
            ("Bash", {"command": "git checkout -- plan/plan.yaml"}, 2),
            ("Bash", {"command": "cat plan/plan.yaml | grep lowd"}, 0),
            ("Bash", {"command": "python3 tools/plan.py show && git diff plan/plan.yaml"}, 0)):
        rc, out = hook(GUARD_FILES, {"tool_name": tool, "tool_input": ti})
        scenario("1: %s `%s` is %s" % (tool, list(ti.values())[0][:46], "refused" if want else "allowed"), rc == want, out)
    w = World([E("lowd")], state={"lowd": step(5, "forged1")})
    rc, err = w.check()
    scenario("1: a step written into the state file with no subagent does not count, and blocks",
             rc == 2 and "not valid" in err and "nothing in flight" in err, err)
    w = World([E("lowd")], state={"lowd": step(5, "a5")}, agents=[("a5", ago(40), "lowd")])
    rc, err = w.check()
    scenario("1: a step whose time is not its subagent's start does not count", rc == 2 and "but subagent a5 started"
             in err, err)
    w = World([E("ng", quotes=[{"said_at": ago(10), "text": "yes run every rung", "decision": "D5"}])],
              decisions=[("D5", "scope", "ng", 1, ago(10), "only the first rung")], state={"ng": step(1)})
    rc, err = w.check()
    scenario("1: a quote that is not its decision's recorded words blocks", rc == 2 and "not the user's recorded" in err,
             err)


def one_agent_per_step():
    """Third review 2: the subagent was started with this entry's step prompt, and counts once."""
    runs = [("FA-ccin2p3-r1", "COMPLETED", "pp", epoch(90), epoch(5)),
            ("FA-ccin2p3-r2", "COMPLETED", "lowd", epoch(90), epoch(5))]
    w = World([E("pp"), E("lowd")], runs=runs, state={"pp": step(20), "lowd": step(20)},
              agents=[("b1", ago(2), "pp")])
    rc, out = w.plan_cmd("step", "lowd", "--agent", "b1")
    scenario("2: a subagent started with another entry's step prompt is refused", rc != 0 and "step prompt" in out, out)
    rc, out = w.plan_cmd("step", "pp", "--agent", "b1")
    rc2, out2 = w.plan_cmd("step", "pp", "--agent", "b1")
    scenario("2: one subagent is one step: recording it again is refused", rc == 0 and rc2 != 0
             and "already recorded" in out2, out + out2)
    w = World([E("pp"), E("lowd")], state={"pp": step(5, "dup"), "lowd": step(5, "dup")})
    rc, err = w.check()
    scenario("2: one subagent recorded on two entries in the state file blocks", rc == 2 and "recorded as 2 steps" in err,
             err)


def git_bypass():
    """Third review 3: the claims hooks cannot be skipped from a command."""
    refused = ("git commit -n -m x", "git commit -anm x", "git commit --no-verify -m x", "git commit --no-ver -m x",
               "git -c core.hooksPath=/dev/null commit -m x", "git -c core.hookspath=/dev/null commit -m x",
               "git config core.hooksPath /dev/null", "git config --unset core.hooksPath",
               "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null git commit -m x",
               "git merge --no-verify side", "git pull", "bash -c 'git commit -n -m x'",
               "eval git commit --no-verify -m x", "cd x && git commit -n -m y", "echo x >> .git/hooks/pre-commit",
               "git config --global core.hooksPath /dev/null", "git -C x config core.hooksPath /dev/null",
               "cd x && git config core.hooksPath /dev/null", "echo core.hooksPath > f",
               "cd x\ngit commit -n -m y", "if true; then git commit -n -m y; fi",
               "cat <<'EOF' > f\nit's\nEOF\ngit commit -n -m y")
    allowed = ("git commit -m 'fix the -n flag'", "git commit -m -n", "git commit -am x", "git log -n 5",
               "git config --get core.hooksPath", "git merge --abort", "git status",
               # a read of core.hooksPath anywhere in a command; commands that only name git, even unparseable ones
               "git config core.hooksPath", "cd /x && git config core.hooksPath",
               "git config --show-origin core.hooksPath", "git -C /x config --get core.hooksPath",
               "python3 -c \"print('git')\" && echo it's", "sed -i \"s/it's/x/\" tools/verify_claims.py && echo git's",
               "grep -n git tools/verify_claims.py", "cat <<'EOF' > f\nit's git\nEOF")
    for c in refused + allowed:
        rc, out = hook(GUARD_GIT, {"tool_name": "Bash", "tool_input": {"command": c}, "cwd": ROOT})
        scenario("3: `%s` is %s" % (c[:60], "refused" if c in refused else "allowed"), rc == (2 if c in refused else 0),
                 out)
    rc, out = hook(GUARD_GIT, {"tool_name": "Edit", "tool_input": {"file_path": "/x/FA/.git/config"}})
    scenario("3: Edit of .git/config is refused", rc == 2, out)
    repo, g, tex, base = repo_with_script()
    g("checkout", "-qb", "side")
    open(tex, "a").write("\nA number committed before the hooks, $0.55$.\n")
    g("commit", "-qam", "side", "--no-verify")
    g("checkout", "-q", "-")
    for c in ("git merge side", "git merge --ff-only side"):
        rc, out = hook(GUARD_GIT, {"tool_name": "Bash", "tool_input": {"command": c}, "cwd": repo})
        scenario("3: a fast-forward `%s` of an unchecked number is refused" % c, rc == 2 and "CLAIM" in out, out)
    g("checkout", "-qb", "clean", base)
    open(os.path.join(repo, "x.txt"), "w").write("x"); g("add", "x.txt"); g("commit", "-qm", "x")
    g("checkout", "-q", "-")
    rc, out = hook(GUARD_GIT, {"tool_name": "Bash", "tool_input": {"command": "git merge clean"}, "cwd": repo})
    scenario("3: merging a branch that leaves docs/results.tex alone is allowed", rc == 0, out)


def parsing():
    """Third review 4: numbers in every LaTeX form are found; a plain allowed number needs a reason."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import verify_claims as v
    got = [t for t, _ in v.numbers(r"\textbf{0.81} ${0.31}$ \num{5.8e-8} 0.4 & 0.6 \\ 2.35x 40k", [])]
    scenario("4: braced, table, e-notation and suffixed numbers are all found",
             all(x in got for x in ("0.81", "0.31", "5.8e-8", "0.4", "0.6", "2.35x", "40k")), got)
    vals = [x for _, x in v.numbers("0.3--0.5 and 0.3-0.5", [])]
    scenario("4: -- and - between numbers are ranges, not minus signs", vals == [0.3, 0.5, 0.3, 0.5], vals)
    got = [t for t, _ in v.numbers(r"tp3\_x on H100, $x_{12}$, catalog\_v3, $2\to4$, \ref{fig:a1}, 2026-10-25", [])]
    scenario("4: identifiers, subscripts, multiplicities, refs and dates are not numbers", got == [], got)
    repo, g, tex, base = repo_with_script()

    def stage(par, head="Old paragraph with 1.23 and no source.\n\n"):
        open(tex, "w").write(head + par + "\n")
        g("add", ".")
        r = subprocess.run([sys.executable, VERIFY, "--root", repo, "--staged"], capture_output=True, text=True)
        return r.returncode, r.stdout
    src = "%% src: python3 analysis/s.py @%s\n%% claim: h3.shape=0.80\n" % base
    for extra in (r"\textbf{0.31}", "${0.31}$", r"0.31 & 1 \\", "2.35x", "40k", "5.8e-8"):
        rc, out = stage(src + "The floor is $0.80$, against %s." % extra)
        scenario("4: an unclaimed %s is refused" % extra, rc == 1 and "is no claim" in out, out)
    rc, out = stage(src + "% allow: 40\nThe floor is $0.80$ over 40 probes.")
    scenario("4: % allow of a plain number without a reason is refused", rc == 1 and "need a reason" in out, out)
    rc, out = stage(src + "% allow: 40 because it is the number of probes, a design count\n"
                    "The floor is $0.80$ over 40 probes.")
    scenario("4: % allow of a plain number with its reason passes", rc == 0, out)
    rc, out = stage("% plan:begin (generated by tools/plan.py view; edit plan/plan.yaml)\n\\begin{itemize}\n"
                    "  \\item \\textbf{x} (running). It ran 40 cells at 0.31.\n\\end{itemize}\n% plan:end")
    scenario("lower: the generated plan block is not checked", rc == 0, out)


def checks_and_sites():
    """Third review 5 and 8: a finishing check must still fail when set; an unreachable site never blocks."""
    w = World([E("x")])
    os.makedirs(os.path.join(w.d, "an"))
    open(os.path.join(w.d, "an", "fig.png"), "w").write("p")
    rc, out = w.plan_cmd("check", "x", "--done-when", "test -f an/fig.png")
    scenario("5: a done_when that already passes in the checkout is refused", rc != 0 and "already passes" in out, out)
    w = World([E("sw")], items=[("sw", "echo 'ssh: connect to host cca: Connection refused' >&2; exit 255")])
    rc, out = w.check()
    scenario("8: an item check that cannot reach its site counts as in flight, with a note", rc == 0
             and "could not run" in out and "Connection refused" in out, out)
    rc, out = w.check()
    scenario("8: the failure is cached, and the note stays", rc == 0 and "could not run" in out, out)
    w = World([E("sw")], items=[("sw", "sleep 30")])
    w.env["PLAN_CHECK_TIMEOUT"] = "2"
    t0 = dt.datetime.now()
    rc, out = w.check()
    took = (dt.datetime.now() - t0).total_seconds()
    scenario("8: an item check that hangs is cut at its timeout, counted in flight", rc == 0 and "no answer in 2 s" in out
             and took < 20, "%s (%.0f s)" % (out, took))


def no_plan_yet():
    """A checkout with no plan (the main one before the switch-over): reads create nothing there (an untracked plan/
    made `site sync` refuse four launches, 2026-10-25), writes are refused."""
    root = tempfile.mkdtemp(prefix="noplan_")
    w = World([E("x")])
    env = dict(w.env, PLAN_ROOT=root)
    for k in ("PLAN_FILE", "PLAN_STATE"):
        env.pop(k)
    outs = [subprocess.run([sys.executable, PLAN] + a, env=env, capture_output=True, text=True)
            for a in (["show"], ["next"], ["show", "--compact"])]
    outs.append(subprocess.run([sys.executable, CHECK], input="{}", env=env, capture_output=True, text=True))
    w2 = subprocess.run([sys.executable, PLAN, "run", "x", "FA-ccin2p3-a"], env=env, capture_output=True, text=True)
    scenario("reads with no plan create nothing; a write is refused", os.listdir(root) == []
             and all(o.returncode == 0 for o in outs) and w2.returncode != 0 and "no plan at" in w2.stderr,
             [os.listdir(root)] + [(o.returncode, o.stderr[-200:]) for o in outs + [w2]])


def lower():
    w = World([E("synthetic", status="held", hold_decision="D43")], decisions=[("D43", "hold", "syn*", 1)])
    scenario("lower: a hold whose covers glob matches the entry passes (the rule of plan.py)", w.check()[0] == 0,
             w.check()[1])
    w = World([E("lowd", next="the DyHPO round")])
    for src in ("startup", "resume", "clear"):
        rc, out = hook(CTX, {"hook_event_name": "SessionStart", "source": src}, env=dict(w.env, CLAUDE_PROJECT_DIR=ROOT))
        ctx = json.loads(out or "{}").get("hookSpecificOutput", {}).get("additionalContext", "") if rc == 0 else ""
        scenario("lower: SessionStart (%s) prints the plan" % src, "THE PLAN" in ctx and "the DyHPO round" in ctx
                 and "just compacted" not in ctx, out)
    live = os.path.join(ROOT, ".claude", "settings.json")
    cp = os.path.join(tempfile.mkdtemp(), "settings.json")
    shutil.copy(live, cp)
    r = subprocess.run([sys.executable, SWITCH, "--settings", cp, "--apply"], capture_output=True, text=True)
    new = json.load(open(cp))
    cmds = [(ev, g.get("matcher", ""), h["command"], h.get("timeout")) for ev, gs in new["hooks"].items() for g in gs
            for h in g["hooks"]]
    old = [h["command"] for gs in json.load(open(live))["hooks"].values() for g in gs for h in g["hooks"]]
    retired = ("plan_guard.py", "continuous_guard.py", "continuous_list_guard.py", "slurm_waiter_guard.sh")
    kept = [c for c in old if not any(x in c for x in retired)]
    ok = (r.returncode == 0 and not any(x in c for _, _, c, _ in cmds for x in retired)
          and ("Stop", "", 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/plan_check.py"', 300) in cmds
          and any("plan_files_guard" in c and m == "Bash" for _, m, c, _ in cmds)
          and any("plan_files_guard" in c and "Edit" in m for _, m, c, _ in cmds)
          and any("git_verify_guard" in c and m == "Bash" for _, m, c, _ in cmds)
          and any(ev == "SessionStart" and m == "" and "context_guard" in c for ev, m, c, _ in cmds)
          and all(any(c == k for _, _, c, _ in cmds) for k in kept))
    scenario("lower: the switch-over's settings retire the four guards, add the new ones (plan check 300 s) and keep "
             "every other hook", ok, r.stdout + r.stderr + json.dumps(cmds)[:600])
    # always from a temporary worktree: run from the main checkout on trunk the script would really merge
    tmp_wt = os.path.join(tempfile.mkdtemp(prefix="switch_wt_"), "wt")
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", tmp_wt, "HEAD"], capture_output=True)
    # the worktree's own HEAD before the run: trunk's may move meanwhile (plan commits of a live session)
    head = subprocess.run(["git", "-C", tmp_wt, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    try:
        r = subprocess.run(["sh", os.path.join(tmp_wt, "tools", "switch_over.sh"), "some-branch"], cwd=tmp_wt,
                           capture_output=True, text=True)
        head_after = subprocess.run(["git", "-C", tmp_wt, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    finally:
        subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", tmp_wt], capture_output=True)
    scenario("lower: the switch-over refuses to run from a worktree (run in a temporary one, nothing merged)",
             r.returncode != 0 and "main checkout" in r.stdout and head_after == head, r.stdout + r.stderr)


def worktree_then_merge():
    """Fourth review 1: a branch that adds a script and its paragraph is checked as it stands, in range mode too."""
    repo, g, tex, base = repo_with_script()
    g("branch", "-m", "trunk")
    g("config", "core.hooksPath", ".githooks")
    wt = os.path.join(tempfile.mkdtemp(prefix="wt_"), "side")
    g("worktree", "add", "-q", "-b", "side", wt)
    gw = lambda *a: subprocess.run(["git", "-C", wt] + list(a), capture_output=True, text=True, env=GIT_ENV)
    open(os.path.join(wt, "analysis", "new.py"), "w").write('import json\nprint(json.dumps({"floor": 0.4321}))\n')
    gw("add", "analysis/new.py"); r0 = gw("commit", "-qm", "the script")
    sha = gw("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(wt, "docs", "results.tex"), "a").write(
        "\n%% src: python3 analysis/new.py @%s\n%% claim: floor=0.43\nThe new floor is $0.43$.\n" % sha)
    r1 = gw("commit", "-qam", "its paragraph")
    rng = "%s..side" % base
    v = lambda: subprocess.run([sys.executable, VERIFY, "--root", repo, "--rev", rng], capture_output=True, text=True)
    r2 = v()
    scenario("1: a paragraph and its new script, committed in a worktree, pass its hook and the range check from the "
             "main checkout", r0.returncode == 0 and r1.returncode == 0 and r2.returncode == 0,
             r0.stderr + r1.stdout + r1.stderr + r2.stdout + r2.stderr)
    g("worktree", "remove", "--force", wt)
    r3 = v()
    left = g("worktree", "list").stdout.strip().splitlines()
    scenario("1: with the branch's worktree gone, the range check runs the script in a temporary worktree at the "
             "branch, and removes it", r3.returncode == 0 and len(left) == 1, r3.stdout + r3.stderr + str(left))
    r4 = g("merge", "--no-ff", "-q", "-m", "merge side", "side")
    scenario("1: merging it into trunk passes the merge hook", r4.returncode == 0, r4.stdout + r4.stderr)


def plan_repo():
    """A checkout with a committed plan, as the main one is after the switch-over."""
    repo, g, tex, base = repo_with_script()
    g("branch", "-m", "trunk")
    os.makedirs(os.path.join(repo, "plan"))
    open(os.path.join(repo, ".gitignore"), "w").write("plan/.state.json\nplan/plan.yaml.lock\nplan/.plan_tmp_*\n"
                                                      "plan/.check_cache.json\n")
    yaml.safe_dump({"entries": [E("x")]}, open(os.path.join(repo, "plan", "plan.yaml"), "w"))
    json.dump({"_claims_verified": base}, open(os.path.join(repo, "plan", ".state.json"), "w"))
    g("add", "."); g("commit", "-qm", "plan", "--no-verify")
    w = World([E("x")])
    env = dict(w.env, PLAN_ROOT=repo, **{k: GIT_ENV[k] for k in GIT_ENV if k.startswith("GIT_")})
    for k in ("PLAN_FILE", "PLAN_STATE"):
        env.pop(k)
    return repo, g, tex, base, env


def plan_commits():
    """Fourth review 2: a plan write leaves the checkout clean, so the next launch's `site sync` goes through."""
    repo, g, tex, base, env = plan_repo()
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "the DyHPO round", *WHY], env=env, capture_output=True,
                       text=True)
    st = g("status", "--porcelain").stdout
    msg = g("log", "-1", "--format=%s").stdout.strip()
    scenario("2: a plan write commits plan/plan.yaml alone; the tree is clean for site sync", r.returncode == 0
             and st == "" and msg == "plan: set-next x", r.stdout + r.stderr + st + msg)
    open(os.path.join(repo, "other.txt"), "w").write("w"); g("add", "other.txt")
    subprocess.run([sys.executable, PLAN, "set-next", "x", "again", *WHY], env=env, capture_output=True, text=True)
    files = g("show", "--name-only", "--format=", "HEAD").stdout.split()
    scenario("2: the plan commit takes plan/plan.yaml only, never other staged work", files == ["plan/plan.yaml"]
             and "A  other.txt" in g("status", "--porcelain").stdout, files)


def test_switch():
    """Fourth review 3: test variables are honoured only under the test switch; quotes and approvals are checked
    against the registry; done entries are rechecked."""
    w = World([E("x")])
    env = dict(w.env)
    env.pop("PLAN_TEST")
    out = subprocess.run([sys.executable, PLAN, "where"], env=env, capture_output=True, text=True).stdout
    scenario("3: PLAN_FILE without PLAN_TEST=1 is ignored", w.plan not in out and "plan file" in out, out)
    r = subprocess.run([sys.executable, PLAN, "where"], env=dict(w.env, PLAN_ROOT=ROOT), capture_output=True, text=True)
    scenario("3: PLAN_TEST=1 with a root outside the temp directory is refused", r.returncode != 0
             and "needs PLAN_ROOT" in r.stderr, r.stdout + r.stderr)
    rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": "PLAN_FILE=/tmp/p.yaml python3 "
                                                                                 "tools/plan.py show"}})
    scenario("3: a PLAN_* variable in a command is refused", rc == 2, out)
    q = lambda **k: dict({"said_at": ago(10), "text": "run every rung please", "decision": "D5"}, **k)
    for name, quote, dec in (
            ("a decision that does not cover the entry", q(), ("D5", "scope", "other", 1, ago(10), "run every rung please")),
            ("a said_at that is not the decision's", q(said_at=ago(300)),
             ("D5", "scope", "ng", 1, ago(10), "run every rung please")),
            ("words said before what they answer", q(answers_after=ago(5)),
             ("D5", "scope", "ng", 1, ago(10), "run every rung please"))):
        w = World([E("ng", quotes=[quote])], decisions=[dec], state={"ng": step(1)})
        rc, err = w.check()
        scenario("3: a quote with %s blocks" % name, rc == 2 and "quote on it" in err, err)
    w = World([E("ng", quotes=[q()])], decisions=[("D5", "scope", "ng", 1, ago(10), "run every rung please")],
              state={"ng": step(1)})
    scenario("3: a quote that is its decision's words, about the entry, at its time, passes", w.check()[0] == 0,
             w.check()[1])
    w = World([E("ng", decisions=["D7"])], decisions=[("D7", "prefer", "ng", 1)], state={"ng": step(1)})
    rc, err = w.check()
    scenario("3: a running entry with no approval in force naming it blocks", rc == 2 and "no approval" in err, err)
    w = World([E("rd", status="done", writeup="abc", done_when="test -f an/none.png", notes_ref="par:readout",
                 evidence={"runs": [], "commit": None, "paths": []})])
    rc, err = w.check()
    rn, out = w.plan_cmd("next")
    scenario("3 (fifth review): a done entry that regressed is a note and a reopen suggestion, never a block",
             rc == 0 and "marked done but" in err and rn == 0 and "SUGGEST rd" in out and "plan reopen rd" in out,
             err + out)
    w = World([E("rd", status="done", writeup="abc", done_when="test -f an/none.png", notes_ref="par:readout",
                 evidence={"runs": [], "commit": None, "paths": []}), E("pp")])
    r = subprocess.run([sys.executable, CHECK], input="{}", env=w.env, capture_output=True, text=True)
    rn, out = w.plan_cmd("next")
    scenario("a check that blocks for another reason keeps its notes off the user's screen, for plan next",
             r.returncode == 2 and "pp:" in r.stderr and "note" not in r.stderr + r.stdout
             and "marked done but" not in r.stderr + r.stdout and "note: rd is marked done but" in out,
             r.stderr + r.stdout + out)


def claims_state():
    """Fourth review 4: the numbers on trunk are checked as state, however a commit got there."""
    repo, g, tex, base, env = plan_repo()
    open(tex, "a").write("\nAn unsourced number, $0.66$.\n")
    g("commit", "-qam", "bad", "--no-verify")
    rc = subprocess.run([sys.executable, CHECK], input="{}", env=env, capture_output=True, text=True)
    scenario("4: a number committed to trunk with the hooks skipped blocks the stop", rc.returncode == 2
             and "fail the claims check" in rc.stderr and "0.66" in rc.stderr, rc.stderr + rc.stdout)
    shutil.copy(PLAN, os.path.join(repo, "tools", "plan.py"))
    origin = tempfile.mkdtemp(prefix="origin_")
    subprocess.run(["git", "init", "-q", "--bare", origin], capture_output=True)
    g("remote", "add", "origin", origin)
    g("push", "-q", "-u", "origin", "%s:refs/heads/trunk" % base)
    g("branch", "-q", "--set-upstream-to=origin/trunk", "trunk")
    remote = lambda: subprocess.run(["git", "-C", origin, "rev-parse", "trunk"], capture_output=True,
                                    text=True).stdout.strip()
    ap = subprocess.run(["bash", os.path.join(ROOT, ".claude", "hooks", "auto_push.sh")], capture_output=True,
                        text=True, env=dict(env, CLAUDE_PROJECT_DIR=repo))
    scenario("4 (sixth review): auto_push runs no check, pushes trunk only up to the last verified commit",
             ap.returncode == 0 and remote() == base and "wait for the claims check" in ap.stdout
             and not os.path.exists(os.path.join(repo, "plan", ".claims_cache.json")), ap.stdout + ap.stderr)
    open(tex, "w").write(open(tex).read().replace("\nAn unsourced number, $0.66$.\n", "\n"))
    g("commit", "-qam", "fix", "--no-verify")
    r = subprocess.run([sys.executable, PLAN, "claims"], env=env, capture_output=True, text=True)
    head = g("rev-parse", "HEAD").stdout.strip()
    mark = json.load(open(os.path.join(repo, "plan", ".state.json"))).get("_claims_verified")
    scenario("4: once a new commit fixes it, the check passes and the watermark moves to trunk's head",
             r.returncode == 0 and mark == head, r.stdout + r.stderr + str(mark))
    ap = subprocess.run(["bash", os.path.join(ROOT, ".claude", "hooks", "auto_push.sh")], capture_output=True,
                        text=True, env=dict(env, CLAUDE_PROJECT_DIR=repo))
    scenario("4: then auto_push pushes trunk's head", ap.returncode == 0 and remote() == head, ap.stdout + ap.stderr)


def fourth_lower():
    rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": "git show HEAD:plan/plan.yaml > /tmp/y"}})
    scenario("lower: reading a plan file into a temp file is allowed (the redirect is not into the plan)", rc == 0, out)
    rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": "cat /tmp/y > plan/plan.yaml"}})
    scenario("lower: a redirect into a plan file is still refused", rc == 2, out)
    rc_src = open(os.path.join(ROOT, ".claude", "hooks", "reply_check.py")).read()
    scenario("lower: reply_check keeps 'do not ask what a decision in force settles', without the self-explanation rule",
             "must not ask the researcher to decide something the decisions above already settle" in rc_src
             and "self-explanation" not in rc_src)
    import importlib.util, tempfile
    spec = importlib.util.spec_from_file_location("reply_check", os.path.join(ROOT, ".claude", "hooks", "reply_check.py"))
    rcm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcm)
    tr = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for rec in ({"type": "user", "message": {"content": "keep it as it is"}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "Joaquin, kept."}]}},
                {"type": "user", "isCompactSummary": True,
                 "message": {"content": "This session is being continued ... 6. All user messages: why would you?"}},
                {"type": "user", "message": {"content": "<system-reminder>an agent report</system-reminder>"}}):
        tr.write(json.dumps(rec) + "\n")
    tr.close()
    msgs, _ = rcm.scan(tr.name)
    scenario("lower: reply_check never takes a compaction summary for a new user message (it quotes old questions)",
             msgs == ["keep it as it is"], str(msgs))
    so = open(os.path.join(ROOT, "tools", "switch_over.sh")).read()
    scenario("lower: the switch-over stops on a failed migration commit and deletes the retired hook files",
             "|| true" not in so and "git rm -q .claude/hooks/plan_guard.py" in so)


def hold_release():
    """Fifth review 1: a hold then a release (the real site edits the hold's covers) leaves nothing standing."""
    w = World([E("syn")], items=[("syn", None)], state={"syn": step(40)},
              transcript=[(ago(20), "user", "hold the synthetic direction for now please"),
                          (ago(2), "user", "release the synthetic direction now please")])
    r1 = w.plan_cmd("answer", "syn", "--kind", "hold", "--quote", "hold the synthetic direction for now please")
    r2 = w.plan_cmd("answer", "syn", "--kind", "release", "--quote", "release the synthetic direction now please")
    held = w.sql("SELECT covers, in_force FROM decisions WHERE kind='hold'")
    rc, err = w.check()
    scenario("1: hold then release: the registry's hold no longer covers it, the entry runs, nothing blocks",
             r1[0] == 0 and r2[0] == 0 and held == [("", 0)] and w.entries()["syn"]["status"] == "running"
             and "quote" not in err, r1[1] + r2[1] + str(held) + err)


def reask_once():
    """Fifth review 2: an answered question asked again is reported at one stop; a quoted question is no re-ask."""
    ask = "Should the small study also run the frozen lower blocks at every rung?"
    done_ask = [{"ask": ask, "at": ago(200), "answered_at": ago(100), "decision": "D5"}]
    w = World([E("lowd", asks=done_ask)], state={"lowd": step(1)},
              transcript=[(ago(10), "assistant", "Joaquin, should the small study also run the frozen lower blocks "
                                                 "at every rung?")])
    rc1, err1 = w.check()
    rc2, err2 = w.check()
    scenario("2: a re-ask blocks at one stop, and not at the next", rc1 == 2 and "asked again" in err1 and rc2 == 0,
             err1 + err2)
    w = World([E("lowd", asks=done_ask)], state={"lowd": step(1)},
              transcript=[(ago(10), "assistant", 'Joaquin, you asked me before: "Should the small study also run the '
                                                 'frozen lower blocks at every rung?" Your answer was yes; it runs.')])
    rc, err = w.check()
    scenario("2: a question quoted in a status message is no re-ask", rc == 0, err)
    w = World([E("lowd", asks=done_ask)], state={"lowd": step(1), "_asks_scanned": ago(5)},
              transcript=[(ago(10), "assistant", "Joaquin, should the small study also run the frozen lower blocks "
                                                 "at every rung?")])
    rc, err = w.check()
    scenario("2: messages before the previous scan are not read again", rc == 0, err)


def reopen():
    """Fifth review 3: plan reopen and plan evidence --drop; any write to a done entry reruns its check."""
    repo = tempfile.mkdtemp(prefix="reopen_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q"); os.makedirs(os.path.join(repo, "docs")); os.makedirs(os.path.join(repo, "an"))
    open(os.path.join(repo, "docs", "results.tex"), "w").write("x\n"); g("add", "."); g("commit", "-qm", "a")
    open(os.path.join(repo, "docs", "results.tex"), "a").write("\\paragraph{rd.}\\label{par:readout} done.\n")
    g("commit", "-qam", "b")
    sha = g("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(repo, "an", "fig.png"), "w").write("p")
    open(os.path.join(repo, "an", "old.png"), "w").write("p")
    rd = E("rd", status="done", writeup=sha, done_when="test -f an/fig.png", notes_ref="par:readout",
           evidence={"runs": [], "commit": sha, "paths": ["an/fig.png", "an/old.png"]})
    w = World([rd])
    w.env["PLAN_ROOT"] = repo
    rc0, err0 = w.check()
    os.remove(os.path.join(repo, "an", "old.png"))
    w.plan_cmd("evidence", "rd", "--drop", "an/old.png")
    rc1, err1 = w.check()
    scenario("3: plan evidence --drop removes a pruned path, and the write reruns the done check (still passing)",
             rc0 == 0 and rc1 == 0 and "marked done" not in err0 + err1
             and w.entries()["rd"]["evidence"]["paths"] == ["an/fig.png"], err0 + err1)
    os.remove(os.path.join(repo, "an", "fig.png"))
    w.plan_cmd("set-next", "rd", "redo the figure", *WHY)
    rc2, err2 = w.check()
    r = w.plan_cmd("reopen", "rd", "--why", "its figure was pruned")
    e = w.entries()["rd"]
    scenario("3: a regression is seen after a write; plan reopen takes it back to running and keeps the write-up",
             rc2 == 0 and "marked done but" in err2 and r[0] == 0 and e["status"] == "running"
             and e["history"][0]["writeup"] == sha and "writeup" not in e, err2 + r[1])


def commit_retry():
    """Lower: a plan write waits out an index.lock; when git stays busy it says how to recover."""
    import threading, time
    repo, g, tex, base, env = plan_repo()
    lock = os.path.join(repo, ".git", "index.lock")
    open(lock, "w").close()
    threading.Timer(2.0, lambda: os.remove(lock)).start()
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "after the lock", *WHY], env=env, capture_output=True,
                       text=True)
    scenario("lower: a plan write retries through a short index.lock and commits", r.returncode == 0
             and g("status", "--porcelain").stdout == "", r.stdout + r.stderr)
    open(lock, "w").close()
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "while git is busy", *WHY], env=env, capture_output=True,
                       text=True)
    os.remove(lock)
    r2 = subprocess.run([sys.executable, PLAN, "commit"], env=env, capture_output=True, text=True)
    scenario("lower: with git busy throughout it fails naming plan commit, which then commits",
             r.returncode == 3 and "tools/plan.py commit" in r.stderr and r2.returncode == 0
             and g("status", "--porcelain").stdout == "" and time.monotonic() - t0 < 40, r.stderr + r2.stderr)


def claims_budget():
    """Lower: a claims check that does not fit the stop's time blocks as incomplete; plan claims finishes it."""
    repo, g, tex, base, env = plan_repo()
    open(os.path.join(repo, "analysis", "slow.py"), "w").write(
        'import json, time\ntime.sleep(8)\nprint(json.dumps({"v": 0.25}))\n')
    g("add", "analysis/slow.py"); g("commit", "-qm", "slow", "--no-verify")
    sha = g("rev-parse", "HEAD").stdout.strip()
    open(tex, "a").write("\n%% src: python3 analysis/slow.py @%s\n%% claim: v=0.25\nThe value is $0.25$.\n" % sha)
    g("commit", "-qam", "par", "--no-verify")
    rc = subprocess.run([sys.executable, CHECK], input="{}", env=dict(env, PLAN_BUDGET="3"), capture_output=True,
                        text=True)
    scenario("lower: the claims check out of time blocks as 'claims check incomplete', not a hook timeout",
             rc.returncode == 2 and "claims check incomplete" in rc.stderr, rc.stderr)
    r = subprocess.run([sys.executable, PLAN, "claims"], env=env, capture_output=True, text=True)
    cache = json.load(open(os.path.join(repo, "plan", ".claims_cache.json")))
    head = g("rev-parse", "HEAD").stdout.strip()
    scenario("lower: plan claims finishes it with no limit, keeps the passed paragraph, moves the watermark",
             r.returncode == 0 and len(cache) == 1
             and json.load(open(os.path.join(repo, "plan", ".state.json")))["_claims_verified"] == head,
             r.stdout + r.stderr)
    # sixth review 1: a script slower than the claims check's own limit inside the stop
    repo, g, tex, base, env = plan_repo()
    open(os.path.join(repo, "analysis", "slow.py"), "w").write(
        'import json, time\ntime.sleep(6)\nprint(json.dumps({"v": 0.25}))\n')
    g("add", "analysis/slow.py"); g("commit", "-qm", "slow", "--no-verify")
    sha = g("rev-parse", "HEAD").stdout.strip()
    open(tex, "a").write("\n%% src: python3 analysis/slow.py @%s\n%% claim: v=0.25\nThe value is $0.25$.\n" % sha)
    g("commit", "-qam", "par", "--no-verify")
    env2 = dict(env, PLAN_CLAIMS_TIMEOUT="2")
    rc = subprocess.run([sys.executable, CHECK], input="{}", env=env2, capture_output=True, text=True)
    r = subprocess.run([sys.executable, PLAN, "claims"], env=env2, capture_output=True, text=True)
    left = subprocess.run(["git", "-C", repo, "worktree", "list"], capture_output=True, text=True).stdout
    scenario("1 (sixth review): a script slower than the stop's claims limit: the stop says incomplete, plan claims "
             "(no limit) completes it", rc.returncode == 2 and "claims check incomplete" in rc.stderr
             and r.returncode == 0 and "verify_claims_" not in left, rc.stderr + r.stdout + r.stderr + left)


def added_only():
    """Policy: only numbers a change adds to a paragraph need a claim; reflows and typo fixes around them need
    none, a number moved in from another paragraph does."""
    repo, g, tex, base = repo_with_script()
    open(tex, "w").write("Old paragraph with 1.23 and no source.\n\n% src: python3 analysis/gone.py @" + base +
                         "\n% claim: x=0.77\nA claimed $0.77$ whose source is gone.\n")
    g("add", "."); g("commit", "-qm", "two paragraphs", "--no-verify")

    def stage(text):
        open(tex, "w").write(text)
        g("add", ".")
        r = subprocess.run([sys.executable, VERIFY, "--root", repo, "--staged"], capture_output=True, text=True)
        return r.returncode, r.stdout
    claimed = "% src: python3 analysis/gone.py @" + base + "\n% claim: x=0.77\n"
    for name, text, want in (
            ("a reflow of an unsourced old number", "Old paragraph\nwith 1.23 and, reworded, no source.\n\n" + claimed
             + "A claimed $0.77$ whose source is gone.\n", 0),
            ("a number moved in from another paragraph (it counts as added there)", claimed +
             "A claimed $0.77$ whose source is gone.\n\nOld paragraph with 1.23 and no source.\n", 1),
            ("a reflow of a claimed paragraph (its claim is not rerun)", "Old paragraph with 1.23 and no source.\n\n" + claimed
             + "A claimed $0.77$ whose source\nis gone.\n", 0),
            ("a typo fix in a claimed paragraph (its claim is not rerun)", "Old paragraph with 1.23 and no source.\n\n"
             + claimed + "A claimed $0.77$ whose source has gone.\n", 0),
            ("a changed number", "Old paragraph with 1.24 and no source.\n\n" + claimed +
             "A claimed $0.77$ whose source is gone.\n", 1)):
        rc, out = stage(text)
        scenario("policy: %s %s" % (name, "is refused" if want else "needs no claim"), rc == want, out)
    # sixth review 2: a changed % src: line reruns its claims
    repo, g, tex, base = repo_with_script()
    open(tex, "w").write("x\n\n%% src: python3 analysis/s.py @%s\n%% claim: h3.shape=0.80\nThe floor is $0.80$.\n"
                         % base)
    g("add", "."); g("commit", "-qm", "claimed", "--no-verify")
    open(os.path.join(repo, "analysis", "s.py"), "w").write(
        'import json\nprint(json.dumps({"h3": {"shape": 0.5}}))\n')
    g("commit", "-qam", "the script changed", "--no-verify")
    sha2 = g("rev-parse", "HEAD").stdout.strip()
    rc, out = stage("x\n\n%% src: python3 analysis/s.py @%s\n%% claim: h3.shape=0.80\nThe floor is $0.80$.\n" % sha2)
    scenario("2 (sixth review): a changed src line reruns its claims", rc == 1 and "recomputed 0.5" in out, out)
    # lower: temporary worktrees a cut-off run left behind are removed at the start of the next
    stale = os.path.join(tempfile.mkdtemp(prefix="verify_claims_"), "wt")
    g("worktree", "add", "-q", "--detach", stale, base)
    os.utime(stale, (1, 1))
    subprocess.run([sys.executable, VERIFY, "--root", repo, "--rev", "%s..HEAD" % base], capture_output=True)
    left = g("worktree", "list").stdout
    scenario("lower: a temporary worktree left by a cut-off run is removed by the next range check",
             stale not in left and not os.path.exists(stale), left)


SEARCHED = ("CLAUDE.md HPO rule 2 and the canonical setup table, and the decisions in force: none says which of the "
            "two pools the small study uses")


def searched_first():
    """A question goes to the user only with what was looked for first (the lr scalings entry waited three weeks on
    a question CLAUDE.md's HPO rule 2 and the measured lr surface settled)."""
    q = "Should the scratch lr window be re-centred on the measured surface?"
    w = World([E("lr")])
    r0 = w.plan_cmd("ask", "lr", "--ask", q)
    r1 = w.plan_cmd("ask", "lr", "--ask", q, "--searched", "read CLAUDE.md")
    r2 = w.plan_cmd("add", "lr2", "--title", "t", "--question", "q", "--next", "n", "--match", "lr2*", "--ask", q,
                    "--searched", "nothing")
    scenario("plan ask (and plan add --ask) without what was searched first, or with a token note, is refused",
             r0[0] != 0 and r1[0] != 0 and "--searched" in r1[1] and r2[0] != 0 and w.entries()["lr"]["status"] ==
             "running" and "lr2" not in w.entries(), r0[1] + r1[1] + r2[1])
    r3 = w.plan_cmd("ask", "lr", "--ask", q, "--searched", SEARCHED)
    e = w.entries()["lr"]
    view, show = w.plan_cmd("view")[1], w.plan_cmd("show")[1]
    scenario("plan ask with what was searched stores it on the entry and its ask, and plan view and show print it",
             r3[0] == 0 and e["searched"] == SEARCHED and e["asks"][-1]["searched"] == SEARCHED
             and "Searched first: CLAUDE.md HPO rule 2" in view and "searched first: CLAUDE.md HPO rule 2" in show,
             r3[1] + view[-400:] + show[-400:])
    tpl = open(os.path.join(ROOT, "plan", "step_prompt.txt")).read()
    scenario("the step prompt has a step agent read CLAUDE.md's setup and HPO rules and the decisions before asking",
             "HPO search-space rules" in tpl and "site decisions --project FA" in tpl and "--searched" in tpl)


def seventh_lower():
    """Seventh review: a source script is bounded on the hook path (plan claims has no limit); a temporary worktree
    a live run uses is never removed by another run's cleanup."""
    repo, g, tex, base = repo_with_script()
    open(os.path.join(repo, "analysis", "slow.py"), "w").write(
        'import json, time\ntime.sleep(3)\nprint(json.dumps({"v": 0.25}))\n')
    g("add", "analysis/slow.py"); g("commit", "-qm", "slow", "--no-verify")
    sha = g("rev-parse", "HEAD").stdout.strip()
    open(tex, "a").write("\n%% src: python3 analysis/slow.py @%s\n%% claim: v=0.25\nThe value is $0.25$.\n" % sha)
    g("add", ".")
    run = lambda *a: subprocess.run([sys.executable, VERIFY, "--root", repo, "--staged"] + list(a),
                                    capture_output=True, text=True)
    r1, r0 = run("--script-timeout", "1"), run("--script-timeout", "0")
    scenario("1: a source script past --script-timeout fails its claim; 0 means no limit", r1.returncode == 1
             and "took over 1 s" in r1.stdout and r0.returncode == 0, r1.stdout + r0.stdout)
    src = open(PLAN).read()
    scenario("1: the hook path defaults to 1800 s per script, plan claims passes no limit",
             "SCRIPT_TIMEOUT = [1800]" in open(VERIFY).read() and '"--script-timeout", "0"' in src)
    owner = subprocess.Popen(["sleep", "60"])
    live_dir = tempfile.mkdtemp(prefix="verify_claims_")
    live = os.path.join(live_dir, "wt")
    g("worktree", "add", "-q", "--detach", live, base)
    open(os.path.join(live_dir, "owner.pid"), "w").write(str(owner.pid))
    os.utime(live, (1, 1))
    subprocess.run([sys.executable, VERIFY, "--root", repo, "--rev", "%s..HEAD" % base], capture_output=True)
    kept = os.path.isdir(live) and live in g("worktree", "list").stdout
    owner.kill(); owner.wait()
    subprocess.run([sys.executable, VERIFY, "--root", repo, "--rev", "%s..HEAD" % base], capture_output=True)
    scenario("2: a temporary worktree whose owning run is alive is kept, however old; once it is gone it is removed",
             kept and not os.path.exists(live) and live not in g("worktree", "list").stdout, g("worktree", "list").stdout)


def hold_without_item():
    """An entry paused before it had a site item (the synthetic prior, paused under a scope decision on 2026-10-14):
    plan answer --kind hold files the item under that scope, then holds it; all or nothing."""
    words = "This should be recorded as a result, and this direction put on pause"
    paused = lambda: E("syn", status="held", hold_decision="D43", decisions=["D43"], match=None,
                       quotes=[{"said_at": ago(600), "text": words, "decision": "D43", "covers": ["synthetic data"]}])
    dec = [("D43", "scope", "synthetic data", 1, ago(600), words)]
    tr = [(ago(600), "user", words)]
    w = World([paused()], decisions=dec, transcript=tr)
    rc0, err0 = w.check()
    r = w.plan_cmd("answer", "syn", "--kind", "hold", "--quote", words)
    e, item = w.entries()["syn"], w.sql("SELECT decision, status, match FROM items WHERE id='syn'")
    rc, err = w.check()
    scenario("a hold on a paused entry with no item: the item is filed under its scope, held, and nothing is flagged",
             rc0 == 2 and r[0] == 0 and item == [("D43", "held", "syn*")] and e["status"] == "held"
             and e["hold_decision"] == "D900" and rc == 0, err0 + r[1] + str(item) + err)
    w = World([paused()], decisions=dec, transcript=tr)
    r = w.plan_cmd("answer", "syn", "--kind", "hold", "--quote", words, env={"STUB_DECIDE_FAIL": "1"})
    scenario("if the hold itself fails, the item made for it is closed and the entry is unchanged",
             r[0] != 0 and w.sql("SELECT status FROM items WHERE id='syn'") == [("done",)]
             and w.entries()["syn"]["hold_decision"] == "D43", r[1])
    w = World([E("syn2", status="held", hold_decision="D44", decisions=["D44"])],
              decisions=[("D44", "prefer", "syn2", 1, ago(600), words)], transcript=tr)
    r = w.plan_cmd("answer", "syn2", "--kind", "forbid", "--quote", words)
    scenario("with no approval or scope in force to file an item under, it is refused before anything is recorded",
             r[0] != 0 and "no approval or scope" in r[1] and "decide" not in w.calls() and "item" not in w.calls(),
             r[1])


def sed_reads():
    for c, want in (("sed -n 140,200p plan/plan.yaml", 0), ("sed -i s/held/running/ plan/plan.yaml", 2),
                    ("sed -ni s/a/b/ plan/plan.yaml", 2), ("sed -n 'w plan/plan.yaml' /tmp/x", 2)):
        rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": c}})
        scenario("live: `%s` is %s" % (c, "refused" if want else "allowed (a read)"), rc == want, out)


def setup_records():
    """A setup entry's record may be CLAUDE.md or a tool (plan kind <id> setup, plan done --record); work entries
    still need their docs/results.tex write-up."""
    repo = tempfile.mkdtemp(prefix="setup_rec_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q"); os.makedirs(os.path.join(repo, "docs")); os.makedirs(os.path.join(repo, "tools"))
    open(os.path.join(repo, "docs", "results.tex"), "w").write("x\n")
    open(os.path.join(repo, "CLAUDE.md"), "w").write("rules\n")
    open(os.path.join(repo, "tools", "t.py"), "w").write("x = 1\n")
    g("add", "."); g("commit", "-qm", "a")
    open(os.path.join(repo, "CLAUDE.md"), "a").write("a new rule\n"); g("commit", "-qam", "the rule")
    rule = g("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(repo, "tools", "t.py"), "a").write("y = 2\n"); g("commit", "-qam", "the tool")
    tool = g("rev-parse", "HEAD").stdout.strip()
    base = dict(done_when="grep -q 'a new rule' CLAUDE.md", evidence={"runs": [], "commit": rule, "paths": ["CLAUDE.md"]})
    w = World([E("setup_x", **base), E("work_x", **base)])
    w.env["PLAN_ROOT"] = repo
    r0 = w.plan_cmd("done", "work_x", "--writeup", rule, "--record", "CLAUDE.md")
    scenario("--record on an entry not marked setup is refused", r0[0] != 0 and "setup entries only" in r0[1], r0[1])
    w.plan_cmd("kind", "setup_x", "setup")
    r1 = w.plan_cmd("done", "setup_x", "--writeup", tool, "--record", "CLAUDE.md")
    scenario("a setup entry's write-up commit must change its record", r1[0] != 0 and "does not change the record"
             in r1[1], r1[1])
    r2 = w.plan_cmd("done", "setup_x", "--writeup", rule, "--record", "CLAUDE.md")
    e = w.entries()["setup_x"]
    rc, err = w.check()
    scenario("a setup entry is done with its record in CLAUDE.md, no results.tex label, and stays done on recheck",
             r2[0] == 0 and e["status"] == "done" and e["record"] == "CLAUDE.md" and "marked done but" not in err,
             r2[1] + err)
    out = os.path.join(tempfile.mkdtemp(), "plan.yaml")
    subprocess.run([sys.executable, MIGRATE, "--out", out, "--root", ROOT], capture_output=True, text=True)
    kinds = {e["id"]: e.get("kind") for e in yaml.safe_load(open(out))["entries"]}
    scenario("the migration marks the setup's own entries setup", all(
        v == "setup" for k, v in kinds.items() if "setup" in k) and any("setup" in k for k in kinds), kinds)


def decision_quotes():
    """An entry added under a decision carries that decision's words, and its step agent is shown them."""
    words = "Okay, go ahead with the small study first round"
    w = World([], decisions=[("D127", "approve", "lowd2", 1, ago(30), words)])
    r = w.plan_cmd("add", "lowd2", "--title", "t", "--question", "q", "--next", "the first round", "--match",
                   "tp3_lowd*", "--decision", "D127")
    e = w.entries()["lowd2"]
    rn, out = w.plan_cmd("next", "--prompts")
    scenario("plan add --decision records the decision's words; the step prompt lists them, not 'none'",
             r[0] == 0 and e["quotes"][0]["text"] == words and words in out, r[1] + out[-600:])
    rc, err = w.check()
    scenario("that quote passes the quote check", "quote on it" not in err, err)
    w = World([E("old", decisions=["D1"])], decisions=[("D1", "approve", "*", 1, ago(30), "go ahead with all of it")])
    rn, out = w.plan_cmd("next", "--prompts")
    scenario("an entry whose decision has no quote on it still shows the decision's words to its step agent",
             "go ahead with all of it" in out, out[-600:])


SITE_STUB_LAUNCH = r'''#!/bin/sh
# a stub of site for the launcher: pick says ccin2p3, run prints the site's commit, submit prints a run id
case "$1" in
  pick) echo "  pick: ccin2p3" ;;
  run) echo "a login notice"; echo "$STUB_SITE_SHA" ;;
  submit) echo "submit $*" >> "$STUB_LOG"; echo "  run FA-ccin2p3-stub0001 (job 1)" ;;
  sync) echo "sync $*" >> "$STUB_LOG" ;;
esac
'''


def plan_commits_and_launches():
    """Every plan write is a trunk commit; a NOSYNC launch against a site checkout that differs only in plan/ runs the
    same code and goes out, one whose code differs is refused (2026-10-25: launches failed or deferred on the race)."""
    repo = tempfile.mkdtemp(prefix="launch_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q"); os.makedirs(os.path.join(repo, "scripts")); os.makedirs(os.path.join(repo, "plan"))
    for f in ("launch_cc_only.sh", "same_code.sh"):
        shutil.copy(os.path.join(ROOT, "scripts", f), os.path.join(repo, "scripts", f))
    open(os.path.join(repo, "job.sh"), "w").write("echo job\n")
    open(os.path.join(repo, "plan", "plan.yaml"), "w").write("entries: []\n")
    g("add", "."); g("commit", "-qm", "code")
    site_at = g("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(repo, "plan", "plan.yaml"), "a").write("# a plan write\n"); g("commit", "-qam", "plan: set-next x")
    plan_only = g("rev-parse", "HEAD").stdout.strip()
    r = lambda *a: subprocess.run(["bash", os.path.join(repo, "scripts", "same_code.sh")] + list(a),
                                  capture_output=True).returncode
    open(os.path.join(repo, "job.sh"), "a").write("echo changed\n"); g("commit", "-qam", "code change")
    code = g("rev-parse", "HEAD").stdout.strip()
    scenario("same_code.sh: a commit apart by plan/ only is the same code; a code change or an unknown commit is not",
             r(site_at, plan_only) == 0 and r(site_at, code) == 1 and r("0" * 40, plan_only) == 1)
    bindir, home = tempfile.mkdtemp(prefix="bin_"), tempfile.mkdtemp(prefix="home_")
    os.makedirs(os.path.join(home, ".local", "share", "ccorch", "launch", "FA"))    # the launcher's submitted list
    open(os.path.join(bindir, "site"), "w").write(SITE_STUB_LAUNCH)
    os.chmod(os.path.join(bindir, "site"), 0o755)
    log = os.path.join(home, "stub.log")
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"], HOME=home, STUB_SITE_SHA=site_at,
               STUB_LOG=log, NOSYNC="1")

    def launch(head, key):
        g("checkout", "-q", head)
        return subprocess.run(["bash", os.path.join(repo, "scripts", "launch_cc_only.sh"), key, "0.1", "a test launch",
                               "job.sh"], capture_output=True, text=True, env=env)
    a = launch(plan_only, "item_a")
    b = launch(code, "item_b")
    sub = open(log).read() if os.path.exists(log) else ""
    scenario("a NOSYNC launch against a site checkout behind by plan commits only goes out; a code difference is "
             "refused", a.returncode == 0 and "run FA-ccin2p3-stub0001" in a.stdout and b.returncode == 2
             and "the code differs" in b.stdout and len(sub.splitlines()) == 1 and "item_a" in sub,
             a.stdout + b.stdout + sub)


def slow_done_checks():
    """A done entry whose check is longer than the stop's time is noted once and cached for the day, not retried at
    every stop; plan recheck finishes it with no limit; plan check --replace gives it a faster check."""
    repo = tempfile.mkdtemp(prefix="slowdone_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q"); os.makedirs(os.path.join(repo, "docs")); os.makedirs(os.path.join(repo, "an"))
    open(os.path.join(repo, "docs", "results.tex"), "w").write("x\n"); g("add", "."); g("commit", "-qm", "a")
    open(os.path.join(repo, "docs", "results.tex"), "a").write("\\paragraph{rd.}\\label{par:readout} done.\n")
    g("commit", "-qam", "b")
    sha = g("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(repo, "an", "fig.png"), "w").write("p")
    rd = E("rd", status="done", writeup=sha, done_when="sleep 4; test -f an/fig.png", notes_ref="par:readout",
           evidence={"runs": [], "commit": sha, "paths": ["an/fig.png"]})
    w = World([rd])
    w.env.update(PLAN_ROOT=repo, PLAN_BUDGET="2")
    rc1, out1 = w.check()
    rc2, out2 = w.check()
    scenario("a done check longer than the stop's time is noted once, then cached for the day like a pass",
             rc1 == 0 and "did not finish" in out1 and "plan.py recheck rd" in out1 and rc2 == 0
             and "did not finish" not in out2, out1 + "|" + out2)
    r = w.plan_cmd("recheck", "rd")
    rc3, out3 = w.check()
    scenario("plan recheck runs it with no limit, and the stops read its result", r[0] == 0 and "0 regressed" in r[1]
             and rc3 == 0 and "did not finish" not in out3, r[1] + out3)
    w.env.pop("PLAN_BUDGET")
    w2 = World([E("run1")])
    w2.env["PLAN_ROOT"] = repo
    tries = (("a running entry", w2, "run1", "test -f an/fig.png", "is for a done entry"),
             ("a trivially true check", w, "rd", "true", "checks nothing"),
             ("a check that fails now", w, "rd", "test -f an/none.png", "reopen it"))
    for name, ww, eid, chk, why in tries:
        r = ww.plan_cmd("check", eid, "--done-when", chk, "--replace")
        scenario("plan check --replace refuses %s" % name, r[0] != 0 and why in r[1], r[1])
    r = w.plan_cmd("check", "rd", "--done-when", "test -f an/fig.png")
    scenario("plan check without --replace on a done entry is refused", r[0] != 0 and "--replace" in r[1], r[1])
    r = w.plan_cmd("check", "rd", "--done-when", "test -f an/fig.png", "--replace")
    e = w.entries()["rd"]
    rc, out = w.check()
    scenario("plan check --replace gives a done entry a fast check that passes, keeping the old one in its history",
             r[0] == 0 and e["done_when"] == "test -f an/fig.png" and e["status"] == "done"
             and e["history"][-1]["was_done_when"].startswith("sleep 4") and rc == 0 and "marked done" not in out,
             r[1] + out)


def tooling_fixes():
    """The review backlog counts code at any depth; run ids are not numbers; a heredoc is judged by what it does."""
    repo = tempfile.mkdtemp(prefix="backlog_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q")
    for d in (".claude/hooks", ".claude/.review_state", "analysis/deep"):
        os.makedirs(os.path.join(repo, d))
    shutil.copy(os.path.join(ROOT, ".claude", "hooks", "review_backlog.sh"), os.path.join(repo, ".claude", "hooks"))
    open(os.path.join(repo, "root_module.py"), "w").write("x = 1\n")
    open(os.path.join(repo, "analysis", "deep", "fit.py"), "w").write("y = 1\n")
    g("add", "."); g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD").stdout.strip()
    for name in ("repo-reviewer", "notes-editor", "claudemd-keeper"):
        open(os.path.join(repo, ".claude", ".review_state", name), "w").write("%s 0 0\n" % base)
    open(os.path.join(repo, "analysis", "deep", "fit.py"), "a").write("".join("z%d = %d\n" % (i, i) for i in range(10)))
    g("commit", "-qam", "deep change")
    st = subprocess.run(["bash", ".claude/hooks/review_backlog.sh", "status"], cwd=repo, capture_output=True,
                        text=True).stdout
    row = next((l.split() for l in st.splitlines() if l.startswith("repo-reviewer")), [])
    scenario("review backlog: a .py change two directories down counts toward the repo-reviewer (lines and commits)",
             row[1:3] == ["10/520", "1/100000"], st)
    # the weighted count (tools/review_weight.py, D171): an analysis-script edit weighs its area's and kind's rates
    for rel in ("tools/review_weight.py", "analysis/review_rates/review_rates.py", "analysis/review_rates/review_rates.json"):
        os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), os.path.join(repo, rel))
    g("add", "."); g("commit", "-qm", "weigher")
    open(os.path.join(repo, ".claude", ".review_state", "repo-reviewer"), "w").write("%s 0 0\n" % g("rev-parse", "HEAD").stdout.strip())
    os.makedirs(os.path.join(repo, "sweep"), exist_ok=True)
    for i in range(5):
        open(os.path.join(repo, "sweep", "sweep_config_fam_p%d.yaml" % i), "w").write("a: 1\n" * 40)
    open(os.path.join(repo, "analysis", "deep", "fit.py"), "a").write("".join("w%d = %d\n" % (i, i) for i in range(100)))
    g("add", "sweep"); g("commit", "-qam", "family and edit")
    st = subprocess.run(["bash", ".claude/hooks/review_backlog.sh", "status"], cwd=repo, capture_output=True,
                        text=True).stdout
    row = next((l.split() for l in st.splitlines() if l.startswith("repo-reviewer")), [])
    n = int(row[1].split("/")[0]) if row else -1
    scenario("review backlog: a generated config family counts once and an analysis edit by its weight", 
             100 < n < 40 + 100 * 1.5 and n != 300, st)
    repo2, g2, tex, base2 = repo_with_script()
    open(tex, "a").write("\n%% src: python3 analysis/s.py @%s\n%% claim: h3.shape=0.80\nThe rerun FA-ccin2p3-60b47206 "
                         "gives $0.80$.\n" % base2)
    g2("add", ".")
    r1 = subprocess.run([sys.executable, VERIFY, "--root", repo2, "--staged"], capture_output=True, text=True)
    open(tex, "a").write("\nAt 13 TeV it holds.\n")
    g2("add", ".")
    r2 = subprocess.run([sys.executable, VERIFY, "--root", repo2, "--staged"], capture_output=True, text=True)
    scenario("verify: a registry run id is not read as numbers; 13 TeV still needs a claim or an allow",
             r1.returncode == 0 and r2.returncode == 1 and "13 is no claim" in r2.stdout, r1.stdout + r2.stdout)
    for c, want in (
            ("python3 - <<'EOF'\nimport yaml\nd = yaml.safe_load(open('plan/plan.yaml'))\nprint(len(d))\nEOF", 0),
            ("python3 - <<'EOF'\nd = open('plan/plan.yaml').read()\nopen('/tmp/out.txt', 'w').write(d)\nEOF", 0),
            ("git commit -F - <<'EOF'\nplan: an edit to plan/plan.yaml\nEOF", 0),
            ("python3 - <<'EOF'\nopen('plan/plan.yaml', 'w').write('x')\nEOF", 2),
            ("python3 - <<'EOF'\nimport json\np = 'plan/.state.json'\njson.dump({}, open(p, 'w'))\nEOF", 2),
            ("bash <<'EOF'\necho x > plan/plan.yaml\nEOF", 2),
            ("cat <<'EOF' > plan/plan.yaml\nentries: []\nEOF", 2)):
        rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": c}})
        scenario("heredoc: `%s` is %s" % (c.replace("\n", " / ")[:70], "refused" if want else "allowed"), rc == want,
                 out)


def reviewed_content():
    """A CLAUDE.md diff the reviewer passed uncommitted is not owed again once committed; a further edit is."""
    repo = tempfile.mkdtemp(prefix="backlog_content_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q")
    os.makedirs(os.path.join(repo, ".claude", "hooks"))
    shutil.copy(os.path.join(ROOT, ".claude", "hooks", "review_backlog.sh"), os.path.join(repo, ".claude", "hooks"))
    open(os.path.join(repo, ".gitignore"), "w").write(".claude/.review_state/\n")
    md = os.path.join(repo, "CLAUDE.md")
    open(md, "w").write("# rules\n")
    g("add", "."); g("commit", "-qm", "base")
    rb = lambda *a, stdin="": subprocess.run(["bash", ".claude/hooks/review_backlog.sh"] + list(a), cwd=repo,
                                             input=stdin, capture_output=True, text=True)
    due = lambda: next((l for l in rb("status").stdout.splitlines() if l.startswith("claudemd-keeper")), "")
    commit = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -qam 'rule'"}})
    rb("init")
    open(md, "a").write("a new rule\n")
    before = due()
    refused = rb("bashpre", stdin=commit).returncode
    rb("begin", "claudemd-keeper"); rb("advance", "claudemd-keeper")
    allowed = rb("bashpre", stdin=commit).returncode
    g("commit", "-qam", "rule")
    after = due()
    scenario("review backlog: a CLAUDE.md edit reviewed uncommitted, then committed, is not due again",
             "DUE" in before and refused == 2 and allowed == 0 and "DUE" not in after,
             before + "\n" + after)
    open(md, "a").write("another rule\n")
    again = due()
    scenario("review backlog: a further CLAUDE.md edit after that commit is due, and its commit refused",
             "DUE" in again and rb("bashpre", stdin=commit).returncode == 2, again)


SM = os.path.join(ROOT, "sweep", "sweep_manager.py")


def sweep_world(entries, sweeps=None, state=None, ended=None, cache_at=None, sites="", **k):
    """A world whose stub site serves sweep_manager's state on ccin2p3 (`sweeps`: name -> facts; None: the site is
    down). `ended` / `cache_at` pre-seed the check cache (when each sweep was first seen ended, and when it was read)."""
    w = World(entries, state=state, **k)
    w.env["PLAN_SWEEP_SITES"] = sites
    if sweeps is not None:
        json.dump({"sweeps": sweeps}, open(os.path.join(w.d, "sweeps_ccin2p3.json"), "w"))
    if cache_at is not None:
        json.dump({"sweeps:ccin2p3": {"at": cache_at, "names": sorted(sweeps or {}), "data": sweeps, "note": None,
                                      "ended": ended or {}}}, open(os.path.join(w.d, ".check_cache.json"), "w"))
    return w


def F(live=0, submitted=8, scripts=8, results=None, items=(), submitted_at=None):
    return {"dir": "/sps/x", "live": live, "submitted": submitted, "scripts": scripts,
            "results": submitted - live if results is None else results, "items": list(items),
            "submitted_at": submitted_at}


def rec(name, m=60):
    return {"site": "ccin2p3", "name": name, "at": ago(m)}


def sweeps():
    """A sweep submitted through sweep_manager (never in runs.db) recorded on its entry: in flight while its jobs are
    in the site's queue, listed when it ends, in flight with a note when the site cannot be read; one no entry records
    is recorded on the entry it fits (the Muon test's two searches ran for hours with no record, 2026-10-26)."""
    st = {"mu": {"sweeps": [rec("mu_search_a")]}}
    w = sweep_world([E("mu")], {"mu_search_a": F(live=3)}, state=st)
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("next")
    rc3, out3 = w.plan_cmd("show")
    scenario("sweeps: a recorded sweep with jobs in the queue is the entry in flight (plan next, the Stop check), and "
             "plan show lists it", rc == 0 and rc2 == 0 and "nothing in flight" not in err + out2
             and "live mu_search_a (ccin2p3, 3 jobs queued or running)" in out3, err + out2 + out3)
    w = sweep_world([E("mu")], None, state=st)
    rc, err = w.check()
    scenario("sweeps: the recorded sweep's site cannot be read: in flight, with a note naming the site and the error",
             rc == 0 and "sweep registry on ccin2p3 could not be read" in err and "Connection refused" in err, err)
    w = sweep_world([E("mu")], {"mu_search_a": F(live=0, results=7)}, state=st, ended={"mu_search_a": ago(10)},
                    cache_at=ago(1), agents=[("mu_s1", ago(5), "mu"), ("mu_s0", ago(15), "mu")])
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("next", "--prompts")
    want = "sweep ended: mu_search_a (ccin2p3, 8 of 8 trials submitted, 7 results)"
    scenario("sweeps: an ended sweep lists the entry with 'sweep ended: <name>' and the step agent's prompt says so",
             rc == 2 and want in err and rc2 == 1 and "Why you were started: " + want in out2, err + out2)
    rc, out = w.plan_cmd("step", "mu", "--agent", "mu_s0")
    rc2, out2 = w.plan_cmd("step", "mu", "--agent", "mu_s1")
    rc3, err3 = w.check()
    scenario("sweeps: a step agent started before the sweep was seen ended is refused, one started after clears it",
             rc != 0 and "before the reason" in out and rc2 == 0 and rc3 == 0, out + out2 + err3)
    w = sweep_world([E("mu")], {"mu_search_a": F(live=0)}, state=st, ended={"mu_search_a": ago(25)},
                    cache_at=ago(30))
    w.check()
    kept = json.load(open(os.path.join(w.d, ".check_cache.json")))["sweeps:ccin2p3"]
    scenario("sweeps: when the site is read again, the time the sweep was first seen ended is kept",
             kept["ended"].get("mu_search_a") == ago(25) and kept["at"] > ago(1), kept)
    w = sweep_world([E("mu")], {}, state={"mu": {"sweeps": [rec("mu_typo")]}}, cache_at=ago(1))
    rc, err = w.check()
    scenario("sweeps: a recorded sweep its site's registry does not know is listed (a step agent looks), not in flight "
             "forever", rc == 2 and "sweep ended: mu_typo (ccin2p3: not in its sweep registry)" in err, err)
    # recording by hand
    w = sweep_world([E("mu")], {"mu_search_a": F(live=2)})
    rc, out = w.plan_cmd("run", "mu", "--sweep", "ccin2p3:/sps/x/sweeps/mu_search_a/")
    got = json.load(open(w.state)).get("mu", {}).get("sweeps") or []
    rc2, out2 = w.plan_cmd("run", "mu", "--sweep", "ccin2p3:mu_nonesuch")
    rc3, out3 = w.plan_cmd("run", "mu", "--sweep", "mu_search_a")
    rc4, out4 = w.plan_cmd("run", "mu")
    scenario("sweeps: plan run --sweep SITE:DIR records the sweep by name; one the registry lacks, a spec with no site "
             "and a run with nothing named are refused",
             rc == 0 and [x["name"] for x in got] == ["mu_search_a"] and rc2 != 0 and "no sweep mu_nonesuch" in out2
             and rc3 != 0 and rc4 != 0, [out, got, out2, out3, out4])
    w = sweep_world([E("mu")], None)
    rc, out = w.plan_cmd("run", "mu", "--sweep", "ccin2p3:mu_search_b")
    scenario("sweeps: with the site down the sweep is recorded anyway, with a note, and counts in flight",
             rc == 0 and "could not be read" in out and w.check()[0] == 0, out)
    w = sweep_world([E("mu", done_when="true")], {"mu_search_a": F(live=1)}, state=st, cache_at=ago(1))
    rc, out = w.plan_cmd("done", "mu", "--writeup", "HEAD")
    scenario("sweeps: plan done is refused while a recorded sweep has jobs in the queue",
             rc != 0 and "sweeps still going: mu_search_a" in out, out)
    # the unrecorded ones
    w = sweep_world([E("mu", match="mu_*"), E("other", match="zz_*")], {"mu_search_c": F(live=4)}, sites="ccin2p3",
                    state={"other": step(5)})
    rc, err = w.check()
    got = json.load(open(w.state)).get("mu", {}).get("sweeps") or []
    scenario("sweeps: a live sweep no entry records is caught and recorded on the running entry whose match names it, "
             "with a note", rc == 0 and [x["name"] for x in got] == ["mu_search_c"] and "recorded on mu" in err, err)
    w = sweep_world([E("mu", match="mu_*"), E("other", match="*_c")], {"mu_search_c": F(live=4)}, sites="ccin2p3",
                    state={"other": step(5), "mu": step(5)})
    rc, err = w.check()
    rc2, out2 = w.plan_cmd("next")
    scenario("sweeps: an unrecorded sweep that fits two running entries blocks the turn and plan next, naming both",
             rc == 2 and "mu_search_c on ccin2p3 runs with no record" in err and "fits mu, other" in err
             and rc2 == 1 and "UNRECORDED" in out2, err + out2)
    w = sweep_world([E("mu", match="mu_*"), E("other", match="*_c")], {"mu_search_c": F(live=4, items=["other"])},
                    sites="ccin2p3", state={"other": step(5), "mu": step(5)})
    rc, err = w.check()
    got = json.load(open(w.state)).get("other", {}).get("sweeps") or []
    scenario("sweeps: the work item its submission named (sweep_manager's stamp) settles which entry it is",
             rc == 0 and [x["name"] for x in got] == ["mu_search_c"], err)
    w = sweep_world([E("mu", match="mu_*")], {"mu_old": F(live=0), "mu_new": F(live=0, items=["mu"],
                                                                                submitted_at=epoch(120))},
                    sites="ccin2p3", state={"mu": step(40)})
    rc, err = w.check()
    got = [x["name"] for x in json.load(open(w.state)).get("mu", {}).get("sweeps") or []]
    scenario("sweeps: an ended sweep submitted within a day is recorded and listed as ended; an old ended one is not "
             "(no noise from the registry's history)", got == ["mu_new"] and "sweep ended: mu_new" in err, err)
    w = sweep_world([E("muon_test", match="*muon*", status="done", writeup="abc")], {"muon_test_search_x": F(live=0)},
                    sites="ccin2p3", state={"muon_test": {"sweeps": [rec("muon_test_search_x")]}})
    w2 = sweep_world([E("muon_test", match="*muon*", status="done", writeup="abc")],
                     {"muon_test_search_y": F(live=2)}, sites="ccin2p3")
    rc, err = w.check()
    rc2, err2 = w2.check()
    scenario("sweeps: a done entry is never reopened by its sweeps, recorded or not; a live one nobody runs is a note",
             w.entries()["muon_test"]["status"] == "done" and "sweep ended" not in err
             and w2.entries()["muon_test"]["status"] == "done" and "muon_test" not in json.load(open(w2.state))
             and rc2 == 0 and "no running entry's match names it" in err2, err + err2)
    # the submission side: sweep_manager stamps the sweep with the work item, and its state feeds the plan
    t = tempfile.mkdtemp(prefix="sm_")
    bins, sw = os.path.join(t, "bin"), os.path.join(t, "sweeps", "mu_search_d")
    os.makedirs(bins); os.makedirs(os.path.join(sw, "jobs")); os.makedirs(os.path.join(sw, "results"))
    for i in range(3):
        open(os.path.join(sw, "jobs", "trial_%04d.sh" % i), "w").write("#!/bin/sh\n")
    open(os.path.join(sw, "results", "hp0000_t1_1.json"), "w").write("{}")
    q = os.path.join(t, "queue")
    open(os.path.join(bins, "sbatch"), "w").write(
        "#!/bin/sh\nn=$(($(cat %s 2>/dev/null | wc -l) + 7001)); echo \"$n PENDING\" >> %s; echo $n\n" % (q, q))
    open(os.path.join(bins, "squeue"), "w").write("#!/bin/sh\ncat %s 2>/dev/null\n" % q)
    open(os.path.join(bins, "scontrol"), "w").write("#!/bin/sh\nexit 0\n")
    for b in ("sbatch", "squeue", "scontrol"):
        os.chmod(os.path.join(bins, b), 0o755)
    env = dict(os.environ, PATH=bins + os.pathsep + os.environ["PATH"], CCORCH_SITE="ccin2p3",
               CCORCH_WORK=J({"items": ["mu"], "active": ["mu_*"], "held": []}))
    reg = os.path.join(t, "registry.json")
    r = subprocess.run([sys.executable, SM, "--registry", reg, "submit", sw, "--seq-batches", "1"], env=env,
                       capture_output=True, text=True)
    stamp = json.load(open(reg))["sweeps"].get("mu_search_d") or {}
    r2 = subprocess.run([sys.executable, SM, "--registry", reg, "state"], env=env, capture_output=True, text=True)
    line = [l for l in r2.stdout.splitlines() if l.startswith("SWEEP_STATE ")]
    facts = json.loads(line[0][len("SWEEP_STATE "):])["sweeps"].get("mu_search_d") if line else {}
    scenario("sweeps: sweep_manager submit stamps the sweep with the work item and time and prints SWEEP_SUBMITTED; "
             "its state reports the queued jobs, scripts and results",
             r.returncode == 0 and "SWEEP_SUBMITTED" in r.stdout and stamp.get("items") == ["mu"]
             and stamp.get("submitted_at") and facts and facts["live"] == 3 and facts["scripts"] == 3
             and facts["results"] == 1, r.stdout + r.stderr + r2.stdout + r2.stderr)
    w = World([E("mu", match="zz_*")])
    w.env["PLAN_SWEEP_SITES"] = "ccin2p3"
    open(os.path.join(w.d, "sweeps_ccin2p3.json"), "w").write(line[0][len("SWEEP_STATE "):] if line else "{}")
    rc, err = w.check()
    got = [x["name"] for x in json.load(open(w.state)).get("mu", {}).get("sweeps") or []]
    open(q, "w").write("")                                    # the queue drains
    r3 = subprocess.run([sys.executable, SM, "--registry", reg, "state", "mu_search_d"], env=env, capture_output=True,
                        text=True)
    open(os.path.join(w.d, "sweeps_ccin2p3.json"), "w").write(r3.stdout.split("SWEEP_STATE ", 1)[-1])
    os.remove(os.path.join(w.d, ".check_cache.json"))
    rc2, err2 = w.check()
    scenario("sweeps: end to end, the stamped sweep (its name outside the entry's globs) is recorded on its item's "
             "entry, in flight while queued, listed when the queue drains",
             rc == 0 and got == ["mu_search_d"] and rc2 == 2 and "sweep ended: mu_search_d (ccin2p3, 3 of 3" in err2,
             err + err2)
    r4 = subprocess.run([sys.executable, SM, "--registry", reg, "submit", sw], env=dict(env, CCORCH_WORK=""),
                        capture_output=True, text=True)
    scenario("sweeps: a submission with no work item is still refused by the work gate (nothing stamped)",
             "refused mu_search_d" in r4.stdout and "SWEEP_SUBMITTED" not in r4.stdout, r4.stdout + r4.stderr)


def open_questions():
    """Only what needs the answer waits (CLAUDE.md 0g; the user, 2026-10-26: "why would you hold up all the catalog
    study for a question you can answer yourself?"). On catalog_coverage a question about one process's top width
    froze the entry's independent work for 8 h: plan ask kept the whole entry waiting."""
    q = "Should the heavy-quark pools take the leading-order width at each mass?"
    blocks = "the heavy-quark width rule; the massless b and the pool check go on"
    w = World([E("cov")])
    rc, out = w.plan_cmd("ask", "cov", "--ask", q, "--searched", SEARCHED, "--blocks", blocks)
    e = w.entries()["cov"]
    rc2, nxt = w.plan_cmd("next", "--prompts")
    scenario("open question: an ask with --blocks keeps the entry running, routed by plan next, and its step prompt "
             "names the question and what waits on it",
             rc == 0 and e["status"] == "running" and e["ask"] == q and e["blocks"] == blocks
             and e["asks"][-1]["blocks"] == blocks and rc2 == 1 and "NEXT cov: nothing in flight" in nxt
             and "Open question to the user" in nxt and q in nxt and "Waits on the answer: " + blocks in nxt,
             out + nxt[-1500:])
    w = World([E("cov", ask=q, blocks=blocks, waiting_since=ago(20), asks=[{"ask": q, "at": ago(20)}])],
              state={"cov": step(1)})
    rc, err = w.check()
    scenario("open question: a running entry's question must still reach the user", rc == 2 and "no message sent" in err,
             err)
    w = World([E("cov", ask=q, blocks=blocks, waiting_since=ago(20), asks=[{"ask": q, "at": ago(20)}])],
              state={"cov": step(1)},
              transcript=[(ago(19), "assistant", "Joaquin, should the heavy-quark pools take the leading-order width "
                                                 "at each mass?")])
    rc, err = w.check()
    scenario("open question: once sent, the running entry with steps in flight lets the turn end", rc == 0, err)
    say = [(ago(19), "assistant", "Joaquin, should the heavy-quark pools take the leading-order width at each mass?"),
           (ago(10), "user", "Yes, take the leading-order width at each mass")]
    w = World([E("cov", ask=q, blocks=blocks, waiting_since=ago(20), asks=[{"ask": q, "at": ago(20)}])],
              state={"cov": step(5)}, transcript=say)
    rc0, err0 = w.check()
    nxt0 = w.plan_cmd("next")[1]
    scenario("open question: a user reply after a running entry's question is listed by plan next (REPLY) and never "
             "blocks the stop", rc0 == 0 and "REPLY cov: the user wrote at" in nxt0, err0 + nxt0)
    rc, out = w.plan_cmd("answer", "cov", "--kind", "scope", "--quote", "Yes, take the leading-order width at each mass")
    e = w.entries()["cov"]
    nxt1 = w.plan_cmd("next")[1]
    pr = World([e]).plan_cmd("next", "--prompts")[1]
    scenario("open question: plan answer (words typed after the ask, before the entry's later step) records the "
             "decision and clears the question; the prompt shows none",
             rc == 0 and e["status"] == "running" and not any(k in e for k in ("ask", "waiting_since", "blocks"))
             and e["asks"][-1]["answered_at"] and e["asks"][-1]["decision"] == "D900" and "REPLY cov" not in nxt1
             and "Open question to the user (only what it blocks waits; CLAUDE.md 0g):\n  none" in pr, out + pr[-800:])
    rc, out = w.plan_cmd("ask", "cov", "--ask", "Should the heavy-quark pools take the leading-order width at each "
                         "mass point?", "--searched", SEARCHED)
    scenario("open question: once answered, asking it again is refused", rc != 0 and "already answered" in out, out)
    why = "every remaining step of the entry builds pools whose card the answer changes"
    w = World([E("cov")], transcript=[(ago(1), "assistant", "Joaquin, should the heavy-quark pools take the "
                                                            "leading-order width at each mass?")])
    rc, out = w.plan_cmd("ask", "cov", "--ask", q, "--searched", SEARCHED, "--holds-all", why)
    e = w.entries()["cov"]
    rc2, nxt = w.plan_cmd("next", "--prompts")
    rc3, out3 = w.plan_cmd("step", "cov", "--agent", "a9")
    scenario("open question: --holds-all keeps the old behaviour (the entry waits, plan next routes no step to it, a "
             "step is refused) and records why",
             rc == 0 and e["status"] == "waiting_user" and e["holds_all"] == why and "NEXT cov" not in nxt
             and rc3 != 0 and w.check()[0] == 0, out + nxt + out3)
    w = World([E("cov")])
    r1 = w.plan_cmd("ask", "cov", "--ask", q, "--searched", SEARCHED, "--holds-all", "it all waits")
    r2 = w.plan_cmd("ask", "cov", "--ask", q, "--searched", SEARCHED, "--holds-all", why, "--blocks", blocks)
    scenario("open question: --holds-all without a reason, or with --blocks, is refused and nothing is recorded",
             r1[0] != 0 and r2[0] != 0 and "ask" not in w.entries()["cov"], r1[1] + r2[1])
    w = World([E("cov", ask=q, blocks=blocks, waiting_since=ago(20), asks=[{"ask": q, "at": ago(20)}])])
    view, show = w.plan_cmd("view")[1], w.plan_cmd("show")[1]
    scenario("open question: plan view and show print a running entry's question and what waits on it",
             "Asked: Should the heavy" in view and "Waits on it: the heavy" in view and "open question" in show
             and "blocks: the heavy" in show, view + show)

DESIGN_GUARD = os.path.join(ROOT, ".claude", "hooks", "design_guard.py")
DESIGN = ["--question", "does the sign head lower the wrong-sign share on the held-out pools",
          "--prior", "the sign head search at one seed (results, sign head section): share 0.12 vs 0.30; this adds "
                     "seeds 1 and 2 and the matched pair head control",
          "--measure", "the wrong-sign share on the held-out pools at the best checkpoint",
          "--check", "on the two finished sign runs the share moved 4x between heads, above its binomial spread",
          "--fixed", "pools, horizon, optimizer, every other HP", "--control", "the same search with the pair head",
          "--seeds", "0 1 2", "--outcome", "yes: share lower by more than the seed spread; no: within it"]


def designs():
    """2026-10-26, the sign head search: jobs designed without first checking that their measurement responds to the
    tested change. A submission for a plan entry needs a design recorded within 24 h and passed by the design reviewer
    (design_guard.py); the reviewer's verdict is the design-reviewer subagent's alone."""
    w = World([E("sg"), E("other")])
    sub = lambda c, who=None: hook(DESIGN_GUARD, dict({"tool_name": "Bash", "tool_input": {"command": c}},
                                                      **({"agent_type": who} if who else {})), env=w.env)
    cmds = {"submit": "site submit auto FA scripts/job.sh --item sg --est-gpu-hours 2",
            "sweep": "site run ccin2p3 FA --item sg -- python sweep/sweep_manager.py submit /sps/x/tp3_sign",
            "generator": "site run ccin2p3 FA --item=sg -- python sweep/generate_sweep.py --config sweep/c.yaml "
                         "--auto-submit",
            "launch script": "NOSYNC=1 bash scripts/launch_cc_only.sh sg 2 note scripts/job.sh",
            "queued launch": "site launch add FA sg_l --item sg --run 'bash scripts/x.sh'"}
    for k, c in cmds.items():
        rc, out = sub(c)
        scenario("design: a %s for an entry with no design is refused, and says what to record" % k,
                 rc == 2 and "plan.py design" in out and "sg" in out, out)
    rc, out = w.plan_cmd("design", "sg", *DESIGN)
    e = w.entries()["sg"]
    scenario("design: plan design records every field with its time", rc == 0 and e["designs"][-1]["seeds"] == "0 1 2"
             and e["designs"][-1]["at"] and "design-reviewer" in out, out)
    rc, out = sub(cmds["submit"])
    scenario("design: a recorded design not yet reviewed does not let the job go, and the refusal says to run the "
             "reviewer", rc == 2 and "waits for the design reviewer" in out and "subagent_type design-reviewer" in out,
             out)
    nxt = w.plan_cmd("next", "--prompts")[1]
    scenario("design: a fresh design is its step agent's to review: plan next routes no review for it",
             "design to review" not in nxt and "The latest design on record" in nxt and "not reviewed yet" in nxt, nxt)
    # the fallback: a design left 40 min without a verdict, its step agent gone
    D40 = dict(zip([k[2:] for k in DESIGN[0::2]], DESIGN[1::2]), at=ago(40))
    wf = World([E("sg", designs=[D40])])
    rc, err = wf.check()
    rc2, nxt = wf.plan_cmd("next", "--prompts")
    scenario("design: fallback, a design without a verdict after 30 min is listed as a design to review, with the "
             "reviewer's prompt, and the stop check counts it like a step to launch",
             rc == 2 and "sg: design to review" in err and rc2 == 1 and "NEXT sg: design to review" in nxt
             and "Design review." in nxt and "Entry: sg" in nxt and "design-pass sg" in nxt, err + nxt)
    started = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    open(os.path.join(wf.ag, "agent-rv1.jsonl"), "w").write(J({"type": "user", "timestamp": started, "message": {
        "role": "user", "content": "Design review. Review it.\n\nEntry: sg (sg)\nQuestion of the entry: q"}}) + "\n")
    rs = wf.plan_cmd("step", "sg", "--agent", "rv1")
    rc, out = wf.plan_cmd("review", "sg", "--agent", "rv1")
    rc3, err3 = wf.check()
    scenario("design: fallback, the reviewer is recorded with plan review (never as a step) and, while it works, the "
             "entry waits on its verdict", rs[0] != 0 and rc == 0 and "design to review" not in err3,
             rs[1] + out + err3)
    for who in (None, "general-purpose"):
        rc, out = sub("python3 tools/plan.py design-pass sg --reviewer ok", who)
        scenario("design: design-pass from %s is refused" % (who or "the main session"), rc == 2
                 and "design-reviewer" in out, out)
    rc, out = sub("python3 tools/plan.py design-pass sg --reviewer ok", "design-reviewer")
    scenario("design: design-pass from the design-reviewer subagent passes the guard", rc == 0, out)
    w.plan_cmd("design-pass", "sg", "--reviewer", "control matched; share moved 4x on the sign runs")
    for k, c in cmds.items():
        rc, out = sub(c)
        scenario("design: the %s passes once the design passed review" % k, rc == 0, out)
    show, pr = w.plan_cmd("show")[1], w.plan_cmd("next", "--prompts")[1]
    scenario("design: plan show prints the latest design and its verdict, and no review is listed after a pass",
             "measure: the wrong-sign share" in show and "passed review" in show and "design to review" not in pr,
             show + pr[-1500:])
    rc, out = sub("site submit auto FA scripts/job.sh --item sg --item other")
    scenario("design: a submission naming a second entry without a design is refused", rc == 2 and "other" in out, out)
    w.plan_cmd("design", "sg", *DESIGN)
    rc, out = sub(cmds["submit"])
    scenario("design: a new design resets the pass", rc == 2 and "waits for the design reviewer" in out, out)
    rc, out = w.plan_cmd("design-fix", "sg", "--reasons", "the control differs in the optimizer too")
    rc2, out2 = sub(cmds["submit"])
    pr = w.plan_cmd("next", "--prompts")[1]
    scenario("design: a FIX keeps the job refused, ends the review listing, and the next step prompt carries the "
             "design and the reasons", rc == 0 and rc2 == 2 and "sent back" in out2 and "design to review" not in pr
             and "The latest design on record" in pr and "control: the same search with the pair head" in pr
             and "review asked for a fix" in pr and "the optimizer too" in pr, out + out2 + pr[-1500:])
    w2 = World([E("sg", designs=[dict(zip([k[2:] for k in DESIGN[0::2]], DESIGN[1::2]), at=ago(25 * 60),
                                      passed_at=ago(25 * 60 - 5), reviewer="r")])])
    rc, out = hook(DESIGN_GUARD, {"tool_name": "Bash", "tool_input": {"command": cmds["submit"]}}, env=w2.env)
    scenario("design: a passed design older than 24 h does not cover a new submission", rc == 2
             and "more than 24 h ago" in out, out)
    # an eval-only re-read of an arm that already ran: REREAD=<run id of the entry>, under its passed design (any age)
    old = dict(D40, at=ago(30 * 60), passed_at=ago(30 * 60 - 5), reviewer="r")
    wr = World([E("sg", designs=[old]), E("nd", designs=[dict(D40, at=ago(30 * 60))])],
               runs=[("FA-ccin2p3-r1", "COMPLETED", "sg", epoch(600), epoch(500)),
                     ("FA-ccin2p3-r2", "COMPLETED", "zz", epoch(600), epoch(500)),
                     ("FA-ccin2p3-r3", "COMPLETED", "nd", epoch(600), epoch(500))])
    rr = lambda c: hook(DESIGN_GUARD, {"tool_name": "Bash", "tool_input": {"command": c}}, env=wr.env)
    ok = rr("REREAD=FA-ccin2p3-r1 site submit auto FA scripts/eval.sh --item sg")
    other = rr("REREAD=FA-ccin2p3-r2 site submit auto FA scripts/eval.sh --item sg")
    unpassed = rr("REREAD=FA-ccin2p3-r3 site submit auto FA scripts/eval.sh --item nd")
    plain = rr("site submit auto FA scripts/eval.sh --item sg")
    scenario("design: a re-read (REREAD=<run of the entry>) under a passed design passes; another entry's run, an "
             "unpassed design, or no marker is refused", ok[0] == 0 and other[0] == 2 and unpassed[0] == 2
             and plain[0] == 2, ok[1] + other[1] + unpassed[1] + plain[1])
    for c in ("site run ccin2p3 FA --item '*' -- python sweep/sweep_manager.py submit /sps/x/a",
              "site submit auto FA scripts/job.sh --item '*'", "site submit auto FA scripts/job.sh --item not_in_plan",
              "site run ccin2p3 FA --item sg -- python sweep/generate_sweep.py --config c.yaml --dry-run",
              "site logs FA-ccin2p3-1 && site poll", "site status ccin2p3", "cat scripts/launch_cc_only.sh",
              "git commit -m 'site submit auto FA x --item sg'", "python3 tools/plan.py show"):
        rc, out = sub(c)
        scenario("design: `%s` passes (upkeep, no plan entry, or no submission)" % c[:60], rc == 0, out)
    w3 = World([E("sg")], state=None)
    os.remove(w3.plan)
    open(w3.plan, "w").write("entries: [: broken")
    r3 = hook(DESIGN_GUARD, {"tool_name": "Bash", "tool_input": {"command": cmds["submit"]}}, env=w3.env)
    r4 = hook(DESIGN_GUARD, {"tool_name": "Bash", "tool_input": {"command": "site poll"}}, env=w3.env)
    scenario("design: an error while checking a submission refuses it; a non-submission still passes",
             r3[0] == 2 and "could not check" in r3[1] and r4[0] == 0, r3[1] + r4[1])
    for i in range(1, len(DESIGN), 2):
        args = list(DESIGN)
        args[i] = "   "
        rc, out = w.plan_cmd("design", "other", *args)
        scenario("design: plan design refuses an empty %s" % DESIGN[i - 1], rc != 0 and DESIGN[i - 1] in out, out)
    args = list(DESIGN)
    args[DESIGN.index("--check") + 1] = "checked"
    rc, out = w.plan_cmd("design", "other", *args)
    scenario("design: a check that does not say how it was verified is refused, nothing is recorded",
             rc != 0 and "existing data" in out and "designs" not in w.entries()["other"], out)


def design_prior():
    """2026-10-26: a pp-trained arm existed and was written up, the entry's decisions named it as the reference, yet a
    comparison was written without it and a rerun proposed ("you completely ignored previous work done ... and you
    were about to run the same thing again"). A design names the prior work on its question, or the terms searched
    to find none; the design reviewer searches again itself."""
    w = World([E("pp", quotes=[{"said_at": ago(60), "text": "build it, run a few checks", "decision": "D117"}])])
    i = DESIGN.index("--prior") + 1
    for k, v in (("an empty --prior", "   "), ("a --prior too short to name a result", "the earlier run"),
                 ("none found without search terms", "none found"), ("none found with one term", "none found: pp")):
        args = list(DESIGN)
        args[i] = v
        rc, out = w.plan_cmd("design", "pp", *args)
        scenario("design prior: %s is refused, nothing is recorded" % k,
                 rc != 0 and "--prior" in out and "designs" not in w.entries()["pp"], out)
    args = list(DESIGN)
    args[i] = "none found; searched pp-trained, trained on pp, pp arm, catalog_v4 pp pools"
    rc, out = w.plan_cmd("design", "pp", *args)
    scenario("design prior: none found with the search terms is recorded", rc == 0
             and "pp-trained" in w.entries()["pp"]["designs"][-1]["prior"], out)
    args = list(DESIGN)
    args[i] = "none of the AdamW seeds of the pp-trained arm reached the e+e- arm's loss (par:v4_vs_pp)"
    rc, out = w.plan_cmd("design", "pp", *args)
    scenario("design prior: a substantive prior opening with a bare 'none' is checked by length, not as a none found",
             rc == 0 and "AdamW seeds" in w.entries()["pp"]["designs"][-1]["prior"], out)
    n0 = len(w.entries()["pp"]["designs"])
    args[i] = "none of it: pp arm"
    rc, out = w.plan_cmd("design", "pp", *args)
    scenario("design prior: a short prior opening with a bare 'none' is refused by the length rule",
             rc != 0 and "at least" in out and "characters" in out and len(w.entries()["pp"]["designs"]) == n0, out)
    rc, out = w.plan_cmd("design", "pp", *DESIGN)
    show = w.plan_cmd("show", "pp")[1]
    pr = w.plan_cmd("next", "--prompts")[1]
    scenario("design prior: the prior work is shown by plan show and carried by the step prompt",
             rc == 0 and "prior: the sign head search at one seed" in show
             and "prior: the sign head search at one seed" in pr, show + pr[-2000:])
    tpl = open(os.path.join(ROOT, "plan", "step_prompt.txt")).read()
    rev = open(os.path.join(ROOT, ".claude", "agents", "design-reviewer.md")).read()
    scenario("design prior: the step prompt asks for --prior and, in a write-up answering a decision, each quote's "
             "result; the reviewer searches on its own and checks every quoted comparison",
             "--prior" in tpl and "names a comparison" in tpl and "searched independently" in rev
             and "grep -n -i" in rev and "site decisions --project FA" in rev and "names a comparison" in rev)
    wf = World([E("pp", quotes=[{"said_at": ago(90), "text": "compare against the pp-trained arm",
                                 "decision": "D119"}],
                  designs=[dict(zip([k[2:] for k in DESIGN[0::2]], DESIGN[1::2]), at=ago(40))])])
    nxt = wf.plan_cmd("next", "--prompts")[1]
    scenario("design prior: the reviewer's fallback prompt carries the prior field and the user's quoted words",
             "Design review." in nxt and "prior: the sign head search" in nxt
             and "compare against the pp-trained arm" in nxt, nxt[-2500:])
    old = World([E("pp", designs=[dict(zip([k[2:] for k in DESIGN[0::2] if k != "--prior"],
                                           [v for k, v in zip(DESIGN[0::2], DESIGN[1::2]) if k != "--prior"]),
                                       at=ago(60))])])
    show = old.plan_cmd("show", "pp")[1]
    scenario("design prior: a design recorded before the field shows it as not recorded",
             "prior: not recorded" in show, show)


def prompt_cap():
    """A step prompt carries the last 8 dead ends and evidence runs / paths, with a line naming how many earlier ones
    exist and where they are (catalog_coverage's whole history made one prompt 17k characters); plan show <id> prints
    them all. A mechanical collect step is told to use tools/collect_run.py."""
    des = [{"tried": "tried %02d" % i, "why": "why %02d" % i, "at": ago(60 - i)} for i in range(12)]
    runs = ["FA-ccin2p3-r%02d" % i for i in range(20)]
    w = World([E("cov", dead_ends=des, evidence={"runs": runs, "commit": "abc", "paths": ["p1", "p2"]})])
    pr = w.plan_cmd("next", "--prompts")[1]
    scenario("step prompt: only the last 8 dead ends and evidence runs, with how many earlier ones and plan show",
             "tried 11: why 11" in pr and "tried 04: why 04" in pr and "tried 03" not in pr
             and "(4 earlier dead ends not shown: python3 tools/plan.py show cov)" in pr
             and "FA-ccin2p3-r19" in pr and "FA-ccin2p3-r12" in pr and "FA-ccin2p3-r11" not in pr
             and "(12 earlier runs not shown: python3 tools/plan.py show cov)" in pr and '"p1"' in pr
             and "earlier paths" not in pr, pr[:2500])
    show = w.plan_cmd("show", "cov")[1]
    scenario("plan show <id> prints that entry's every dead end and its whole evidence",
             "tried 00" in show and "FA-ccin2p3-r00" in show and "FA-ccin2p3-r19" in show, show[-1500:])
    tpl = open(os.path.join(ROOT, "plan", "step_prompt.txt")).read()
    scenario("the step prompt sends a mechanical collect step to tools/collect_run.py",
             "tools/collect_run.py" in tpl and os.path.exists(os.path.join(ROOT, "tools", "collect_run.py")))


def parallel_steps():
    """Independent next steps run in parallel (the user, 2026-10-26: "what is the point of setting up all of this to
    do parallel work if you are going to queue stuff like that?"): an entry holds several numbered next steps, each
    listed, launched and in flight on its own, so a free step is listed while another step of the entry runs."""
    two = {1: "analyse the density gap", 2: "submit the pp-trained Muon arm"}
    w = World([E("par", next=dict(two))])
    rc, out = w.check()
    nx = w.plan_cmd("next")[1]
    scenario("two independent next steps, nothing in flight: both listed, and the stop check counts both",
             rc == 2 and "par step 1" in out and "par step 2" in out and "--next 2" in out
             and "NEXT par step 1 of 2" in nx and "NEXT par step 2 of 2" in nx and "2 step(s) to launch" in nx, out + nx)
    w = World([E("par", next=dict(two))], state={"par": {"steps": [{"agent": "w1", "at": ago(10), "next": 1}]}})
    rc, out = w.check()
    nx = w.plan_cmd("next")[1]
    scenario("step 1's agent works: step 2 is still listed (a free step is never queued behind it)",
             rc == 2 and "par step 2" in out and "par step 1" not in out and "1 step(s) to launch" in nx, out + nx)
    w = World([E("par", next=dict(two))], state={"par": {"steps": [{"agent": "w1", "at": ago(10), "next": 1},
                                                                   {"agent": "w2", "at": ago(5), "next": 2}]}})
    scenario("both steps' agents work: nothing to launch", w.check()[0] == 0, w.check()[1])
    w = World([E("par", next=dict(two))], runs=[("FA-ccin2p3-j2", "RUNNING", "par", epoch(30), None)],
              state={"par": {"runs": ["FA-ccin2p3-j2"], "run_steps": {"FA-ccin2p3-j2": 2}}})
    nx = w.plan_cmd("next")[1]
    scenario("a live job recorded on step 2 holds step 2 only; step 1 is listed",
             "NEXT par step 1 of 2" in nx and "step 2 of 2" not in nx, nx)
    w = World([E("par", next=dict(two))], runs=[("FA-ccin2p3-j0", "RUNNING", "par", epoch(30), None)])
    nx = w.plan_cmd("next")[1]
    scenario("a live job recorded without a step number belongs to the first step: step 2 is listed",
             "NEXT par step 2 of 2" in nx and "step 1 of 2" not in nx, nx)
    w = World([E("par", next=dict(two))], runs=[("FA-ccin2p3-j2", "COMPLETED", "par", epoch(60), epoch(5))],
              state={"par": {"runs": ["FA-ccin2p3-j2"], "run_steps": {"FA-ccin2p3-j2": 2},
                             "steps": [{"agent": "w1", "at": ago(10), "next": 1}]}})
    nx = w.plan_cmd("next")[1]
    scenario("step 2's job ended while step 1's agent works: step 2 is listed for the ended run, step 1 is not",
             "NEXT par step 2 of 2" in nx and "runs ended: FA-ccin2p3-j2" in nx and "step 1 of 2" not in nx, nx)
    # the prompts: each names its own step, the other as running in parallel, and the no-chaining rule
    w = World([E("par", next=dict(two))])
    pr = w.plan_cmd("next", "--prompts")[1]
    p2 = pr[pr.index("NEXT par step 2 of 2"):]
    scenario("each listed step has its own prompt: its step as {next}, the other named as running in parallel",
             "Your step (number 1 of the entry's next steps): analyse the density gap" in pr
             and "Your step (number 2 of the entry's next steps): submit the pp-trained Muon arm" in p2
             and "  1: analyse the density gap" in p2 and "set-next par --next 2" in p2 and "run par <run id> --next 2"
             in p2 and 'never chained with "then"' in p2, pr[:3000])
    # recording: plan step needs --next on a several-step entry, names an existing step, and matches the prompt
    w = World([E("par", next=dict(two))], agents=[("a1", ago(2), "par"), ("a2", ago(2), "par"), ("a3", ago(2), "par")])
    rc1, o1 = w.plan_cmd("step", "par", "--agent", "a1")
    rc2, o2 = w.plan_cmd("step", "par", "--agent", "a1", "--next", "3")
    rc3, o3 = w.plan_cmd("step", "par", "--agent", "a1", "--next", "2")
    rec = json.load(open(w.state))["par"]["steps"]
    nx = w.plan_cmd("next")[1]
    scenario("plan step: --next needed with several steps, refused for a step the entry lacks, recorded on its step",
             rc1 != 0 and "--next N" in o1 and rc2 != 0 and "no next step 3" in o2 and rc3 == 0
             and rec[-1].get("next") == 2 and "NEXT par step 1 of 2" in nx and "step 2 of 2" not in nx,
             o1 + o2 + o3 + nx)
    pa = os.path.join(w.ag, "agent-a2.jsonl")
    m = json.loads(open(pa).readline())
    m["message"]["content"] += "\nYour step (number 1 of the entry's next steps): analyse the density gap"
    open(pa, "w").write(json.dumps(m) + "\n")
    rc4, o4 = w.plan_cmd("step", "par", "--agent", "a2", "--next", "2")
    scenario("plan step refuses an agent started with another step's prompt", rc4 != 0 and "step 1, not 2" in o4, o4)
    w.plan_cmd("run", "par", "FA-ccin2p3-x9", "--next", "2")
    scenario("plan run --next records which step the job belongs to",
             json.load(open(w.state))["par"].get("run_steps") == {"FA-ccin2p3-x9": 2})
    # set-next / add-next / drop-next: stable numbers, a step agent touches only its own step
    w = World([E("par", next="one thing")], state={"par": {"steps": [{"agent": "w1", "at": ago(10)}]}})
    scenario("an entry holding one string is in flight while its step agent (recorded without a number) works",
             w.check()[0] == 0, w.check()[1])
    w.plan_cmd("set-next", "par", "first", *WHY, "--also", "second", *WHY, "--also", "third", *WHY)
    a = w.entries()["par"]["next"]
    w.plan_cmd("set-next", "par", "second, then its follow-up", "--next", "2", *WHY)
    b = w.entries()["par"]["next"]
    w.plan_cmd("drop-next", "par", "1")
    w.plan_cmd("add-next", "par", "fourth", *WHY)
    c = w.entries()["par"]["next"]
    show = w.plan_cmd("show", "par")[1]
    scenario("set-next --also holds independent steps; --next N replaces only N; drop and add keep the numbers",
             a == {1: "first", 2: "second", 3: "third"} and b == {1: "first", 2: "second, then its follow-up",
                                                                  3: "third"}
             and c == {2: "second, then its follow-up", 3: "third", 4: "fourth"} and "next 4: fourth" in show,
             [a, b, c, show])
    w.plan_cmd("drop-next", "par", "2"); w.plan_cmd("drop-next", "par", "3")
    d1 = w.entries()["par"]["next"]
    w.plan_cmd("set-next", "par", "only", *WHY)
    d2 = w.entries()["par"]["next"]
    rc5, o5 = w.plan_cmd("drop-next", "par", "7")
    w.plan_cmd("set-next", "par", "x", *WHY, "--also", "y", *WHY)
    d3 = w.entries()["par"]["next"]
    scenario("one step left keeps its number, a plain set-next replaces it in place, --also numbers past the records "
             "(none here: from 1), a missing step refused",
             d1 == {4: "fourth"} and d2 == {4: "only"} and d3 == {1: "x", 2: "y"} and rc5 != 0, [d1, d2, d3, o5])
    w = World([E("par", next={1: "a", 2: "b"})], state={"par": {"steps": [{"agent": "w9", "at": ago(10), "next": 2}]}})
    w.plan_cmd("drop-next", "par", "2")
    w.plan_cmd("add-next", "par", "c", *WHY)
    scenario("a new step never reuses the number of a dropped step an agent record still names",
             w.entries()["par"]["next"] == {1: "a", 3: "c"}, w.entries()["par"]["next"])
    w = World([E("par", next={1: "A", 2: "B"})], state={"par": {"steps": [{"agent": "w8", "at": ago(10), "next": 2}]}})
    w.plan_cmd("set-next", "par", "C", *WHY, "--also", "D", *WHY)
    nx = w.plan_cmd("next")[1]
    scenario("a plain set-next --also over numbered records numbers past them: an old agent never holds a new step",
             w.entries()["par"]["next"] == {3: "C", 4: "D"} and "NEXT par step 3 of 2" in nx and "step 4 of 2" in nx, [w.entries()["par"]["next"], nx])
    # feeds and branches: required on set-next / add-next, stored per step, shown in the prompt and plan show
    w = World([E("fb", next="legacy step")])
    pr0 = w.plan_cmd("next", "--prompts")[1]
    sh0 = w.plan_cmd("show", "fb")[1]
    scenario("a step set before feeds and branches stays readable; its prompt and plan show flag it",
             "Your step (number 1 of the entry's next steps): legacy step" in pr0
             and "no feeds/branches recorded: record them with plan set-next before working" in pr0
             and "no feeds/branches recorded" in sh0 and "next: legacy step" in sh0, pr0[:2000] + sh0)
    r1 = w.plan_cmd("set-next", "fb", "measure the gap")
    r2 = w.plan_cmd("set-next", "fb", "measure the gap", "--feeds", "whether the gap closes at full horizon",
                    "--if", "it closes: run the long arm")
    r3 = w.plan_cmd("add-next", "fb", "a side check", "--feeds", "whether the gap closes at full horizon",
                    "--if", "it closes: run the long arm", "--if", "it stays:")
    r4 = w.plan_cmd("set-next", "fb", "A", *WHY, "--also", "B")
    scenario("set-next and add-next refuse a step without --feeds, with one branch, with a branch naming no action, "
             "or an --also step without its own; nothing is written",
             all(r[0] != 0 for r in (r1, r2, r3, r4)) and "--feeds" in r1[1] and "at least two --if" in r1[1]
             and "(got 1)" in r2[1] and "it stays:" in r3[1] and "'B'" in r4[1]
             and w.entries()["fb"]["next"] == "legacy step", [r1, r2, r3, r4])
    rc, o = w.plan_cmd("set-next", "fb", "measure the gap", "--feeds", "whether the gap closes at full horizon",
                       "--if", "it closes: run the long arm", "--if", "it stays: write it up as a floor",
                       "--also", "the side check", *WHY)
    e = w.entries()["fb"]
    pr = w.plan_cmd("next", "--prompts")[1]
    sh = w.plan_cmd("show", "fb")[1]
    p2 = pr[pr.index("NEXT fb step 2 of 2"):]
    scenario("a step's feeds and branches are stored with it and shown in its own prompt and in plan show",
             rc == 0 and e["step_why"][1]["feeds"] == "whether the gap closes at full horizon"
             and e["step_why"][2]["if"] == list(WHY[3::2])
             and "Feeds (the decision of the entry's question this step settles): whether the gap closes" in pr
             and "  if it stays: write it up as a floor" in pr
             and "Feeds (the decision of the entry's question this step settles): whether the arm is adopted" in p2
             and "  if it loses: drop it and record a dead end" in p2
             and "Your step serves the decision under 'Feeds'" in pr and "no feeds/branches recorded" not in pr
             and "feeds: whether the gap closes at full horizon" in sh and "if it wins: adopt it in the template" in sh,
             o + pr[:2500] + sh)
    w.plan_cmd("drop-next", "fb", "1")
    w.plan_cmd("add-next", "fb", "a third", "--feeds", "which floor the long arm reaches",
               "--if", "below 1e-6: adopt", "--if", "above: stop and report")
    e = w.entries()["fb"]
    scenario("drop-next takes the step's feeds with it; add-next stores its own",
             set(e["step_why"]) == {2, 3} and e["step_why"][3]["if"] == ["below 1e-6: adopt", "above: stop and report"],
             e.get("step_why"))


SEND = os.path.join(ROOT, ".claude", "hooks", "send_check.py")


def sent_figures():
    """send_check.py (2026-10-27, the user: "no cryptic unexplained names"): a figure or deck sent with SendUserFile
    is refused when its text holds a code; a glossary term passes; a clean figure passes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "docs"))

    def fig(name, xlabel, ylabel="median MSE", legend=None):
        f, ax = plt.subplots(figsize=(3, 2))
        ax.plot([1, 2], [1, 2], label=legend)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        f.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight"); f.savefig(os.path.join(d, name + ".png"))
        plt.close(f)
        return os.path.join(d, name + ".pdf")

    def send(path, glossary=""):
        open(os.path.join(d, "docs", "glossary.txt"), "w").write(glossary)
        r = subprocess.run([sys.executable, SEND], input=J({"tool_name": "SendUserFile", "cwd": d,
                                                              "tool_input": {"files": [path]}}),
                           capture_output=True, text=True, env=dict(os.environ, CLAUDE_PROJECT_DIR=d))
        return r.returncode, r.stderr

    rc, err = send(fig("arm", "training events", legend="tp3_baseline"))
    scenario("a figure with an arm name is refused, naming the string and the page",
             rc == 2 and "'tp3_baseline'" in err and "page 1" in err and "recipe or arm prefix" in err, err)
    rc, err = send(fig("codes", "held-out axis H2p5, my call D145"))
    scenario("held-out axis and decision codes are refused", rc == 2 and "'H2p5'" in err and "'D145'" in err, err)
    stub = os.path.join(d, "stub.txt")
    open(stub, "w").write("from FA-ccin2p3-cb09b906, trial_0012 at 3abde40c\nconfig val_loss_no_reg\n")
    r = subprocess.run([sys.executable, SEND, "--scan", stub], capture_output=True, text=True,
                       env=dict(os.environ, CLAUDE_PROJECT_DIR=d))
    scenario("run ids, trial names, hashes and snake_case identifiers are found (text stub)",
             r.returncode == 1 and all(x in r.stdout for x in ("'FA-ccin2p3-cb09b906'", "'trial_0012'", "'3abde40c'",
                                                               "'val_loss_no_reg'")), r.stdout)
    rc, err = send(fig("ratio", "error ratio"))
    rc2, err2 = send(fig("ratio_ok", "error, fine-tune over scratch"))
    rc3, err3 = send(fig("slash_ok", "fine-tune / scratch"))
    scenario("a ratio that does not say what over what is refused; one that does passes",
             rc == 2 and "ratio or gain" in err and rc2 == 0 and rc3 == 0, [err, err2, err3])
    p = fig("gloss", "catalog_v3, seed 42")
    rc, err = send(p)
    rc2, err2 = send(p, "catalog_v3\tthe third catalog\n")
    scenario("a code is refused until the glossary lists it, then passes", rc == 2 and rc2 == 0, [err, err2])
    rc3, err3 = send(p, "# PROPOSED, NOT YET ACCEPTED\ncatalog_v3\tthe third catalog\n")
    scenario("a term below the glossary's PROPOSED marker is still refused", rc3 == 2, err3)
    rc, err = send(fig("clean", "training events D", "MSE, dimensionless target", legend="with the context vector"))
    scenario("a clean figure passes", rc == 0, err)
    rc, err = send(os.path.join(d, "clean.png"))
    os.remove(os.path.join(d, "arm.pdf"))
    rc2, err2 = send(os.path.join(d, "arm.png"))
    scenario("a .png is read through its sibling .pdf, and refused without one",
             rc == 0 and rc2 == 2 and "no sibling .pdf" in err2, [err, err2])
    r = subprocess.run([sys.executable, SEND], input=J({"tool_name": "Read", "tool_input": {"file_path": p}}),
                       capture_output=True, text=True, env=dict(os.environ, CLAUDE_PROJECT_DIR=d))
    scenario("another tool is left alone", r.returncode == 0, r.stderr)
    shutil.rmtree(d)


def the_suite_itself():
    """The exit code reflects every scenario, and every group of scenarios is run (a group defined but never called
    once let the suite exit 0 with failures: main() was split by an edit)."""
    import ast
    tree = ast.parse(open(os.path.abspath(__file__)).read())
    groups = {f.name for f in tree.body if isinstance(f, ast.FunctionDef) and any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "scenario" for n in ast.walk(f))}
    listed = {g.__name__ for g in GROUPS}
    scenario("every function that runs scenarios is in GROUPS", groups <= listed, sorted(groups - listed))
    code = ("import sys; sys.path.insert(0, %r); import test_setup_scenarios as t; "
            "t.GROUPS[:] = [lambda: t.scenario('a failing scenario', False)]; sys.exit(t.main())"
            % os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    scenario("a failing scenario makes the suite exit 1", r.returncode == 1, r.stdout + r.stderr)


def reviewer_resume():
    """reviewer_resume.py: a blocking review's re-check continues the same reviewer; a fresh one goes after release."""
    d = tempfile.mkdtemp(prefix="resume_"); st = os.path.join(d, ".claude", ".review_state"); os.makedirs(st)
    hook = os.path.join(ROOT, ".claude", "hooks", "reviewer_resume.py")
    env = dict(os.environ, CLAUDE_PROJECT_DIR=d)
    run = lambda ev: subprocess.run(["python3", hook], input=json.dumps(ev), capture_output=True, text=True, env=env)
    launch = {"tool_name": "Agent", "tool_input": {"subagent_type": "repo-reviewer", "prompt": "review"}}
    run(dict(launch, hook_event_name="PostToolUse", tool_response={"content": "agentId: abc123 (internal)"}))
    open(os.path.join(st, "repo-reviewer.lock"), "w").close()
    r = run(dict(launch, hook_event_name="PreToolUse"))
    scenario("a fresh reviewer is refused while the pillar's lock is held, naming the one to continue",
             r.returncode == 2 and "abc123" in r.stderr, r.stderr)
    os.remove(os.path.join(st, "repo-reviewer.lock"))
    r2 = run(dict(launch, hook_event_name="PreToolUse"))
    r3 = run({"tool_name": "Agent", "hook_event_name": "PreToolUse", "tool_input": {"subagent_type": "general-purpose"}})
    scenario("after release a fresh reviewer goes; other agents are never touched", r2.returncode == 0 and r3.returncode == 0,
             [r2.stderr, r3.stderr])


CC_NOTICE = ("The main session does not operate on CC-IN2P3 while its login notice forbids generative AI tools, and "
             "the regeneration waits on it: should I ask CC for an exception?")


def idle_wait():
    """2026-10-29: every entry waited on its own question for 7 hours with the supervisor stopped; the stop check
    refuses that idle turn."""
    held = lambda eid: E(eid, status="waiting_user", ask=CC_NOTICE, waiting_since=ago(20), holds_all="x" * 40)
    sent = [(ago(19), "assistant", "Joaquin, " + CC_NOTICE)]
    w = World([held("a"), held("b")], transcript=sent, supervisor=False)
    rc, err = w.check()
    scenario("the stop check blocks when every entry waits on the user and no supervisor runs",
             rc == 2 and "site supervise" in err and "every live entry waits on the user" in err, err)
    rc, err = World([held("a"), held("b")], transcript=sent).check()
    scenario("with the supervisor running it passes", rc == 0, err)
    part = E("c", ask=CC_NOTICE, waiting_since=ago(20), next="regenerate the pp pools on CC-IN2P3",
             blocks="regenerate the pp pools on CC-IN2P3")
    w = World([held("a"), part], transcript=sent, state={"c": step(1)}, supervisor=False)
    rc, err = w.check()
    scenario("a running entry whose question blocks all its next steps counts as waiting",
             rc == 2 and "every live entry waits" in err, err)
    w = World([held("a"), E("d")], transcript=sent, state={"d": step(1)}, supervisor=False)
    rc, err = w.check()
    scenario("one entry that can move keeps the turn free of that block", "every live entry waits" not in err, err)


def main_session_work():
    """The main session works no entry itself (0i) beyond 3 work calls per user message (main_session_guard); without a
    transcript the budget is full, so the table below is the classification. A subagent's calls pass and never count.
    The subagent marker is agent_id/agent_type, as in the hook inputs lock_guard logs."""
    sub = {"agent_id": "a1b2", "agent_type": "general-purpose"}
    scratch = "/tmp/claude-1000/x/scratchpad"
    for who, tool, ti, want in (
            ("main", "Bash", {"command": "site submit ccin2p3 FA scripts/job_probe.sh --item lowd"}, 2),
            ("sub", "Bash", {"command": "site submit ccin2p3 FA scripts/job_probe.sh --item lowd"}, 0),
            ("main", "Bash", {"command": "site poll"}, 0),
            ("main", "Edit", {"file_path": "/home/joaquin/work/FA/tools/plan.py"}, 2),
            ("main", "Edit", {"file_path": "/home/joaquin/work/FA/CLAUDE.md"}, 0),
            ("sub", "Edit", {"file_path": "/home/joaquin/work/FA/tools/plan.py"}, 0),
            ("main", "Write", {"file_path": scratch + "/note.txt"}, 0),
            ("main", "Bash", {"command": "NOSYNC=1 site sync ccin2p3 FA && site runs"}, 2),
            ("main", "Bash", {"command": "cd /home/joaquin/work/FA && sbatch x.sh"}, 2),
            ("main", "Bash", {"command": ".venv-cpu/bin/pytest tests/"}, 2),
            ("main", "Bash", {"command": "python3 tools/plan.py next --prompts"}, 0),
            ("main", "Bash", {"command": "python3 tools/agent_costs.py"}, 0),
            ("main", "Bash", {"command": "bash -c 'site run ccin2p3 FA -- ls'"}, 2),
            ("main", "Bash", {"command": "sed -i s/a/b/ tools/plan.py"}, 2),
            ("main", "Bash", {"command": "echo x > tools/plan.py"}, 2),
            ("main", "Bash", {"command": "git commit -m \"$(cat <<'EOF'\nno site submit here > x\nEOF\n)\" && git push"}, 0),
            ("main", "Bash", {"command": "git log --oneline -3 2>&1 | head; grep -n site tools/plan.py > /dev/null"}, 0),
            ("main", "Bash", {"command": "bash .claude/hooks/review_backlog.sh status"}, 0),
            ("main", "Bash", {"command": "site events --wait --project FA"}, 0)):
        inp = dict(sub if who == "sub" else {}, tool_name=tool, tool_input=ti, scratchpad_dir=scratch)
        rc, out = hook(MAIN_GUARD, inp)
        scenario("main session guard: %s %s `%s` is %s" % (who, tool, list(ti.values())[0][:50].replace("\n", " "),
                 "refused" if want else "allowed"), rc == want and (not want or "step agent's work" in out), out)

    # the budget: 3 work calls per user message (the user, 2026-10-29: "you should still be able to do those small
    # things in some situations"), counted from the transcript since the user's last message
    import tempfile
    d = tempfile.mkdtemp()
    tp = os.path.join(d, "t.jsonl")
    k = [0]

    def user(text, **kw):
        return dict(kw, type="user", message={"role": "user", "content": text})

    def call(name, ti, sidechain=False):
        k[0] += 1
        return {"type": "assistant", "isSidechain": sidechain, "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu%d" % k[0], "name": name, "input": ti}]}}

    def result():
        return {"type": "user", "toolUseResult": {}, "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu%d" % k[0], "content": "ok"}]}}

    edit = {"file_path": "/home/joaquin/work/FA/tools/plan.py"}
    sub_work = ("Bash", {"command": "site sync ccin2p3 FA"})

    def probe(records, inp_extra=None):
        with open(tp, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        inp = dict(inp_extra or {}, tool_name="Edit", tool_input=edit, scratchpad_dir=scratch, transcript_path=tp,
                   tool_use_id="tu_now")
        return hook(MAIN_GUARD, inp)

    def works(n):
        out = []
        for _ in range(n):
            out += [call("Edit", edit), result()]
        return out

    noise = [user("<task-notification>agent done</task-notification>"), user("Stop hook feedback: x"),
             user("<system-reminder>r</system-reminder>", isMeta=True)]
    for name, recs, want in (
            ("the 1st work call after the user's message passes", [user("do it")], 0),
            ("the 3rd work call passes", [user("do it")] + works(2), 0),
            ("the 4th work call is refused", [user("do it")] + works(3), 2),
            ("a new user message resets the count", [user("do it")] + works(3) + [user("now this")] + works(2), 0),
            ("a mid-turn user message resets the count", [user("do it")] + works(3) + [{
                "type": "user", "toolUseResult": {}, "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "x", "content":
                    "The user sent a new message while you were working:\nalso this"}]}}], 0),
            ("task notifications, hook feedback and reminders are not user messages",
             [user("do it")] + works(1) + noise[:1] + works(1) + noise[1:] + works(1), 2),
            ("reads never count", [user("do it")] + works(2) + [call("Bash", {"command": "site poll"}), result(),
              call("Read", {"file_path": "x"}), result(), call("Bash", {"command": "git log -3"}), result()], 0),
            ("subagent (sidechain) calls never count", [user("do it")] + works(2) + [
                call(*sub_work, sidechain=True), result(), call(*sub_work, sidechain=True)], 0),
            ("the current call is not counted against itself", [user("do it")] + works(2) + [
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "tu_now", "name": "Edit",
                                                               "input": edit}]}}], 0),
            ("parallel work calls in one message each count", [user("do it"), {"type": "assistant", "message": {
                "content": [{"type": "tool_use", "id": "p%d" % i, "name": "Edit", "input": edit} for i in range(3)]}}],
             2)):
        rc, out = probe(recs)
        scenario("main session budget: %s" % name, rc == want and (not want or "3 work calls already" in out), out)
    rc, out = probe([user("do it")] + works(5), {"agent_id": "a1", "agent_type": "general-purpose"})
    scenario("main session budget: a subagent's call passes whatever the main session used", rc == 0, out)


def duplicate_check():
    """The commit check of near-duplicate functions (tools/dupcheck.py in .githooks/pre-commit, the user's 2026-10-29
    call on one-use scripts): a staged copy of an existing loader (variables renamed, other paths) is refused naming
    the original; a new distinct function, and a copy with `# dup-ok: <reason>` on its def line, pass; editing one of
    a pair that was already duplicated is not refused for that old pair. The step prompt carries analysis/common's
    index and the line on reuse."""
    repo = tempfile.mkdtemp(prefix="dup_")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q")
    for d in ("analysis", "tools", ".githooks"):
        os.makedirs(os.path.join(repo, d))
    for f in ("tools/dupcheck.py", "tools/verify_claims.py", ".githooks/pre-commit"):
        shutil.copy(os.path.join(ROOT, f), os.path.join(repo, f))
    g("config", "core.hooksPath", ".githooks")
    loader = ("import glob, json, os\n\n\ndef load_runs(path):\n"
              "    out = {}\n"
              "    for f in sorted(glob.glob(os.path.join(path, \"results\", \"*.json\"))):\n"
              "        r = json.load(open(f))\n"
              "        if r.get(\"val_loss\") is None:\n"
              "            continue\n"
              "        key = os.path.basename(f)[:-5]\n"
              "        out[key] = r[\"val_loss\"] * r[\"prepd_std\"] ** 2\n"
              "    return out\n")
    open(os.path.join(repo, "analysis", "a.py"), "w").write(loader)
    g("add", "."); g("commit", "-qm", "base", "--no-verify")

    def commit(name, text):
        open(os.path.join(repo, "analysis", name), "w").write(text)
        g("add", "analysis/" + name)
        r = g("commit", "-qm", "add " + name)
        if r.returncode:
            g("reset", "-q", "analysis/" + name); os.remove(os.path.join(repo, "analysis", name))
        return r.returncode, r.stdout + r.stderr
    copy = (loader.replace("load_runs(path)", "read_sweep(d)").replace("(path,", "(d,").replace("out", "res")
            .replace("\"results\"", "\"res_dir\"").replace("key", "k"))
    rc, out = commit("b.py", copy)
    scenario("dupcheck: a copied loader (renamed variables, another path) is refused, naming the original",
             rc != 0 and "analysis/b.py:read_sweep" in out and "analysis/a.py:load_runs" in out, out)
    distinct = ("import numpy as np\n\n\ndef stats(curves):\n"
                "    if not curves:\n        return None\n"
                "    f = np.log10([c[0] for c in curves])\n    b = np.log10([min(c) for c in curves])\n"
                "    lo, hi = np.quantile(f - b, [0.1, 0.9])\n"
                "    turned = [c[-1] / min(c) for c in curves]\n"
                "    return {\"first\": float(np.median(f)), \"improve\": float(np.median(f - b)), \"q\": [lo, hi],\n"
                "            \"turn\": float(np.median(turned))}\n")
    rc, out = commit("c.py", distinct)
    scenario("dupcheck: a new distinct function passes", rc == 0, out)
    rc, out = commit("d.py", copy.replace("def read_sweep(d):", "def read_sweep(d):  # dup-ok: a test of the override"))
    scenario("dupcheck: a copy with `# dup-ok: <reason>` on its def line passes", rc == 0, out)
    p = os.path.join(repo, "analysis", "d.py")
    text = open(p).read().replace("  # dup-ok: a test of the override", "").replace(
        "        k = os.path.basename(f)[:-5]\n", "        k = os.path.basename(f)[:-5].lower()\n")
    open(p, "w").write(text)
    g("add", "analysis/d.py")
    r = g("commit", "-qm", "edit d")
    scenario("dupcheck: editing one of an already duplicated pair is not refused for that old pair",
             r.returncode == 0, r.stdout + r.stderr)
    os.makedirs(os.path.join(repo, "analysis", "common"))
    open(os.path.join(repo, "analysis", "common", "readers.py"), "w").write(
        "import numpy as np\n\n\ndef best_checkpoint(d):\n    i = int(np.argmin(d[\"val_loss_no_reg\"]))\n"
        "    return i, float(d[\"val_loss_no_reg\"][i])\n")
    g("add", "."); g("commit", "-qm", "common", "--no-verify")
    rc, out = commit("e.py", "import numpy as np\n\n\ndef at_best(m):\n    j = int(np.argmin(m[\"val_loss_no_reg\"]))\n"
                     "    return j, float(m[\"val_loss_no_reg\"][j])\n")
    scenario("dupcheck: a short copy of an analysis/common function is refused, naming it",
             rc != 0 and "analysis/common/readers.py:best_checkpoint" in out, out)
    g("mv", "analysis/d.py", "analysis/moved d.py")
    r = g("commit", "-qm", "move d")
    scenario("dupcheck: moving a file (to a path with a space) is not writing its functions anew",
             r.returncode == 0, r.stdout + r.stderr)
    # an old pair, committed past the check: a small edit of one of them keeps it >= 0.8 similar and is not refused
    open(os.path.join(repo, "analysis", "old_dup.py"), "w").write(loader.replace("load_runs", "g"))
    g("add", "."); g("commit", "-qm", "old pair", "--no-verify")
    p = os.path.join(repo, "analysis", "a.py")
    text = open(p).read().replace("        key = os.path.basename(f)[:-5]\n", "        key = os.path.basename(f)[:-5].lower()\n")
    open(p, "w").write(text)
    still = subprocess.run([sys.executable, os.path.join(repo, "tools", "dupcheck.py"), "--root", repo, p],
                           capture_output=True, text=True).stdout
    g("add", "analysis/a.py")
    r = g("commit", "-qm", "edit a")
    scenario("dupcheck: a small edit of one of a committed duplicate pair (still a near copy) is not refused",
             r.returncode == 0 and "analysis/a.py:load_runs  " in still and "no match" not in still,
             r.stdout + r.stderr + still)
    w = World([E("cov")])
    pr = w.plan_cmd("next", "--prompts")[1]
    scenario("the step prompt carries analysis/common's index and the line on reuse",
             "best_trial(trials" in pr and "Before writing a loader, picker or statistic, use analysis/common" in pr
             and "readers_index.py failed" not in pr, pr[-2500:])



HOOKS = os.path.join(ROOT, ".claude", "hooks")


def sh_hook(name, inp, env=None, cwd=None):
    r = subprocess.run(["bash", os.path.join(HOOKS, name)],
                       input=json.dumps(inp), capture_output=True, text=True, env=env or os.environ, cwd=cwd)
    return r.returncode, r.stderr + r.stdout


def command_reading():
    """The measured false refusals (tools/agent_calls.py --refusals, analysis/agent_calls/refusal_verdicts.json): the
    Bash guards read the raw text, so a heredoc body, a quoted string or a grep pattern that only named a refused form
    was refused. They now read the command through .claude/hooks/cmdparse.py: each measured false refusal passes, and
    the refusal it stands next to still fires."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmdparse", os.path.join(HOOKS, "cmdparse.py"))
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    shared = all("import cmdparse" in open(os.path.join(HOOKS, f)).read() or "from cmdparse import" in
                 open(os.path.join(HOOKS, f)).read() or "cmdparse.py" in open(os.path.join(HOOKS, f)).read()
                 for f in ("git_verify_guard.py", "plan_files_guard.py", "lock_guard.py", "design_guard.py",
                           "placement_guard.sh", "plot_guard.sh", "review_backlog.sh"))
    own = [f for f in ("git_verify_guard.py", "plan_files_guard.py", "design_guard.py")
           if "def heredocs(" in open(os.path.join(HOOKS, f)).read() or "def segments(cmd):" in
           open(os.path.join(HOOKS, f)).read()]
    scenario("cmdparse: every Bash guard reads the command through the one helper, none keeps its own copy",
             shared and not own, own)
    scenario("cmdparse: a python heredoc is data, a bash heredoc and sh -c are code, quotes are blanked",
             cp.commands("python3 - <<'EOF'\nprint('sbatch it''s')\nEOF\nbash -c 'sbatch x.sh'") ==
             [["python3", "-", "<<", "EOF"], ["bash", "-c", "sbatch x.sh"], ["sbatch", "x.sh"]]
             and "sbatch y" in cp.scan_text("bash <<'EOF'\nsbatch y\nEOF")
             and "sbatch" not in cp.scan_text("echo 'sbatch q' # it's sbatch"),
             cp.commands("python3 - <<'EOF'\nprint('x')\nEOF\nbash -c 'sbatch x.sh'"))

    # 1. git_verify_guard
    k = "--no-" + "verify"
    for c, want in (
            ("python3 - <<'EOF'\ns = '''it's'''\nprint(\"don't\")\nEOF\ngit add x && git commit -m 'edit'", 0),
            ("cat > f.py <<'EOF'\nrun(['git', 'config', 'core.hooksPath', '/dev/null'])  # it's a test\nEOF\n"
             "git diff --stat", 0),
            ("git diff --stat && echo it's", 0),
            ("git commit -m 'never pass %s here'" % k, 0),
            ("git commit %s -m x" % k, 2),
            ("echo $'it\\'s'; git commit %s -m x" % k, 2),      # valid bash shlex cannot parse
            ("python3 - <<'EOF'\nprint('it''s')\nEOF\ngit commit -n -m y", 2),
            ("bash <<'EOF'\ngit commit -n -m y\nEOF", 2),
            ("git -c core.hooksPath=/dev/null commit -m x && echo it's", 2),
            ("git cherry-pick HEAD 2>&1 | tail -3", 0),          # a redirection is not a commit range
            ("git cherry-pick HEAD~2..HEAD 2>&1 >/dev/null", 2),
            ("git commit -n -m y 2>&1 &>log <in", 2)):
        rc, out = hook(GUARD_GIT, {"tool_name": "Bash", "tool_input": {"command": c}, "cwd": ROOT})
        scenario("git verify: `%s` is %s" % (c.replace("\n", " / ")[:70], "refused" if want else "allowed"),
                 rc == want, out)

    # 2. plan_files_guard
    pv = "PLAN_" + "FILE"
    for c, want in (
            ("grep -i lowd plan/plan.yaml", 0), ("grep -in status plan/plan.yaml | head", 0),
            ("awk '/id:/{print $2}' plan/plan.yaml", 0), ("awk '$1==\"id:\" || $1==\"x\"' plan/plan.yaml", 0),
            ("grep -n %s tools/plan.py tests/test_setup_scenarios.py" % pv, 0),
            ("python3 - <<'EOF'\nopen('tests/t.py', 'a').write('env %s=/tmp/p.yaml')\nEOF" % pv, 0),
            ("sed -i s/held/running/ plan/plan.yaml", 2), ("perl -pi -e s/a/b/ plan/plan.yaml", 2),
            ("awk '{print > \"plan/plan.yaml\"}' /tmp/x", 2), ("awk -i inplace '{print}' plan/plan.yaml", 2),
            ("%s=/tmp/p.yaml python3 tools/plan.py show" % pv, 2), ("export %s=/tmp/p.yaml" % pv, 2)):
        rc, out = hook(GUARD_FILES, {"tool_name": "Bash", "tool_input": {"command": c}})
        scenario("plan files: `%s` is %s" % (c.replace("\n", " / ")[:70], "refused" if want else "allowed"),
                 rc == want, out)

    # 3. placement_guard (a temporary HOME: no placement file is read or spent)
    env = dict(os.environ, HOME=tempfile.mkdtemp(prefix="home_"))
    sb = "sb" + "atch"
    for c, want in (
            ("python3 - <<'EOF'\nopen('tests/t.py', 'a').write('%s job.sh\\n')\nEOF" % sb, 0),
            ("cat > tests/t.txt <<'EOF'\n%s job.sh\nEOF" % sb, 0), ("echo '%s job.sh'" % sb, 0),
            ("%s job.sh" % sb, 2), ("bash -c '%s job.sh'" % sb, 2), ("ssh cc <<'EOF'\n%s j.sh\nEOF" % sb, 2),
            ("site submit ccin2p3 FA scripts/job_probe.sh --item x", 2)):
        rc, out = sh_hook("placement_guard.sh", {"tool_name": "Bash", "tool_input": {"command": c}}, env=env)
        scenario("placement: `%s` is %s" % (c.replace("\n", " / ")[:70], "refused" if want else "allowed"),
                 rc == want, out)

    # 4. lock_guard and design_guard: a call is a command word, not a mention
    lock = os.path.join(HOOKS, "lock_guard.py")
    lenv = dict(os.environ, CLAUDE_PROJECT_DIR=tempfile.mkdtemp(prefix="lock_"))
    rb = "review_backlog.sh"
    for c, caller, want in (
            ("git commit -m 'notes: the reviewer ran %s begin notes-editor'" % rb, "", 0),
            ("cat > a.py <<'EOF'\nsubprocess.run(['bash', '%s', 'advance', 'x'])\nEOF" % rb, "", 0),
            ("python3 - <<'EOF'\nimport json\njson.dump({'k': '%s begin in a clone'}, open('/tmp/v.json', 'w'))\nEOF" % rb,
             "", 0),
            ("bash .claude/hooks/%s advance repo-reviewer" % rb, "", 2),
            ("cd /x && bash .claude/hooks/%s advance repo-reviewer" % rb, "general-purpose", 2),
            ("bash .claude/hooks/%s advance repo-reviewer" % rb, "repo-reviewer", 0)):
        inp = {"tool_name": "Bash", "tool_input": {"command": c}}
        if caller:
            inp["agent_type"] = caller
        rc, out = hook(lock, inp, env=lenv)
        scenario("lock: `%s` from %s is %s" % (c.replace("\n", " / ")[:60], caller or "the main session",
                                               "refused" if want else "allowed"), rc == want, out)
    dg = os.path.join(HOOKS, "design_guard.py")
    dp = "design-" + "pass"
    for c, want in (("cat > .claude/agents/x.md <<'EOF'\nthen run python3 tools/plan.py %s <entry>\nEOF" % dp, 0),
                    ("grep -n 'tools/plan.py %s' .claude/agents/design-reviewer.md" % dp, 0),
                    ("echo site submit auto FA j.sh --item nosuchentry", 0),
                    ("python3 tools/plan.py %s lowd" % dp, 2)):
        rc, out = hook(dg, {"tool_name": "Bash", "tool_input": {"command": c}})
        scenario("design: `%s` from the main session is %s" % (c.replace("\n", " / ")[:60],
                                                             "refused" if want else "allowed"), rc == want, out)

    # the review of this fix: what bash runs inside double quotes, combined shell flags, interpreter programs, quoted
    # verbs and assignments are still code (each was refused by the raw reading and let through by the first rewrite)
    rb_adv = "bash .claude/hooks/%s advance repo-reviewer" % rb
    nv = "--no-" + "verify"
    for path, c, extra, want in (
            (os.path.join(HOOKS, "placement_guard.sh"), 'jid="$(%s --parsable job.sh)"' % sb, None, 2),
            (os.path.join(HOOKS, "placement_guard.sh"), "bash -lc 'cd x; %s job.sh'" % sb, None, 2),
            (os.path.join(HOOKS, "placement_guard.sh"), "python3 -c \"import os; os.system('cd x; %s job.sh')\"" % sb,
             None, 2),
            (os.path.join(HOOKS, "placement_guard.sh"), 'echo "id `%s job.sh`"' % sb, None, 2),
            (os.path.join(HOOKS, "placement_guard.sh"), "cat > t.txt <<EOF\n$(%s job.sh)\nEOF" % sb, None, 2),
            (lock, 'x="$(%s)"' % rb_adv, None, 2), (lock, "bash -lc '%s'" % rb_adv, None, 2),
            (lock, "python3 -c \"import os; os.system('%s')\"" % rb_adv, None, 2),
            (lock, "echo '%s' > /tmp/a.sh; bash /tmp/a.sh" % rb_adv, None, 2),
            (dg, 'python3 tools/plan.py "%s" foo' % dp, None, 2),
            (GUARD_GIT, 'env "GIT_CONFIG_COUNT=1" git commit -m x', None, 2),
            (GUARD_GIT, "bash -lc 'git commit %s -m x'" % nv, None, 2),
            (GUARD_GIT, 'x="$(git commit %s -m x)"' % nv, None, 2),
            (GUARD_FILES, 'env "PLAN_ROOT=/tmp" "PLAN_TEST=1" python3 tools/plan.py next', None, 2),
            (GUARD_FILES, "perl -e 'open my $f, \">\", shift; print $f 1' plan/plan.yaml", None, 2),
            (GUARD_FILES, "ruby -e 'File.write(ARGV[0], \"x\")' plan/plan.yaml", None, 2),
            (GUARD_FILES, "sed -f s.sed plan/plan.yaml", None, 2)):
        inp = {"tool_name": "Bash", "tool_input": {"command": c}, "cwd": ROOT}
        if path.endswith(".sh"):
            rc, out = sh_hook(os.path.basename(path), inp, env=env)
        else:
            rc, out = hook(path, inp, env=lenv if path == lock else None)
        scenario("still refused (%s): `%s`" % (os.path.basename(path), c.replace("\n", " / ")[:60]), rc == want, out)
    scenario("cmdparse: a git commit inside \"$(...)\" is code for the commit gate (review_backlog.sh bashpre)",
             "git commit -m y" in cp.scan_text('x="$(git commit -m y)"'), cp.scan_text('x="$(git commit -m y)"'))

    # 5. plot_guard: reading a script is no submission
    d = tempfile.mkdtemp(prefix="plot_")
    bad = os.path.join(d, "bad.sh")
    off = "plot" + "=" + "false"
    open(bad, "w").write("#!/bin/bash\npython run.py exp_name=x %s\n" % off)
    for c, want in (("grep -rn plot %s/*.sh" % d, 0), ("cat %s" % bad, 0), ("head -3 %s && ls %s" % (bad, d), 0),
                    ("sed -n 1,5p %s" % bad, 0), ("grep -n '%s' run.py" % off, 0),
                    ("%s %s" % (sb, bad), 2), ("site submit auto FA %s --item x" % bad, 2),
                    ("python run.py exp_name=y %s" % off, 2)):
        rc, out = sh_hook("plot_guard.sh", {"tool_name": "Bash", "tool_input": {"command": c}})
        scenario("plot: `%s` is %s" % (c.replace(d, "<tmp>")[:60], "refused" if want else "allowed"), rc == want, out)

    # 6. review_backlog gate inside a worktree reads the main checkout's watermarks
    repo = tempfile.mkdtemp(prefix="wtgate_")
    g = lambda *a, cwd=repo: subprocess.run(["git", "-C", cwd] + list(a), capture_output=True, text=True, env=GIT_ENV)
    g("init", "-q")
    os.makedirs(os.path.join(repo, ".claude", "hooks"))
    shutil.copy(os.path.join(HOOKS, "review_backlog.sh"), os.path.join(repo, ".claude", "hooks"))
    open(os.path.join(repo, ".gitignore"), "w").write(".claude/.review_state/\nwt/\n")
    open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
    g("add", "."); g("commit", "-qm", "base")
    old = g("rev-parse", "HEAD").stdout.strip()
    open(os.path.join(repo, "a.py"), "a").write("".join("y%d = %d\n" % (i, i) for i in range(700)))
    g("commit", "-qam", "big")
    new = g("rev-parse", "HEAD").stdout.strip()
    st = os.path.join(repo, ".claude", ".review_state")
    os.makedirs(st)
    for name in ("repo-reviewer", "notes-editor", "claudemd-keeper"):
        open(os.path.join(st, name), "w").write("%s 0 0\n" % new)
    wt = os.path.join(repo, "wt")
    g("worktree", "add", "-q", "-b", "feat", wt)
    os.makedirs(os.path.join(wt, ".claude", ".review_state"))
    for name in ("repo-reviewer", "notes-editor", "claudemd-keeper"):     # the worktree's stale copy
        open(os.path.join(wt, ".claude", ".review_state", name), "w").write("%s 0 0\n" % old)
    genv = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
    gate = lambda root: subprocess.run(["bash", os.path.join(root, ".claude", "hooks", "review_backlog.sh"), "gate"],
                                       input=json.dumps({"tool_input": {"file_path": os.path.join(root, "a.py")}}),
                                       capture_output=True, text=True, cwd=root, env=genv)
    r1 = gate(wt)
    scenario("review gate: an edit in a worktree is judged on the main checkout's watermark, not its stale copy",
             r1.returncode == 0, r1.stdout + r1.stderr)
    for name in ("repo-reviewer",):
        open(os.path.join(st, name), "w").write("%s 0 0\n" % old)
    r2, r3 = gate(wt), gate(repo)
    scenario("review gate: a real backlog of 700 lines still refuses the edit, in the worktree and the main checkout",
             r2.returncode == 2 and r3.returncode == 2 and "repo-reviewer" in r2.stderr, r2.stdout + r2.stderr)

    # 7. the step prompt asks for a placement before every named submission
    tpl = open(os.path.join(ROOT, "plan", "step_prompt.txt")).read()
    scenario("step prompt: site pick right before every named site submit, CPU jobs too",
             "Run `site pick FA --jobs N ...` right before every `site submit <site>` (CPU jobs too): placement_guard "
             "refuses a named submission with no plan unit left." in tpl)


def structured_blocks():
    """The steps an open question holds are recorded, not guessed (review of 2026-10-29): plan ask takes
    --blocks-steps, required when the entry has more than one next step, and prints what it holds; blocked_steps reads
    only that field, the free-text parse (ranges, "or") kept for questions recorded before it."""
    three = {1: "analyse the density gap", 2: "submit the pp-trained Muon arm", 3: "collect the default share"}
    q = "Should the default share use the forward rotation on the e+e- pools as well?"
    w = World([E("blk", next=dict(three))])
    rc, out = w.plan_cmd("ask", "blk", "--ask", q, "--searched", SEARCHED, "--blocks", "steps 1 and 3")
    scenario("structured blocks: an ask on a three-step entry without --blocks-steps is refused, nothing recorded",
             rc != 0 and "--blocks-steps" in out and "ask" not in w.entries()["blk"], out)
    rc, out = w.plan_cmd("ask", "blk", "--ask", q, "--searched", SEARCHED, "--blocks", "the rotation",
                         "--blocks-steps", "1,3")
    e = w.entries()["blk"]
    nx = w.plan_cmd("next")[1]
    scenario("structured blocks: --blocks-steps 1,3 is recorded, printed back, and only step 2 is listed",
             rc == 0 and e["blocks_steps"] == [1, 3] and "step 1 HELD" in out and "step 2 moves" in out
             and "step 3 HELD" in out and "NEXT blk step 2 of 3" in nx and "step 1 of 3" not in nx
             and "step 3 of 3" not in nx, out + nx)
    w = World([E("blk", next=dict(three))])
    rc, out = w.plan_cmd("ask", "blk", "--ask", q, "--searched", SEARCHED, "--blocks-steps", "4")
    scenario("structured blocks: a step the entry does not have is refused", rc != 0 and "no next step 4" in out, out)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import plan as P
    cases = {"steps 1-3": {1, 2, 3}, "steps 1 to 3": {1, 2, 3}, "step 1 or 3": {1, 3}, "steps 1, 2 and 3": {1, 2, 3},
             "step 2's design": {2}}
    got = {}
    for text, want in cases.items():
        legacy = E("old", next=dict(three), ask=q, waiting_since=ago(20), blocks=text)
        got[text] = P.blocked_steps(legacy)
    scenario("structured blocks: a question recorded before the field falls back to the text, ranges and 'or' "
             "included", all(got[t] == w_ for t, w_ in cases.items()), got)
    rec = E("new", next=dict(three), ask=q, waiting_since=ago(20), blocks="steps 1-3", blocks_steps=[2])
    scenario("structured blocks: a recorded field wins over the text", P.blocked_steps(rec) == {2},
             P.blocked_steps(rec))


GROUPS = [liveness, event_ownership, files, answers, adding, questions, holds, done, migration, claims, searched_first, seventh_lower,
          hold_without_item, sed_reads, hold_release, reask_once, reopen, commit_retry, claims_budget, added_only,
          worktree_then_merge, plan_commits, test_switch, claims_state, fourth_lower, direct_writes,
          one_agent_per_step, git_bypass, parsing, checks_and_sites, no_plan_yet, lower, setup_records,
          decision_quotes, plan_commits_and_launches, slow_done_checks, tooling_fixes, reviewer_resume, reviewed_content, sweeps, open_questions,
          designs, design_prior, prompt_cap, parallel_steps, sent_figures, idle_wait, the_suite_itself, main_session_work,
          duplicate_check, command_reading, structured_blocks]


def main():
    for g in GROUPS:
        g()
    print("%d/%d scenarios pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if RESULTS and all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
