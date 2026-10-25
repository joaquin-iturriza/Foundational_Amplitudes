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
                 state=None, agents=()):
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
        # at the step's time with that entry's prompt, unless its agent id starts with "forged"
        agents = list(agents) + [(x["agent"], x["at"], eid) for eid, v in (state or {}).items() if eid[:1] != "_"
                                 for x in (v or {}).get("steps") or [] if not x["agent"].startswith("forged")]
        for aid, started, eid in agents:
            p = os.path.join(self.ag, "agent-%s.jsonl" % aid)
            if not os.path.exists(p):
                open(p, "w").write(J({"type": "user", "timestamp": started, "message": {
                    "role": "user", "content": "You are a step agent.\n\nEntry: %s (t)\nQuestion: q" % eid}}) + "\n")
        open(j("site"), "w").write(STUB)
        os.chmod(j("site"), 0o755)
        self.env = dict(os.environ, PLAN_TEST="1", PLAN_FILE=self.plan, PLAN_STATE=self.state, PLAN_RUNS_DB=self.db,
                        PLAN_TRANSCRIPTS=self.tr, PLAN_AGENTS=self.ag, PLAN_SITE=j("site"), PLAN_ROOT=self.d,
                        PLAN_SITE_MODULE=shutil.which("site"),
                        PLAN_EVENTS_CMD="printf '%s\\n'" % events.replace("'", ""))

    def check(self):
        r = subprocess.run([sys.executable, CHECK], input="{}", env=self.env, capture_output=True, text=True)
        return r.returncode, r.stderr + r.stdout

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
    w = World([E("lowd")], state={"lowd": step(50)}, agents=[("a2early", ago(40), "lowd")])
    rc, out = w.plan_cmd("step", "lowd", "--agent", "a2early")
    scenario("B1: nothing in flight since 20 min ago: a subagent started 40 min ago is refused", rc != 0, out)
    ask = "Should the readout run on every rung now?"
    wait = dict(status="waiting_user", ask=ask, waiting_since=ago(30), asks=[{"ask": ask, "at": ago(30)}])
    w = World([E("ng", **wait)], agents=[("a3", ago(1), "ng")],
              transcript=[(ago(29), "assistant", "Joaquin, should the readout run on every rung now?"),
                          (ago(3), "user", "Yes, run it on every rung please")])
    rc, out = w.plan_cmd("step", "ng", "--agent", "a3")
    scenario("B1: a waiting entry takes no step; it blocks until answered", rc != 0 and w.check()[0] == 2, out)
    runs = [("FA-ccin2p3-old", "RUNNING", "pp", 1, None)] + [("FA-ccin2p3-n%02d" % i, "COMPLETED", "pp", 100 + i, 200)
                                                              for i in range(40)]
    w = World([E("pp")], runs=runs, state={"pp": step(1)})
    scenario("a live run older than 40 newer ones is seen", w.check()[0] == 0, w.check()[1])
    for s in ("UNKNOWN", "HELD", "CANCELLED by 123", "OUT_OF_MEMORY"):
        w = World([E("delta")], runs=[("FA-ccin2p3-u", s, "delta", epoch(100), None)])
        scenario("a run in state %s is not live" % s, w.check()[0] == 2, w.check()[1])


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
    w = World([E("ng", **wait())], transcript=say)
    rc, err = w.check()
    scenario("a user message after the question blocks until it reaches the entry", rc == 2 and "the user wrote" in err,
             err)
    rc, out = w.plan_cmd("answer", "ng", "--kind", "scope", "--quote", "Yes, run it on every rung please", "--set",
                         "next=run it")
    e = w.entries()["ng"]
    scenario("plan answer records the words and updates the entry in one call",
             rc == 0 and e["status"] == "running" and e["next"] == "run it" and e["quotes"][0]["decision"] == "D900"
             and e["asks"][0]["answered_at"], out)
    w = World([E("ng", **wait())], transcript=say[:1] + [(ago(3), "user", "Unrelated: how is the pp generation going")])
    w.plan_cmd("unanswered", "ng")
    scenario("plan unanswered leaves it waiting, unblocked", w.check()[0] == 0, w.check()[1])
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
               "eval git commit --no-verify -m x", "cd x && git commit -n -m y", "echo x >> .git/hooks/pre-commit")
    allowed = ("git commit -m 'fix the -n flag'", "git commit -m -n", "git commit -am x", "git log -n 5",
               "git config --get core.hooksPath", "git merge --abort", "git status")
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
    try:
        r = subprocess.run(["sh", os.path.join(tmp_wt, "tools", "switch_over.sh"), "some-branch"], cwd=tmp_wt,
                           capture_output=True, text=True)
        head_after = subprocess.run(["git", "-C", tmp_wt, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    finally:
        subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", tmp_wt], capture_output=True)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
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
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "the DyHPO round"], env=env, capture_output=True,
                       text=True)
    st = g("status", "--porcelain").stdout
    msg = g("log", "-1", "--format=%s").stdout.strip()
    scenario("2: a plan write commits plan/plan.yaml alone; the tree is clean for site sync", r.returncode == 0
             and st == "" and msg == "plan: set-next x", r.stdout + r.stderr + st + msg)
    open(os.path.join(repo, "other.txt"), "w").write("w"); g("add", "other.txt")
    subprocess.run([sys.executable, PLAN, "set-next", "x", "again"], env=env, capture_output=True, text=True)
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
    w.plan_cmd("set-next", "rd", "redo the figure")
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
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "after the lock"], env=env, capture_output=True,
                       text=True)
    scenario("lower: a plan write retries through a short index.lock and commits", r.returncode == 0
             and g("status", "--porcelain").stdout == "", r.stdout + r.stderr)
    open(lock, "w").close()
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, PLAN, "set-next", "x", "while git is busy"], env=env, capture_output=True,
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


GROUPS = [liveness, files, answers, adding, questions, holds, done, migration, claims, searched_first, seventh_lower,
          hold_without_item, sed_reads, hold_release, reask_once, reopen, commit_retry, claims_budget, added_only,
          worktree_then_merge, plan_commits, test_switch, claims_state, fourth_lower, direct_writes,
          one_agent_per_step, git_bypass, parsing, checks_and_sites, no_plan_yet, lower, setup_records,
          decision_quotes, the_suite_itself]


def main():
    for g in GROUPS:
        g()
    print("%d/%d scenarios pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if RESULTS and all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
