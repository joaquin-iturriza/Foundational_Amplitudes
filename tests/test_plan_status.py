"""plan status and its Stop gate on a frozen plan fixture (tests/fixtures/plan_status): catalog_coverage's owner stopped,
its steps 11 and 14 waiting on two runs that ended, no live job, no open event; independent of the live plan and runs.db."""
import json, os, shutil, sqlite3, subprocess, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDED = ("FA-ccin2p3-f8d114ce", "FA-ccin2p3-37ebce98")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "plan_status")
RUNS_SCHEMA = """CREATE TABLE runs (id TEXT PRIMARY KEY, project TEXT, site TEXT, branch TEXT, commit_sha TEXT,
  job TEXT, sched_id TEXT, state TEXT, remote_path TEXT, submitted INTEGER, finished INTEGER, est_gpu_hours REAL,
  note TEXT, item TEXT, expect TEXT, summary TEXT, summary_state TEXT, running_seen INTEGER, launch TEXT);
CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, project TEXT, item TEXT, kind TEXT,
  key TEXT, msg TEXT, acked INTEGER, ack_note TEXT, mute_until INTEGER, tell TEXT);
CREATE TABLE decisions (id TEXT PRIMARY KEY, project TEXT, kind TEXT, quote TEXT, said_at TEXT, covers TEXT,
  note TEXT, recorded INTEGER, in_force INTEGER DEFAULT 1);
CREATE TABLE items (project TEXT, id TEXT, title TEXT, decision TEXT, scope TEXT, est_gpu_hours REAL,
  match TEXT, check_cmd TEXT, ready TEXT, notes TEXT, status TEXT, created INTEGER, updated INTEGER, progress TEXT,
  waiting TEXT, PRIMARY KEY (project, id));"""


def world(tmp_path):
    (tmp_path / "plan").mkdir()
    (tmp_path / ".claude").mkdir()
    for f in ("plan.yaml", ".state.json"):
        shutil.copy(os.path.join(FIXTURE, f), tmp_path / "plan")
    db = tmp_path / "runs.db"
    c = sqlite3.connect(db)
    c.executescript(RUNS_SCHEMA)
    for r in ENDED:
        c.execute("insert into runs (id, project, site, state, item) values (?, 'FA', 'ccin2p3', 'COMPLETED', "
                  "'catalog_coverage')", (r,))
    c.execute("insert into decisions (id, project, kind, quote, said_at, covers, recorded, in_force) values "
              "('D228', 'FA', 'approve', 'fix setup problems during the weekend hold', '2026-10-30T10:00:00+00:00', "
              "'agent_redesign', 1700000000, 1)")
    c.commit()
    env = dict(os.environ, PLAN_TEST="1", PLAN_ROOT=str(tmp_path), PLAN_RUNS_DB=str(db), PLAN_EVENTS_CMD="true",
               PLAN_SITE_MODULE=shutil.which("site"))
    return env


def test_status_stalled_and_gate(tmp_path):
    env = world(tmp_path)
    out = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plan.py"), "status"], env=env,
                         capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("catalog_coverage"))
    assert "STALLED" in line and "steps 11, 14 wait on runs that ended" in line, out
    gate = os.path.join(REPO, ".claude", "hooks", "plan_check.py")
    r = subprocess.run([sys.executable, gate], input="{}", env=env, capture_output=True, text=True)
    # a stalled entry whose owner is stopped is a note (what it waits on is in its hand-back); one with no owner refuses
    assert r.returncode == 2 and "multi_step is STALLED" in r.stderr, r.stderr
    assert "catalog_coverage is STALLED" not in r.stderr, r.stderr
    r = subprocess.run([sys.executable, gate], input=json.dumps({"agent_id": "x"}), env=env, capture_output=True,
                       text=True)
    assert "STALLED" not in r.stderr


def test_supervisor_stalled_event(tmp_path):
    """site_runends.stalled_pass: an event on the stalled scratch plan, none on the live plan, cleared on recovery."""
    import importlib.machinery, importlib.util
    env = world(tmp_path)
    bindir = os.path.expanduser("~/.local/bin")
    sys.path.insert(0, bindir)
    def load(name, path):
        ld = importlib.machinery.SourceFileLoader(name, path)
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        ld.exec_module(mod)
        return mod
    m = load("site_tool", os.path.join(bindir, "site"))
    import site_runends as R
    db = tmp_path / "ev.db"
    c = sqlite3.connect(db)
    c.executescript(sqlite3.connect(os.path.expanduser("~/.local/share/ccorch/runs.db")).execute(
        "select sql from sqlite_master where name='events'").fetchone()[0])
    pr = {"local": REPO, "run_ends": {}}
    old = dict(os.environ)
    try:
        os.environ.update(env)
        R.stalled_pass(m, c, "FA", pr)
        assert c.execute("select count(*) from events where item='catalog_coverage'").fetchone()[0] == 1
        row = c.execute("select item,kind,key,msg from events where acked is null").fetchone()
        assert row[:3] == ("catalog_coverage", "STALLED", "stalled:catalog_coverage"), row
        assert R.stalled_pass(m, c, "FA", pr) == 0              # one open event per key
        for k in list(os.environ):
            if k.startswith("PLAN_"):
                del os.environ[k]
        assert R.stalled_pass(m, c, "FA", pr) == 0              # live plan: nothing stalled, the event clears
        assert c.execute("select count(*) from events where acked is null").fetchone()[0] == 0
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_timed_wait_is_waiting(tmp_path):
    """agent_redesign's step waits on a day of clean running (names no run), its owner reads as stopped in the scratch
    world: WAITING, not STALLED; the supervisor and the gate ignore it."""
    env = world(tmp_path)
    out = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plan.py"), "status"], env=env,
                         capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("agent_redesign"))
    assert " WAITING " in line and "STALLED" not in line, out
    r = subprocess.run([sys.executable, os.path.join(REPO, ".claude", "hooks", "plan_check.py")], input="{}", env=env,
                       capture_output=True, text=True)
    assert "agent_redesign is STALLED" not in r.stderr


def test_ask_on_held_entry_and_no_step(tmp_path):
    """plan ask records a question on a held entry (it stays held) and one holding no step (--blocks-steps none) of an
    entry with several next steps; a running entry's question still needs --blocks-steps."""
    env = world(tmp_path)
    p = tmp_path / "plan" / "plan.yaml"
    d = yaml.safe_load(open(p))
    for e in d["entries"]:
        if e["id"] == "agent_redesign":
            e["status"] = "held"
            for k in ("ask", "waiting_since", "blocks", "blocks_steps", "holds_all"):
                e.pop(k, None)
    yaml.safe_dump(d, open(p, "w"))
    plan = [sys.executable, os.path.join(REPO, "tools", "plan.py")]
    r = subprocess.run(plan + ["ask", "agent_redesign", "--ask", "Fix the wait_guard watch regex during the hold?",
                               "--searched", "CLAUDE.md and site decisions FA: none on repairs during a hold",
                               "--blocks", "the wait_guard fix only", "--blocks-steps", "none"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "step numbers the answer gates" in r.stderr, r.stderr
    r = subprocess.run(plan + ["ask", "agent_redesign", "--ask", "Fix the wait_guard watch regex during the hold?",
                               "--searched", "CLAUDE.md and site decisions FA: none on repairs during a hold",
                               "--blocks", "nothing", "--blocks-steps", "none"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    e = next(x for x in yaml.safe_load(open(p))["entries"] if x["id"] == "agent_redesign")
    assert e["status"] == "held" and e["blocks_steps"] == [] and e["asks"][-1]["blocks"] == "nothing"
    r = subprocess.run(plan + ["withdraw", "agent_redesign", "--decision", "D228", "--why",
                               "D228 approves setup fixes during the weekend hold"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    e = next(x for x in yaml.safe_load(open(p))["entries"] if x["id"] == "agent_redesign")
    assert e["status"] == "held" and "ask" not in e and e["asks"][-1]["decision"] == "D228"
    multi = next((x for x in yaml.safe_load(open(p))["entries"] if x.get("status") == "running"
                  and isinstance(x.get("next"), dict) and len(x["next"]) > 1 and x["id"] != "agent_redesign"), None)
    if multi is None:
        return
    q = ["--ask", "Is the second figure needed for the memo at all?", "--searched", "docs/results.tex, CLAUDE.md and site decisions FA: none settles it"]
    r = subprocess.run(plan + ["ask", multi["id"]] + q, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "--blocks-steps none" in r.stderr, r.stderr
    r = subprocess.run(plan + ["ask", multi["id"]] + q + ["--blocks", "nothing", "--blocks-steps", "none"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    e = next(x for x in yaml.safe_load(open(p))["entries"] if x["id"] == multi["id"])
    assert e["status"] == "running" and e["blocks_steps"] == []


def test_brief_lists_open_question(tmp_path):
    """plan brief prints an open question and what it blocks under WAITING ON JOAQUIN, do not do:."""
    env = world(tmp_path)
    p = tmp_path / "plan" / "plan.yaml"
    d = yaml.safe_load(open(p))
    for e in d["entries"]:
        if e["id"] == "catalog_coverage":
            e.update(ask="Redraw the zero-shot arms on option A or B?", waiting_since="2026-11-03T10:00:00+00:00",
                     searched="none", blocks="the redraw of arms_zero_shot_a/b", blocks_steps=[])
    yaml.safe_dump(d, open(p, "w"))
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plan.py"), "brief", "catalog_coverage"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "WAITING ON JOAQUIN, do not do:" in r.stdout
    assert "Redraw the zero-shot arms on option A or B?" in r.stdout and "the redraw of arms_zero_shot_a/b" in r.stdout
