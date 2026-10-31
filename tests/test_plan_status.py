"""plan status and its Stop gate on a scratch copy of the live plan: catalog_coverage's owner stopped, its steps 11 and
14 waiting on two runs that ended, no live job, no open event."""
import json, os, shutil, sqlite3, subprocess, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDED = ("FA-ccin2p3-f8d114ce", "FA-ccin2p3-37ebce98")


def world(tmp_path):
    (tmp_path / "plan").mkdir()
    (tmp_path / ".claude").mkdir()
    shutil.copy(os.path.join(REPO, "plan", "plan.yaml"), tmp_path / "plan")
    shutil.copy(os.path.join(REPO, "plan", ".state.json"), tmp_path / "plan")
    db = tmp_path / "runs.db"
    shutil.copy(os.path.expanduser("~/.local/share/ccorch/runs.db"), db)
    c = sqlite3.connect(db)
    c.execute("update runs set state='COMPLETED' where project='FA' and item='catalog_coverage'")
    c.commit()
    d = yaml.safe_load(open(tmp_path / "plan" / "plan.yaml"))
    for e in d["entries"]:
        if e["id"] == "catalog_coverage":
            e["owner"] = {"agent": "astopped0000", "state": "working", "since": "2026-10-30T18:15:20+00:00"}
            e["step_waits"] = {11: "wait on " + ENDED[0], 14: "wait on " + ENDED[1]}
            e.pop("ask", None)
    yaml.safe_dump(d, open(tmp_path / "plan" / "plan.yaml", "w"))
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
    assert r.returncode == 2 and "catalog_coverage is STALLED" in r.stderr, r.stderr
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
    assert r.returncode == 0, r.stderr
    e = next(x for x in yaml.safe_load(open(p))["entries"] if x["id"] == "agent_redesign")
    assert e["status"] == "held" and e["blocks_steps"] == [] and e["asks"][-1]["blocks"] == "the wait_guard fix only"
    multi = next((x for x in yaml.safe_load(open(p))["entries"] if x.get("status") == "running"
                  and isinstance(x.get("next"), dict) and len(x["next"]) > 1 and x["id"] != "agent_redesign"), None)
    if multi is None:
        return
    q = ["--ask", "Is the second figure needed for the memo at all?", "--searched", "docs/results.tex, CLAUDE.md and site decisions FA: none settles it"]
    r = subprocess.run(plan + ["ask", multi["id"]] + q, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "--blocks-steps none" in r.stderr, r.stderr
    r = subprocess.run(plan + ["ask", multi["id"]] + q + ["--blocks", "the figure", "--blocks-steps", "none"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    e = next(x for x in yaml.safe_load(open(p))["entries"] if x["id"] == multi["id"])
    assert e["status"] == "running" and e["blocks_steps"] == []
