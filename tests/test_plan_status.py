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
