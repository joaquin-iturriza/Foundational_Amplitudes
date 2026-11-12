"""A loop disables only the condition of plan_check that refused (disabled.json "plan_check.py:<condition>"), and a
core condition (ledger: armed waiter and supervisor; plan_state: owners, open events) is never disabled. Replay of
I-531d1e89 (figures_handed_back refused twice and switched off the whole Stop gate).

    .venv-cpu/bin/python tests/test_dispatch_conditions.py
"""
import _tmpdir  # noqa: F401
import json, os, shutil, subprocess, sys, tempfile
import test_plan_check_reply_disabled as base

HOOKS = base.HOOKS
STUB = '''import os, sys
sys.stdin.read()
sys.stderr.write(open(os.path.join(os.path.dirname(__file__), "msg.txt")).read())
sys.exit(2)
'''
FIG = "PLAN CHECK: the turn cannot end yet.\n  figures handed back for Joaquin not sent: a.png\nDISPATCH-CONDITIONS disableable=figures_handed_back core=\n"
WAITER = "PLAN CHECK: the turn cannot end yet.\n  no armed waiter\nDISPATCH-CONDITIONS disableable= core=ledger\n"


def box():
    d = tempfile.mkdtemp(prefix="dispcond_")
    shutil.copy(os.path.join(HOOKS, "dispatch.py"), d)
    open(os.path.join(d, "plan_check.py"), "w").write(STUB)
    open(os.path.join(d, "tr.jsonl"), "w").close()
    env = dict(os.environ, DISPATCH_STATE=os.path.join(d, "st.json"), DISPATCH_PROJECTS=os.path.join(d, "none.toml"),
               CLAUDE_PROJECT_DIR=d)

    def run(msg, session):
        open(os.path.join(d, "msg.txt"), "w").write(msg)
        inp = {"session_id": session, "transcript_path": os.path.join(d, "tr.jsonl"), "cwd": d, "hook_event_name": "Stop"}
        return subprocess.run([sys.executable, os.path.join(d, "dispatch.py"), "plan_check.py"], input=json.dumps(inp),
                              capture_output=True, text=True, env=env, cwd=d).returncode
    dis = lambda: json.load(open(os.path.join(d, "disabled.json"))) if os.path.exists(os.path.join(d, "disabled.json")) else {}
    return run, dis


def test_loop_on_figures_disables_only_that_condition():
    run, dis = box()
    assert run(FIG, "s") == 2 and run(FIG, "s") == 0
    assert list(dis()) == ["plan_check.py:figures_handed_back"], dis()
    assert run(WAITER, "s2") == 2          # the gate is still on: a turn without an armed waiter is refused


def test_loop_on_core_condition_is_not_disabled():
    run, dis = box()
    assert [run(WAITER, "s") for _ in range(3)] == [2, 2, 2]
    assert dis() == {}


def test_plan_check_skips_disabled_condition_but_not_core():
    off = {"plan_check.py:figures_handed_back": {}, "plan_check.py:ledger": {}, "plan_check.py:context": {}}
    pc = base.world(off)
    import stop_conditions as sc
    sc.ledger = lambda *a, **k: (["no armed waiter"], [])
    sc.context = lambda *a, **k: (["context"], [])
    pc.REFUSED.clear()
    bad, _ = pc.stop_conditions(base.INP, [])
    assert "no armed waiter" in bad and "context" not in bad, bad    # ledger is core: its entry in disabled.json is ignored
    assert pc.REFUSED == {"ledger": True}, pc.REFUSED


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
