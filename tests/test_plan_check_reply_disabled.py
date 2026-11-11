"""plan_check runs reply_check's verdict at Stop (stop_conditions.reply); while reply_check.py is listed in
.claude/hooks/disabled.json (incident I-edf7c73a, kept off by D335) it is skipped there too. Tonight's case
(2026-11-09 23:45): with reply_check disabled, plan_check still refused the turn with "REPLY CHECK: your question
"what do you mean you can't kill it?" is not answered". The verdict is replayed (reply_check calls a model); the other
Stop parts are stubbed, so only the gate is tested. A temporary world (PLAN_TEST=1); nothing live is read or written.

    .venv-cpu/bin/python tests/test_plan_check_reply_disabled.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import importlib, json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS = os.path.join(ROOT, ".claude", "hooks")
TONIGHT = ("REPLY CHECK: your question \"what do you mean you can't kill it?\" is not answered. The reply says the "
           "kill is done and explains a refusal, but the explanation is muddled (this check runs once per user message).")


def world(disabled):
    t = tempfile.mkdtemp(prefix="reply_disabled_")
    os.makedirs(os.path.join(t, ".claude", "hooks"))
    with open(os.path.join(t, ".claude", "settings.json"), "w") as fh:
        json.dump({"hooks": {"Stop": [{"hooks": [{"type": "command", "command":
                   "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/dispatch.py\" plan_check.py"}]}]}}, fh)
    if disabled is not None:
        with open(os.path.join(t, ".claude", "hooks", "disabled.json"), "w") as fh:
            json.dump(disabled, fh)
    old = {k: v for k, v in os.environ.items() if k.startswith("PLAN_")}
    for k in old:
        del os.environ[k]
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=t, PLAN_SITE="true")
    try:
        sys.path[:0] = [os.path.join(ROOT, "tools"), HOOKS]
        import plan, stop_conditions, plan_check
        importlib.reload(plan)
        sc = importlib.reload(stop_conditions)
        pc = importlib.reload(plan_check)
    finally:
        for k in ("PLAN_TEST", "PLAN_ROOT", "PLAN_SITE"):
            os.environ.pop(k, None)
        os.environ.update(old)
    for part in ("figure_pairs", "context", "ledger", "incidents"):
        setattr(sc, part, lambda *a, **k: ([], []))
    sc.reply = lambda inp: ([TONIGHT], [])
    return pc


INP = {"hook_event_name": "Stop", "session_id": "s", "transcript_path": "/nonexistent"}


def test_disabled_reply_check_does_not_refuse():
    pc = world({"reply_check.py": {"incident": "I-edf7c73a", "why": "D335"}})
    bad, _ = pc.stop_conditions(INP, [])
    assert not any("REPLY CHECK" in b for b in bad), bad


def test_enabled_reply_check_still_refuses():
    for dis in (None, {}, {"other_hook.py": {}}):
        pc = world(dis)
        bad, _ = pc.stop_conditions(INP, [])
        assert any("what do you mean you can't kill it?" in b for b in bad), (dis, bad)


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k, flush=True)
    print("%d passed" % n)
