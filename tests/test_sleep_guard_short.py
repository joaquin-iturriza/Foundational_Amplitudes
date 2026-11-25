"""~/.claude/hooks/sleep_guard.sh (D558): a plain foreground sleep of at most 5 s outside a loop passes; longer sleeps,
loops, tail -f stay refused. Skipped where the user-level hook is absent."""
import _tmpdir  # noqa: F401
import json, os, subprocess
import pytest

HOOK = os.path.expanduser("~/.claude/hooks/sleep_guard.sh")


def rc(cmd):
    return subprocess.run(["bash", HOOK], input=json.dumps({"tool_input": {"command": cmd}}),
                          capture_output=True, text=True).returncode


@pytest.mark.skipif(not os.path.exists(HOOK), reason="no user-level hook")
def test_short_sleep_passes_rest_refused():
    for c in ("sleep 0; cat f", "sleep 1; tail -25 f", "kill 1 2; sleep 5; kill -0 1"):
        assert rc(c) == 0, c
    for c in ("until grep -q x f; do sleep 5; done", "while true; do sleep 1; done", "sleep 30",
              "timeout 60 tail -f /dev/null", "sleep 6", "sleep 2; sleep 30"):
        assert rc(c) == 2, c
