"""One reviewer per backlog, resumed reviewers included: replay of 2026-11-24, two claudemd-keepers both ran
`begin` in the same second and the second one's `advance` swept the other's backlog."""
import importlib.util, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))


def guard(state):
    spec = importlib.util.spec_from_file_location("review_guard", os.path.join(ROOT, ".claude", "hooks", "review_guard.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.STATE, m.LOG = state, os.path.join(state, "lock_calls.log")
    return m


def call(m, agent, verb):
    return m.bash({"agent_type": "claudemd-keeper", "agent_id": agent,
                   "tool_input": {"command": "bash .claude/hooks/review_backlog.sh %s claudemd-keeper" % verb}})


def test_second_reviewer_refused():
    with tempfile.TemporaryDirectory() as st:
        m = guard(st)
        assert call(m, "a177", "begin") == 0
        open(os.path.join(st, "claudemd-keeper.lock"), "w").close()     # what review_backlog.sh begin writes
        assert call(m, "a313", "begin") == 2                             # the 19:26:20 double begin
        assert call(m, "a313", "advance") == 2                           # the 19:28:37 advance by the other one
        assert call(m, "a177", "advance") == 0                           # the holder releases its own lock
        os.unlink(os.path.join(st, "claudemd-keeper.lock"))              # advance removed it: a new reviewer may begin
        assert call(m, "a313", "begin") == 0
        assert "agent=a313" in open(m.LOG).read()


if __name__ == "__main__":
    test_second_reviewer_refused()
    print("ok")
