"""tools/main_turns.py: the Bash classifier of the main session's calls (cases from its hand check)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import main_turns as m

CASES = [('site ack E1 --note "a; rm x > y.txt"', "orchestration"),        # quoted text is not run
         ('grep -ln "x" a 2>/dev/null', "read"),                              # ln as a flag, not a command
         ('python3 tools/plan.py brief x --add "y" > $S/b.txt', "orchestration"),
         ('python3 $S/tokens_since.py 1', "read"),                            # scratchpad counter
         ("python3 -c \"import json;s=json.load(open('.claude/settings.json'))\"", "read"),
         ("python3 - <<'EOF'\np='x'; open(p,'w').write('1')\nEOF", "inline"),
         ("python3 - <<'EOF'\nprint(open('x').read())\nEOF", "read"),
         ('git commit -m "x" -- plan/plan.yaml', "orchestration"),
         ('git commit -m "x" -- tools/a.py', "inline"),
         ('cp a b && sed -n 1p a', "inline"),
         ('site item set FA x', "orchestration"),
         ('python3 tools/usage_report.py --all', "inline"),
         ('python3 tools/usage_windows.py --help', "read"),
         ('site submit --item x ccin2p3 FA job.sh', "inline")]


def test_bash_kind():
    bad = [(c, w, m.bash_kind(c)) for c, w in CASES if m.bash_kind(c) != w]
    assert not bad, bad


def test_triggers():
    assert m.trigger_of("Stop hook feedback: x", None, {}) == "stop_hook"
    assert m.trigger_of("<task-notification>\n<task-id>a0123456789abcdef</task-id>", None, {}) == "agent_done"
    assert m.trigger_of("<task-notification>\n<task-id>bx</task-id>\n<tool-use-id>toolu_1</tool-use-id>", None,
                        {"toolu_1": "site events --wait --project FA"}) == "waiter"
    assert m.trigger_of("Another Claude session sent a message: <agent-message from=\"a1\">", {"kind": "peer"}, {}) == "handback"
    assert m.trigger_of("Yes, go ahead", {"kind": "human"}, {}) == "user"


if __name__ == "__main__":
    test_bash_kind()
    test_triggers()
    print("ok")
