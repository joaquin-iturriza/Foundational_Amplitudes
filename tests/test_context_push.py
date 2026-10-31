"""The push hook's output limit, pending parts, and the 'agents: no' / 'on: write' header keys (agent_redesign step 5)."""
import importlib.util, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, ".claude", "hooks", "context_push.py")


def hook(sections):
    d, st = tempfile.mkdtemp(prefix="secs_"), tempfile.mkdtemp(prefix="push_")
    for name, (head, body) in sections.items():
        open(os.path.join(d, name + ".txt"), "w").write(head + "\n---\n" + body)
    os.environ.update(CONTEXT_DIRS=d, CONTEXT_STATE=st, CLAUDE_PROJECT_DIR=ROOT)
    sys.path.insert(0, os.path.dirname(HOOK))
    spec = importlib.util.spec_from_file_location("cp_test", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def call(m, who, tool, ti, mode="post"):
    inp = {"session_id": "s", "tool_name": tool, "tool_input": ti,
           "hook_event_name": "PreToolUse" if mode == "pre" else "PostToolUse"}
    if who != "main":
        inp["agent_id"] = who
    return m.respond(inp, mode)[1] or ""


def test_limit_and_parts():
    big = "".join("line %05d %s\n" % (i, "x" * 60) for i in range(200))          # ~14.6k chars
    m = hook({"s1": ("cmds: \\bfoo\\b", big), "s2": ("cmds: \\bfoo\\b", "small text\n")})
    outs = [call(m, "A", "Bash", {"command": "foo"}, "pre")]
    outs += [call(m, "A", "Read", {"file_path": "/nowhere"}) for _ in range(3)]
    assert all(len(o) <= m.LIMIT for o in outs), [len(o) for o in outs]
    joined = "".join(outs)
    assert "part 1 of 2" in joined and "part 2 of 2" in joined and "REFERENCE s2" in joined
    assert sum(joined.count("line %05d " % i) for i in range(200)) == 200       # every line once
    assert outs[-1] == ""                                                         # nothing left
    assert call(m, "A", "Bash", {"command": "foo"}, "pre") == ""                  # once per agent


def test_agents_no():
    m = hook({"plan": ("agents: no\ncmds: tools/plan\\.py", "plan text\n")})
    assert call(m, "A", "Bash", {"command": "python3 tools/plan.py show x"}, "pre") == ""
    assert "REFERENCE plan" in call(m, "main", "Bash", {"command": "python3 tools/plan.py show x"}, "pre")


def test_on_write():
    m = hook({"hk": ("on: write\npaths: .claude/hooks/**\ncmds: \\bhook_incident\\.py\\b", "hooks text\n")})
    assert call(m, "A", "Read", {"file_path": os.path.join(ROOT, ".claude/hooks/x.py")}) == ""
    assert call(m, "A", "Bash", {"command": "grep -n foo .claude/hooks/x.py"}, "pre") == ""
    assert "REFERENCE hk" in call(m, "A", "Edit", {"file_path": os.path.join(ROOT, ".claude/hooks/x.py")}, "pre")
    assert "REFERENCE hk" in call(m, "B", "Bash", {"command": "python3 tools/hook_incident.py x --why y"}, "pre")


if __name__ == "__main__":
    for t in (test_limit_and_parts, test_agents_no, test_on_write):
        t()
        print("PASS", t.__name__)
