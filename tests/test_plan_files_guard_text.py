"""plan_files_guard.py: a plan file named only as text (a sed pattern on another file, a triple-quoted string in a
heredoc script) is not a write to it (2026-11-09: two refusals of edits to scripts/remove_worktree.sh); real writes
still refuse.

    .venv-cpu/bin/python tests/test_plan_files_guard_text.py
    GUARD=<path to a copy> ...      tests that copy instead of the installed hook
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.environ.get("GUARD") or os.path.join(ROOT, ".claude", "hooks", "plan_files_guard.py")
P, S = "plan/" + "plan.yaml", "plan/" + ".state.json"
CASES = [
    ("sed -i on a script, its pattern names the plan state",
     "sed -i 's#a|%s) x#b#' scripts/remove_worktree.sh" % S, False),
    ("python heredoc, a triple-quoted text names the plan state",
     "python3 - <<'EOF'\np='scripts/remove_worktree.sh'; s=open(p).read()\no='''x|%s) continue ;; esac'''\n"
     "n='''y'''\nopen(p,'w').write(s.replace(o,n))\nEOF" % S, False),
    ("sed -i on the plan", "sed -i 's/a/b/' %s" % P, True),
    ("sed w into the plan", "sed -n 'w %s' notes.txt" % P, True),
    ("awk print into the plan", "awk '{print > \"%s\"}' x" % P, True),
    ("python heredoc writing the plan", "python3 - <<'EOF'\nopen('%s','w').write('x')\nEOF" % P, True),
    ("python heredoc, a variable set to the plan path", "python3 - <<'EOF'\np = '%s'\nopen(p,'w').write('x')\nEOF" % P,
     True),
    ("redirect into the plan", "echo x > %s" % P, True),
    ("grep the plan", "grep -n foo %s" % P, False),
    # 2026-11-10: heredocs patching tests/*.py, the test code inside a multi-line triple-quoted string
    ("python heredoc, test code in a triple-quoted string writes a variable named plan (a temp path)",
     "python3 - <<'EOF'\np='tests/test_x.py'\ns=open(p).read()\nadd='''\ndef test_y():\n"
     "    plan = os.path.join(t, \"plan.yaml\")\n    open(plan, \"w\").write(json.dumps({\"entries\": []}))\n'''\n"
     "open(p,'w').write(s+add)\nEOF", False),
    ("python heredoc, test code in a triple-quoted string writes PLAN in a test world",
     "python3 - <<'EOF'\np='tests/test_x.py'\ns=open(p).read()\nnew='''TMP = tempfile.mkdtemp()\n"
     "PLAN = os.path.join(TMP, \"plan\", \"plan.yaml\")\nopen(PLAN, \"w\").write(src)\n'''\n"
     "open(p,'w').write(s.replace('x', new))\nEOF", False),
    ("python heredoc, a real plan write after a triple-quoted string",
     "python3 - <<'EOF'\nnote='''text'''\nopen('%s','w').write('x')\nEOF" % P, True),
    ("python heredoc, a real plan write through a variable after a multi-line string",
     "python3 - <<'EOF'\ndoc='''a\nb'''\nq = '%s'\nopen(q,'w').write(doc)\nEOF" % P, True),
]


def refused(cmd):
    r = subprocess.run([sys.executable, GUARD], input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd},
                                                                   "hook_event_name": "PreToolUse"}),
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 2 or '"deny"' in r.stdout


def test_cases():
    bad = [(name, want) for name, cmd, want in CASES if refused(cmd) != want]
    assert not bad, bad


if __name__ == "__main__":
    test_cases()
    print("ok test_cases (%d cases)" % len(CASES))
