"""plan check --before: passes_at runs a check in a detached worktree at a commit.

    .venv-cpu/bin/python tests/test_plan_check_before.py
"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402

old = subprocess.run(["git", "-C", ROOT, "rev-parse", "145d14839^"], capture_output=True, text=True).stdout.strip()
assert plan.passes_at("test -f tools/plan.py", old)
assert not plan.passes_at("test -f tests/test_plan_check_before.py", old)   # this file did not exist then
assert "plan_before_" not in subprocess.run(["git", "-C", ROOT, "worktree", "list"], capture_output=True, text=True).stdout
print("ok")
