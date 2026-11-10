#!/usr/bin/env python3
"""worktree_fold_guard.sh refuses a command that deletes a worktree holding unfolded results, whatever form the rm
takes (repo-reviewer 2026-11-10 on 7395041f: /bin/rm, rm -f -r, xargs rm and command rm went past it), and lets prose
naming a worktree path, or a non-recursive rm, through.

Run:  python3 tests/test_worktree_fold_guard.py   (pytest also collects it)"""
import json
import os
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, ".claude", "hooks", "worktree_fold_guard.sh")

BLOCKED = [
    "rm -rf worktrees/wt-x",
    "/bin/rm -rf worktrees/wt-x",
    "rm -f -r worktrees/wt-x",
    "ls | xargs rm -rf worktrees/wt-x",
    "command rm -rf worktrees/wt-x",
    "\\rm -rf worktrees/wt-x",
    "sudo rm --recursive worktrees/wt-x",
    "cd /x && rm worktrees/wt-x -rf",
    "env FOO=1 rm -Rf worktrees/wt-x",
    "git worktree remove worktrees/wt-x",
]
PASSED = [
    "python3 tools/plan.py handoff e --state 'removed: rm -rf worktrees/wt-x after the fold'",
    "rm -f worktrees/wt-x/sweeps/a.json",
    "echo rm -rf worktrees/wt-x",
    "rm -rf worktrees/wt-other",          # no such worktree
    "ls worktrees/wt-x",
]


def run(repo, cmd):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=repo)
    return subprocess.run(["bash", HOOK], input=json.dumps({"tool_input": {"command": cmd}}),
                          capture_output=True, text=True, env=env).returncode


def make_repo():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "worktrees", "wt-x", "sweeps"))
    open(os.path.join(d, "worktrees", "wt-x", "sweeps", "result.json"), "w").write("{}\n")
    return d


def test_deletions_blocked():
    d = make_repo()
    try:
        bad = [c for c in BLOCKED if run(d, c) != 2]
        assert not bad, bad
    finally:
        shutil.rmtree(d)


def test_others_pass():
    d = make_repo()
    try:
        bad = [c for c in PASSED if run(d, c) != 0]
        assert not bad, bad
    finally:
        shutil.rmtree(d)


def test_folded_worktree_passes():
    d = make_repo()
    try:
        os.makedirs(os.path.join(d, "sweeps"))
        open(os.path.join(d, "sweeps", "result.json"), "w").write("{}\n")
        assert run(d, "rm -rf worktrees/wt-x") == 0
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    test_deletions_blocked()
    test_others_pass()
    test_folded_worktree_passes()
    print("ok")
