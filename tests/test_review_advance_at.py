#!/usr/bin/env python3
"""review_backlog.sh `advance <reviewer> --at <commit>` records the commit the reviewer read, so the merge guard
(tools/review_state.py) accepts a worktree branch's merge; `advance` without --at stays at HEAD.

Run:  python3 tests/test_review_advance_at.py   (pytest also collects it)"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHO = "repo-reviewer"


def sh(cwd, *a, check=True):
    return subprocess.run(list(a), cwd=cwd, capture_output=True, text=True, check=check)


def make_repo():
    d = tempfile.mkdtemp()
    g = lambda *a: sh(d, "git", "-c", "user.name=t", "-c", "user.email=t@t", *a)
    g("init", "-q", "-b", "trunk")
    os.makedirs(os.path.join(d, ".claude", "hooks"))
    os.makedirs(os.path.join(d, "tools"))
    for f in (".claude/hooks/review_backlog.sh", ".claude/hooks/cmdparse.py"):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), os.path.join(d, f))
    shutil.copy(os.path.join(ROOT, "tools", "review_state.py"), os.path.join(d, "tools"))
    open(os.path.join(d, "a.py"), "w").write("x = 1\n")
    g("add", "-A")
    g("commit", "-qm", "base")
    return d, g


def advance(d, *a, check=True):
    return sh(d, "bash", ".claude/hooks/review_backlog.sh", "advance", WHO, *a, check=check)


def unreviewed(d, rng):
    return sh(d, sys.executable, "tools/review_state.py", "unreviewed", rng, "--root", d, check=False).returncode


def watermark(d):
    return open(os.path.join(d, ".claude", ".review_state", WHO)).read().split()[0]


def test_advance_at_branch_tip():
    d, g = make_repo()
    try:
        base = g("rev-parse", "HEAD").stdout.strip()
        advance(d)
        assert watermark(d) == base            # without --at: HEAD, unchanged behaviour
        g("branch", "feat")
        g("checkout", "-q", "feat")
        open(os.path.join(d, "a.py"), "w").write("x = 2\n")
        g("commit", "-qam", "change")
        tip = g("rev-parse", "HEAD").stdout.strip()
        g("checkout", "-q", "trunk")
        assert unreviewed(d, "trunk..feat") == 1   # watermark at trunk's head: refused
        advance(d)                                  # the misfire: trunk's head
        assert watermark(d) == base and unreviewed(d, "trunk..feat") == 1
        advance(d, "--at", tip)                     # main checkout, branch tip
        assert watermark(d) == tip and unreviewed(d, "trunk..feat") == 0
        advance(d, "--at", "feat")                  # a branch name resolves too
        assert watermark(d) == tip
        assert advance(d, "--at", "nope", check=False).returncode == 2
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    test_advance_at_branch_tip()
    print("ok")
