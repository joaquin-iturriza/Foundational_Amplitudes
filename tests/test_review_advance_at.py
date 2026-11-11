#!/usr/bin/env python3
"""review_backlog.sh `advance <reviewer> --at <commit>` records the commit the reviewer read, so the merge guard
(tools/review_state.py) accepts a worktree branch's merge; `advance` without --at stays at HEAD. The watermark (what
the gate measures the checkout's backlog from) moves only forward along the checkout's history: a branch tip that is
not in it is listed in .passed only (2026-11-10: sigma-gen's pass overwrote trunk's watermark).

Run:  python3 tests/test_review_advance_at.py   (pytest also collects it)"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
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
        advance(d, "--at", tip)                     # main checkout, branch tip: listed, watermark stays on trunk
        assert watermark(d) == base and unreviewed(d, "trunk..feat") == 0
        advance(d, "--at", "feat")                  # a branch name resolves too
        assert watermark(d) == base
        g("merge", "-q", "--no-edit", "feat")       # once merged, a pass at trunk's head moves it forward
        advance(d)
        assert watermark(d) == tip
        assert advance(d, "--at", "nope", check=False).returncode == 2
    finally:
        shutil.rmtree(d)


def test_two_branches_passing_in_sequence():
    d, g = make_repo()
    try:
        advance(d)
        base = watermark(d)
        tips = []
        for br, val in (("f1", "2"), ("f2", "3")):
            g("checkout", "-q", "-b", br, "trunk")
            open(os.path.join(d, "a.py"), "w").write("x = %s\n" % val)
            g("commit", "-qam", br)
            tips.append(g("rev-parse", "HEAD").stdout.strip())
            g("checkout", "-q", "trunk")
        advance(d, "--at", "f1")
        advance(d, "--at", "f2")                    # neither branch is in trunk: the watermark stays on trunk
        assert watermark(d) == base
        assert unreviewed(d, "trunk..f1") == 0      # f1 still passed (listed)
        assert unreviewed(d, "trunk..f2") == 0
        g("checkout", "-q", "-b", "f3", "trunk")
        open(os.path.join(d, "a.py"), "w").write("x = 4\n")
        g("commit", "-qam", "f3")
        g("checkout", "-q", "trunk")
        assert unreviewed(d, "trunk..f3") == 1      # never passed: refused
        open(os.path.join(d, "a.py"), "w").write("x = 5\n")
        g("commit", "-qam", "trunk work")
        head = g("rev-parse", "HEAD").stdout.strip()
        advance(d)
        assert watermark(d) == head
        advance(d, "--at", base)                    # an older tip never moves the watermark back
        assert watermark(d) == head
    finally:
        shutil.rmtree(d)


def write_watermark(d, sha):
    os.makedirs(os.path.join(d, ".claude", ".review_state"), exist_ok=True)
    open(os.path.join(d, ".claude", ".review_state", WHO), "w").write("%s 0 0\n" % sha)


def test_diverged_branch_pass_keeps_trunk_watermark():
    """tonight's case (2026-11-10): trunk passed at f5ac899c, then sigma-gen's worktree pass `advance --at` its tip
    5e5b2788 (a sibling of trunk) overwrote the watermark and trunk's reviewed commits came back as backlog"""
    d, g = make_repo()
    try:
        g("checkout", "-q", "-b", "sigma-gen")
        open(os.path.join(d, "a.py"), "w").write("x = 'sigma'\n")
        g("commit", "-qam", "sigma-gen work")
        sg = g("rev-parse", "HEAD").stdout.strip()
        g("checkout", "-q", "trunk")
        open(os.path.join(d, "b.py"), "w").write("y = 1\n")
        g("add", "b.py")
        g("commit", "-qm", "trunk work")
        trunk_pass = g("rev-parse", "HEAD").stdout.strip()
        advance(d)
        assert watermark(d) == trunk_pass
        advance(d, "--at", "sigma-gen")             # from a worktree in real life; same refs here
        assert watermark(d) == trunk_pass           # was sg before the fix
        assert unreviewed(d, "trunk..sigma-gen") == 0
        assert sg in open(os.path.join(d, ".claude", ".review_state", WHO + ".passed")).read().split()
        # a watermark already stranded off trunk (the state the bug left) is replaced by the next trunk pass
        write_watermark(d, sg)
        advance(d, "--at", trunk_pass)
        assert watermark(d) == trunk_pass
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    test_advance_at_branch_tip()
    test_two_branches_passing_in_sequence()
    test_diverged_branch_pass_keeps_trunk_watermark()
    print("ok")
