"""The repository stays at trunk, main and live worktrees (agent_redesign step 20, D334). Each case runs in a temporary
git world (a bare origin and a clone holding copies of the scripts under test); the real repository is not touched.
  remove_worktree.sh   a merged branch is deleted here and on origin; a remote copy with commits trunk lacks is kept;
                       a merged branch with a stale upstream is deleted (git branch -d refused it, 2026-11-04)
  auto_push.sh         trunk is pushed; a feature branch is never published
  repo_hygiene.py      an ended agent's merged worktree is folded and removed (--fold-agents), a running one is kept,
                       an ended one with unmerged commits is listed; merged local and remote branches are listed;
                       a clean world exits 0
  model_trial.py       setup makes a detached worktree (no zz_trial_* branch)

    .venv-cpu/bin/python tests/test_repo_hygiene.py
"""
import json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, cwd, check=True, env=None):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise AssertionError("%s -> %d\n%s%s" % (cmd, r.returncode, r.stdout, r.stderr))
    return r.stdout + r.stderr


def world():
    t = tempfile.mkdtemp(prefix="repo_hygiene_")
    sh("git init -q --bare -b trunk origin.git", t)
    sh("git clone -q origin.git repo 2>/dev/null; true", t)
    r = os.path.join(t, "repo")
    sh("git checkout -q -b trunk 2>/dev/null || true; git config user.email t@t; git config user.name t", r)
    for f in ("scripts/remove_worktree.sh", "scripts/fold_worktree.sh", "tools/repo_hygiene.py",
              ".claude/hooks/auto_push.sh"):
        os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
        shutil.copy(os.path.join(ROOT, f), os.path.join(r, f))
    open(os.path.join(r, ".gitignore"), "w").write("worktrees/\n.claude/worktrees/\n")
    sh("git add -A && git commit -qm base && git push -q -u origin trunk", r)
    return t, r


def feature(r, path, br, merge=True, push=False):
    sh("git worktree add -q -b %s %s trunk" % (br, path), r)
    w = os.path.join(r, path)
    open(os.path.join(w, br + ".txt"), "w").write(br)
    sh("git add -A && git commit -qm %s" % br, w)
    if push:
        sh("git push -q -u origin %s" % br, w)
    if merge:
        sh("git merge -q --no-edit %s" % br, r)
    return w


def remote_heads(t):
    return set(l.split("refs/heads/")[1] for l in sh("git ls-remote --heads origin.git", t).split("\n") if l)


def test_remove_worktree_deletes_remote_copy():
    t, r = world()
    feature(r, "worktrees/wt-a", "a", merge=True, push=True)          # merged, remote copy in trunk
    w = feature(r, "worktrees/wt-b", "b", merge=False, push=True)     # remote copy gets a commit trunk lacks
    open(os.path.join(w, "b2.txt"), "w").write("x")
    sh("git add -A && git commit -qm b2 && git push -q", w)
    sh("git reset -q --hard HEAD~1", w)                               # local b back to its first commit
    sh("git merge -q --no-edit b", r)
    out = sh("bash scripts/remove_worktree.sh worktrees/wt-a", r)
    assert "deleted origin/a" in out and "a" not in remote_heads(t), out
    assert sh("git branch --list a", r).strip() == ""
    out = sh("bash scripts/remove_worktree.sh worktrees/wt-b", r)
    assert "origin/b has commits trunk lacks; kept" in out and "b" in remote_heads(t), out
    assert sh("git branch --list b", r).strip() == "", "local merged b deleted although its upstream is ahead"


def test_auto_push_trunk_only():
    t, r = world()
    feature(r, "worktrees/wt-c", "c", merge=False)
    open(os.path.join(r, "t.txt"), "w").write("t")
    sh("git add -A && git commit -qm t", r)
    out = sh("bash .claude/hooks/auto_push.sh", r, env=dict(os.environ, CLAUDE_PROJECT_DIR=r))
    assert "c" not in remote_heads(t), out
    assert sh("git rev-parse trunk", r) == sh("git rev-parse trunk", os.path.join(t, "origin.git")), out


def transcript(tdir, aid, ended, age):
    d = os.path.join(tdir, "sess", "subagents")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "agent-%s.jsonl" % aid)
    msg = {"type": "assistant", "message": {"stop_reason": "end_turn" if ended else "tool_use", "content": []}}
    open(f, "w").write(json.dumps(msg) + "\n")
    os.utime(f, (time.time() - age, time.time() - age))


def test_repo_hygiene_folds_ended_agents_and_lists_the_rest():
    t, r = world()
    tdir = os.path.join(t, "transcripts")
    ended, running, unmerged = "a" + "1" * 16, "a" + "2" * 16, "a" + "3" * 16
    for aid, merge in ((ended, True), (running, True), (unmerged, False)):
        feature(r, ".claude/worktrees/agent-%s" % aid, "worktree-agent-%s" % aid, merge=merge)
    transcript(tdir, ended, True, 3600)
    transcript(tdir, running, True, 60)            # wrote a minute ago: still running
    transcript(tdir, unmerged, True, 3600)
    sh("git branch merged_left trunk && git push -q origin merged_left", r)
    env = dict(os.environ, HYGIENE_TRANSCRIPTS=tdir)
    p = subprocess.run([sys.executable, "tools/repo_hygiene.py", "--fold-agents", "--no-fetch"], cwd=r,
                       capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    sh("git fetch -q --prune origin", r)
    assert p.returncode == 1, out
    assert "folded and removed .claude/worktrees/agent-%s" % ended in out, out
    assert not os.path.isdir(os.path.join(r, ".claude/worktrees/agent-" + ended)), out
    assert os.path.isdir(os.path.join(r, ".claude/worktrees/agent-" + running)), out
    assert "agent_unmerged" in out and unmerged in out, out
    assert "branch_merged    merged_left" in out, out
    p = subprocess.run([sys.executable, "tools/repo_hygiene.py", "--no-fetch"], cwd=r, capture_output=True,
                       text=True, env=env)
    assert "remote_merged    origin/merged_left" in p.stdout, p.stdout
    # clean world: only trunk
    t2, r2 = world()
    p = subprocess.run([sys.executable, "tools/repo_hygiene.py", "--no-fetch"], cwd=r2, capture_output=True,
                       text=True, env=env)
    assert p.returncode == 0 and "repo_hygiene: clean" in p.stdout, p.stdout


def test_model_trial_setup_is_detached():
    src = open(os.path.join(ROOT, "tools", "model_trial.py")).read()
    body = src[src.index("def setup("):src.index("\n\n\n", src.index("def setup("))]
    assert '"--detach"' in body and '"-b"' not in body, body


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k, flush=True)
    print("%d passed" % n)
