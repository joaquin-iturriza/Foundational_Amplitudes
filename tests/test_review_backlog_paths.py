"""review_backlog.sh bashpre (D558): a commit naming its paths after `--` is refused only for a changed file of the overdue
pillar; a commit run in another repository is not checked; a commit without `--` keeps the whole-pillar rule."""
import _tmpdir  # noqa: F401
import json, os, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo():
    repo = tempfile.mkdtemp(prefix="rbp_")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, env=env)
    g("init", "-q", "-b", "trunk")
    os.makedirs(os.path.join(repo, ".claude", "hooks")); os.makedirs(os.path.join(repo, "tools"))
    for f in ("review_backlog.sh", "cmdparse.py"):
        shutil.copy(os.path.join(ROOT, ".claude", "hooks", f), os.path.join(repo, ".claude", "hooks"))
    for f in ("review_state.py", "attribution.py", "review_weight.py"):
        shutil.copy(os.path.join(ROOT, "tools", f), os.path.join(repo, "tools"))
    open(os.path.join(repo, ".gitignore"), "w").write(".claude/.review_state/\n")
    for f in ("CLAUDE.md", "a.py", "b.py"):
        open(os.path.join(repo, f), "w").write("x = 1\n")
    g("add", "."); g("commit", "-qm", "base")
    run = lambda *a, stdin="": subprocess.run(["bash", ".claude/hooks/review_backlog.sh"] + list(a), cwd=repo,
                                              input=stdin, capture_output=True, text=True)
    run("init")
    return repo, run


def _pre(run, cmd):
    return run("bashpre", stdin=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}})).returncode


def test_named_paths_outside_overdue_pillar_pass():
    repo, run = _repo()
    open(os.path.join(repo, "CLAUDE.md"), "a").write("rule\n")          # claudemd-keeper overdue (1 line)
    assert _pre(run, "git commit -m x -- a.py b.py") == 0
    assert _pre(run, "git commit -m x -- CLAUDE.md") == 2
    assert _pre(run, "git commit -m x -- a.py CLAUDE.md") == 2
    assert _pre(run, "git commit -qam x") == 2                           # no `--`: whole pillar
    other = tempfile.mkdtemp(prefix="rbo_")
    assert _pre(run, "cd %s && git commit -m x -- CLAUDE.md" % other) == 0
    assert _pre(run, "git -C %s commit -m x" % other) == 0


def test_repo_reviewer_overdue_named_file_refused():
    repo, run = _repo()
    open(os.path.join(repo, "a.py"), "a").write("".join("y%d = %d\n" % (i, i) for i in range(600)))
    assert _pre(run, "git commit -m x -- a.py") == 2
    assert _pre(run, "git commit -m x -- b.py") == 0
    assert _pre(run, "git commit -qam x") == 2


def test_covering_pathspecs_refused():
    repo, run = _repo()
    open(os.path.join(repo, "a.py"), "a").write("".join("y%d = %d\n" % (i, i) for i in range(600)))
    assert _pre(run, "git commit -m x -- .") == 2
    assert _pre(run, "git commit -m x -- '*.py'") == 2
