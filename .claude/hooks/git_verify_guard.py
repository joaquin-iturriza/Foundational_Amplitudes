#!/usr/bin/env python3
"""PreToolUse convenience (setup redesign, D121): catches, at the command, the usual ways of skipping the claims check
of .githooks (pre-commit, pre-merge-commit), so a problem is seen before it is committed. It is not the gate and can
be got around (a script, an alias); the gate is a check of state: plan_check.py and auto_push.sh run
tools/verify_claims.py over every trunk commit since the last verified one (`plan claims`). Refused here:
  - git commit -n / --no-verify, git merge --no-verify;
  - git -c core.hooksPath=..., --config-env for it, git config of core.hooksPath (set, unset, any scope), any GIT_CONFIG_*
    variable (they inject config), and Edit/Write of .git/config or .git/hooks;
  - git pull (fetch, then merge: the merge is checked);
  - the same inside sh -c, bash -c, eval or a heredoc fed to a shell;
  - git merge <ref> and git cherry-pick <commit> whose incoming change to docs/results.tex fails
    tools/verify_claims.py (a fast-forward merge runs no hook, so the range is checked here, ff or not);
  - git merge <ref> into trunk while the range changes a pillar's files its reviewer has not passed
    (tools/review_state.py; agent redesign B6);
  - a git commit whose -m message attributes the work to an assistant (tools/attribution.py; B7).
The command is read through cmdparse.py (shared by the Bash guards): heredoc bodies fed to python, cat, git ... are
data and quoted strings are values, so a body or a message that only names these forms is not refused. A command that
cannot be parsed is refused only when a risky form appears outside quotes and heredocs.
A merge's range check gets CHECK_S seconds; past that the merge is let through (the state check catches it later).
Installed with the switch-over (tools/switch_over.sh), which sets core.hooksPath once, before this guard exists."""
import json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmdparse  # noqa: E402

WRAPPERS = {"command", "env", "sudo", "nice", "time", "nohup", "exec", "then", "do", "else", "elif", "if", "while",
            "until", "!", "{", "xargs"}
# for a command cmdparse cannot parse (an unbalanced quote outside any heredoc, $'...'): refused only when one of the
# risky forms appears in its code, outside heredoc bodies and quoted strings (scan_text); a heredoc with an apostrophe
# before an ordinary git add/commit parses once its body is taken out
RISKY = re.compile(r"core\.hookspath|--no-ve|\bGIT_CONFIG|\bcommit\b[^;&|\n]*\s-(?!-)[A-Za-z]*n|"
                   r"\bgit\b[^;&|\n]*\s(?:pull|merge|cherry-pick)\b", re.I)
RISKY_C = re.compile(r"\bgit\b[^;&|\n]*\s-c\s*['\"]?core\.", re.I)      # on the code with its quotes kept
READ_OPTS = {"--get", "--get-all", "--show-origin", "--show-scope", "--local", "--global", "--system", "--worktree"}
GLOBAL_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--exec-path"}
COMMIT_ARG = set("mFCct")                      # git commit short options that take a value
CHECK_S = 90             # seconds for the claims check of a merge's range; registered with a 120 s hook timeout


def run_segments(cmd):
    """Every simple command bash would run (cmdparse.all_segments: sh -c / bash -lc / eval inner commands and command
    substitutions included), with redirections (2>&1, > f, < f) dropped; None when the text cannot be parsed."""
    segs = cmdparse.all_segments(cmd)
    if segs is None:
        return None
    out = [k for k in (cmdparse.strip_redirs(seg) for seg in segs) if k]
    return out


def no_verify(tok):
    return tok.startswith("--no-ve") and "--no-verify".startswith(tok.split("=")[0])


def git_call(seg):
    """(-C dir or None, [-c k=v...], subcommand, its args) when the segment runs git, else None."""
    i = 0
    while i < len(seg) and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", seg[i]) or os.path.basename(seg[i]) in WRAPPERS):
        i += 1
    if i >= len(seg) or os.path.basename(seg[i]) != "git":
        return None
    i, cdir, confs = i + 1, None, []
    while i < len(seg) and seg[i].startswith("-"):
        t = seg[i]
        name = t.split("=")[0]
        val = t.split("=", 1)[1] if "=" in t and name in GLOBAL_ARG else None
        if name in GLOBAL_ARG and val is None:
            i += 1
            val = seg[i] if i < len(seg) else ""
        if name == "-C":
            cdir = os.path.join(cdir or "", val)
        elif name in ("-c", "--config-env"):
            confs.append(val)
        i += 1
    return (cdir, confs, seg[i] if i < len(seg) else "", seg[i + 1:])


def commit_skips(args):
    skip = False
    for t in args:
        if skip:
            skip = False
            continue
        if t == "--":
            break
        if no_verify(t):
            return True
        if t.startswith("-") and not t.startswith("--") and len(t) > 1:
            for j, ch in enumerate(t[1:]):
                if ch == "n":
                    return True
                if ch in COMMIT_ARG:
                    skip = j == len(t) - 2          # the value is the next token
                    break
        elif t in ("--message", "--file", "--reuse-message", "--reedit-message", "--template", "--author", "--date",
                   "--cleanup", "--fixup", "--squash", "--trailer", "--pathspec-from-file"):
            skip = True
    return False


def refs(args, with_value=("-m", "-s", "-X", "--strategy", "--strategy-option", "-F", "--file", "--into-name")):
    out, skip = [], False
    for t in args:
        if skip:
            skip = False
        elif t in with_value:
            skip = True
        elif not t.startswith("-"):
            out.append(t)
    return out


def attributed(repo, args):
    """a `git commit -m/--message` whose message attributes the work to an assistant (tools/attribution.py, the
    pattern .githooks/commit-msg applies to every commit)."""
    msgs, k = [], 0
    while k < len(args):
        t = args[k]
        if t in ("-m", "--message") and k + 1 < len(args):
            msgs.append(args[k + 1])
            k += 2
            continue
        if t.startswith("--message="):
            msgs.append(t.split("=", 1)[1])
        elif t.startswith("-m") and len(t) > 2 and not t.startswith("--"):
            msgs.append(t[2:])
        k += 1
    if not msgs:
        return None
    top = git(repo, "rev-parse", "--show-toplevel").stdout.strip() or repo
    sys.path.insert(0, os.path.join(top, "tools"))
    try:
        import attribution
    except ImportError:
        return None
    bad = attribution.offending("\n\n".join(msgs))
    return ("a commit message attributing the work to an assistant (CLAUDE.md rule 4): %s" % "; ".join(bad)) if bad else None


def heredoc_attributed(cmd, cwd):
    """a commit message given as a heredoc (`git commit -F - <<'EOF'`) that attributes the work to an assistant."""
    _, bodies = cmdparse.heredocs(cmd)
    for line, body in bodies:
        if cmdparse.feeder(line) == "git" and re.search(r"\bcommit\b", line):
            why = attributed(cwd, ["-m", body])
            if why:
                return why
    return None


def reads_hookspath(sub, args):
    """`git config [--get|--show-origin|scope...] core.hooksPath` or `git config get core.hooksPath`: a read."""
    a = [x.lower() for x in args]
    if a[:1] == ["get"]:
        a = a[1:]
    return sub == "config" and [x for x in a if x not in READ_OPTS] == ["core.hookspath"]


def check_range(repo, rng):
    v = os.path.join(repo, "tools", "verify_claims.py")
    if not os.path.exists(v):
        return None
    try:
        r = subprocess.run([sys.executable, v, "--root", repo, "--rev", rng], capture_output=True, text=True,
                           timeout=CHECK_S)
    except subprocess.TimeoutExpired:
        sys.stderr.write("git verify guard: the claims check of %s took over %d s; let through (plan_check checks "
                         "trunk afterwards)\n" % (rng, CHECK_S))
        return None
    return None if r.returncode == 0 else (r.stdout + r.stderr).strip()[-1500:]


def unreviewed_merge(repo, base, ref):
    """Review before a merge (agent redesign B6): a merge into trunk (fast-forward or not) of a range that changes a
    pillar's files the pillar's reviewer has not passed (tools/review_state.py) is refused; the threshold gate stays
    the frequency, this is the floor."""
    if git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip() != "trunk":
        return None
    top = git(repo, "rev-parse", "--show-toplevel").stdout.strip()
    rs = os.path.join(top, "tools", "review_state.py")
    if not os.path.exists(rs):
        return None
    r = subprocess.run([sys.executable, rs, "unreviewed", "%s..%s" % (base, ref), "--root", top], capture_output=True,
                       text=True, timeout=60)
    if r.returncode == 0:
        return None
    return ("merging %s into trunk brings changes no reviewer has passed (%s): run the pillar's reviewer on its "
            "backlog (it advances its watermark on a pass), then merge" % (ref, "; ".join(r.stdout.split("\n")[:3]).strip()))


def git(repo, *a):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True)


def refusal(cmd, cwd, depth=0):
    code, scan = cmdparse.code_text(cmd), cmdparse.scan_text(cmd)
    if re.search(r"\bGIT_CONFIG\w*", code):     # quotes kept: env "GIT_CONFIG_COUNT=1" git ... injects too
        return "a GIT_CONFIG_* variable (it injects git config, core.hooksPath included)"
    hp = re.search(r"core\.hookspath", scan, re.I)
    if not hp and "git" not in code:
        return None
    segs = run_segments(cmd)                  # all_segments takes the heredocs out itself
    if segs is None:
        m = RISKY.search(scan) or RISKY_C.search(code)
        return "a command that cannot be parsed, with %r outside quotes and heredocs (run it on its own)" % (
            m.group(0)[:40]) if m else None
    for seg in segs if hp else ():
        if any(re.search(r"core\.hookspath", t, re.I) for t in seg):
            g = git_call(seg)
            words = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
            shell = bool(words) and os.path.basename(words[0]) in ("sh", "bash", "zsh", "dash", "eval")
            if not shell and not (g and not g[1] and reads_hookspath(g[2], g[3])):
                return ("core.hooksPath named in a command (only reading it, `git config core.hooksPath`, "
                        "is allowed)")
    for seg in segs:
        words = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
        prog = os.path.basename(words[0]) if words else ""
        inner = (cmdparse.inner_commands(seg) or [None])[0]
        if inner is not None:
            why = refusal(inner, cwd, depth + 1) if depth < 4 else "nested shells"
            if why:
                return why
            continue
        g = git_call(seg)
        if not g:
            continue
        cdir, confs, sub, args = g
        repo = os.path.join(cwd, cdir) if cdir else cwd
        if any(c.split("=")[0].lower() in ("core.hookspath",) for c in confs):
            return "git -c core.hooksPath (it skips the claims hooks)"
        if sub == "config" and not reads_hookspath(sub, args) and any(a.lower().split("=")[0] == "core.hookspath" for a in args):
            return "git config core.hooksPath (set once by tools/switch_over.sh)"
        if sub == "commit" and commit_skips(args):
            return "git commit -n / --no-verify (it skips the claims check)"
        if sub == "commit":
            why = attributed(repo, args)
            if why:
                return why
        if sub == "merge" and any(no_verify(a) for a in args):
            return "git merge --no-verify"
        if sub == "pull":
            return "git pull (git fetch, then git merge, so the incoming change is checked)"
        if sub == "merge" and not any(a in ("--abort", "--continue", "--quit") for a in args):
            for ref in refs(args):
                base = git(repo, "merge-base", "HEAD", ref).stdout.strip()
                if base:
                    why = check_range(repo, "%s..%s" % (base, ref))
                    if why:
                        return "merging %s brings numbers the claims check refuses:\n%s" % (ref, why)
                    why = unreviewed_merge(repo, base, ref)
                    if why:
                        return why
        if sub == "cherry-pick" and not any(a in ("--abort", "--continue", "--quit", "--skip") for a in args):
            for ref in refs(args, with_value=("-m", "--mainline", "-X", "--strategy", "--strategy-option")):
                if ".." in ref:
                    return "cherry-pick of a range (pick commits one by one, each is checked)"
                why = check_range(repo, "%s^..%s" % (ref, ref))
                if why:
                    return "cherry-picking %s brings numbers the claims check refuses:\n%s" % (ref, why)
    return None


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    tool, ti = inp.get("tool_name", ""), inp.get("tool_input") or {}
    why = None
    if tool in ("Edit", "Write", "MultiEdit"):
        p = (ti.get("file_path") or "").replace(os.sep, "/")
        if re.search(r"(?:^|/)\.git/(?:config$|hooks/)", p):
            why = "%s of %s" % (tool, p)
    elif tool == "Bash":
        cmd = ti.get("command") or ""
        code = cmdparse.code_text(cmd)         # a heredoc body that only names .git/hooks writes nothing there
        if re.search(r"\.git/(?:config\b|hooks/)", code) and re.search(r">|\btee\b|\bsed\b.*-i|\bcp\b|\bmv\b|\brm\b|\bln\b", code):
            why = "a write to .git/config or .git/hooks"
        else:
            why = refusal(cmd, inp.get("cwd") or os.getcwd()) or heredoc_attributed(cmd, inp.get("cwd") or os.getcwd())
    if why:
        sys.stderr.write("git verify guard: refused (%s).\n" % why)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
