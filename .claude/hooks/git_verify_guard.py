#!/usr/bin/env python3
"""PreToolUse convenience (setup redesign, D121): catches, at the command, the usual ways of skipping the claims check
of .githooks (pre-commit, pre-merge-commit), so a problem is seen before it is committed. It is not the gate and can
be got around (a script, an alias); the gate is a check of state: plan_check.py and auto_push.sh run
tools/verify_claims.py over every trunk commit since the last verified one (`plan claims`). Refused here:
  - git commit -n / --no-verify, git merge --no-verify;
  - git -c core.hooksPath=..., --config-env for it, git config of core.hooksPath (set, unset, any scope), any GIT_CONFIG_*
    variable (they inject config), and Edit/Write of .git/config or .git/hooks;
  - git pull (fetch, then merge: the merge is checked);
  - the same inside sh -c, bash -c or eval;
  - git merge <ref> and git cherry-pick <commit> whose incoming change to docs/results.tex fails
    tools/verify_claims.py (a fast-forward merge runs no hook, so the range is checked here, ff or not).
A merge's range check gets CHECK_S seconds; past that the merge is let through (the state check catches it later).
Installed with the switch-over (tools/switch_over.sh), which sets core.hooksPath once, before this guard exists."""
import json, os, re, shlex, subprocess, sys

WRAPPERS = {"command", "env", "sudo", "nice", "time", "nohup", "exec"}
GLOBAL_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--exec-path"}
COMMIT_ARG = set("mFCct")                      # git commit short options that take a value
CHECK_S = 90             # seconds for the claims check of a merge's range; registered with a 120 s hook timeout


def segments(cmd):
    lx = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
    lx.whitespace_split = True
    out, cur = [], []
    for t in lx:
        if t and set(t) <= set(";&|()<>"):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
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


def git(repo, *a):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True)


def refusal(cmd, cwd, depth=0):
    if re.search(r"\bGIT_CONFIG\w*", cmd):
        return "a GIT_CONFIG_* variable (it injects git config, core.hooksPath included)"
    if re.search(r"core\.hookspath", cmd, re.I) and not re.fullmatch(
            r"\s*git\s+config\s+(?:--get\s+)?core\.hookspath\s*", cmd, re.I):
        return "core.hooksPath named in a command (only reading it, `git config --get core.hooksPath`, is allowed)"
    if "git" not in cmd:
        return None
    try:
        segs = segments(cmd)
    except ValueError:
        return "a git command that cannot be parsed"
    for seg in segs:
        words = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
        prog = os.path.basename(words[0]) if words else ""
        inner = None
        if prog in ("sh", "bash", "zsh", "dash") and "-c" in words:
            k = words.index("-c") + 1
            inner = words[k] if k < len(words) else ""
        elif prog == "eval":
            inner = " ".join(words[1:])
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
        reading = [a.lower() for a in args] in (["core.hookspath"], ["--get", "core.hookspath"])
        if sub == "config" and not reading and any(a.lower().split("=")[0] == "core.hookspath" for a in args):
            return "git config core.hooksPath (set once by tools/switch_over.sh)"
        if sub == "commit" and commit_skips(args):
            return "git commit -n / --no-verify (it skips the claims check)"
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
        if re.search(r"\.git/(?:config\b|hooks/)", cmd) and re.search(r">|\btee\b|\bsed\b.*-i|\bcp\b|\bmv\b|\brm\b|\bln\b", cmd):
            why = "a write to .git/config or .git/hooks"
        else:
            why = refusal(cmd, inp.get("cwd") or os.getcwd())
    if why:
        sys.stderr.write("git verify guard: refused (%s).\n" % why)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
