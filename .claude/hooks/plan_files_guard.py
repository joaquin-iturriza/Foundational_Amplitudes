#!/usr/bin/env python3
"""PreToolUse guard (setup redesign, D121): the plan's files change only through tools/plan.py, which checks what it
records (the user's words, real step agents, machine checks). Refused:
  - Edit, Write, MultiEdit or NotebookEdit of plan/plan.yaml, plan/.state.json, plan/.check_cache.json or the lock;
  - a Bash command that names one of those files (with or without plan/, so a cd into plan/ counts) and is not a run
    of tools/plan.py or tools/plan_migrate.py, unless it only reads it (cat, head, tail, less, grep, wc, diff, git
    diff/log/show/status/blame).
plan/step_prompt.txt is code (the step agent's template), edited like any file. A write the command text cannot show
(a script that opens the file) is not caught here: plan_check re-validates every recorded step and quote on read."""
import json, os, re, shlex, sys

DATA = r"(?:plan/)?(?:plan\.yaml(?:\.lock)?|\.state\.json|\.check_cache\.json|\.plan_tmp_\w*)"
READERS = {"cat", "head", "tail", "less", "more", "grep", "rg", "wc", "diff", "file", "stat", "ls", "md5sum",
           "sha256sum"}
GIT_READ = {"diff", "log", "show", "status", "blame", "ls-files", "cat-file"}
TOOLS = re.compile(r"(?:^|/)tools/plan(?:_migrate)?\.py$")


def segments(cmd):
    """The simple commands of a shell line (split at ; && || | & and newlines), each as tokens, or None when the
    line cannot be parsed (then any mention of a plan file is refused)."""
    try:
        lx = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
        lx.whitespace_split = True
        toks = list(lx)
    except ValueError:
        return None
    out, cur = [], []
    for t in toks:
        if t and set(t) <= set(";&|()"):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def bash_refusal(cmd):
    if re.search(r"\bPLAN_[A-Z_]+", cmd):
        return "a PLAN_* variable (they point tools/plan.py at a test world, honoured only with PLAN_TEST=1 in tests)"
    if not re.search(DATA, cmd):
        return None
    segs = segments(cmd)
    if segs is None:
        return "a command naming a plan file that cannot be parsed"
    for seg in segs:
        if not any(re.search(DATA, t) for t in seg):
            continue
        targets = [seg[i + 1] for i, t in enumerate(seg[:-1]) if re.fullmatch(r"\d?(?:>|>>|>\||<>)", t)]
        targets += [t.lstrip("0123456789>|") for t in seg if re.match(r"\d?>", t) and len(t.lstrip("0123456789>|"))]
        if any(re.search(DATA + "$", x) for x in targets):
            return "a redirection into a plan file"
        seg = [t for i, t in enumerate(seg) if not re.fullmatch(r"\d?(?:>|>>|>\||<>|<)", t)
               and not (i and re.fullmatch(r"\d?(?:>|>>|>\||<>|<)", seg[i - 1]))]
        if not any(re.search(DATA, t) for t in seg):
            continue
        words = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]   # drop VAR=value prefixes
        if not words:
            continue
        prog = os.path.basename(words[0])
        if prog.startswith("python"):
            script = next((w for w in words[1:] if not w.startswith("-")), "")
            if TOOLS.search(script) and "-c" not in words[1:words.index(script)]:
                continue
            return "python on a plan file other than tools/plan.py"
        if prog in READERS and "-i" not in words and not any(w.startswith("--in-place") for w in words):
            continue
        if prog == "git" and len(words) > 1:
            sub = next((w for w in words[1:] if not w.startswith("-")), "")
            if sub in GIT_READ:
                continue
        return "%s on a plan file" % prog
    return None


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    tool, ti = inp.get("tool_name", ""), inp.get("tool_input") or {}
    why = None
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = ti.get("file_path") or ti.get("notebook_path") or ""
        if re.search(r"(?:^|/)plan/" + DATA + "$", path.replace(os.sep, "/")):
            why = "%s of %s" % (tool, path)
    elif tool == "Bash":
        why = bash_refusal(ti.get("command") or "")
    if why:
        sys.stderr.write("plan files guard: refused (%s). The plan changes only through tools/plan.py (plan add, "
                         "answer, ask, step, check, evidence, set-next, done ...), which checks what it records.\n" % why)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
