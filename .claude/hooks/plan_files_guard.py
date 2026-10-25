#!/usr/bin/env python3
"""PreToolUse guard (setup redesign, D121): the plan's files change only through tools/plan.py, which checks what it
records (the user's words, real step agents, machine checks). Refused:
  - Edit, Write, MultiEdit or NotebookEdit of plan/plan.yaml, plan/.state.json, plan/.check_cache.json or the lock;
  - a Bash command that names one of those files (with or without plan/, so a cd into plan/ counts) and is not a run
    of tools/plan.py or tools/plan_migrate.py, unless it only reads it (cat, head, tail, less, grep, wc, diff, git
    diff/log/show/status/blame).
Heredoc bodies are taken out before the command is read: a body fed to an interpreter (python, perl, ...) is refused
only when a line of it writes and names a plan file (itself, or a variable set to its path); a shell's body is read
as a command; any other body (a commit message, cat's text) is data.
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


HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
INTERP = re.compile(r"^(?:python[0-9.]*|perl|ruby|node)$")
SHELLS = {"bash", "sh", "zsh", "dash"}
WRITE = re.compile(r"""open\([^)\n]*,\s*(?:mode\s*=\s*)?['"][^'"]*[wax+]|\.write_(?:text|bytes)\(|"""
                   r"""\bos\.(?:replace|rename|remove|unlink)\(|\bshutil\.|\.(?:unlink|rename|replace)\(|>\s*\$?\w""")


def heredocs(cmd):
    """(the command with its heredoc bodies taken out, [(the line that feeds a body, the body)])."""
    lines, out, bodies, i = cmd.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in HEREDOC.finditer(line):
            tabs, word, body = m.group(1), m.group(3), []
            while i < len(lines) and (lines[i].lstrip("\t") if tabs else lines[i]) != word:
                body.append(lines[i])
                i += 1
            i += 1                                     # the closing word
            bodies.append((line, "\n".join(body)))
    return "\n".join(out), bodies


def writes_plan(body):
    """A line of an interpreter's script that writes and names a plan file, itself or through a variable set to one."""
    # variables set to a plan file's PATH (p = "plan/plan.yaml", Path(...), os.path.join(...)), not to its contents
    names = {m.group(1) for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=([^=\n].*)$", body, re.M)
             if re.search(DATA, m.group(2)) and not re.search(r"\bopen\(|\.read|\bload\(|safe_load", m.group(2))}
    for line in body.splitlines():
        if WRITE.search(line) and (re.search(DATA, line) or any(re.search(r"\b%s\b" % re.escape(n), line)
                                                                 for n in names)):
            return line.strip()
    return None


def heredoc_refusal(line, body):
    """Why a heredoc body is refused (None: it is not): an interpreter's script that writes a plan file, or a shell
    script that would be refused as a command. Any other body (a commit message, cat's text) is data."""
    if not re.search(DATA, body):
        return None
    segs = segments(HEREDOC.split(line)[0]) or [[]]
    words = [t for t in (segs[-1] if segs else []) if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
    prog = os.path.basename(words[0]) if words else ""
    if INTERP.match(prog):
        w = writes_plan(body)
        return "a %s script fed by a heredoc that writes a plan file (%s)" % (prog, w[:80]) if w else None
    if prog in SHELLS:
        return bash_refusal(body)
    return None


def bash_refusal(cmd):
    if re.search(r"\bPLAN_[A-Z_]+", cmd):
        return "a PLAN_* variable (they point tools/plan.py at a test world, honoured only with PLAN_TEST=1 in tests)"
    cmd, bodies = heredocs(cmd)
    for line, body in bodies:
        why = heredoc_refusal(line, body)
        if why:
            return why
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
        if prog == "sed" and not any(w == "-i" or re.match(r"-[a-zA-Z]*i", w) or w.startswith("--in-place")
                                     for w in words[1:]):
            # a read (sed -n 1,9p FILE): the plan file may only be an input, after the script, never in the script
            # (sed's w command writes)
            script = next((w for w in words[1:] if not w.startswith("-")), "")
            if not re.search(DATA, script):
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
