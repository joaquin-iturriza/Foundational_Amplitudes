#!/usr/bin/env python3
"""PreToolUse guard (setup redesign, D121): the plan's files change only through tools/plan.py, which checks what it
records (the user's words, real step agents, machine checks). Refused:
  - Edit, Write, MultiEdit or NotebookEdit of plan/plan.yaml, plan/.state.json, plan/.check_cache.json or the lock;
  - a Bash command that names one of those files (with or without plan/, so a cd into plan/ counts) and is not a run
    of tools/plan.py or tools/plan_migrate.py, unless it only reads it (cat, head, tail, less, grep, wc, diff, git
    diff/log/show/status/blame; sed without an in-place flag or -f and with the file outside its script; awk whose
    program has no redirection, pipe or system());
  - a PLAN_* variable set (an assignment or export, outside quotes and data heredocs).
The command is read through cmdparse.py, shared by the Bash guards. Heredoc bodies are taken out before it is read: a body fed to an interpreter (python, perl, ...) is refused
only when a line of it writes and names a plan file (itself, or a variable set to its path); a shell's body is read
as a command; any other body (a commit message, cat's text) is data.
plan/step_prompt.txt is code (the step agent's template), edited like any file. A write the command text cannot show
(a script that opens the file) is not caught here: plan_check re-validates every recorded step and quote on read."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmdparse import SHELLS, all_segments, code_text, feeder, heredocs, prog as prog_of, segments, words as cmd_words  # noqa: E402

DATA = r"(?:plan/)?(?:plan\.yaml(?:\.lock)?|\.state\.json|\.check_cache\.json|\.plan_tmp_\w*)"
READERS = {"cat", "head", "tail", "less", "more", "grep", "rg", "wc", "diff", "file", "stat", "ls", "md5sum",
           "sha256sum"}
GIT_READ = {"diff", "log", "show", "status", "blame", "ls-files", "cat-file"}
TOOLS = re.compile(r"(?:^|/)tools/plan(?:_migrate)?\.py$")
# a PLAN_* variable set: an assignment before the program (env included: env "PLAN_ROOT=/x" ...) or an export, read from
# the parsed tokens of every command bash would run (quotes removed by the parser, data heredocs out); a PLAN_ROOT
# inside a grep pattern or a heredoc that writes a test is no assignment. PLAN_SET is the fallback for an unparsable
# command, on its code with quotes kept.
DECLARE = {"export", "declare", "typeset", "readonly", "local"}
PLAN_SET = re.compile(r"(?:^|[\s;&|(`])(?:PLAN_[A-Z_]+=|export\s+(?:-\w+\s+)*(?:[A-Za-z_]\w*(?:=\S*)?\s+)*PLAN_[A-Z_]+)")
INPLACE = re.compile(r"^(?:-[a-zA-Z]*i|--in-place)")
AWK = {"awk", "gawk", "mawk", "nawk"}


def plan_var_set(cmd):
    segs = all_segments(cmd)
    if segs is None:
        return bool(PLAN_SET.search(code_text(cmd)))
    for seg in segs:
        ws = cmd_words(seg)
        start = len(seg) - len(ws)                 # assignments and wrappers (env ...) before the program
        for k, t in enumerate(seg):
            if re.match(r"^PLAN_[A-Z_]+(?:=|$)", t) and (k < start or prog_of(ws) in DECLARE):
                return True
    return False


def editor_read(prog, words):
    """sed or awk only reading: no in-place flag, no -f program file, and the plan file is no part of the program
    (sed's w); an awk program with no >, no pipe and no system(), never -i inplace. perl and ruby are never readers
    here: their program can write to a file given as an argument (open my $f, ">", shift)."""
    opts = [w for w in words[1:] if w.startswith("-")]
    if prog in AWK:
        if "-f" in words or any(w == "-i" or w.startswith("--include") for w in opts):
            return False
        rest, k = words[1:], 0
        while k < len(rest) and rest[k].startswith("-"):
            k += 2 if rest[k] in ("-v", "-F") else 1
        script = rest[k] if k < len(rest) else ""
        return not re.search(r">|(?<!\|)\|(?!\|)|\bsystem\s*\(", script)
    if prog != "sed" or any(INPLACE.match(w) or re.fullmatch(r"-[a-zA-Z]*f|--file(?:=.*)?", w) for w in opts):
        return False
    script = next((w for w in words[1:] if not w.startswith("-")), "")
    return not re.search(DATA, script)


INTERP = re.compile(r"^(?:python[0-9.]*|perl|ruby|node)$")
WRITE = re.compile(r"""open\([^)\n]*,\s*(?:mode\s*=\s*)?['"][^'"]*[wax+]|\.write_(?:text|bytes)\(|"""
                   r"""\bos\.(?:replace|rename|remove|unlink)\(|\bshutil\.|\.(?:unlink|rename|replace)\(|>\s*\$?\w""")


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
    prog = feeder(line)
    if INTERP.match(prog):
        w = writes_plan(body)
        return "a %s script fed by a heredoc that writes a plan file (%s)" % (prog, w[:80]) if w else None
    if prog in SHELLS:
        return bash_refusal(body)
    return None


KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}", "done", "fi", "time"}


def named(t):
    """A token that is the plan file itself or a path to it (git show c:plan/plan.yaml): no whitespace, so a quoted
    message that merely mentions it is not one."""
    return bool(re.search(DATA, t)) and not re.search(r"\s", t)


def c_writes(seg):
    """An interpreter run with -c whose program text writes a plan file."""
    ws = cmd_words(seg)
    if not ws or not INTERP.match(os.path.basename(ws[0])):
        return False
    return any(writes_plan(w) for i, w in enumerate(ws[1:], 1) if ws[i - 1] in ("-c", "-e"))


def mentions(seg):
    """The segment names a plan file: as a path argument, in an -c program that writes it, or in an awk or sed program."""
    if any(named(t) for t in seg) or c_writes(seg):
        return True
    p = prog_of(cmd_words(seg))
    return (p == "sed" or p in AWK) and any(re.search(DATA, t) for t in seg)


def bash_refusal(cmd):
    if plan_var_set(cmd):
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
        while seg and seg[0] in KEYWORDS:           # a loop's or conditional's keyword is no command
            seg = seg[1:]
        if not mentions(seg):
            continue
        targets = [seg[i + 1] for i, t in enumerate(seg[:-1]) if re.fullmatch(r"\d?(?:>|>>|>\||<>)", t)]
        targets += [t.lstrip("0123456789>|") for t in seg if re.match(r"\d?>", t) and len(t.lstrip("0123456789>|"))]
        if any(re.search(DATA + "$", x) for x in targets):
            return "a redirection into a plan file"
        seg = [t for i, t in enumerate(seg) if not re.fullmatch(r"\d?(?:>|>>|>\||<>|<)", t)
               and not (i and re.fullmatch(r"\d?(?:>|>>|>\||<>|<)", seg[i - 1]))]
        if not mentions(seg):
            continue
        words = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]   # drop VAR=value prefixes
        if not words:
            continue
        prog = os.path.basename(words[0])
        if prog.startswith("python"):
            if c_writes(words):
                return "python -c writing a plan file"
            script = next((w for w in words[1:] if not w.startswith("-")), "")
            if not any(named(t) for t in words[1:]):
                continue                           # the plan file is only inside a quoted string (a message)
            if TOOLS.search(script) and "-c" not in words[1:words.index(script)]:
                continue
            return "python on a plan file other than tools/plan.py"
        if prog in READERS:                        # none of them writes (grep -i is ignore-case)
            continue
        if (prog == "sed" or prog in AWK) and editor_read(prog, words):
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
