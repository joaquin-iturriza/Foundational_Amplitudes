#!/usr/bin/env python3
"""The one reading of a Bash command that the PreToolUse guards share (the false refusals measured by
tools/agent_calls.py --refusals: guards matched the raw text, so a heredoc's body or a quoted string that only NAMED
sbatch, core.hooksPath, review_backlog.sh advance or plan.py design-pass was read as running it).

  heredocs(cmd)   -> (the command without its heredoc bodies, [(the line that feeds a body, the body)])
  feeder(line)    -> the program a heredoc on that line is fed to
  is_code(line)   -> a body fed to a shell (bash, sh, ssh, eval, or piped into one) is code; to python, cat, tee, git
                     ... it is data
  code_text(cmd)  -> the command with data bodies taken out and code bodies kept as further lines
  blank_quotes(s) -> s with every quoted string emptied ('...', "...", $'...' become '' or ""), so a regex over it sees
                     command words only (scan_text adds back what runs inside them)
  segments(cmd)   -> the simple commands (split at ; && || | & ( ) and newlines), each as shlex tokens, or None when
                     the text cannot be parsed
  words(seg)      -> a segment's command words: VAR=value prefixes and wrappers (env, sudo, time, then, do ...) dropped
  substitutions(s)-> the $(...) and `...` bash runs, double-quoted ones included
  inner_commands  -> what sh -c / bash -lc / eval / ssh host run; inline_programs: python -c, perl/ruby/node -e
  all_segments    -> every simple command bash would run, as tokens (inner commands and substitutions included)
  commands(cmd)   -> the same as words; None when any part cannot be parsed
  scan_text(cmd)  -> code_text with quotes blanked, plus the inner commands, the substitutions and the inline programs
                     (raw): the text a regex guard matches
CLI for the shell guards: `cmdparse.py scan` (stdin: the command; prints scan_text), `cmdparse.py submitted-scripts`
(the .sh arguments of sbatch / site submit commands), `cmdparse.py runpy` (the words of every command that runs
run.py, readers excluded). Each exits 3 when the command cannot be parsed, so the caller falls back."""
import os, re, shlex, sys

HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
SHELLS = {"bash", "sh", "zsh", "dash"}
CODE_FEEDERS = SHELLS | {"ssh", "eval", "source", "."}
SEP = set(";&|()\n")
WRAPPERS = {"command", "env", "sudo", "nice", "time", "nohup", "exec", "then", "do", "else", "elif", "if", "while",
            "until", "!", "{", "}", "xargs", "timeout", "stdbuf"}
READERS = {"cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "wc", "diff", "file", "stat", "ls",
           "md5sum", "sha256sum", "echo", "printf", "find", "sed", "awk", "git", "cut", "sort", "uniq", "tr", "jq"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


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


def strip_comments(s):
    """s without its shell comments (a # starting a word, outside quotes, to the end of the line) and with its
    backslash-newline continuations joined."""
    out, i, n, q = [], 0, len(s), None
    while i < n:
        c = s[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                out.append(s[i:i + 2])
                i += 2
                continue
            if c == q:
                q = None
        elif c == "\\" and i + 1 < n:
            if s[i + 1] != "\n":
                out.append(s[i:i + 2])
            i += 2
            continue
        elif c in "'\"":
            q = c
        elif c == "#" and (i == 0 or s[i - 1] in " \t\n;&|()"):
            while i < n and s[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def segments(cmd):
    """The simple commands of a shell text (split at ; && || | & ( ) and newlines, comments dropped), each as tokens,
    or None when the text cannot be parsed."""
    try:
        lx = shlex.shlex(strip_comments(cmd), posix=True, punctuation_chars=";&|<>()\n")
        lx.whitespace, lx.commenters, lx.whitespace_split = " \t\r", "", True
        toks = list(lx)
    except ValueError:
        return None
    out, cur = [], []
    for t in toks:
        if t and set(t) <= SEP:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def words(seg):
    """The command words of a segment: leading VAR=value assignments and wrappers dropped (a wrapper's own options too,
    timeout's duration included)."""
    i = 0
    while i < len(seg):
        t = seg[i]
        if ASSIGN.match(t):
            i += 1
        elif os.path.basename(t) in WRAPPERS:
            w = os.path.basename(t)
            i += 1
            while i < len(seg) and (seg[i].startswith("-") or (w == "env" and ASSIGN.match(seg[i]))):
                i += 1
            if w == "timeout" and i < len(seg):
                i += 1                                 # the duration
        else:
            break
    return seg[i:]


def prog(ws):
    return os.path.basename(ws[0]) if ws else ""


def feeder(line):
    """The program the first heredoc on this line is fed to."""
    segs = segments(HEREDOC.split(line)[0]) or [[]]
    return prog(words(segs[-1])) if segs else ""


def is_code(line):
    """A heredoc body fed to a shell (or to a program piped into one) is code; anything else is data."""
    return feeder(line) in CODE_FEEDERS or bool(re.search(r"\|\s*(?:\S*/)?(?:ba|z|da)?sh\b", line))


def code_text(cmd):
    """The command with data heredoc bodies taken out and code bodies kept (recursively) as further lines. A data body
    under an unquoted delimiter (<<EOF) still runs its command substitutions; they are kept as lines too."""
    top, bodies = heredocs(cmd)
    out = [top]
    for line, b in bodies:
        if is_code(line):
            out.append(code_text(b))
        elif not (HEREDOC.search(line).group(2)):
            out += [code_text(x) for x in substitutions(b, in_heredoc=True)]
    return "\n".join(out)


def substitutions(s, in_heredoc=False):
    """The contents of every $(...) and `...` in s that bash runs: outside single quotes, inside double quotes too
    ($(( )) arithmetic excluded). In a heredoc body single quotes are plain characters."""
    out, i, n, dq = [], 0, len(s), False
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "'" and not dq and not in_heredoc:
            j = s.find("'", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == '"' and not in_heredoc:
            dq = not dq
        elif c == "$" and s[i + 1:i + 2] == "(" and s[i + 2:i + 3] != "(":
            depth, j = 1, i + 2
            while j < n and depth:
                depth += {"(": 1, ")": -1}.get(s[j], 0)
                j += 1
            out.append(s[i + 2:j - 1] if not depth else s[i + 2:])
            i = j
            continue
        elif c == "`":
            j = s.find("`", i + 1)
            j = n if j < 0 else j
            out.append(s[i + 1:j])
            i = j + 1
            continue
        i += 1
    return out


def blank_quotes(s):
    """s with the contents of every quoted string removed ('a b' -> '', "x" -> "", $'..' -> ''); a backslash-escaped
    character outside quotes is dropped too. An unterminated quote empties the rest of the line."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'" or (c == "$" and i + 1 < n and s[i + 1] == "'"):
            if c == "$":
                i += 1
            j = i + 1
            while j < n and s[j] != "'":
                j += 2 if (c == "$" and s[j] == "\\") else 1
            out.append("''")
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            out.append('""')
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


INTERP = re.compile(r"^(?:python[0-9.]*|perl|ruby|node)$")


def inner_commands(seg):
    """The command strings a segment runs as shell code: sh -c '<inner>' (combined flags too: bash -lc, sh -ec) and
    eval <args>."""
    ws = words(seg)
    for k, t in enumerate(ws):
        b = os.path.basename(t)
        if b in SHELLS:
            for j in range(k + 1, len(ws)):
                if not ws[j].startswith("-") or ws[j] == "--":
                    break
                if re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", ws[j]):
                    return [ws[j + 1]] if j + 1 < len(ws) else []
            return []
        if b == "eval" and k == 0:
            return [" ".join(ws[1:])]
    if prog(ws) == "ssh":                      # ssh host <cmd>: the remote command is shell code
        rest = [w for w in ws[1:] if not w.startswith("-")]
        return [" ".join(rest[1:])] if len(rest) > 1 else []
    return []


def inline_programs(seg):
    """The program text an interpreter runs from its arguments: python -c '<code>', perl/ruby/node -e '<code>'. It is
    not shell, but a string in it can be (os.system('sbatch ...')): the regex guards read it as text."""
    ws = words(seg)
    p = prog(ws)
    if not INTERP.match(p):
        return []
    flag = r"-[a-zA-Z]*c" if p.startswith("python") else r"-[a-zA-Z]*e"
    for j, w in enumerate(ws[1:], 1):
        if re.fullmatch(flag, w):
            return [ws[j + 1]] if j + 1 < len(ws) else []
        if not w.startswith("-"):
            break
    return []


def all_segments(cmd, depth=0):
    """Every simple command bash would run, as tokens: the code's own (data heredocs out), the inner commands of
    sh -c / eval / ssh, and every command substitution, quoted or not. None when any part cannot be parsed."""
    if depth > 6:
        return None
    code = code_text(cmd)
    segs = segments(code)
    if segs is None:
        return None
    out = list(segs)
    more = [x for seg in segs for x in inner_commands(seg)] + substitutions(code)
    for x in more:
        sub = all_segments(x, depth + 1)
        if sub is None:
            return None
        out += sub
    return out


def commands(cmd):
    """Every simple command bash would run, as its words; None when any part cannot be parsed."""
    segs = all_segments(cmd)
    return None if segs is None else [words(seg) for seg in segs]


def scan_text(cmd, depth=0):
    """For a regex guard: the code with quotes blanked, plus (recursively) the inner commands of sh -c / eval / ssh and
    every command substitution, and, as raw text, the programs of python -c / perl -e."""
    code = code_text(cmd)
    parts = [blank_quotes(strip_comments(code))]
    if depth < 6:
        segs = segments(code) or []
        parts += [scan_text(x, depth + 1) for seg in segs for x in inner_commands(seg)]
        parts += [scan_text(x, depth + 1) for x in substitutions(code)]
        parts += [x for seg in segs for x in inline_programs(seg)]
    return "\n".join(parts)


def submitted_scripts(cmds):
    """The .sh arguments of every sbatch / site submit command."""
    out = []
    for ws in cmds:
        p = prog(ws)
        if p == "sbatch" or (p == "site" and ws[1:2] == ["submit"]):
            out += [w for w in ws[1:] if w.endswith(".sh")]
    return out


def runs_runpy(cmds):
    """The commands that run run.py (python run.py, site run ... -- python run.py), readers of it excluded."""
    return [ws for ws in cmds if prog(ws) not in READERS and any(os.path.basename(w) == "run.py" for w in ws)]


def main(argv):
    mode = argv[1] if len(argv) > 1 else "scan"
    cmd = sys.stdin.read()
    if mode == "scan":
        print(scan_text(cmd))
        return 0
    cmds = commands(cmd)
    if cmds is None:
        return 3
    if mode == "submitted-scripts":
        print("\n".join(submitted_scripts(cmds)))
    elif mode == "runpy":
        print("\n".join(" ".join(ws) for ws in runs_runpy(cmds)))
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
