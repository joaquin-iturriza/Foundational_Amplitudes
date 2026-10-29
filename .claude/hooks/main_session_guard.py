#!/usr/bin/env python3
"""The main session works no entry itself (CLAUDE.md 0i), within a budget: up to 3 work calls per user message.

The user (2026-10-29): "It seems a bit to extreme, you should still be able to do those small things in some
situations". So the main session may make BUDGET work calls since the user's last message; the 4th and later are
refused. The number 3, its per-message scope and that submissions count like any other work call are **open
(unsourced)**: set by the main session's request, not by the user; to be confirmed when the hook's registration is
proposed to the user. Non-work calls never count. The user's last message is found from the end of the transcript
(`transcript_path`) with reply_check.py's rule: a "user" record whose text is not a tool_result, not isMeta, not a
compaction summary and does not start with one of its SKIP prefixes (system reminders, task notifications, hook
feedback, command output); a message sent mid-turn (MIDTURN, inside a tool_result) counts as one too. The work calls
counted are the main session's tool_use records after it (sidechain records skipped; the current call, by its
tool_use_id, never counted), classified by the same rules as the current call. No transcript: the budget is full.

PreToolUse(Bash|Edit|Write|MultiEdit). A tool call made inside a subagent carries `agent_id` and `agent_type` in its
hook input; the main session's carries neither (read from the real inputs logged by lock_guard.py in
.claude/.review_state/lock_calls.log: every subagent call, general-purpose included, has both keys, the main
session's has none). Every subagent call passes, and never counts. Work, in the main session:
  - Bash: `site submit|sync|run|pick|env|copy|fetch`, `sbatch`, `scancel`, `condor_submit`, any python or pytest
    run except `python3 tools/plan.py ...` and `python3 tools/agent_costs.py`, a project script run through
    bash/sh/./ (scripts/, sweep/, tools/, .claude/hooks/) except `review_backlog.sh begin|status`, and the shell
    writes that stand in for an Edit (`sed -i`, `perl -i`, `tee`, a `>`/`>>` redirection to a file other than
    /dev/null or the session scratchpad).
  - Edit/Write/MultiEdit of any file but a CLAUDE.md or one under the session scratchpad.
Everything else passes: reads (cat, grep, ls, git log/diff/status/show, site runs/poll/status/logs/events/ack/
decisions/items/digest/where/list), git add/commit/push, `site supervise`, `site events --wait`.
Quoted strings and heredoc bodies are data (a commit message naming `site submit` passes), except the script of
`bash -c` / `sh -c` / `eval`, which is read as a command. The user's own `!` commands run no hook.
Python on purpose: a crash here exits 1, which does not block (hook_syntax_guard.py).
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reply_check import MIDTURN, SKIP, texts  # noqa: E402  (the same reading of "the user's message")

BUDGET = 3
MSG = ("main session: %d work calls already since the user's last message; this is a step agent's work (CLAUDE.md 0i):"
       " launch a fresh agent with a written prompt." % BUDGET)
CHUNK = 1 << 16
SITE_WORK = {"submit", "sync", "run", "pick", "env", "copy", "fetch"}
PY_OK = re.compile(r"^(?:\S*/)?python3?\s+(?:\S*/)?tools/(?:plan\.py(?:\s|$)|agent_costs\.py(?:\s|$))")
SCRIPT_DIRS = re.compile(r"(?:^|/)(?:scripts|sweep|tools|\.claude/hooks)/\S+")
BACKLOG_OK = re.compile(r"review_backlog\.sh\s+(?:begin|status)\b")
WRAP = re.compile(r"(?:\b(?:bash|sh|zsh)\s+(?:-\w+\s+)*-c|\beval)\s*$")


def strip_data(cmd):
    """The command with heredoc bodies and quoted strings blanked; the scripts of bash -c / eval returned apart."""
    inner = []
    out, i, n = [], 0, len(cmd)
    heredocs = []
    while i < n:
        c = cmd[i]
        if c in "'\"":
            j = i + 1
            while j < n and cmd[j] != c:
                j += 2 if (c == '"' and cmd[j] == "\\") else 1
            body = cmd[i + 1:j]
            if WRAP.search("".join(out)[-60:]):
                inner.append(body)
            out.append(c + " " * len(body) + c)
            i = j + 1
            continue
        m = re.match(r"<<-?\s*(['\"]?)(\w+)\1", cmd[i:])
        if m:
            heredocs.append(m.group(2))
            out.append("<<" + m.group(2))
            i += m.end()
            continue
        if c == "\n" and heredocs:
            out.append("\n")
            i += 1
            while heredocs and i < n:
                e = cmd.find("\n", i)
                e = n if e < 0 else e
                if cmd[i:e].strip() == heredocs[0]:
                    heredocs.pop(0)
                i = e + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), inner


def segments(cmd):
    return [s.strip() for s in re.split(r"\|\||&&|[;|&\n]|\$\(|`|\(|\)", cmd) if s.strip()]


def bash_refusal(cmd, scratch, depth=0):
    text, inner = strip_data(cmd)
    for seg in segments(text):
        seg = re.sub(r"^(?:(?:env|command|exec|time|nohup|sudo|eval)\s+|[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*", "", seg)
        toks = seg.split()
        if not toks:
            continue
        head = os.path.basename(toks[0])
        if head == "site" and len(toks) > 1 and toks[1] in SITE_WORK:
            return "site %s" % toks[1]
        if head in ("sbatch", "scancel", "condor_submit"):
            return head
        if re.match(r"^(?:python[\d.]*|pytest|py\.test)$", head) or (head in ("uv", "poetry") and "run" in toks[1:3]):
            if not PY_OK.match(seg):
                return "a python/pytest run"
        if head in ("bash", "sh", "zsh", "source", ".") or toks[0].startswith("./"):
            target = toks[0] if toks[0].startswith("./") else next((t for t in toks[1:] if not t.startswith("-")), "")
            if SCRIPT_DIRS.search(target) and not BACKLOG_OK.search(seg):
                return "running %s" % target
        if head in ("sed", "perl") and any(re.match(r"^-\w*i", t) for t in toks[1:]):
            return "%s -i" % head
        if head == "tee" and any(not t.startswith("-") and t != "/dev/null" for t in toks[1:]):
            return "tee to a file"
        for tgt in re.findall(r"(?<![<>&\d])\d?>>?\s*([^\s&|;]+)", seg):
            if tgt.startswith("&") or tgt == "/dev/null" or (scratch and os.path.abspath(tgt).startswith(scratch)):
                continue
            return "a redirection to %s" % tgt
    if depth < 3:
        for body in inner:
            r = bash_refusal(body, scratch, depth + 1)
            if r:
                return r
    return None


def path_ok(path, scratch):
    if not path:
        return True
    p = os.path.abspath(path)
    if os.path.basename(p) == "CLAUDE.md":
        return True
    return bool(scratch) and (p == scratch or p.startswith(scratch.rstrip("/") + "/"))


def work(tool, ti, scratch):
    """Why this call is work, or None."""
    if tool == "Bash":
        return bash_refusal(ti.get("command", "") or "", scratch)
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = ti.get("file_path") or ti.get("notebook_path") or ""
        return None if path_ok(path, scratch) else "%s of %s" % (tool, path)
    return None


def lines_backwards(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos, rest = f.tell(), b""
        while pos > 0:
            step = min(CHUNK, pos)
            pos -= step
            f.seek(pos)
            parts = (f.read(step) + rest).split(b"\n")
            rest = parts[0]
            for ln in reversed(parts[1:]):
                yield ln
        yield rest


def is_user_message(r):
    if r.get("type") != "user":
        return False
    for t in texts((r.get("message") or {}).get("content")):
        t = t.strip()
        if MIDTURN in t:
            return True
        if (t and not t.startswith(SKIP) and r.get("toolUseResult") is None and not r.get("isMeta")
                and not r.get("isCompactSummary")):
            return True
    return False


def used(path, scratch, current_id):
    """The main session's work calls since the user's last message, reading the transcript from its end."""
    seen = set()
    for ln in lines_backwards(path):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("isSidechain"):
            continue
        if is_user_message(r):
            break
        if r.get("type") != "assistant":
            continue
        for c in (r.get("message") or {}).get("content") or []:
            if (isinstance(c, dict) and c.get("type") == "tool_use" and c.get("id") != current_id
                    and work(c.get("name", ""), c.get("input") or {}, scratch)):
                seen.add(c.get("id"))
        if len(seen) >= BUDGET:
            break
    return len(seen)


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    if inp.get("agent_id") or inp.get("agent_type"):
        return 0
    tool, ti = inp.get("tool_name", ""), inp.get("tool_input") or {}
    scratch = os.path.abspath(inp["scratchpad_dir"]) if inp.get("scratchpad_dir") else ""
    why = work(tool, ti, scratch)
    if not why:
        return 0
    path = inp.get("transcript_path")
    n = used(path, scratch, inp.get("tool_use_id")) if path and os.path.exists(path) else BUDGET
    if n < BUDGET:
        return 0
    print("BLOCKED (%s). %s" % (why, MSG), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
