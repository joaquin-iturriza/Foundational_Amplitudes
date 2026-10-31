#!/usr/bin/env python3
"""Context guard: before the main session's context is compacted, what the compaction would lose is on the plan or
the ledger (Joaquin's approval of "1 2 and 3", 2026-10-31; it replaces the docs/results.tex notes of 2026-10-05).

Over the main session's messages since its last compaction, the turn cannot end, and an automatic compaction is held,
while
  - a question the main session put to the user has no `plan ask` (or `plan answer`, `plan withdraw`) recording it;
  - an answer or decision of the user's has no `site decide` (or `plan answer`) recording it;
  - a commitment in a main-session message ("I'll do X") has no plan step (`plan add-next|set-next|step|wait|add`)
    and was not done later in the window.
A model (reply_check's sonnet call) reads the user's messages, the main session's text and the recording commands,
and lists each omission with a quote; an item stands only when its quote is in the window. Any failure of the model
passes: the check never stalls the compaction or the turn.

  PreCompact    (trigger "auto") omissions found: refuse (exit 2) with the list, record WANTED. None: let it through.
  Stop          a compaction is wanted (WANTED, or the context at notes_at()) and omissions are found: block once per
                state of the transcript (CHECKED holds the size checked), with the list.
  PostToolUse   a compaction is wanted: the instruction as additional context (no model call).
  SessionStart  every source: print the plan (tools/plan.py show --compact); after a compaction also clear the
                markers and say that the summary is not the plan.

When auto-compaction fires is Claude Code's call (env CLAUDE_CODE_AUTO_COMPACT_WINDOW in settings.json); a hook
cannot start one, only hold it. Held, it is retried on the next auto-compaction check.
"""
import json, os, re, sys

REPO = os.environ.get("CLAUDE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FLUSHED = os.path.join(REPO, ".claude", ".notes_flushed")
WANTED = os.path.join(REPO, ".claude", ".compact_wanted")

CHECKED = os.path.join(REPO, ".claude", ".compact_checked")

INSTR = ("CONTEXT GUARD: the context is near compaction. Before it, record on the plan and the ledger what it would "
         "lose: every question you put to Joaquin (`python3 tools/plan.py ask <entry> ...`), every answer or decision "
         "of his (`site decide FA --kind K --quote \"<his words>\"`, or `plan answer`), every 'I'll do X' you told "
         "him that is not done (a plan step: `plan add-next <entry> ...`). The check lists what it finds missing.")
AFTER = ("CONTEXT GUARD: this context was just compacted. The summary above is a lossy record of the CONVERSATION (what "
         "was in flight), not of the plan: work only from the plan printed below, not from the summary's "
         "pending list. CLAUDE.md's ground rules hold (0: start every message with 'Joaquin'; 0b: never narrow the scope). "
         "Re-check the running batches with `site` before acting on any number in the summary.")


def plan_text():
    """The plan itself (tools/plan.py show --compact): the only source of truth, printed so no summary stands
    between the fresh context and it (setup redesign, D121)."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plan.py"), "show", "--compact"],
                           capture_output=True, text=True, timeout=60)
        return "THE PLAN (plan/plan.yaml; launch a step agent for each entry `plan next` lists, now):\n" + r.stdout + r.stderr
    except Exception as e:                      # noqa: BLE001
        return "THE PLAN could not be printed (%r): run python3 tools/plan.py show now." % e


def notes_at():
    """Context size at which the notes are asked for: 50k below the auto-compaction window (Joaquin, 2026-10-30:
    "Reduce the threshold of compaction to 400k, and the notes updating a bit lower")."""
    try:
        return int(os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") or 400000) - 50000
    except ValueError:
        return 350000


def context_tokens(path):
    """The session's current context: the last assistant call's input, cache read and cache write, from the
    transcript's tail. Unreadable: 0."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (1 << 20)))
            lines = f.read().split(b"\n")
    except (OSError, TypeError):
        return 0
    for line in reversed(lines):
        if b'"usage"' not in line:
            continue
        try:
            u = json.loads(line)["message"]["usage"]
        except (ValueError, KeyError, TypeError):
            continue
        return (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0))
    return 0


RECORDING = re.compile(r"plan\.py\s+(ask|answer|withdraw|unanswered|add-next|set-next|step|wait|add|describe)\b|"
                       r"\bsite\s+decide\b")


def window(path, lines=None):
    """(user messages, main-session text blocks, recording commands) since the transcript's last compaction, each a
    (time, text) list. The user's messages are reply_check's: typed prompts and mid-turn messages, no wake-ups."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import reply_check as rc
    users, says, recs = [], [], []
    for line in (open(path, errors="replace") if lines is None else lines):
        if '"compact_boundary"' in line:
            users, says, recs = [], [], []
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or r.get("isSidechain"):
            continue
        ts, m = r.get("timestamp", ""), r.get("message") or {}
        if r.get("type") == "attachment" or (r.get("type") == "user" and rc.MIDTURN in line):
            users += [(ts, p) for p in rc.midturn(r)]
        elif r.get("type") == "user" and not (r.get("isMeta") or r.get("isCompactSummary")) \
                and r.get("toolUseResult") is None:
            for t in rc.texts(m.get("content")):
                t = t.strip()
                if t and not t.startswith(rc.SKIP) and '"tool_result"' not in line:
                    users.append((ts, t))
        elif r.get("type") == "assistant" and isinstance(m.get("content"), list):
            for x in m["content"]:
                if not isinstance(x, dict):
                    continue
                if x.get("type") == "text" and x.get("text", "").strip():
                    says.append((ts, x["text"].strip()))
                elif x.get("type") == "tool_use" and x.get("name") == "Bash":
                    cmd = (x.get("input") or {}).get("command") or ""
                    if RECORDING.search(cmd):
                        recs.append((ts, cmd[:600]))
                elif x.get("type") == "tool_use" and x.get("name") in ("SendMessage", "Agent"):
                    i = x.get("input") or {}      # work handed to an agent is recorded with its owner
                    recs.append((ts, "%s to %s: %s" % (x["name"], i.get("to") or i.get("description") or "",
                                                       str(i.get("message") or i.get("prompt") or "")[:600])))
    return users, says, recs


def check_prompt(users, says, recs, budget=150000, steps=None):
    """The model's brief: the plan's entries and steps, then the window, oldest first, as one time-ordered log cut
    from the front to `budget` chars."""
    steps = plan_text() if steps is None else steps
    log = sorted([(t, "JOAQUIN", x) for t, x in users] + [(t, "ASSISTANT", x) for t, x in says] +
                 [(t, "RECORDED", x) for t, x in recs])
    body = "\n\n".join("[%s %s]\n%s" % (t[:16], who, x[:4000]) for t, who, x in log)
    body = body[-budget:]
    return f"""You check the bookkeeping of an assistant before its context is compacted (what is not recorded is
lost). The log below is its conversation with the researcher Joaquin since the last compaction, oldest first:
JOAQUIN is his message, ASSISTANT a message the assistant wrote to him, RECORDED a command the assistant ran that
records something (`plan.py ask` records a question put to Joaquin; `site decide` or `plan.py answer` records his
answer or decision in his words; `plan.py add-next|set-next|step|wait|add` records a step of work; a SendMessage or
Agent call hands work to an agent that owns it). The plan's entries and their next steps, as they stand now:
<<<
{steps[:20000]}
>>>

Answer ONLY with JSON: {{"questions": [...], "answers": [...], "commitments": [...]}}, each item
{{"quote": "<copied word for word from the log>", "why": "<one short sentence>"}}. List:
- questions: a question the ASSISTANT put to Joaquin that needs his answer (a decision only he can make), that no
  later RECORDED `plan.py ask`/`answer`/`withdraw` records and that he did not answer later in the log in a way
  RECORDED by `site decide` or `plan.py answer`. Rhetorical questions, questions answered in the same message, and
  "should I go on?" courtesy questions do not count.
- answers: a message of JOAQUIN that answers a question or decides how work goes (approves, refuses, scopes,
  prefers, forbids), for which no later RECORDED `site decide` or `plan.py answer` carries his words. Questions of
  his, complaints and remarks that decide nothing do not count.
- commitments: an ASSISTANT sentence to Joaquin promising work not done in the same message ("I'll run X", "next I
  will ...", "X comes next"), that no later ASSISTANT message reports done and no RECORDED plan step covers.
  Work that finished later in the log does not count.
Quote exactly (the quote is searched for in the log). When nothing is missing, answer with three empty lists.

<<<
{body}
>>>
"""


def items_of(v, users, says):
    """[(kind, quote, why)] of the model's answer whose quote is in the window (questions and commitments in the
    assistant's text, answers in the user's): a quote the window does not hold is dropped."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import reply_check as rc
    pool = {"questions": rc.norm("\n".join(x for _, x in says)), "commitments": rc.norm("\n".join(x for _, x in says)),
            "answers": rc.norm("\n".join(x for _, x in users))}
    out = []
    for kind in ("questions", "answers", "commitments"):
        for it in (v.get(kind) or []) if isinstance(v, dict) else []:
            q = str((it or {}).get("quote", "")).strip() if isinstance(it, dict) else ""
            if len(q) >= 8 and rc.norm(q) in pool[kind]:
                out.append((kind, q, str(it.get("why", "")).strip()))
    return out


def pending(path, ask=None, steps=None):
    """The omissions of the window as refusal lines; [] when none, or when the check cannot run (it never stalls)."""
    try:
        users, says, recs = window(path)
        if not users and not says:
            return []
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import reply_check as rc
        v = (ask or rc.ask_model)(check_prompt(users, says, recs, steps=steps))
        if v is None:
            return []
        how = {"questions": "a question to Joaquin with no `plan ask`", "answers": "Joaquin's words with no "
               "`site decide`", "commitments": "a commitment with no plan step"}
        return ["%s: \"%s\" (%s)" % (how[k], q[:300], w) for k, q, w in items_of(v, users, says)]
    except Exception:                   # noqa: BLE001
        return []


def stop_items(inp, ask=None, repo=REPO, steps=None):
    """Stop: the refusal lines, once per state of the transcript (its size), while a compaction is wanted."""
    path = inp.get("transcript_path")
    if inp.get("agent_id") or not path or not os.path.exists(path):
        return []
    wanted, checked = (os.path.join(repo, ".claude", x) for x in (".compact_wanted", ".compact_checked"))
    if not os.path.exists(wanted) and context_tokens(path) < notes_at():
        return []
    open(wanted, "w").close()
    size = os.path.getsize(path)
    try:
        last = json.load(open(checked))
    except (OSError, ValueError):
        last = {}
    if last.get("size") == size:
        return []                       # this state was checked: never block twice on it
    got = pending(path, ask, steps)
    json.dump({"size": size, "items": got}, open(checked, "w"))
    return [INSTR + "\n" + "\n".join("  - " + g for g in got)] if got else []


def _clear(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    ev = inp.get("hook_event_name", "")
    if ev == "PreCompact":
        if inp.get("trigger") != "auto" or inp.get("agent_id"):
            return 0
        open(WANTED, "w").close()
        got = pending(inp.get("transcript_path"))
        if not got:
            return 0
        sys.stderr.write(INSTR + "\n" + "\n".join("  - " + g for g in got) + "\n")
        return 2
    if ev == "SessionStart":
        ctx = plan_text()                     # every source (startup, resume, clear, compact) starts from the plan
        if inp.get("source") == "compact":
            for p in (FLUSHED, WANTED, CHECKED):
                _clear(p)
            ctx = AFTER + "\n\n" + ctx
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
        return 0
    if ev == "Stop":
        if inp.get("stop_hook_active"):
            return 0
        got = stop_items(inp)
        if got:
            print(json.dumps({"decision": "block", "reason": got[0]}))
    elif ev == "PostToolUse" and not inp.get("agent_id"):
        if os.path.exists(WANTED) or context_tokens(inp.get("transcript_path")) >= notes_at():
            open(WANTED, "w").close()
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": INSTR}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
