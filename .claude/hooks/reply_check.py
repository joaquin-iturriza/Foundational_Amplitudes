#!/usr/bin/env python3
"""Stop hook: does the reply answer the user? (2026-10-15, after the user's "you keep ignoring my questions")

Runs only on a turn that answers a NEW message typed by the user (never on an event wake-up) and blocks at most once
per user message, so it cannot loop. A separate small model (claude -p --model haiku, run from /tmp so no project
settings or hooks load) reads the user's latest message(s), the reply that ends the turn and the decisions in force
(`site decisions`), and checks:
  1. every question the user asked is answered directly (not deflected, not answered with a plan);
  2. the reply does not ask the user something a decision in force already settles.
On "no" the turn is blocked once with the reason. Any failure of the check itself (no CLI, timeout, unparsable
answer) lets the turn end: the check must never be what stalls work.
State: .claude/.reply_check_state (hash of the last user message checked).
"""
import hashlib, json, os, re, subprocess, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
STATE = os.path.join(ROOT, ".claude", ".reply_check_state")
SKIP = ("<system-reminder", "[SYSTEM NOTIFICATION", "Stop hook feedback", "<task-notification", "<command-",
        "<local-command", "Caveat:", "Goal check-in", "[Request interrupted", "Another Claude session sent")
MIDTURN = "The user sent a new message while you were working:"


def texts(content):
    if isinstance(content, str):
        return [content]
    out = []
    for c in content or []:
        if isinstance(c, dict):
            if c.get("type") == "text":
                out.append(c.get("text", ""))
            elif c.get("type") == "tool_result":
                cc = c.get("content")
                out += [cc] if isinstance(cc, str) else texts(cc)
    return out


def scan(path):
    """The messages typed since the current turn began (its prompt, then any sent mid-turn), and the last assistant text."""
    msgs, reply = [], ""
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = r.get("message") or {}
        if r.get("type") == "user":
            for t in texts(m.get("content")):
                t = t.strip()
                if MIDTURN in t:
                    for part in t.split(MIDTURN)[1:]:
                        part = part.split("This is how Claude Code surfaces")[0].strip()
                        if part:
                            msgs.append(part)
                elif t and not t.startswith(SKIP) and r.get("toolUseResult") is None and not r.get("isMeta"):
                    msgs = [t]        # a new prompt opens the window: earlier messages were answered in earlier turns
        elif r.get("type") == "assistant":
            tx = "".join(texts(m.get("content"))).strip()
            if tx:
                reply = tx
    return msgs, reply


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    path = inp.get("transcript_path")
    if not path or not os.path.exists(path):
        return 0
    msgs, reply = scan(path)
    reply = inp.get("last_assistant_message") or reply
    if not msgs or not reply:
        return 0
    recent = msgs[-3:]
    h = hashlib.sha1(recent[-1].encode()).hexdigest()
    try:
        st = json.load(open(STATE))
    except (OSError, ValueError):
        st = {}
    if st.get("hash") == h:
        return 0                      # this message was already checked; never block twice
    json.dump({"hash": h}, open(STATE, "w"))
    try:
        dec = subprocess.run(["site", "decisions", "--project", "FA"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        dec = ""
    prompt = f"""You check one reply of an assistant to a researcher. Answer ONLY with JSON: {{"ok": true}} or
{{"ok": false, "reason": "<one or two sentences naming exactly what is missing>"}}.

The researcher's latest message(s), oldest first:
<<<
{chr(10).join(recent)[-6000:]}
>>>

The assistant's reply:
<<<
{reply[-8000:]}
>>>

Decisions the researcher already made (in force):
<<<
{dec[-6000:]}
>>>

Rules:
1. Every question the researcher asked (explicit, or an obvious "why"/"what"/"so?" demand) must be answered directly
   in the reply. A plan, an apology, or an answer to a different question does not count. Instructions without a
   question need no answer.
2. The reply must not ask the researcher to decide something the decisions above already settle.
3. Be strict on 1 and 2 and ignore everything else (style, length, whether the work is right).
If both rules hold, answer {{"ok": true}}."""
    try:
        out = subprocess.run(["claude", "-p", "--setting-sources", "project", "--model", "haiku",
                              "--no-session-persistence", "--disallowed-tools", "Bash,Read,Edit,Write,Agent,WebFetch"],
                             input=prompt, capture_output=True, text=True, timeout=90, cwd="/tmp").stdout
        v = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    except Exception:
        return 0
    if v.get("ok"):
        return 0
    print("REPLY CHECK: " + str(v.get("reason", "the reply does not answer the user's question")) +
          " Answer that directly in a revised reply (this check runs once per user message).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
