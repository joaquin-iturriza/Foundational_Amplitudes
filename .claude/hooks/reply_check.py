#!/usr/bin/env python3
"""Stop hook: does the reply answer the user? (2026-10-15, after the user's "you keep ignoring my questions")

Runs only on a turn that answers a NEW message typed by the user (never on an event wake-up) and blocks at most once
per user message, so it cannot loop. A separate small model (claude -p --model haiku, run from /tmp so no project
settings or hooks load) reads the user's latest message(s), the reply that ends the turn and the decisions in force
(`site decisions`), and checks:
  1. every question the user asked is answered directly (not deflected, not answered with a plan); only the user's own
     questions count, not those inside text they pasted from others (2026-10-22: a pasted chat's "can we meet?",
     addressed to two collaborators, was flagged and answered with an empty reply);
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
    """The messages typed since the current turn began (its prompt, then any sent mid-turn), and EVERY assistant text
    block since that prompt, joined: an answer written before a tool call is as much the reply as the last block
    (2026-10-25: reading only the last block failed answers that sat before a tool call, three times in a row)."""
    msgs, replies = [], []
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
                elif (t and not t.startswith(SKIP) and r.get("toolUseResult") is None and not r.get("isMeta")
                      and not r.get("isCompactSummary")):
                    # a compaction summary is written as a user record; it quotes the user's old questions, and was
                    # checked as a new message (2026-10-26: three questions answered that morning were flagged again)
                    msgs = [t]        # a new prompt opens the window: earlier messages were answered in earlier turns
                    replies = []
        elif r.get("type") == "assistant":
            tx = "".join(texts(m.get("content"))).strip()
            if tx:
                replies.append(tx)
    return msgs, "\n\n".join(replies)


STOP = set("the a an and or of to in on for with is are be it that this what why how do does did you your i me my we "
           "not no yes so but if as at by from was were has have had can could would should will just about".split())


def relevant(dec, text, keep=12):
    """only the decisions that bear on this exchange (the user, 2026-10-25: "only decisions that are relevant"): each
    decision line scored by the words it shares with the user's messages and the reply, the best `keep` with any
    overlap; the whole list (~35 KB) made every check slow and noisy"""
    words = {w for w in re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower()) if w not in STOP}
    blocks = re.split(r"\n(?=\s*D\d+ )", dec)
    scored = []
    for b in blocks:
        bw = {w for w in re.findall(r"[a-z][a-z0-9_+-]{2,}", b.lower()) if w not in STOP}
        n = len(words & bw)
        if n >= 2:
            scored.append((n, b.strip()))
    return "\n".join(b for _, b in sorted(scored, key=lambda x: -x[0])[:keep])


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    path = inp.get("transcript_path")
    if not path or not os.path.exists(path):
        return 0
    msgs, reply = scan(path)
    reply = reply or inp.get("last_assistant_message") or ""
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
    dec = relevant(dec, "\n".join(recent) + "\n" + reply)
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
   question need no answer. Only the researcher's OWN questions count. A message often contains text the researcher
   pasted or quoted from other people (a chat thread with names like "Suprio Dubey:", an email, a review, a quoted
   earlier reply of the assistant, text inside <pasted_content> tags): questions or requests in that text are
   addressed to those people, not to the assistant, and need no answer unless the researcher's own words ask for one
   (e.g. "can you answer this?"). When the researcher's own words around a paste ask something ("We should have
   results for this, right?"), that is the question to check. Never name a question from pasted text in your reason, and never fail a
   reply because it leaves one unanswered.
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
          " Answer that directly in a revised reply (this check runs once per user message). If the check is wrong"
          " (the question was not the researcher's, e.g. it sits in text they pasted from someone else, or the reply"
          " does answer it), do not manufacture an answer: end the turn with no new reply.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
