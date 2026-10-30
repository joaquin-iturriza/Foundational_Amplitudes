#!/usr/bin/env python3
"""Stop hook: does the reply answer the user? (2026-10-15, after the user's "you keep ignoring my questions")

Runs only on a turn that answers a NEW message typed by the user (never on an event wake-up) and blocks at most once
per user message, so it cannot loop. A separate model (claude -p --model sonnet, run from /tmp so no project settings
or hooks load) reads the user's latest message(s) and every text block the assistant wrote since (not its thinking),
and checks that every question of the user's own is answered directly; questions inside text they pasted from others
do not count (2026-10-22: a pasted chat's "can we meet?" was flagged).
A refusal must quote the question word for word; it stands only when that quote is in the user's message and is a
question (its sentence ends with "?", or it asks explicitly to explain), and its reason does not say the reply answers
or is fine (2026-10-30: haiku refused statements and refused against its own reasoning, 109 refusals in one session).
On a refusal that stands the turn is blocked once. Any failure of the check itself (no CLI, timeout, unparsable
answer) lets the turn end: the check must never be what stalls work.
State: .claude/.reply_check_state (hash of the last user message checked).
"""
import hashlib, json, os, re, subprocess, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
STATE = os.path.join(ROOT, ".claude", ".reply_check_state")
TIMEOUT = 120            # seconds for the sonnet call; plan_check, which runs it, has 300
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


def midturn(r):
    """A message the user typed while the turn was running: an attachment record `queued_command` of human origin
    (Claude Code shows it to the model as a text block opening with MIDTURN, beside a tool result). The MIDTURN text is
    taken only from a plain text block of a user record, never from a tool result: a tool result holding this file's
    source (it defines MIDTURN) was read as a message of the user's (2026-10-30, "Why does this hook keep
    misfiring?": the check judged the reply against reply_check.py's source)."""
    if r.get("type") == "attachment":
        a = r.get("attachment") or {}
        o = a.get("origin")
        if a.get("type") == "queued_command" and isinstance(o, dict) and o.get("kind") == "human":
            p = a.get("prompt")
            p = "\n".join(texts(p)) if not isinstance(p, str) else p
            return [p.strip()] if p and p.strip() else []
        return []
    c = (r.get("message") or {}).get("content")
    out = []
    for t in ([c] if isinstance(c, str) else [x.get("text", "") for x in c or [] if isinstance(x, dict)
                                               and x.get("type") == "text"]):
        for part in t.split(MIDTURN)[1:]:
            part = part.split("This is how Claude Code surfaces")[0].strip()
            if part:
                out.append(part)
    return out


def scan(path):
    """The messages typed since the current turn began (its prompt, then any sent mid-turn), and EVERY assistant text
    block since that prompt, joined: an answer written before a tool call is as much the reply as the last block
    (2026-10-25: reading only the last block failed answers that sat before a tool call, three times in a row).
    Thinking blocks are not part of the reply: the user does not read them."""
    msgs, replies = [], []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = r.get("message") or {}
        if r.get("type") == "attachment":
            for part in midturn(r):
                if part not in msgs:
                    msgs.append(part)
        elif r.get("type") == "user":
            if r.get("isMeta") or r.get("isCompactSummary"):
                continue
            mid = midturn(r)
            for part in mid:
                if part not in msgs:
                    msgs.append(part)
            if mid:
                continue
            for t in texts(m.get("content")):
                t = t.strip()
                if (t and not t.startswith(SKIP) and r.get("toolUseResult") is None and not r.get("isMeta")
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


def check(inp, ask=None):
    """(2, the reason) when the reply leaves the user's question unanswered, else (0, ""). Imported by plan_check.py
    (agent redesign B1); the standalone hook runs it through main(). `ask` stands in for the model in tests. Any
    exception passes the turn: plan_check fails closed, so an error here would block it."""
    try:
        return _check(inp, ask)
    except Exception:
        return 0, ""


def _check(inp, ask):
    path = inp.get("transcript_path")
    if not path or not os.path.exists(path):
        return 0, ""
    msgs, reply = scan(path)
    reply = reply or inp.get("last_assistant_message") or ""
    if not msgs or not reply:
        return 0, ""
    recent = msgs[-3:]
    h = hashlib.sha1(recent[-1].encode()).hexdigest()
    try:
        st = json.load(open(STATE))
    except (OSError, ValueError):
        st = {}
    if st.get("hash") == h:
        return 0, ""                  # this message was already checked; never block twice
    json.dump({"hash": h}, open(STATE, "w"))
    v = (ask or ask_model)(prompt(recent, reply))
    if v is None:
        return 0, ""
    ok, why = verdict(v, recent)
    if ok:
        return 0, ""
    return 2, ("REPLY CHECK: your question \"" + why + "\" is not answered. " + str(v.get("reason", "")).strip() +
          " Answer that directly in a revised reply (this check runs once per user message). If the check is wrong"
          " (the question was not the researcher's, e.g. it sits in text they pasted from someone else, or the reply"
          " does answer it), do not manufacture an answer: end the turn with one line to the user naming this check, what it"
          " flagged and why that is wrong (CLAUDE.md 0g).")


def prompt(recent, reply):
    return f"""You check one reply of an assistant to a researcher. Answer ONLY with JSON: {{"ok": true}} or
{{"ok": false, "question": "<the researcher's unanswered question, copied word for word from their message>",
  "reason": "<one sentence: what the reply leaves out>"}}.

The researcher's latest message(s), oldest first:
<<<
{chr(10).join(recent)[-6000:]}
>>>

The assistant's reply:
<<<
{reply[-8000:]}
>>>

Rules:
1. A question is a sentence of the researcher's own that ends with "?", or an explicit request to explain ("explain
   why ...", "tell me what ..."). A statement, a complaint, an instruction or a remark is not a question, even when it
   sounds dissatisfied ("I don't see what the question is", "you didn't answer"): do not turn it into one.
2. A message may hold several questions, even within one sentence ("What is X? And why Y, are you saying Z?"): each
   is a question of its own. Every such question must be answered directly somewhere in the reply (the reply may be several messages; an answer
   in an early one counts). A plan, an apology, or an answer to a different question does not count. Read the whole
   reply before deciding; if the reply does answer, the verdict is {{"ok": true}}.
3. Only the researcher's OWN questions count. Text they pasted or quoted from other people (a chat thread with names
   like "Suprio Dubey:", an email, a review, a quoted earlier reply of the assistant, text inside <pasted_content>
   tags) is addressed to those people and needs no answer unless the researcher's own words ask for one.
4. Ignore everything else (style, length, whether the work is right, whether the reply asks something).
5. On "ok": false, "question" must be copied exactly from the researcher's message, so it can be found there.
"""


def ask_model(text):
    """sonnet's verdict as a dict, or None on any failure (no CLI, timeout, no JSON): the check never stalls a turn.
    haiku invented questions from statements and refused against its own reasoning (2026-10-30, 109 refusals)."""
    try:
        out = subprocess.run(["claude", "-p", "--setting-sources", "project", "--model", "sonnet",
                              "--no-session-persistence", "--disallowed-tools", "Bash,Read,Edit,Write,Agent,WebFetch"],
                             input=text, capture_output=True, text=True, timeout=TIMEOUT, cwd="/tmp").stdout
        v = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def norm(s):
    """lower case, curly quotes made straight, each whitespace run one character: a newline when the run holds one
    (a line break ends a sentence), else a space"""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').lower()
    return re.sub(r"\s+", lambda m: "\n" if "\n" in m.group(0) else " ", s).strip()


EXPLAIN = re.compile(r"\b(explain|tell me (why|what|how|where|when|which|who|whether|if)|"
                     r"(want|like|need|wanna) to know|let me know (why|what|how|where|when|which|whether|if))\b", re.I)
SELF_OK = re.compile(r"\b(the reply|it|this) (does|did) (answer|address)|\breply answers (it|this|that|both|all)\b|"
                     r"\banswers (it|this|that|both|all)\b|\b(should|would) be (ok|okay|fine)\b|"
                     r"\b(reply|answer|it|this) (is|seems|looks) (ok|okay|fine)\b|\bno question\b|\basked nothing\b|"
                     r"\bno (direct )?question (was|is) asked\b|\bcheck passes\b", re.I)


def self_ok(reason):
    """the reason says the reply answers or is fine: a SELF_OK phrase not negated by the few words before it ("the
    reply never answers it" is a refusal, "the reply answers all three" is not)"""
    for m in SELF_OK.finditer(reason):
        before = reason[max(0, m.start() - 30):m.start()].lower()
        near = " ".join(re.split(r"[,;.:]", before)[-1].split()[-3:])     # the same clause, three words back
        if re.search(r"\b(never|not|no|neither|nor|hardly|barely)\b|n't\b", near):
            continue
        after = reason[m.end():]
        if re.search(r"\b(but|yet|though|however|except)\b,?\s+(\S+\s+){0,4}?(not|never|doesn't|does not|didn't|did not|"
                     r"skips|misses|leaves|omits|ignores|fails|without|no)\b", after, re.I):
            continue          # "addresses X but not Y": the reason names a gap after granting a part
        return True
    return False


def verdict(v, recent):
    """(True, "") to pass, (False, the quoted question) to refuse. A refusal stands only when its "question" is found
    word for word (whitespace, case and curly quotes aside) in the researcher's messages AND is a question: the
    sentence it sits in ends with "?" (the next of . ! ? or a line break after the quote is a "?"), or the quote
    explicitly asks to explain. A refusal whose own reason says the reply answers or is ok is discarded (2026-10-30:
    haiku made questions of statements, "I really don't see what's the question here", and refused after reasoning
    "...the reply answers all three")."""
    if v.get("ok") is not False:
        return True, ""
    if self_ok(str(v.get("reason", ""))):
        return True, ""
    q = re.sub(r"\s+", " ", norm(str(v.get("question", "")))).strip(" \"'")
    if len(q) < 4:
        return True, ""
    msg = norm("\n".join(recent))
    at = msg.replace("\n", " ").find(q)
    if at < 0:
        return True, ""
    end = re.search(r"[.!?\n]", msg[at + len(q) - 1:])
    if (end and end.group(0) == "?") or EXPLAIN.search(q):
        return False, str(v.get("question")).strip()
    return True, ""


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    rc, msg = check(inp)
    if rc:
        print(msg, file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
