"""The reply check (.claude/hooks/reply_check.py) on exchanges of the main session's transcript of 15 to 30 Oct
(71602535-...): each case is the slice of the transcript from the user's message to the refusal the old haiku check
gave, kept in tests/fixtures/reply_check/<refusal line>.jsonl (tool outputs cut to 3000 characters, thinking kept).

    python3 tests/test_reply_check.py            the scan and the verdict logic, the model stubbed (seconds)
    python3 tests/test_reply_check.py --model    also each case through sonnet (minutes; run it in the background)
    python3 tests/test_reply_check.py --build    rebuild the fixtures from the transcript

FALSE: refusals the old check gave wrongly; the new check must pass them. REAL: refusals that were right (the next
reply answered what was missing); the new check must still refuse them."""
import json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import reply_check as RC  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures", "reply_check")
TRANSCRIPT = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA/71602535-0955-42ad-a58f-23dc320798dc.jsonl")

# refusal line in the transcript: (what the old check flagged, the stub verdict haiku's refusal maps to)
FALSE = {
    136915: ("a statement read as a question (30 Oct)",
             {"ok": False, "question": "I really don't see what's the question here",
              "reason": "the reply never states what question the assistant was asking"}),
    136008: ("\"you didn't answer\", answered before (30 Oct)",
             {"ok": False, "question": "I asked how was this not part of what I asked for, you didn't answer",
              "reason": "it never directly answers why the 57 pools were not covered"}),
    134510: ("a statement of fact read as a request (30 Oct)",
             {"ok": False, "question": "I just wanted to know where the tokens went",
              "reason": "the reply never says where they went"}),
    133900: ("its reason says the reply answers all three (30 Oct)",
             {"ok": False, "question": "where exactly did the context of the project go?",
              "reason": "the reply answers all three, but the opening line is a promise rather than the answer"}),
    # real questions the reply did answer: only the model can pass these, so no stub
    135899: ("both numbered requests answered under their own headings (30 Oct)", None),
    133565: ("\"Why is there a question there not resolved?\", answered first thing (30 Oct)", None),
    131986: ("its reason says no question was asked and the check passes (29 Oct)",
             {"ok": False, "question": "Yes, trimmed to their logs, and yes fix 2 and 3",
              "reason": "The researcher asked nothing explicit, so no question needed answering; this check passes"}),
}
REAL = {
    73434: ("\"why 20 minutes\" left without a reason (22 Oct; the next reply gave one)",
            {"ok": False, "question": "And why 20 minutes",
             "reason": "the reply never gives a reason for that interval"}),
    130834: ("\"What do you mean adding a user pause\" left unexplained (29 Oct; the next reply explained it)",
             {"ok": False, "question": "What do you mean adding a user pause to the stop hooks?",
              "reason": "the reply never explains what that pause was"}),
    83830: ("\"What do you mean 'if we ever trim a finished chapter'?\" left unexplained (23 Oct)",
            {"ok": False, "question": "What do you mean \"if we ever trim a finished chapter\"?",
             "reason": "the reply never explains that phrase"}),
    93102: ("\"Isn't /goal supposed to do this exactly? What do you mean?\" not addressed (24 Oct)",
            {"ok": False, "question": "Isn't /goal supposed to do this exactly?",
             "reason": "the reply never addresses /goal"}),
    96475: ("\"Why would we invent something if it's a settled convention?\" not answered (24 Oct)",
            {"ok": False, "question": "Why would we invent something if it's a settled convention?",
             "reason": "the reply never answers it directly"}),
    116802: ("\"why would you do it again?\" about the pretraining sweep, not answered (26 Oct)",
             {"ok": False, "question": "But you already swept a pretraining, why would you do it again?",
              "reason": "the reply never says whether the pretraining is rerun"}),
    121663: ("\"why is this stated like this rather than shown?\" about slide 5, not answered (27 Oct)",
             {"ok": False, "question": "why is this stated like this rather than shown?",
              "reason": "the reply only commits to changing it to a plot"}),
    122296: ("\"how many times did it find something?\" given no count (29 Oct)",
             {"ok": False, "question": "out of the times it triggered, how many times did it find something?",
              "reason": "the reply gives no count"}),
}
# the live misfire of 30 Oct 11:07: the main session had read reply_check.py's source with sed, and the source's
# MIDTURN constant made that tool output a "message of the user's"; the answer the main session meant sat in a
# thinking block, the visible reply was one plan line
LIVE = 136982
LIVE_Q = "Why does this hook keep misfiring? Aren't the changes supposed to fix that?"
RESULTS = []


def scenario(name, ok, info=""):
    RESULTS.append(bool(ok))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "\n      " + str(info)[:800]))


def build():
    """one pass over the transcript: for each case, the records from the turn's prompt to its refusal"""
    want = set(FALSE) | set(REAL) | {LIVE}
    os.makedirs(FIX, exist_ok=True)
    start, buf = None, []
    for i, line in enumerate(open(TRANSCRIPT), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("type")
        if t == "user" and r.get("toolUseResult") is None and not r.get("isMeta") \
                and not r.get("isCompactSummary"):
            c = (r.get("message") or {}).get("content")
            first = c if isinstance(c, str) else next((x.get("text", "") for x in c or [] if isinstance(x, dict)
                                                       and x.get("type") == "text"), "")
            if first.strip() and not first.strip().startswith(RC.SKIP):
                start, buf = i, []
        if i in want:
            open(os.path.join(FIX, "%d.jsonl" % i), "w").writelines(json.dumps(x) + "\n" for x in buf)
            print("case %d: lines %s to %d, %d records" % (i, start, i, len(buf)))
        if t not in ("user", "assistant", "attachment"):
            continue
        if t == "attachment" and (r.get("attachment") or {}).get("type") != "queued_command":
            continue
        keep = {k: r[k] for k in ("type", "timestamp", "isMeta", "isCompactSummary", "attachment") if k in r}
        if r.get("toolUseResult") is not None:
            keep["toolUseResult"] = True
        m = r.get("message")
        if m is not None:
            c = m.get("content")
            if isinstance(c, list):
                c = [dict((k, v) for k, v in x.items() if k != "signature") for x in c if isinstance(x, dict)]
                for x in c:
                    if x.get("type") == "tool_result":
                        cc = x.get("content")
                        x["content"] = (cc[:3000] if isinstance(cc, str) else
                                        [dict(y, text=y.get("text", "")[:3000]) for y in cc or [] if isinstance(y, dict)])
                    if x.get("type") == "tool_use":
                        x["input"] = {"_": json.dumps(x.get("input"))[:300]}
            keep["message"] = {"role": m.get("role"), "content": c}
        buf.append(keep)


def path(n):
    return os.path.join(FIX, "%d.jsonl" % n)


def run(n, ask):
    """check() on case n with a fresh state file and `ask` for the model; (rc, msg, the prompt the model saw)"""
    seen = []
    with tempfile.TemporaryDirectory() as d:
        RC.STATE = os.path.join(d, "state")
        rc, msg = RC.check({"transcript_path": path(n)}, ask=lambda p: (seen.append(p), ask(p))[1])
    return rc, msg, seen[0] if seen else ""


def stubbed():
    for n, (what, v) in FALSE.items():
        if v is None:
            continue
        rc, msg, _ = run(n, lambda p, v=v: v)
        scenario("false refusal %d passes, model stubbed with haiku's refusal: %s" % (n, what), rc == 0, msg)
    for n, (what, v) in REAL.items():
        rc, msg, _ = run(n, lambda p, v=v: v)
        scenario("real catch %d still refuses, model stubbed: %s" % (n, what), rc == 2 and "REPLY CHECK" in msg, msg)

    # the quote logic on its own
    m = ["What supervisor are you talking about? And why 20 minutes, are you telling me X?\n\nI don't see it"]
    V = lambda q, reason="missing": {"ok": False, "question": q, "reason": reason}
    for q, want in (("why 20 minutes", False), ("WHAT   supervisor are you\ntalking about?", False),
                    ("I don't see it", True), ("why is the supervisor slow?", True), ("", True), ("why", True)):
        scenario("quote %r: %s" % (q, "refuses" if not want else "passes"), RC.verdict(V(q), m)[0] is want,
                 RC.verdict(V(q), m))
    scenario("an explicit request to explain refuses without a \"?\"",
             not RC.verdict(V("please explain why the run stopped"), ["please explain why the run stopped."])[0])
    for reason in ("Actually, the reply does answer why", "so this should be ok", "the reply answers both parts",
                   "no question needed answering"):
        scenario("reason %r is discarded" % reason, RC.verdict(V("why 20 minutes", reason), m)[0])
    for reason in ("the reply never answers it directly", "it does not answer this", "the reply doesn't answer it",
                   "The reply answers the first part, but never answers why"):
        scenario("reason %r is kept" % reason, RC.verdict(V("why 20 minutes", reason), m)[0] is False)
    scenario("a refusal without ok: false passes", RC.verdict({"reason": "x", "question": "why 20 minutes"}, m)[0])
    scenario("a failing model lets the turn end", run(73434, lambda p: None)[0] == 0)

    # once per message
    with tempfile.TemporaryDirectory() as d:
        RC.STATE = os.path.join(d, "state")
        v = REAL[73434][1]
        a = RC.check({"transcript_path": path(73434)}, ask=lambda p: v)[0]
        b = RC.check({"transcript_path": path(73434)}, ask=lambda p: v)[0]
        scenario("once per message: refuses, then passes the same message", (a, b) == (2, 0), (a, b))

    # the scan: the live misfire, and the reply read across every text block of the turn
    msgs, reply = RC.scan(path(LIVE))
    scenario("live case: the user's message is the only message (no tool output read as one)", msgs == [LIVE_Q],
             [x[:80] for x in msgs])
    scenario("live case: the reply is the visible text, the thinking block is not in it",
             reply.startswith("Joaquin, the reply-check fix is recorded") and "Haiku" not in reply, reply[:300])
    with tempfile.TemporaryDirectory() as d:
        t = os.path.join(d, "t.jsonl")
        recs = [json.loads(x) for x in open(path(LIVE))]
        first = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text":
                 "Joaquin, it misfires because the changes never touched its judgment: haiku invents questions."}]}}
        k = next(i for i, r in enumerate(recs) if r["type"] == "assistant")
        open(t, "w").writelines(json.dumps(r) + "\n" for r in recs[:k] + [first] + recs[k:])
        msgs, reply = RC.scan(t)
        scenario("answer in the first text block, a plan line last: the reply holds both",
                 "never touched its judgment" in reply and reply.rstrip().endswith("touching the same file."), reply)
        mid = {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "And how long will it take?",
                                                    "origin": {"kind": "human"}}}
        bot = {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "<task-notification>x",
                                                    "origin": {"kind": "task-notification"}}}
        open(t, "w").writelines(json.dumps(r) + "\n" for r in recs[:k] + [mid, bot] + recs[k:])
        msgs, _ = RC.scan(t)
        scenario("a message typed mid-turn is read; a task notification is not",
                 msgs == [LIVE_Q, "And how long will it take?"], msgs)


def model():
    """each case through the real model (sonnet); the verdict and its JSON are printed"""
    for group, cases, want in (("false refusal", FALSE, 0), ("real catch", REAL, 2)):
        for n, (what, _) in cases.items():
            got = {}
            def ask(p):
                got["v"] = RC.ask_model(p)
                return got["v"]
            rc, msg, _ = run(n, ask)
            scenario("model: %s %d %s: %s" % (group, n, "passes" if want == 0 else "refuses", what), rc == want,
                     "%s | %s" % (json.dumps(got.get("v")), msg))
            print("      model said: %s" % json.dumps(got.get("v"))[:400])


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
        sys.exit(0)
    stubbed()
    if "--model" in sys.argv:
        model()
    print("%d/%d passed" % (sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
