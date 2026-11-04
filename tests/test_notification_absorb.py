"""notification_absorb.py (UserPromptSubmit; agent_redesign next 15, D279): it blocks an agent's completion
notification only when the agent's report already came by its hand-back, and passes everything else.
  unit:   recorded notification texts (delivered, waiting on its own work, a background command, a notification with
          no hand-back in the transcript, typed text, the hook-input path through stdin)
  replay: every turn-starting user record and every queued message of the main-session transcripts on disk, each
          judged on its transcript cut at that record; prints the counts by trigger class and verdict, and fails if
          anything but an agent's delivered notification would be blocked, or a delivered one after its hand-back
          would pass. Read-only over ~/.claude/projects/-home-joaquin-work-FA; skipped when it is absent.
    python3 tests/test_notification_absorb.py [unit|replay] [--since ISO]"""
import collections, glob, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, ".claude", "hooks", "notification_absorb.py")
sys.path.insert(0, os.path.dirname(HOOK))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import notification_absorb as na  # noqa: E402

PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
A = "a6cc542156004bfd9"
DELIVERED = ("<task-notification>\n<task-id>%s</task-id>\n<tool-use-id>toolu_01FRyT1tmhjRQ63UsCFuUDWG</tool-use-id>\n"
             "<output-file>/tmp/x/tasks/%s.output</output-file>\n<status>completed</status>\n<summary>Agent \"Catalog "
             "coverage\" finished</summary>\n<note>A task-notification fires each time this agent stops with no live "
             "background children of its own.</note>\n<result>This agent's report was delivered to you as a message "
             "from \"%s\" (its SubagentHandback call). Read it there; it is not repeated here.\n</result>\n"
             "<usage><subagent_tokens>30962</subagent_tokens></usage>\n</task-notification>") % (A, A, A)
WAITING = ("<task-notification>\n<task-id>%s</task-id>\n<status>completed</status>\n<summary>Agent \"x\" is waiting "
           "on its own background work</summary>\n<result>This agent has not reported yet: it is waiting on its own "
           "background work and will deliver its report through SubagentHandback when that finishes.\n</result>\n"
           "</task-notification>") % A
INLINE = ("<task-notification>\n<task-id>%s</task-id>\n<status>completed</status>\n<summary>Agent \"x\" finished"
          "</summary>\n<result>Joaquin, the figure is done.</result>\n</task-notification>") % A
KILLED = DELIVERED.replace("<status>completed</status>", "<status>killed</status>")
BASH = ("<task-notification>\n<task-id>bqjn1kqfw</task-id>\n<tool-use-id>toolu_01X</tool-use-id>\n<status>completed"
        "</status>\n<summary>Background command \"site events --wait\" completed (exit code 0)</summary>\n"
        "</task-notification>")
PREAMBLE = ("[SYSTEM NOTIFICATION - NOT USER INPUT]\nThis is an automated background-task event, NOT a message from "
            "the user.\n\n")


def transcript(tmp, with_handback):
    p = os.path.join(tmp, "t.jsonl")
    recs = [{"type": "user", "message": {"role": "user", "content": "go"}}]
    if with_handback:
        recs.append({"type": "attachment", "attachment": {"type": "queued_command", "prompt":
                     "<agent-message from=\"%s\">\n[Subagent hand-back] The text below is the final report" % A}})
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def unit():
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        hb, nohb = transcript(tmp, True), None
        os.makedirs(os.path.join(tmp, "n"))
        nohb = transcript(os.path.join(tmp, "n"), False)
        cases = [("delivered after its hand-back", DELIVERED, hb, True),
                 ("delivered, harness preamble first", PREAMBLE + DELIVERED, hb, True),
                 ("delivered, no hand-back in the transcript", DELIVERED, nohb, False),
                 ("delivered, no transcript path", DELIVERED, None, False),
                 ("waiting on its own background work", WAITING, hb, False),
                 ("report inline in the notification", INLINE, hb, False),
                 ("agent killed", KILLED, hb, False),
                 ("background command", BASH, hb, False),
                 ("typed text quoting a notification late", "x" * 2000 + DELIVERED, hb, False),
                 ("typed text", "Joaquin here: why did it wake twice?", hb, False),
                 ("delivered + a second notification (inline)", DELIVERED + "\n" + INLINE.replace(A, "a0123456789abcdef"),
                  hb, False),
                 ("delivered + trailing system-reminder", DELIVERED + "\n<system-reminder>Goal check-in</system-reminder>",
                  hb, False),
                 ("delivered + trailing whitespace", DELIVERED + "\n\n", hb, True)]
        for name, prompt, tp, want in cases:
            got, why = na.verdict(prompt, tp)
            print("%-45s block=%-5s %s" % (name, got, why))
            if got != want:
                fails.append(name)
        # the hook-input path: stdin JSON, the block JSON on stdout, exit 0; a crash exits 1 without blocking
        inp = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": DELIVERED, "transcript_path": hb})
        r = subprocess.run([sys.executable, HOOK], input=inp, capture_output=True, text=True)
        ok = r.returncode == 0 and json.loads(r.stdout).get("decision") == "block"
        print("%-45s rc=%d %s" % ("stdin: delivered", r.returncode, r.stdout.strip()))
        if not ok:
            fails.append("stdin delivered")
        r = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "UserPromptSubmit",
                           "prompt": "hello", "transcript_path": hb}), capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip():
            fails.append("stdin typed")
        # through the dispatcher, twice the same block with nothing between: both blocks stand, no incident (a prompt
        # absorbed is not a loop); DISPATCH_SITE stubbed so a wrong ALERT would not reach the ledger
        inc = os.path.join(os.path.dirname(HOOK), "incidents.jsonl")
        n0 = sum(1 for _ in open(inc)) if os.path.exists(inc) else 0
        env = dict(os.environ, DISPATCH_STATE=os.path.join(tmp, "state.json"), DISPATCH_SITE="true")
        inp2 = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": DELIVERED, "transcript_path": hb,
                           "session_id": "s-test", "cwd": tmp})
        outs = [subprocess.run([sys.executable, os.path.join(os.path.dirname(HOOK), "dispatch.py"),
                                "notification_absorb.py"], input=inp2, capture_output=True, text=True, env=env)
                for _ in range(2)]
        n1 = sum(1 for _ in open(inc)) if os.path.exists(inc) else 0
        ok = all(o.returncode == 0 and '"block"' in o.stdout for o in outs) and n1 == n0
        print("%-45s %s (incidents %d -> %d)" % ("dispatch: the same block twice", "ok" if ok else "WRONG", n0, n1))
        if not ok:
            fails.append("dispatch twice")
        r = subprocess.run([sys.executable, HOOK], input="not json", capture_output=True, text=True)
        print("%-45s rc=%d (non-blocking)" % ("stdin: bad input", r.returncode))
        if r.returncode != 1 or r.stdout.strip():
            fails.append("bad input")
    return fails


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def replay(since=None):
    """Every turn-starting user record and every queued message of the main transcripts, judged on its transcript
    cut at that record. Returns the failures; prints the counts by (kind, verdict)."""
    import main_turns as mt
    paths = sorted(glob.glob(os.path.join(PROJ, "*.jsonl")))
    if not paths:
        print("replay: no transcripts, skipped")
        return []
    counts, fails = collections.Counter(), []
    with tempfile.TemporaryDirectory() as tmp:
        cut = os.path.join(tmp, "cut.jsonl")
        for p in paths:
            with open(p, errors="replace") as f:
                lines = f.readlines()
            out = open(cut, "w")
            handed = set()
            for line in lines:
                out.write(line)
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("timestamp") or ""
                if d.get("type") == "user":
                    c = (d.get("message") or {}).get("content")
                    if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                        txt = "".join(text_of(b.get("content")) for b in c if isinstance(b, dict))
                        handed |= set(na.re.findall(r'agent-message from="(a[0-9a-f]{16})"', txt))
                        continue
                    prompt, where = text_of(c), "turn"
                elif d.get("type") == "attachment" and (d.get("attachment") or {}).get("type") == "queued_command":
                    a = d["attachment"]
                    prompt, where = a.get("prompt") if isinstance(a.get("prompt"), str) else text_of(a.get("prompt")), "queued"
                else:
                    continue
                if not prompt:
                    continue
                for m in na.re.finditer(r'agent-message from="(a[0-9a-f]{16})"[^>]*>\s*\[Subagent hand-back\]', prompt):
                    handed.add(m.group(1))
                if since and t < since:
                    continue
                out.flush()
                kind = mt.trigger_of(prompt, d.get("origin"), {})
                block, why = na.verdict(prompt, cut)
                m = na.AGENT.search(prompt[:2000])
                deliv = bool(na.DELIVERED.search(prompt)) and "<status>completed</status>" in prompt
                counts[(where, kind, "delivered" if kind == "agent_done" and deliv else "-", block)] += 1
                if block and not (kind == "agent_done" and deliv):
                    fails.append(("blocked a %s that is not a hand-back delivery" % kind, p, t))
                if kind == "agent_done" and deliv and m and m.group(1) in handed and not block:
                    fails.append(("passed a delivered notification after its hand-back", p, t, why))
            out.close()
    for k, v in sorted(counts.items()):
        print("replay %-7s %-12s %-10s block=%-5s %d" % (k + (v,)))
    return fails


if __name__ == "__main__":
    args = sys.argv[1:]
    since = None
    if "--since" in args:
        since = args[args.index("--since") + 1]
        args = [a for a in args if a not in ("--since", since)]
    groups = args or ["unit", "replay"]
    fails = []
    if "unit" in groups:
        fails += unit()
    if "replay" in groups:
        fails += replay(since)
    for f in fails:
        print("FAIL", f)
    print("%s: %d failure(s)" % ("FAIL" if fails else "PASS", len(fails)))
    sys.exit(1 if fails else 0)
