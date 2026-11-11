#!/usr/bin/env python3
"""UserPromptSubmit: absorb the second wake of an agent end (agent_redesign next 15, D279).

Each background subagent's end gives the main session two signals: its SubagentHandback message (the report) and the
harness's <task-notification> "Agent ... finished", whose <result> says the report "was delivered to you as a message
from <id> (its SubagentHandback call)". Mid-turn the notification is folded into the running turn at no cost; when it
arrives after the turn ended it starts a turn of its own, and in 4 of 32 such cases since 2026-10-31 the model was
called on it at the full context (main_turns.py agent_done).

This hook blocks that prompt, so no model call is made on it, only when all hold:
  - the prompt is a <task-notification> whose task id is an agent id (a + 16 hex) and whose status is completed;
  - its result says the report was delivered through SubagentHandback (nothing in it is new);
  - the prompt is that one notification and nothing else (no second notification, no reminder after it);
  - the transcript holds a hand-back message from that same agent (the report did reach this session).
Anything else (a notification with the report inline, an agent waiting on its own work, a failed or killed agent,
a background command, typed text) passes untouched. A crash exits 1 (non-blocking: the prompt goes through).

Hand-back batching (D324, Joaquin 2026-11-07: "10 minutes seems reasonable"): a prompt that is one agent's hand-back
and nothing else, arriving in the main session (no agent_id) while its events waiter (`site events --wait`, pid file
~/.local/share/ccorch/events_waiter.FA.pid) is alive, is written to ~/.local/share/ccorch/spool/FA/ and opened as a
routine HANDBACK event (`site event`); the prompt is blocked. The waiter collects routine events for a fixed
--debounce window (600 s) from the first, then prints each spooled report in full and acks it: hand-backs, run ends and
other routine events of one 10-minute window wake the main session once. No waiter alive, a failed `site event`, a
prompt holding anything else (typed text, a second message): the prompt goes through untouched.
    python3 notification_absorb.py            (hook input on stdin)
    python3 notification_absorb.py --check <transcript.jsonl> <prompt file>    the verdict, for tests and replays
"""
import hashlib, json, os, re, subprocess, sys, time

AGENT = re.compile(r"<task-id>(a[0-9a-f]{16})</task-id>")
DELIVERED = re.compile(r"report was delivered to you as a message from \"?(a[0-9a-f]{16})\"? \(its SubagentHandback call\)")


def handed_back(transcript, agent):
    """True when the transcript holds a hand-back message from `agent` (a queued message or a turn-starting one)."""
    tag = 'agent-message from=\\"%s\\"' % agent      # as written inside a JSON string
    tag_plain = 'agent-message from="%s"' % agent
    try:
        with open(transcript, errors="replace") as f:
            for line in f:
                if (tag in line or tag_plain in line) and "Subagent hand-back" in line:
                    return True
    except OSError:
        return False
    return False


def verdict(prompt, transcript):
    """(block, why) for one prompt."""
    s = prompt or ""
    i = s.find("<task-notification>")
    if i < 0 or i > 1500:          # the harness's notification preamble may come first, nothing else
        return False, "not a task-notification"
    s = s[i:]
    end = s.find("</task-notification>")
    if end < 0 or s.count("<task-notification>") != 1 or s[end + len("</task-notification>"):].strip():
        return False, "more than one notification, or text after it"      # anything else in it may be new
    s = s[:end]
    m = AGENT.search(s[:300])
    if not m:
        return False, "not an agent's notification"
    agent = m.group(1)
    if "<status>completed</status>" not in s:
        return False, "status is not completed"
    d = DELIVERED.search(s)
    if not d or d.group(1) != agent:
        return False, "the result is not a SubagentHandback delivery"
    if not transcript or not handed_back(transcript, agent):
        return False, "no hand-back from %s in the transcript" % agent
    return True, agent


# the hook receives the bare `<agent-message ...>` (absorb.log 2026-11-11T19:55Z); the transcript records it with the
# harness's "Another Claude session sent a message:" line before it: both match
HANDBACK = re.compile(r'\A(?:Another Claude session sent a message:\s*)?<agent-message from="(a[0-9a-f]{16})">\s*'
                      r'\[Subagent hand-back\]')
TRAILER = 'That "other Claude session" is an agent working inside this same session'
CCORCH = os.path.expanduser(os.environ.get("NOTIFY_CCORCH_HOME", "~/.local/share/ccorch"))
PROJECT = "FA"
SITE_BIN = os.path.expanduser(os.environ.get("NOTIFY_SITE_BIN", "~/.local/bin/site"))


def proc_start(pid):
    """(clock ticks since boot as written in /proc/<pid>/stat field 22, start as epoch seconds) or None."""
    try:
        ticks = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[19]
        btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime "))
        return ticks, btime + int(ticks) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError, StopIteration):
        return None


def waiter_alive(project=PROJECT):
    """The project's events waiter is alive: its pid file names a live `site events --wait` process whose start ticks
    match the ones it recorded (a reused pid does not pass). Liveness only: the earlier test that it started after
    `site` last changed went false at every edit of `site` until the waiter was re-armed (agent_redesign step 23);
    every waiter since 2026-11-09 knows HANDBACK."""
    try:
        f = open(os.path.join(CCORCH, "events_waiter.%s.pid" % project)).read().split()
        pid = int(f[0])
        cmd = open("/proc/%d/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError, IndexError):
        return False
    st = proc_start(pid)
    if not ("events" in cmd and "--wait" in cmd) or not st or (len(f) > 1 and f[1] != st[0]):
        return False
    return True


def handback_of(prompt):
    """The agent id when the prompt is exactly one hand-back (and the harness's standard note after it), else None."""
    s = (prompt or "").strip()
    m = HANDBACK.match(s)
    if not m or s.count("<agent-message") != 1:
        return None
    end = s.find("</agent-message>")
    if end < 0:
        return None
    rest = s[end + len("</agent-message>"):].strip()
    if rest and not rest.startswith(TRAILER):
        return None
    if rest.count("<") or rest.count("[") or "\n\n" in rest:     # anything after the standard note may be new
        return None
    return m.group(1)


def spool(prompt, agent, site=("site",)):
    """(ok, event line): the report written to the spool and opened as a HANDBACK event."""
    d = os.path.join(CCORCH, "spool", PROJECT)
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(prompt.encode()).hexdigest()[:10]
    path = os.path.join(d, "%d_%s_%s.txt" % (time.time(), agent, h))
    with open(path, "w") as f:
        f.write(prompt)
    body = prompt.split("The report follows:", 1)[-1].strip().splitlines()
    first = " ".join((body[0] if body else "").split())[:160]
    r = subprocess.run(list(site) + ["event", "--project", PROJECT, "--kind", "HANDBACK", "--key",
                                     "handback:%s:%s" % (agent, h), "--msg",
                                     "hand-back from %s: %s | report: %s" % (agent, first, path)],
                       capture_output=True, text=True, timeout=60)
    ok = r.returncode == 0 and re.search(r"\bE\d+\b", r.stdout or "")
    return bool(ok), (r.stdout or r.stderr or "").strip()[-200:]


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        with open(sys.argv[3]) as f:
            print(json.dumps(dict(zip(("block", "why"), verdict(f.read(), sys.argv[2])))))
        return 0
    d = json.load(sys.stdin)
    if d.get("hook_event_name", "UserPromptSubmit") != "UserPromptSubmit":
        return 0
    prompt = str(d.get("prompt") or "")
    agent = None if d.get("agent_id") else handback_of(prompt)
    if agent:
        # agent_redesign step 23: no hand-back was ever spooled (no spool dir, no HANDBACK event since 10-31); one line
        # per hand-back says which condition held, so the next ones show why
        try:
            with open(os.path.join(CCORCH, "absorb.log"), "a") as f:
                f.write("%s %s waiter_alive=%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), agent,
                                                     waiter_alive()))
        except OSError:
            pass
    elif not d.get("agent_id") and "<agent-message" in prompt[:400]:
        try:
            with open(os.path.join(CCORCH, "absorb.log"), "a") as f:
                f.write("%s not-one-handback %r\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prompt[:120]))
        except OSError:
            pass
    if agent and waiter_alive():
        ok, line = spool(prompt, agent)
        if ok:
            print(json.dumps({"decision": "block", "reason": "hand-back of %s spooled for the 10-minute batch (%s); "
                              "the events waiter delivers it" % (agent, line)}))
            return 0
    block, why = verdict(prompt, d.get("transcript_path"))
    if block:
        print(json.dumps({"decision": "block",
                          "reason": "agent %s finished; its report was already delivered by its hand-back" % why}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:      # never block on a crash
        print("notification_absorb.py: %s" % e, file=sys.stderr)
        sys.exit(1)
