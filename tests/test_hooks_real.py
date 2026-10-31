"""The agent redesign's hooks on the real states of the night of 29 to 30 Oct (tests/fixtures/hooks_20261029/, built by
tools/hook_fixtures.py; the night's transcripts read in place). Nothing live is touched: every hook runs from a copy in
a temporary hooks directory or with its state, registry and `site` pointed at temporary stubs.
    python3 tests/test_hooks_real.py [group ...]"""
import json, os, re, shutil, sqlite3, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS = os.path.join(ROOT, ".claude", "hooks")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hook_fixtures as HF  # noqa: E402

MAIN = "71602535-0955-42ad-a58f-23dc320798dc"          # the night's main session
RESULTS = []
J = lambda x: json.dumps(x, separators=(",", ":"))


def scenario(name, ok, info=""):
    RESULTS.append(bool(ok))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "\n      " + str(info)[:1500].replace("\n", "\n      ")))


# ------------------------------------------------------------------ A1
def fixtures():
    got = HF.counts()
    for k, v in HF.EXPECTED.items():
        scenario("A1: the fixture reproduces the night's %s count (%d)" % (k, v), got.get(k) == v, got)
    led = HF.load()
    ids = [e["id"] for e in led["events"] if e["project"] == "FA"]
    scenario("A1: the window's FA events are ids 2188 to 2416", (min(ids), max(ids)) == (2188, 2416), (min(ids), max(ids)))
    named = set()
    for e in led["events"]:
        named.update(re.findall(r"\bFA-[a-z0-9]+-[0-9a-f]{8}\b", e["msg"] or ""))
    have = {r["id"] for r in led["runs"]}
    scenario("A1: every run an event names is in the fixture's runs", named <= have, sorted(named - have)[:5])
    for sub in ("plan_before", "plan_head"):
        p = os.path.join(HF.DEFAULT_OUT, sub)
        scenario("A1: %s has the plan and its state" % sub, os.path.getsize(os.path.join(p, "plan.yaml")) > 1000
                 and os.path.exists(os.path.join(p, ".state.json")))


# ------------------------------------------------------------------ A2, A3: a sandbox for the dispatcher
STUB_SITE = r'''#!/usr/bin/env python3
import os, sqlite3, sys
d = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(d, "site_calls.log"), "a").write(repr(sys.argv[1:]) + "\n")
if sys.argv[1:2] == ["event"]:       # as the real site: the ALERT is an open event (it changes the dispatcher's state)
    c = sqlite3.connect(os.path.join(d, "runs.db"))
    c.execute("INSERT INTO events (ts, project, kind, key, msg) VALUES (1, 'FA', 'ALERT', 'k', 'm')")
    c.commit()
print("  E9001 FA ALERT: stub")
'''
STUB_HOOK = r'''#!/usr/bin/env python3
import os, sys
d = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(d, "ran.log"), "a").write(os.path.basename(__file__) + " " + " ".join(sys.argv[1:]) + "\n")
sys.stdin.read()
msg = open(os.path.join(d, "message.txt")).read() if os.path.exists(os.path.join(d, "message.txt")) else ""
if msg:
    sys.stderr.write(msg)
    sys.exit(2)
'''


class Box:
    """a hooks directory with the real dispatch.py and stub hooks, a git repo as the project, a registry, a transcript."""

    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="dispatch_")
        j = lambda *x: os.path.join(self.d, *x)
        self.hooks, self.repo, self.tr = j("hooks"), j("repo"), j("transcript.jsonl")
        os.makedirs(self.hooks)
        shutil.copy(os.path.join(HOOKS, "dispatch.py"), self.hooks)
        for name in ("plan_check.py", "lock_guard.py"):
            open(os.path.join(self.hooks, name), "w").write(STUB_HOOK)
        open(j("site"), "w").write(STUB_SITE)
        os.chmod(j("site"), 0o755)
        os.makedirs(os.path.join(self.repo, "plan"))
        open(os.path.join(self.repo, "plan", "plan.yaml"), "w").write("entries: []\n")
        g = lambda *a: subprocess.run(["git", "-C", self.repo] + list(a), capture_output=True, check=True,
                                      env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))
        g("init", "-q"); g("add", "-A"); g("commit", "-qm", "init")
        self.git = g
        open(j("projects.toml"), "w").write('[projects.FA]\nlocal = "%s"\n' % self.repo)
        c = sqlite3.connect(j("runs.db"))
        c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts INTEGER, project TEXT, item TEXT, kind TEXT, key TEXT,"
                  " msg TEXT, acked INTEGER, ack_note TEXT, mute_until INTEGER, tell TEXT)")
        c.commit()
        open(self.tr, "w").close()
        self.env = dict(os.environ, DISPATCH_SITE=j("site"), DISPATCH_RUNS_DB=j("runs.db"),
                        DISPATCH_PROJECTS=j("projects.toml"), DISPATCH_STATE=j("state.json"),
                        CLAUDE_PROJECT_DIR=self.repo)

    def say(self, msg):
        p = os.path.join(self.hooks, "message.txt")
        if msg:
            open(p, "w").write(msg)
        elif os.path.exists(p):
            os.remove(p)

    def append(self, recs):
        with open(self.tr, "a") as f:
            for r in recs:
                f.write(J(r) + "\n")

    def tool_use(self, name="Bash", inp=None):
        self.append([{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "x", "name": name,
                                                                    "input": inp or {"command": "ls"}}]}}])

    def run(self, hook="plan_check.py", event="Stop", tool=None, session="s1", args=()):
        inp = {"session_id": session, "transcript_path": self.tr, "cwd": self.repo, "hook_event_name": event}
        if tool:
            inp.update(tool_name=tool[0], tool_input=tool[1])
        r = subprocess.run([sys.executable, os.path.join(self.hooks, "dispatch.py"), hook] + list(args),
                           input=J(inp), capture_output=True, text=True, env=self.env, cwd=self.repo)
        return r.returncode, r.stdout + r.stderr

    def disabled(self):
        try:
            return json.load(open(os.path.join(self.hooks, "disabled.json")))
        except OSError:
            return {}

    def incidents(self):
        p = os.path.join(self.hooks, "incidents.jsonl")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    def site_calls(self):
        p = os.path.join(self.d, "site_calls.log")
        return open(p).read().splitlines() if os.path.exists(p) else []

    def ran(self):
        p = os.path.join(self.hooks, "ran.log")
        return open(p).read().splitlines() if os.path.exists(p) else []


def night_stop_refusals():
    """[(time, message, records since the previous refusal)] of plan_check's Stop refusals in the main session."""
    out, since = [], []
    for _, r in HF.transcript_records(name=MAIN):
        errs = r.get("hookErrors") if r.get("subtype") == "stop_hook_summary" else None
        pc = [e for e in errs or [] if isinstance(e, str) and "/plan_check.py" in e.split("]:")[0]]
        if pc:
            out.append((r["timestamp"], pc[0].split("]: ", 1)[1], since))
            since = []
        elif r.get("type") in ("assistant", "user"):
            since.append(r)
    return out


def incident_tool():
    """A2: tools/hook_incident.py disables, records once, re-enables."""
    b = Box()
    cli = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hook_incident.py"), "--hooks-dir",
                                     b.hooks] + list(a), capture_output=True, text=True, env=b.env, cwd=b.repo)
    r = cli("plan_check.py", "--why", "refuses for runs already acked")
    inc, calls = b.incidents(), b.site_calls()
    scenario("A2: disabling plan_check.py writes disabled.json and incidents.jsonl and opens one ALERT",
             r.returncode == 0 and "plan_check.py" in b.disabled() and len(inc) == 1
             and inc[0]["why"] == "refuses for runs already acked" and len(calls) == 1 and "'ALERT'" in calls[0]
             and "'--project', 'FA'" in calls[0], (r.stdout, r.stderr, inc, calls))
    r2 = cli("plan_check.py", "--why", "again")
    scenario("A2: a second --why on a disabled hook adds no event and no incident",
             r2.returncode == 0 and len(b.site_calls()) == 1 and len(b.incidents()) == 1, (r2.stdout, b.site_calls()))
    rc, out = b.run()
    scenario("A3: a hook in disabled.json is not run (exit 0, a note)", rc == 0 and "disabled" in out and not b.ran(),
             (rc, out, b.ran()))
    r3 = cli("--resolve", "plan_check.py")
    scenario("A2: --resolve removes it", r3.returncode == 0 and "plan_check.py" not in b.disabled(), r3.stdout)
    rc, out = b.run()
    scenario("A3: once resolved it runs again", rc == 0 and b.ran() == ["plan_check.py "], (rc, out, b.ran()))
    r4 = cli("--list")
    scenario("A2: --list shows the incident and its resolution", r4.returncode == 0 and "incidents (2" in r4.stdout,
             r4.stdout)
    r5 = cli("no_such_hook.py", "--why", "x")
    scenario("A2: a hook that does not exist is refused", r5.returncode != 0, r5.stderr)
    # (a): an incident opened by hand carries the hook's last refusal the dispatcher saw
    b = Box()
    b.say("lock guard: only repo-reviewer advances its pillar")
    cmd = ("Bash", {"command": "bash .claude/hooks/review_backlog.sh advance repo-reviewer"})
    b.tool_use(*cmd)
    b.run("lock_guard.py", "PreToolUse", cmd)
    r = _incident_cli(b, "lock_guard.py", "--why", "refuses the reviewer's own advance")
    inc = b.incidents()
    msg = (b.site_calls() or [""])[-1]
    scenario("(a) an incident opened by hand (hook_incident.py --why) carries the last refusal: the call, its text, "
             "the session; its ALERT is for the main session", r.returncode == 0 and len(inc) == 1
             and inc[0]["call"] == list(cmd) and inc[0]["last_refusal"].startswith("lock guard: only")
             and inc[0]["session"] == "s1" and "review_backlog.sh advance" in msg and "lock guard: only" in msg
             and "for the main session to fix" in msg and "--replay " + inc[0]["id"] in r.stdout, (r.stdout, inc, msg))


def _incident_cli(b, *a):
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hook_incident.py"), "--hooks-dir", b.hooks]
                          + list(a), capture_output=True, text=True, env=b.env, cwd=b.repo)


def incident_for_main_session(b, rec, call, hook, refusal_head, fixed_by, still_on):
    """Joaquin 2026-10-30 ("why would it directly reach me intead of trying to fix it first if the fix is obvious?"):
    an incident is an ALERT the main session acts on, carrying what a fix needs (hook, event, call refused, refusal
    text, count, session); it reaches the user only through `site ack --needs-user`; the fix is tested on the recorded
    refusal with --replay."""
    alert = [c for c in b.site_calls() if "'event'" in c]
    msg = alert[-1] if alert else ""
    need = ["HOOK INCIDENT " + rec["id"], "for the main session to fix (not addressed to the user)", hook,
            "%s time(s) in a row" % rec.get("count"), "session s1", re.sub(r"\s+", " ", refusal_head)[:40], "--replay " + rec["id"],
            "--needs-user", "'--kind', 'ALERT'"]
    if call:
        need += [call[0], call[1]["file_path"]]
    else:
        need += ["the turn could not end", "--resolve"]
    scenario("(a) %s: the incident's ALERT is for the main session and carries the hook, the call refused, the refusal "
             "text, the count (%s) and the session, with the fix-first route and --needs-user as the only way to the "
             "user" % (hook, rec.get("count")), alert and all(x in msg for x in need) and "reaches the user" not in msg
             and rec.get("count") == 2 and rec.get("session") == "s1" and (rec.get("call") == list(call) if call
                                                                            else rec.get("call") is None),
             ([x for x in need if x not in msg], msg[:1500]))
    r = _incident_cli(b, "--show", rec["id"])
    scenario("(a) %s: --show prints the whole record (its recorded hook input included)" % hook,
             r.returncode == 0 and json.loads(r.stdout).get("input", {}).get("hook_event_name"), r.stdout[:400])
    r = _incident_cli(b, "--replay", rec["id"])
    scenario("(a) %s: --replay before a fix: the recorded refusal is still refused, the same text (exit 2)" % hook,
             r.returncode == 2 and "still refused (the same text)" in r.stdout, (r.returncode, r.stdout[:600]))
    fixed_by()
    r = _incident_cli(b, "--replay", rec["id"])
    scenario("(a) %s: --replay after the fix: the recorded call passes (exit 0)" % hook,
             r.returncode == 0 and "passes now" in r.stdout, (r.returncode, r.stdout[:600]))
    if still_on:
        incident_reaches_user_only_by_needs_user(msg)
    if not still_on:
        r = _incident_cli(b, "--resolve", hook)
        rc, _ = b.run(hook)
        scenario("(a) %s: --resolve once the replay passes, and it runs again" % hook,
                 r.returncode == 0 and rc == 0 and hook not in b.disabled(), (r.stdout, rc))


def temp_site_home():
    """a HOME for the live site tool with an empty ledger: its own runs.db, FA registered with the ledger on, the real
    sites.toml. Nothing live is touched."""
    h = tempfile.mkdtemp(prefix="sitehome_")
    os.makedirs(os.path.join(h, ".config", "ccorch"))
    open(os.path.join(h, ".config", "ccorch", "projects.toml"), "w").write(
        '[projects.FA]\nlocal = "%s"\nledger = true\n' % ROOT)
    shutil.copy(os.path.expanduser("~/.config/ccorch/sites.toml"), os.path.join(h, ".config", "ccorch"))
    env = dict(os.environ, HOME=h)
    return h, (lambda *a: subprocess.run(["site"] + list(a), capture_output=True, text=True, env=env, cwd=h))


def incident_reaches_user_only_by_needs_user(alert_call):
    """the incident's ALERT, opened through the live site tool in a scratch ledger: acked with what was fixed it is
    routine in `site digest` (not shown to the user as a call); only `site ack --needs-user` puts it under Needs you."""
    m = re.search(r"'--msg', '(.*)'\]$", alert_call)
    msg = m.group(1) if m else "HOOK INCIDENT"
    _, site = temp_site_home()
    site("event", "--project", "FA", "--kind", "ALERT", "--key", "hook-incident:fixed", "--msg", msg)
    site("event", "--project", "FA", "--kind", "ALERT", "--key", "hook-incident:needs", "--msg", msg)
    open_ = site("events", "--project", "FA").stdout
    a1 = site("ack", "E1", "--note", "md_guard was right (a new .md); told the agent to write into docs/results.tex")
    a2 = site("ack", "E2", "--note", "the fix is not obvious", "--needs-user", "whether md_guard should allow it")
    dig = site("digest", "--project", "FA").stdout
    need = dig.split("Needs you:", 1)[-1].split("Findings:", 1)[0]
    scenario("(a) the md_guard incident's ALERT in a scratch ledger: while open it waits for the main session; acked "
             "with the fix it is routine (not a call for the user); only the --needs-user ack is listed under Needs you",
             a1.returncode == 0 and a2.returncode == 0 and "HOOK INCIDENT" in open_ and "E1" not in need
             and "E2" in need and "NEEDS YOU: whether md_guard should allow it" in need
             and "ALERT x1" in dig, (a1.stderr, a2.stderr, dig))


def dispatcher():
    """A3: pass-through, loops, and the night's repeated plan_check refusals."""
    b = Box()
    rc, out = b.run()
    scenario("A3: a hook that passes passes through (exit 0, run once)", rc == 0 and len(b.ran()) == 1, (rc, out))
    b.say("PLAN CHECK: x")
    rc, out = b.run()
    scenario("A3: a refusal passes through (exit 2 and its message)", rc == 2 and "PLAN CHECK: x" in out, (rc, out))
    rc, out = b.run(hook="missing_guard.py")
    scenario("A3: a missing hook script: exit 0 with a note", rc == 0 and "no hook missing_guard.py" in out, (rc, out))

    # the night: the first pair of identical plan_check refusals with only text between (the acked-runs demand), and
    # the pair with a SendMessage between
    refs = night_stop_refusals()
    pairs = [(refs[k - 1], refs[k]) for k in range(1, len(refs)) if refs[k][1] == refs[k - 1][1]]
    quiet = [p for p in pairs if not any(
        b_.get("type") == "tool_use" for r in p[1][2] for b_ in ((r.get("message") or {}).get("content") or [])
        if isinstance(b_, dict))]
    busy = [p for p in pairs if p not in quiet]
    scenario("A3: the night has identical consecutive plan_check refusals, with and without a tool call between",
             quiet and busy, (len(pairs), len(quiet), len(busy)))
    if quiet:
        (t1, msg, _), (t2, _, between) = quiet[0]
        b = Box()
        b.say(msg)
        rc1, _ = b.run()
        b.append(between)
        rc2, out2 = b.run()
        inc = b.incidents()
        scenario("A3: the acked-runs demand (%s, %s): plan_check disabled at the second identical refusal, an incident "
                 "and one ALERT opened, plan_check's refusal let through (exit 0)" % (t1[11:19], t2[11:19]),
                 rc1 == 2 and rc2 == 0 and "plan_check.py" in b.disabled() and len(inc) == 1
                 and inc[0]["session"] == "s1" and inc[0]["last_refusal"].startswith(msg[:200])
                 and len(b.site_calls()) == 1, (rc1, rc2, out2, inc))
        rc3, out3 = b.run()
        scenario("A3: the next Stop skips it", rc3 == 0 and "disabled" in out3 and len(b.ran()) == 2, (rc3, b.ran()))
        if inc:
            incident_for_main_session(b, inc[0], None, "plan_check.py", msg.strip()[:60], fixed_by=lambda: b.say(None),
                                      still_on=False)
    if busy:
        (t1, msg, _), (t2, _, between) = busy[0]
        b = Box()
        b.say(msg)
        rc1, _ = b.run()
        b.append(between)
        rc2, _ = b.run()
        scenario("A3: the same refusal twice with a tool call between (%s, %s) is no loop" % (t1[11:19], t2[11:19]),
                 rc1 == 2 and rc2 == 2 and not b.disabled() and not b.incidents(), (rc1, rc2, b.disabled()))

    # the state changed between two identical refusals: no loop
    for what in ("commit", "plan", "event"):
        b = Box()
        b.say("PLAN CHECK: same")
        b.run()
        if what == "commit":
            open(os.path.join(b.repo, "f"), "w").write("x")
            b.git("add", "f"); b.git("commit", "-qm", "f")
        elif what == "plan":
            open(os.path.join(b.repo, "plan", "plan.yaml"), "a").write("# changed\n")
        else:
            c = sqlite3.connect(os.path.join(b.d, "runs.db"))
            c.execute("INSERT INTO events (ts, project, kind, key, msg) VALUES (1, 'FA', 'RUN', 'k', 'm')")
            c.commit()
        rc2, _ = b.run()
        scenario("A3: identical refusals with a %s between are no loop" % {"commit": "new commit", "plan": "plan change",
                                                                         "event": "new open event"}[what],
                 rc2 == 2 and not b.disabled(), (rc2, b.disabled()))

    # PreToolUse: the identical command refused twice in a row (its own tool call in the transcript between) is a loop;
    # another call between is not; another session is not the same loop
    cmd = ("Bash", {"command": "bash .claude/hooks/review_backlog.sh advance repo-reviewer"})
    # a tool-call guard refusing the identical retry is doing its job: it stays on, the refusal stands, and the first
    # repeat opens one incident (the real md_guard.sh, a Write of a new .md file)
    b = Box()
    shutil.copy(os.path.join(HOOKS, "md_guard.sh"), b.hooks)
    write = ("Write", {"file_path": os.path.join(b.repo, "NOTES_NEW.md"), "content": "x"})
    rcs = []
    for _ in range(5):
        b.tool_use(*write)
        rcs.append(b.run("md_guard.sh", "PreToolUse", write)[0])
    inc = b.incidents()
    scenario("A3: a retried md_guard refusal stays refused (five times), is never disabled, and opens one incident "
             "with one ALERT", rcs == [2] * 5 and not b.disabled() and len(inc) == 1 and inc[0].get("kept")
             and inc[0]["hook"] == "md_guard.sh" and len(b.site_calls()) == 1 and "hook-incident:" + inc[0]["id"]
             in b.site_calls()[0], (rcs, b.disabled(), inc, b.site_calls()))
    incident_for_main_session(b, inc[0], write, "md_guard.sh", "BLOCKED by md_guard", fixed_by=lambda: (os.makedirs(
        os.path.join(b.repo, ".claude"), exist_ok=True), open(os.path.join(b.repo, ".claude", "md_allowlist.txt"), "w")
        .write("NOTES_NEW.md\n")), still_on=True)
    b = Box()
    b.say("lock guard: only repo-reviewer advances its pillar")
    b.tool_use(*cmd)
    rc1, _ = b.run("lock_guard.py", "PreToolUse", cmd)
    b.tool_use(*cmd)
    rc2, out2 = b.run("lock_guard.py", "PreToolUse", cmd)
    scenario("A3: the same for lock_guard: the identical retry is refused again, the guard stays on",
             rc1 == 2 and rc2 == 2 and not b.disabled() and len(b.incidents()) == 1, (rc1, rc2, out2))
    b = Box()
    b.say("lock guard: refused")
    b.run("lock_guard.py", "PreToolUse", cmd)
    b.tool_use("Read", {"file_path": "/x"})
    b.tool_use(*cmd)
    rc2, _ = b.run("lock_guard.py", "PreToolUse", cmd)
    scenario("A3: the same PreToolUse refusal with another tool call between is no repeat (no incident)",
             rc2 == 2 and not b.disabled() and not b.incidents(), b.incidents())
    b = Box()
    b.say("lock guard: refused")
    b.run("lock_guard.py", "PreToolUse", cmd)
    rc2, _ = b.run("lock_guard.py", "PreToolUse", ("Bash", {"command": "other"}))
    scenario("A3: a PreToolUse refusal of a different command is no repeat (no incident)",
             rc2 == 2 and not b.disabled() and not b.incidents(), b.incidents())
    b = Box()
    b.say("PLAN CHECK: same")
    b.run(session="s1")
    rc2, _ = b.run(session="s2")
    scenario("A3: the same refusal in another session is no loop", rc2 == 2 and not b.disabled(), b.disabled())
    b = Box()
    b.say("PLAN CHECK: same")
    for _ in range(2):
        r = subprocess.run([sys.executable, os.path.join(b.hooks, "dispatch.py"), "plan_check.py"],
                           input=J({"hook_event_name": "Stop"}), capture_output=True, text=True, env=b.env)
    scenario("A3: without session_id or transcript_path no loop is declared", r.returncode == 2 and not b.disabled(),
             r.returncode)
    b = Box()
    b.say("PLAN CHECK: x")
    env = dict(b.env, DISPATCH_STATE=os.path.join(b.d, "no", "such", "dir", "state.json"))
    r = subprocess.run([sys.executable, os.path.join(b.hooks, "dispatch.py"), "plan_check.py"],
                       input=J({"session_id": "s", "transcript_path": b.tr, "cwd": b.repo}), capture_output=True,
                       text=True, env=env)
    scenario("A3: the loop bookkeeping fails after the hook refused: the refusal stands (exit 2, its message)",
             r.returncode == 2 and "PLAN CHECK: x" in r.stderr and "loop check failed" in r.stderr,
             (r.returncode, r.stderr))
    r = subprocess.run([sys.executable, os.path.join(b.hooks, "dispatch.py"), "plan_check.py"], input="[1]",
                       capture_output=True, text=True, env=b.env)
    scenario("A3: a hook input that is no JSON object: the refusal stands", r.returncode == 2, (r.returncode, r.stderr))
    b = Box()
    raw = "BLOCKED \u2265 \u2014 ok\n".encode() + b"\xff\xfe not utf-8\n"
    open(os.path.join(b.hooks, "bytes_guard.sh"), "w").write(
        "cat >/dev/null; printf '%s' '{\"x\": 1}'; printf '" + "".join("\\%03o" % c for c in raw) + "' >&2; exit 2\n")
    r = subprocess.run([sys.executable, os.path.join(b.hooks, "dispatch.py"), "bytes_guard.sh"], input=b"{}",
                       capture_output=True, env=dict(b.env, PYTHONIOENCODING="ascii", LC_ALL="C"))
    direct = subprocess.run(["bash", os.path.join(b.hooks, "bytes_guard.sh")], input=b"{}", capture_output=True)
    scenario("A3: stdout, stderr and exit code pass through byte for byte (non-UTF-8 bytes, an ASCII locale)",
             (r.returncode, r.stdout, r.stderr) == (direct.returncode, direct.stdout, direct.stderr) == (2, b'{"x": 1}', raw),
             (r.returncode, r.stdout, r.stderr))
    b = Box()
    open(os.path.join(b.hooks, "json_block.sh"), "w").write(
        "cat >/dev/null; echo '{\"decision\":\"block\",\"reason\":\"BATCHED REVIEW DUE\"}'\n")
    rc1, out1 = b.run("json_block.sh", args=("check",))
    rc2, _ = b.run("json_block.sh", args=("check",))
    scenario("A3: a JSON block (review_backlog.sh check's form) passes through, and repeated is a loop keyed with its "
             "arguments", rc1 == 0 and '"decision":"block"' in out1 and rc2 == 0 and "json_block.sh check" in b.disabled(),
             (out1, b.disabled()))


SCEN_PATCH = r'''
import os, runpy, subprocess, sys
HOOKS = os.path.join(%(root)r, ".claude", "hooks")
DISPATCH = os.path.join(HOOKS, "dispatch.py")
_run, _popen = subprocess.run, subprocess.Popen
def _wrap(argv):
    if not isinstance(argv, (list, tuple)):
        return argv
    argv = list(argv)
    for k in (0, 1):
        if k < len(argv) and isinstance(argv[k], str) and os.path.dirname(os.path.abspath(argv[k])) == HOOKS \
                and os.path.basename(argv[k]) != "dispatch.py" and (k == 0 or os.path.basename(argv[0]).startswith(("python", "bash", "sh"))):
            open(os.environ["WRAP_LOG"], "a").write(os.path.basename(argv[k]) + "\n")
            return [sys.executable, DISPATCH, os.path.basename(argv[k])] + argv[k + 1:]
    return argv
def run(args, *a, **k):
    return _run(_wrap(args), *a, **k)
class Popen(_popen):
    def __init__(self, args, *a, **k):
        super().__init__(_wrap(args), *a, **k)
subprocess.run, subprocess.Popen = run, Popen
sys.argv = [%(scen)r]
runpy.run_path(%(scen)r, run_name="__main__")
'''


def scenarios_via_dispatch():
    """A3: the scenarios of tests/test_setup_scenarios.py with every hook it runs from .claude/hooks/ run through the
    dispatcher give the same results as run directly (the suite has failures of its own on trunk: compared, not
    assumed green)."""
    scen = os.path.join(ROOT, "tests", "test_setup_scenarios.py")
    d = tempfile.mkdtemp(prefix="scen_dispatch_")
    env = dict(os.environ, DISPATCH_STATE=os.path.join(d, "state.json"), WRAP_LOG=os.path.join(d, "wrapped.log"))
    res = {}
    for mode in ("direct", "dispatch"):
        cmd = [sys.executable, scen] if mode == "direct" else \
            [sys.executable, "-c", SCEN_PATCH % {"root": ROOT, "scen": scen}]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
        res[mode] = [l for l in r.stdout.splitlines() if l.startswith(("PASS  ", "FAIL  "))]
    diff = [(a, b_) for a, b_ in zip(res["direct"], res["dispatch"]) if a != b_]
    wrapped = open(env["WRAP_LOG"]).read().split() if os.path.exists(env["WRAP_LOG"]) else []
    names = sorted(set(wrapped))
    scenario("A3: the suite's hook runs went through the dispatcher (%d runs of %d hooks: %s)"
             % (len(wrapped), len(names), ", ".join(names)), len(names) >= 5, wrapped[:5])
    n_fail = sum(1 for l in res["direct"] if l.startswith("FAIL"))
    scenario("A3: test_setup_scenarios through the dispatcher: the same %d results as direct (%d of them fail on "
             "trunk already)" % (len(res["direct"]), n_fail),
             res["direct"] and len(res["direct"]) == len(res["dispatch"]) and not diff, diff[:5])


def switch_a():
    """A4: the registration diff of switch-over A: every hook through the dispatcher, nothing added or removed, both
    files; --revert puts them back."""
    d = tempfile.mkdtemp(prefix="switch_a_")
    proj, user = os.path.join(d, "p.json"), os.path.join(d, "u.json")
    shutil.copy(os.path.join(ROOT, ".claude", "settings.json"), proj)
    shutil.copy(os.path.expanduser("~/.claude/settings.json"), user)
    before = {f: json.load(open(f)) for f in (proj, user)}
    sw = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "tools", "switch_settings.py"), "--settings", proj,
                                    "--user-settings", user] + list(a), capture_output=True, text=True)
    r = sw("--switch", "A", "--apply")
    cmds = lambda s: [(ev, g.get("matcher"), h["command"], h.get("timeout")) for ev, gs in s["hooks"].items()
                      for g in gs for h in g["hooks"]]
    for f, prefix in ((proj, 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/dispatch.py" '),
                      (user, 'python3 "$HOME/.claude/hooks/dispatch.py" ')):
        old, new = cmds(before[f]), cmds(json.load(open(f)))
        # the old command ends with "/.claude/hooks/<hook>[ args]" (a closing quote dropped); the new one is the
        # dispatcher followed by exactly that "<hook>[ args]"
        done = all(o[2].startswith(prefix) for o in old)      # settings already switched over: apply is a no-op
        same = len(old) == len(new) and (old == new if done else True) and all(o[0] == n[0] and o[1] == n[1] and o[3] == n[3] and n[2].startswith(prefix)
                                            and (done or o[2].replace('"', "").endswith("/.claude/hooks/" + n[2][len(prefix):]))
                                            for o, n in zip(old, new))
        scenario("A4: %s: all %d registrations run through the dispatcher, same events, matchers and timeouts"
                 % ("project" if f == proj else "user", len(new)), r.returncode == 0 and same,
                 [(o[2], n[2]) for o, n in zip(old, new)][:3])
        rest = lambda s: {k: v for k, v in s.items() if k != "hooks"}
        scenario("A4: %s: nothing but the hooks changed" % ("project" if f == proj else "user"),
                 rest(json.load(open(f))) == rest(before[f]))
    again = sw("--switch", "A")
    scenario("A4: applied twice is a no-op (an already dispatched command is kept)", again.stdout.strip() == "",
             again.stdout[:300])
    sw("--revert", "A")
    scenario("A4: --revert A restores both files", all(json.load(open(f)) == before[f] for f in (proj, user)))
    u = os.path.expanduser("~/.claude/hooks/dispatch.py")
    scenario("A4: the user-level dispatcher is installed and is the same file as the project's",
             os.path.exists(u) and open(u).read() == open(os.path.join(HOOKS, "dispatch.py")).read(), u)


# ================================================================== B: the hook set
SUBAGENTS = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA/%s/subagents" % MAIN)
STATE_HEAD = os.path.join(HF.DEFAULT_OUT, "plan_head", "." + "state.json")


def _load_hook(name):
    import importlib.util
    sys.path.insert(0, HOOKS)
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(HOOKS, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _iso(t):
    import datetime as dt
    return dt.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def agent_spans(state):
    """{entry: [(start, end)]}: each recorded step agent's first to last transcript record, read from the night's
    subagent transcripts (in place)."""
    out = {}
    for eid, v in state.items():
        if eid.startswith("_") or not isinstance(v, dict):
            continue
        for x in v.get("steps") or []:
            f = os.path.join(SUBAGENTS, "agent-%s.jsonl" % x.get("agent"))
            if not os.path.exists(f):
                continue
            ts = [_iso(m.group(1)) for line in open(f, errors="replace")
                  for m in [re.search(r'"timestamp":"([^"]+)"', line)] if m]
            if ts:
                out.setdefault(eid, []).append((min(ts), max(ts)))
    return out


def stop_events_replay():
    """B1: the 227 events replayed event by event through plan_check's event split: at each event's time, every event
    open then (not yet acked) is a refusal unless a step agent of its entry was working at that time (a note)."""
    pc = _load_hook("plan_check.py")
    led = HF.load()
    spans = agent_spans(json.load(open(STATE_HEAD)))
    ev = [e for e in led["events"] if e["project"] == "FA"]
    refusals = notes = moments = 0
    wrong = []
    for e in ev:
        t = e["ts"]
        open_ = [x for x in ev if x["ts"] <= t and (x["acked"] is None or x["acked"] > t)]
        working = {k for k, sp in spans.items() if any(a <= t <= b for a, b in sp)}
        lines = ["E%d  10-29 00:00 FA       %-9s %-24s %s" % (x["id"], x["kind"], x["item"] or "-", x["msg"] or "")
                 for x in open_]
        r, n = pc.split_events(lines, working)
        moments += 1
        refusals += len(r)
        notes += len(n)
        want_r = {x["id"] for x in open_ if (x["item"] or "-") not in working}
        got_r = {int(l.split()[2][1:]) for l in r}
        if want_r != got_r:
            wrong.append((e["id"], sorted(want_r ^ got_r)[:5]))
    scenario("B1: the 227 events replayed: at %d event times, %d open-event refusals (no working agent) and %d notes "
             "(a step agent of the entry working)" % (moments, refusals, notes), moments == 227 and not wrong,
             wrong[:5])
    r, n = pc.split_events(["E1  10-29 20:52 FA       RUN       catalog_coverage         x ended FAILED"],
                           {"catalog_coverage"})
    scenario("B1: an open event of an entry whose agent is still working is a note, not a refusal",
             not r and len(n) == 1, (r, n))


def stop_acked_runs():
    """B1: the acked-runs demand. Every RUN event of the night was acked with a note naming its run: plan.acked_runs
    counts all of them handled, so no step is demanded for those run ends."""
    d = tempfile.mkdtemp(prefix="acked_")
    led = HF.load()
    c = sqlite3.connect(os.path.join(d, "runs.db"))
    for t in ("events", "runs"):
        rows = led[t]
        cols = sorted({k for r in rows for k in r})
        c.execute("CREATE TABLE %s (%s)" % (t, ",".join(cols)))
        c.executemany("INSERT INTO %s (%s) VALUES (%s)" % (t, ",".join(cols), ",".join("?" * len(cols))),
                      [[r.get(k) for k in cols] for r in rows])
    c.commit()
    runs = sorted({re.search(r"\b(FA-[a-z0-9]+-[0-9a-f]{8})\b", e["msg"]).group(1) for e in led["events"]
                   if e["kind"] == "RUN"})
    code = ("import json, sys; sys.path.insert(0, %r); import plan; print(json.dumps(sorted(plan.acked_runs(set(%r)))))"
            % (os.path.join(ROOT, "tools"), runs))
    shutil.copy(os.path.join(HF.DEFAULT_OUT, "plan_before", "plan.yaml"), d)
    env = dict(os.environ, PLAN_TEST="1", PLAN_ROOT=d, PLAN_FILE=os.path.join(d, "plan.yaml"),
               PLAN_RUNS_DB=os.path.join(d, "runs.db"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=d)
    got = json.loads(r.stdout or "[]")
    scenario("B1: the acked-runs demand: of the %d runs whose end events were acked, %d count as handled (a note "
             "naming them), so none is demanded" % (len(runs), len(got)), got == runs and len(runs) > 200,
             (r.stderr[-500:], sorted(set(runs) - set(got))[:5]))


def stop_conditions():
    """B1: the conditions plan_check takes over, each on its own."""
    sc = _load_hook("stop_conditions.py")
    d = tempfile.mkdtemp(prefix="stopc_")
    os.makedirs(os.path.join(d, ".claude"))
    import time as _t
    b, _ = sc.figure_pairs(d)                                    # first run: the baseline
    os.utime(os.path.join(d, ".claude", ".figpair_last"), (_t.time() - 10, _t.time() - 10))
    os.makedirs(os.path.join(d, "analysis"))
    open(os.path.join(d, "analysis", "f.png"), "w").write("x")
    b1, _ = sc.figure_pairs(d)
    open(os.path.join(d, "analysis", "f.pdf"), "w").write("x")
    b2, _ = sc.figure_pairs(d)
    os.makedirs(os.path.join(d, "runs"))
    open(os.path.join(d, "runs", "g.png"), "w").write("x")
    b3, _ = sc.figure_pairs(d)
    scenario("B1: a .png saved without its .pdf is a refusal; with it, or under runs/ (pruned), it passes",
             not b and len(b1) == 1 and "analysis/f.pdf" in b1[0] and not b2 and not b3, (b1, b2, b3))
    # context_guard's Stop branch (Joaquin's approval of "1 2 and 3", 2026-10-31): near compaction, a question to the
    # user with no plan ask, an answer with no site decide or a commitment with no plan step refuses; the model is a
    # stub here (the real call was run on the session's two compaction windows of 2026-10-31)
    tr = os.path.join(d, "main.jsonl")
    rec = lambda t, role, text: {"type": role, "timestamp": "2026-10-31T0%d:00:00Z" % t,
                                 "message": {"role": role, "content": [{"type": "text", "text": text}]}}
    open(tr, "w").write("".join(json.dumps(r) + "\n" for r in (
        {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-10-31T00:00:00Z"},
        rec(1, "user", "Run the mixture arms on both seeds."),
        rec(2, "assistant", "Joaquin, should the held-out pools be generated closer to the families? I'll run the "
                             "ablation after the verdict."))))
    found = lambda v: (lambda prompt: v)
    miss = {"questions": [{"quote": "should the held-out pools be generated closer to the families?", "why": "w"}],
            "answers": [{"quote": "Run the mixture arms on both seeds.", "why": "w"}],
            "commitments": [{"quote": "I'll run the ablation after the verdict.", "why": "w"},
                            {"quote": "a sentence that is nowhere in the window", "why": "w"}]}
    inp = {"transcript_path": tr}
    c0, _ = sc.context(inp, d, ask=found(miss), steps="")
    open(os.path.join(d, ".claude", ".compact_wanted"), "w").close()
    c1, _ = sc.context(inp, d, ask=found(miss), steps="")
    c2, _ = sc.context(inp, d, ask=found(miss), steps="")
    c3, _ = sc.context({"stop_hook_active": True, "transcript_path": tr}, d, ask=found(miss), steps="")
    open(tr, "a").write(json.dumps(rec(3, "assistant", "Recorded.")) + "\n")
    c4, _ = sc.context(inp, d, ask=found({"questions": [], "answers": [], "commitments": []}), steps="")
    open(tr, "a").write(json.dumps(rec(4, "assistant", "More.")) + "\n")
    c5, _ = sc.context(inp, d, ask=lambda prompt: None, steps="")
    scenario("B1: context_guard's Stop branch: no compaction wanted -> nothing; wanted -> one refusal naming the "
             "question, the answer and the commitment (a quote not in the window dropped); the same transcript state "
             "-> not again; stop_hook_active -> nothing; the model finds nothing, or fails -> passes",
             not c0 and len(c1) == 1 and all(x in c1[0] for x in ("no `plan ask`", "no `site decide`",
                                                                  "no plan step")) and "nowhere" not in c1[0]
             and not c2 and not c3 and not c4 and not c5, (c0, c1, c2, c3, c4, c5))
    tr2 = tr + ".later"
    shutil.copy(tr, tr2)
    open(tr2, "a").write(json.dumps(rec(5, "assistant", "Again.")) + "\n")
    c6, n6 = sc.context({"transcript_path": tr2}, d, ask=found(miss), steps="")
    open(tr2, "a").write(json.dumps(rec(6, "assistant", "Joaquin, may I widen the pools?")) + "\n")
    more = dict(miss, questions=miss["questions"] + [{"quote": "may I widen the pools?", "why": "w"}])
    c7, _ = sc.context({"transcript_path": tr2}, d, ask=found(more), steps="")
    scenario("B1: context_guard's Stop branch refuses once per omission: the same omissions at a new transcript "
             "state pass with a note; a new omission refuses again",
             not c6 and n6 and "no `plan ask`" in n6[0] and len(c7) == 1 and "widen the pools" in c7[0], (c6, n6, c7))
    sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
    import context_guard as cg
    u, says, recs = cg.window(tr)
    p_ = cg.check_prompt(u, says, recs, steps="STEPS")
    scenario("B1: the check's window starts at the last compaction and holds the user's message, the assistant's "
             "text and the plan's steps", len(u) == 1 and len(says) == 3 and "STEPS" in p_ and
             "Run the mixture arms" in p_, (u, says))
    import io, reply_check
    saved = (cg.WANTED, cg.CHECKED, cg.FLUSHED, reply_check.ask_model, cg.plan_text, sys.stdin, sys.stderr)
    cg.WANTED, cg.CHECKED, cg.FLUSHED = (os.path.join(d, ".claude", x) for x in ("w2", "c2", "f2"))
    cg.plan_text = lambda: ""
    pre = []
    for v in (miss, {"questions": [], "answers": [], "commitments": []}, None):
        reply_check.ask_model = lambda prompt, v=v: v
        sys.stdin, sys.stderr = io.StringIO(json.dumps({"hook_event_name": "PreCompact", "trigger": "auto",
                                                        "transcript_path": tr})), io.StringIO()
        pre.append((cg.main(), sys.stderr.getvalue()))
    sys.stdin = io.StringIO(json.dumps({"hook_event_name": "PreCompact", "trigger": "manual", "transcript_path": tr}))
    reply_check.ask_model = lambda prompt: miss
    pre.append((cg.main(), ""))
    (cg.WANTED, cg.CHECKED, cg.FLUSHED, reply_check.ask_model, cg.plan_text, sys.stdin, sys.stderr) = saved
    scenario("B1: PreCompact (auto) is held with the list while something is unrecorded; it goes through when "
             "nothing is, when the model fails, and on a manual /compact",
             pre[0][0] == 2 and "no `plan ask`" in pre[0][1] and [x[0] for x in pre[1:]] == [0, 0, 0], pre)
    for code, err, want, what in (
            (0, "", 0, "passes"),
            (2, "site ledger: this turn cannot end yet.\nFA has active work items and the supervisor is not running "
                "(last pass never): start it", 1, "supervisor down"),
            (2, "site ledger: this turn cannot end yet.\nno waiter is armed: run `site events --wait --project FA` in "
                "the background", 1, "waiter not armed"),
            (1, "Traceback", 1, "guard broken")):
        stub = os.path.join(d, "site_%d_%d" % (code, len(err)))
        open(stub, "w").write("#!/bin/sh\ncat >&2 <<'EOF'\n%s\nEOF\nexit %d\n" % (err, code))
        os.chmod(stub, 0o755)
        r, n = sc.ledger(d, [stub])
        scenario("B1: ledger (`site guard --stop`) %s: %s" % (what, "a refusal" if want else "no refusal"),
                 len(r) == want and (not want or err.splitlines()[-1][:20] in r[0]), r)
    lg = os.path.join(d, "ledger_guard.sh")
    open(lg, "w").write("#!/bin/sh\nexit 0\n")
    no = sc.ledger_deferred(lg)
    open(lg, "a").write("# grep -qF 'LEDGER_MARK: ledger: site guard --stop' plan_check.py\n")
    scenario("B1: plan_check asks the ledger only once ledger_guard.sh defers to it (never both)",
             not no and sc.ledger_deferred(lg), no)
    hd = os.path.join(d, "hooks")
    os.makedirs(hd)
    r, n = sc.incidents(hd)
    json.dump({"plan_check.py": {"why": "loop", "time": "2026-10-30T05:00"}}, open(os.path.join(hd, "disabled.json"), "w"))
    r2, n2 = sc.incidents(hd)
    scenario("B1: an open incident is a note, never a refusal", not r and not n and not r2 and len(n2) == 1, (r2, n2))
    tr = os.path.join(d, "t.jsonl")
    with open(tr, "w") as f:
        f.write(J({"type": "user", "message": {"content": "why is the queue empty?"}}) + "\n")
        f.write(J({"type": "assistant", "message": {"content": [{"type": "text", "text": "Joaquin, it is not."}]}}) + "\n")
    code = ("import sys, json; sys.path.insert(0, %r); import stop_conditions as sc; print(json.dumps(sc.reply(%r)))"
            % (HOOKS, {"transcript_path": tr}))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PATH="/usr/bin:/bin", CLAUDE_PROJECT_DIR=d))
    scenario("B1: the reply check with no `claude` CLI on PATH passes",
             r.returncode == 0 and json.loads(r.stdout or "[[1]]")[0] == [], r.stdout + r.stderr)
    proj = os.path.join(d, ".claude", "settings.json")
    shutil.copy(os.path.join(ROOT, ".claude", "settings.json"), proj)
    names = ("figure_pair_guard.sh", "reply_check.py", "context_guard.py")
    before = [sc.registered_at_stop(h, d) for h in names]
    open(os.path.join(d, "u.json"), "w").write('{"hooks": {}}\n')
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "switch_settings.py"), "--settings", proj,
                    "--user-settings", os.path.join(d, "u.json"), "--switch", "B", "--apply"], capture_output=True)
    after = [sc.registered_at_stop(h, d) for h in names]
    scenario("B1: each taken-over condition runs only once its standalone hook is no longer registered for Stop "
             "(registered now: %s; after switch-over B: %s)" % (before, after),
             before in ([True] * 3, [False] * 3) and after == [False, False, False], (before, after))   # live settings may already be switched


def stop_opener():
    """B1: the opener on the night's main-session transcript: each turn whose closing text lacks "Joaquin" is refused
    once at its first stop, never twice for one user message; a turn that has it is never refused."""
    sc = _load_hook("stop_conditions.py")
    recs, state = [], {}
    refused, per_prompt, lacking, stops, wrong = 0, {}, 0, 0, 0
    for _, r in HF.transcript_records(name=MAIN):
        if r.get("subtype") == "stop_hook_summary":
            stops += 1
            prompt, last = sc.turn_of(recs[-400:])
            b, _ = sc.opener({}, state, records=recs[-400:])
            lacks = bool(last) and not sc.starts_with_opener(last)
            lacking += lacks
            if b:
                refused += 1
                per_prompt[prompt] = per_prompt.get(prompt, 0) + 1
                wrong += not lacks
            continue
        recs.append(r)
    twice = [k for k, v in per_prompt.items() if v > 1]
    scenario("B1: the night's main-session stops (%d): %d ended with text lacking the opener, %d refused, none twice "
             "for one user message, none that had it" % (stops, lacking, refused),
             stops > 50 and refused >= 1 and not twice and not wrong and refused <= lacking, (twice, wrong))
    b, _ = sc.opener({"agent_id": "a1"}, {}, records=[{"type": "assistant", "message": {"content": [
        {"type": "text", "text": "done"}]}}])
    scenario("B1: the opener is never asked of a subagent", not b, b)
    gate = open(os.path.join(ROOT, ".claude", "hooks", "plan_check.py")).read()
    scenario("B1: the opener is not registered in the Stop gate (2026-10-31: it is Joaquin's drift signal)",
             "sc.opener(" not in gate, "sc.opener( found in plan_check.py")


NIGHT_SUBMIT = re.compile(r"site\s+submit|sweep_manager\.py\s+submit|launch_\w+\.sh|\bsbatch\b|condor_submit|"
                          r"site\s+launch\s+add|site\s+run\b")


def _night_commands():
    out = []
    for _, r in HF.transcript_records():
        if r.get("type") != "assistant":
            continue
        for b in (r.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                c = (b.get("input") or {}).get("command", "")
                if NIGHT_SUBMIT.search(c):
                    out.append(c)
    return list(dict.fromkeys(out))


def _guard_env(d):
    home = os.path.join(d, "home")
    pl = os.path.join(home, ".local", "share", "ccorch", "placement")
    os.makedirs(pl, exist_ok=True)
    import time as _t
    json.dump({"ts": int(_t.time()), "quota": {"ccin2p3": 1000, "lxplus": 1000}},
              open(os.path.join(pl, "FA.json"), "w"))
    return dict(os.environ, HOME=home, SUBMIT_TEST="1", CLAUDE_PROJECT_DIR=ROOT,
                SUBMIT_RUNS_DB=os.path.join(d, "none.db"), PATH=os.environ["PATH"])


def _rc(argv, inp, env):
    return subprocess.run(argv, input=J(inp), capture_output=True, text=True, env=env, cwd=ROOT).returncode


def submit_replay():
    """B2: every submission-like command of the night's transcripts through the four old guards and through
    submit_guard: the same verdict per check and overall (the budget is new and reported apart)."""
    cmds = _night_commands()
    old = {"placement": ["bash", os.path.join(HOOKS, "placement_guard.sh")],
           "design": [sys.executable, os.path.join(HOOKS, "design_guard.py")],
           "hpo": ["bash", os.path.join(HOOKS, "hpo_guard.sh")],
           "plot": ["bash", os.path.join(HOOKS, "plot_guard.sh")]}
    diff, refused, budget_refused = [], 0, []
    for c in cmds:
        inp = {"tool_name": "Bash", "tool_input": {"command": c}, "cwd": ROOT}
        verdicts = {}
        for name, argv in old.items():
            d = tempfile.mkdtemp(prefix="sub_")
            o = _rc(argv, inp, _guard_env(d))
            d = tempfile.mkdtemp(prefix="sub_")
            n = _rc([sys.executable, os.path.join(HOOKS, "submit_guard.py")], inp,
                    dict(_guard_env(d), SUBMIT_ONLY=name))
            verdicts[name] = (o, n)
            if (o == 2) != (n == 2):
                diff.append((name, o, n, c[:160]))
        refused += any(o == 2 for o, _ in verdicts.values())
        d = tempfile.mkdtemp(prefix="sub_")
        if _rc([sys.executable, os.path.join(HOOKS, "submit_guard.py")], inp,
               dict(_guard_env(d), SUBMIT_ONLY="budget")) == 2:
            budget_refused.append(c[:120])
    scenario("B2: the night's %d submission commands: submit_guard's verdict equals the old guard's for each of "
             "placement, design, hpo and plot (%d refused by the old guards)" % (len(cmds), refused),
             len(cmds) >= 20 and not diff, diff[:5])
    scenario("B2: the budget on the night's commands: %d refused (they are CPU workers or carry --est-gpu-hours)"
             % len(budget_refused), not budget_refused, budget_refused[:3])


def submit_budget():
    """B2: the GPU budget cases."""
    d = tempfile.mkdtemp(prefix="budget_")
    gpu, cpu = os.path.join(d, "gpu.sh"), os.path.join(d, "cpu.sh")
    open(gpu, "w").write("#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --time=10:00:00\npython run.py\n")
    open(cpu, "w").write("#!/bin/bash\n#SBATCH --cpus-per-task=16\npython tools/lhe_parts.py work\n")
    db = os.path.join(d, "runs.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE runs (id TEXT, project TEXT, site TEXT, item TEXT, est_gpu_hours REAL, submitted INTEGER)")
    c.execute("CREATE TABLE items (project TEXT, id TEXT, est_gpu_hours REAL)")
    import time as _t
    c.executemany("INSERT INTO items VALUES ('FA', ?, ?)", [("small", 20), ("approved", 35), ("split", None),
                                                           ("jz", None), ("cpu", None)])
    c.execute("INSERT INTO runs VALUES ('FA-ccin2p3-00000001', 'FA', 'ccin2p3', 'split', 20, ?)", (int(_t.time()) - 1800,))
    c.commit()
    env = dict(_guard_env(d), SUBMIT_ONLY="budget", SUBMIT_RUNS_DB=db)
    sub = lambda cmd: _rc([sys.executable, os.path.join(HOOKS, "submit_guard.py")],
                          {"tool_name": "Bash", "tool_input": {"command": cmd}}, env)
    for what, cmd, want in (
            ("35 GPU-h for an item recorded at 20", "site submit auto FA %s --item small --est-gpu-hours 35" % gpu, 2),
            ("12 GPU-h on Jean Zay without approval", "site submit jeanzay FA %s --item jz --est-gpu-hours 12" % gpu, 2),
            ("12 GPU-h through auto --allow-jeanzay", "site submit auto FA %s --item jz --allow-jeanzay "
                                                     "--est-gpu-hours 12" % gpu, 2),
            ("the second of two 20 GPU-h submissions 30 min apart", "site submit auto FA %s --item split "
                                                                    "--est-gpu-hours 20" % gpu, 2),
            ("a GPU job without --est-gpu-hours", "site submit auto FA %s --item small" % gpu, 2),
            ("CPU generation workers without an estimate", "site submit ccin2p3 FA %s --item cpu --env Q=x" % cpu, 0),
            ("35 GPU-h for an item recorded at 35", "site submit auto FA %s --item approved --est-gpu-hours 35" % gpu, 0),
            ("10 GPU-h for an unrecorded item", "site submit auto FA %s --item jz --est-gpu-hours 10" % gpu, 0),
            ("a --check dry run", "site submit auto FA %s --item small --check" % gpu, 0),
            ("two 20 GPU-h submissions in one command", "site submit auto FA %s --item jz --est-gpu-hours 20; "
                                                        "site submit auto FA %s --item jz --est-gpu-hours 20" % (gpu, gpu), 2),
            ("a 4 GPU-h submission in a loop of 10 (seq 10)", "for i in $(seq 10); do site submit auto FA %s --item jz "
                                                             "--est-gpu-hours 4; done" % gpu, 2),
            ("a 4 GPU-h submission in a loop of 5 ({1..5})", "for i in {1..5}; do site submit auto FA %s --item jz "
                                                            "--est-gpu-hours 4; done" % gpu, 0),
            ("a GPU submission in a loop whose count is not readable", "for f in $(ls cfg); do site submit auto FA %s "
                                                                      "--item jz --est-gpu-hours 1; done" % gpu, 2),
            ("CPU workers in a loop of 30", "for i in $(seq 30); do site submit ccin2p3 FA %s --item cpu; done" % cpu, 0),
            ("one GPU submit whose --note says 'for the parallel test'",
             "site submit auto FA %s --item jz --est-gpu-hours 1 --note \"probe for the parallel test\"" % gpu, 0),
            ("6 GPU-h in nested loops of 3 x 2 (36 GPU-h)", "for i in 1 2 3; do for j in $(seq 2); do site submit auto "
                                                           "FA %s --item jz --est-gpu-hours 6; done; done" % gpu, 2),
            ("an unrelated seq 100 before a loop of one", "seq 100 >/dev/null; for i in 1; do site submit auto FA %s "
                                                          "--item jz --est-gpu-hours 12; done" % gpu, 0),
            ("seq 3 | xargs of 6 GPU-h (18)", "seq 3 | xargs -I{} site submit auto FA %s --item jz --est-gpu-hours 6"
             % gpu, 0),
            ("a C-style for loop of 6 (count unreadable)", "for ((i=0;i<6;i++)); do site submit auto FA %s --item jz "
                                                          "--est-gpu-hours 6; done" % gpu, 2),
            ("a C-style loop around a loop of 2", "for ((i=0;i<6;i++)); do for j in 1 2; do site submit auto FA %s "
                                                  "--item jz --est-gpu-hours 1; done; done" % gpu, 2),
            ("a loop of 6 inside bash -c", "bash -c \"for i in 1 2 3 4 5 6; do site submit auto FA %s --item jz "
                                           "--est-gpu-hours 6; done\"" % gpu, 2),
            ("one submit inside bash -c", "bash -c \"site submit auto FA %s --item jz --est-gpu-hours 6\"" % gpu, 0),
            ("a GPU submit inside xargs sh -c (count unreadable)", "seq 40 | xargs -I{} sh -c 'site submit auto FA %s "
                                                                  "--item jz --est-gpu-hours 1'" % gpu, 2)):
        rc = sub(cmd)
        scenario("B2: budget: %s is %s" % (what, "refused" if want else "allowed"), rc == want, (rc, cmd))
    code = ("import sys; sys.path.insert(0, %r); import submit_guard as g\n"
            "def boom(*a): raise RuntimeError('x')\n"
            "g.hpo = boom\n"
            "print(len(g.problems({'tool_input': {'command': sys.argv[1]}})))" % HOOKS)
    crash = lambda cmd: subprocess.run([sys.executable, "-c", code, cmd], capture_output=True, text=True,
                                       env=dict(env, SUBMIT_ONLY="")).stdout.strip()
    scenario("B2: a check that crashes refuses a command that submits and passes one that does not (the others still run)",
             crash("site submit auto FA j.sh --item x") != "0" and crash("ls -la") == "0",
             (crash("site submit auto FA j.sh --item x"), crash("ls -la")))


SCEN_MAP_PATCH = r'''
import json, os, runpy, subprocess, sys
HOOKS = os.path.join(%(root)r, ".claude", "hooks")
MAP = json.loads(os.environ["SCEN_MAP"])
_run, _popen = subprocess.run, subprocess.Popen
def _wrap(argv, kw):
    if not isinstance(argv, (list, tuple)):
        return argv
    argv = list(argv)
    for k in (0, 1):
        if k < len(argv) and isinstance(argv[k], str) and os.path.dirname(os.path.abspath(argv[k])) == HOOKS \
                and os.path.basename(argv[k]) in MAP:
            new, only = MAP[os.path.basename(argv[k])]
            if only:
                env = dict(kw.get("env") or os.environ, SUBMIT_TEST="1", SUBMIT_ONLY=only)
                kw["env"] = env
            return [sys.executable, os.path.join(HOOKS, new)] + argv[k + 1:]
    return argv
def run(args, *a, **k):
    return _run(_wrap(args, k), *a, **k)
class Popen(_popen):
    def __init__(self, args, *a, **k):
        super().__init__(_wrap(args, k), *a, **k)
subprocess.run, subprocess.Popen = run, Popen
sys.argv = [%(scen)r]
import importlib.util
spec = importlib.util.spec_from_file_location("scen", %(scen)r)
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
names = os.environ.get("SCEN_GROUPS", "").split(",")
t.GROUPS[:] = [g for g in t.GROUPS if g.__name__ in names]
sys.exit(t.main())
'''


def scenarios_mapped(mapping, groups):
    """the named groups of test_setup_scenarios.py run with the old hooks and with each mapped to its replacement:
    [(direct line, mapped line)] that differ, and the number of results."""
    scen = os.path.join(ROOT, "tests", "test_setup_scenarios.py")
    res = {}
    for mode in ("direct", "mapped"):
        env = dict(os.environ, SCEN_MAP=json.dumps(mapping if mode == "mapped" else {}), SCEN_GROUPS=",".join(groups))
        r = subprocess.run([sys.executable, "-c", SCEN_MAP_PATCH % {"root": ROOT, "scen": scen}], capture_output=True,
                           text=True, env=env, cwd=ROOT, timeout=1800)
        res[mode] = [l for l in r.stdout.splitlines() if l.startswith(("PASS  ", "FAIL  "))]
    return [(a, b_) for a, b_ in zip(res["direct"], res["mapped"]) if a != b_], len(res["direct"]), \
        len(res["direct"]) == len(res["mapped"])


def submit_scenarios():
    """B2: the placement and design scenarios of test_setup_scenarios.py (command_reading, designs, design_prior) give
    the same results with placement_guard.sh and design_guard.py replaced by submit_guard.py (that check only)."""
    diff, n, same_n = scenarios_mapped({"placement_guard.sh": ["submit_guard.py", "placement"],
                                        "design_guard.py": ["submit_guard.py", "design"]},
                                       ["command_reading", "designs", "design_prior"])
    scenario("B2: %d scenarios of placement and design: the same results through submit_guard" % n,
             n > 20 and same_n and not diff, diff[:5])


def review_merged():
    """B3: lock_guard's and reviewer_resume's scenarios against review_guard.py; one reviewer per backlog."""
    diff, n, same_n = scenarios_mapped({"lock_guard.py": ["review_guard.py", ""],
                                        "reviewer_resume.py": ["review_guard.py", ""]},
                                       ["command_reading", "reviewer_resume"])
    scenario("B3: %d scenarios of lock_guard and reviewer_resume: the same results against review_guard.py" % n,
             n > 10 and same_n and not diff, diff[:5])
    d = tempfile.mkdtemp(prefix="rg_")
    st = os.path.join(d, ".claude", ".review_state")
    os.makedirs(st)
    env = dict(os.environ, CLAUDE_PROJECT_DIR=d)
    rg = lambda ev: subprocess.run([sys.executable, os.path.join(HOOKS, "review_guard.py")], input=J(ev),
                                   capture_output=True, text=True, env=env)
    launch = {"tool_name": "Agent", "tool_input": {"subagent_type": "claudemd-keeper", "prompt": "review"}}
    rg(dict(launch, hook_event_name="PostToolUse", tool_response={"content": "agentId: keeper1 (internal)"}))
    open(os.path.join(st, "claudemd-keeper.lock"), "w").close()
    r = rg(dict(launch, hook_event_name="PreToolUse"))
    scenario("B3: a second claudemd-keeper launch while the first holds the lock is refused, naming it",
             r.returncode == 2 and "keeper1" in r.stderr, r.stderr)
    import time as _t
    old = _t.time() - 181 * 60
    os.utime(os.path.join(st, "claudemd-keeper.lock"), (old, old))
    r = rg(dict(launch, hook_event_name="PreToolUse"))
    scenario("B3: with the lock stale (181 min) a fresh launch passes", r.returncode == 0, r.stderr)
    r = rg({"tool_name": "Bash", "tool_input": {"command": "bash .claude/hooks/review_backlog.sh advance claudemd-keeper"}})
    r2 = rg({"tool_name": "Bash", "agent_type": "claudemd-keeper",
             "tool_input": {"command": "bash .claude/hooks/review_backlog.sh advance claudemd-keeper"}})
    scenario("B3: advance from the main session is refused, from the pillar's own reviewer it passes and is logged",
             r.returncode == 2 and r2.returncode == 0 and "caller=claudemd-keeper" in open(os.path.join(
                 st, "lock_calls.log")).read(), (r.stderr, r2.stderr))


def _repo_with_backlog(prefix):
    repo = tempfile.mkdtemp(prefix=prefix)
    g = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True,
                                  env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))
    g("init", "-q", "-b", "trunk")
    os.makedirs(os.path.join(repo, ".claude", "hooks"))
    os.makedirs(os.path.join(repo, "tools"))
    for f in ("review_backlog.sh",):
        shutil.copy(os.path.join(HOOKS, f), os.path.join(repo, ".claude", "hooks"))
    for f in ("review_state.py", "attribution.py", "review_weight.py"):
        shutil.copy(os.path.join(ROOT, "tools", f), os.path.join(repo, "tools"))
    open(os.path.join(repo, ".gitignore"), "w").write(".claude/.review_state/\n")
    open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
    g("add", "."); g("commit", "-qm", "base")
    rb = lambda *a, stdin="": subprocess.run(["bash", ".claude/hooks/review_backlog.sh"] + list(a), cwd=repo,
                                             input=stdin, capture_output=True, text=True)
    rb("init")
    return repo, g, rb


def review_modes():
    """B4: the Stop nudge and bashpost are gone: still registered until switch-over B, they are no-ops (exit 0, no
    output); gate and bashpre still refuse an overdue pillar."""
    repo, g, rb = _repo_with_backlog("rb4_")
    open(os.path.join(repo, "a.py"), "a").write("".join("y%d = %d\n" % (i, i) for i in range(600)))
    commit = J({"tool_name": "Bash", "tool_input": {"command": "git commit -qam x"}})
    edit = J({"tool_name": "Edit", "tool_input": {"file_path": os.path.join(repo, "a.py")}})
    c, bp = rb("check", stdin="{}"), rb("bashpost", stdin=commit)
    pre, gate = rb("bashpre", stdin=commit), rb("gate", stdin=edit)
    scenario("B4: an overdue pillar: check and bashpost say nothing (exit 0), bashpre refuses its commit, gate its edit",
             (c.returncode, c.stdout, bp.returncode, bp.stdout) == (0, "", 0, "") and pre.returncode == 2
             and gate.returncode == 2, (c.stdout[:200], bp.stdout[:200], pre.returncode, gate.returncode))


def review_paths():
    """B5: the code backlog leaves out plan bookkeeping and generated data and recipe lists, keeps the generation code,
    and counts .claude/settings.json. On the night's generation commits (5bb09c87^..6220a189)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("review_state", os.path.join(ROOT, "tools", "review_state.py"))
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    new_paths = rs.pillars(ROOT)["repo-reviewer"]
    d = tempfile.mkdtemp(prefix="rb5_")
    os.makedirs(os.path.join(d, ".claude", "hooks"))
    open(os.path.join(d, ".claude", "hooks", "review_backlog.sh"), "w").write(
        subprocess.run(["git", "-C", ROOT, "show", "61c4d6a0:.claude/hooks/review_backlog.sh"], capture_output=True,
                       text=True).stdout)
    old_paths = rs.pillars(d)["repo-reviewer"]
    names = lambda paths: set(subprocess.run(["git", "-C", ROOT, "-c", "core.quotePath=false", "diff", "--name-only",
                                              "5bb09c87^", "6220a189", "--"] + paths, capture_output=True,
                                             text=True).stdout.split())
    o, n = names(old_paths), names(new_paths)
    gone = sorted(o - n)
    keep = {"tools/lhe_probe.py", "tools/lhe_parts.py", "recipes/gen_catalog_v4_default.py"}
    scenario("B5: on the night's generation range the code backlog drops %d generated files (%s...) and keeps the "
             "generation code" % (len(gone), ", ".join(gone[:3])),
             "recipes/catalog_v4_pp_processes.yaml" in gone and keep <= n and not any(
                 x.endswith(".py") for x in gone), (gone[:8], keep - n))
    s = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", ".claude/settings.json"],
                       capture_output=True, text=True).stdout.strip()
    scenario("B5: a .claude/settings.json edit now counts toward the code backlog",
             ".claude/settings.json" in names_of(s, new_paths) and ".claude/settings.json" not in names_of(s, old_paths),
             (s, new_paths))


def names_of(sha, paths):
    return set(subprocess.run(["git", "-C", ROOT, "diff", "--name-only", sha + "^", sha, "--"] + paths,
                              capture_output=True, text=True).stdout.split())


def review_before_merge():
    """B6: a merge into trunk of unreviewed code is refused (fast-forward or not), and passes after the reviewer's
    advance; `plan done` refuses an entry whose write-up commit is outside a passed backlog."""
    repo, g, rb = _repo_with_backlog("rb6_")
    g("checkout", "-qb", "feat")
    open(os.path.join(repo, "a.py"), "a").write("z = 2\n")
    g("commit", "-qam", "feature")
    g("checkout", "-q", "trunk")
    guard = lambda cmd: subprocess.run([sys.executable, os.path.join(HOOKS, "git_verify_guard.py")],
                                       input=J({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": repo}),
                                       capture_output=True, text=True)
    r1, r1ff = guard("git merge --no-ff feat"), guard("git merge feat")
    g("checkout", "-q", "feat")
    rb("advance", "repo-reviewer")
    g("checkout", "-q", "trunk")
    r2 = guard("git merge --no-ff feat")
    scenario("B6: merging unreviewed code into trunk is refused (--no-ff and fast-forward), after the reviewer's "
             "advance it passes", r1.returncode == 2 and r1ff.returncode == 2 and "no reviewer has passed" in r1.stderr
             and r2.returncode == 0, (r1.stderr[-300:], r1ff.returncode, r2.stderr[-300:]))
    import importlib.util
    spec = importlib.util.spec_from_file_location("review_state", os.path.join(repo, "tools", "review_state.py"))
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    open(os.path.join(repo, "a.py"), "a").write("w = 3\n")
    g("commit", "-qam", "unreviewed write-up")
    head = g("rev-parse", "HEAD").stdout.strip()
    scenario("B6: a commit past the watermark is unreviewed (what plan done refuses), the reviewed one is not",
             rs.unreviewed(head, repo) and not rs.unreviewed(g("rev-parse", "feat").stdout.strip(), repo),
             rs.unreviewed(head, repo))


def attribution():
    """B7: no attribution in a commit."""
    repo, g, rb = _repo_with_backlog("rb7_")
    os.makedirs(os.path.join(repo, ".githooks"))
    shutil.copy(os.path.join(ROOT, ".githooks", "commit-msg"), os.path.join(repo, ".githooks"))
    g("config", "core.hooksPath", ".githooks")
    lines = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
             "Claude-Session: https://claude.ai/code/session_x")
    for ln in lines:
        msg = os.path.join(repo, "msg.txt")
        open(msg, "w").write("a change\n\n%s\n" % ln)
        open(os.path.join(repo, "a.py"), "a").write("#\n")
        r = g("commit", "-qa", "-F", msg)
        scenario("B7: `git commit -F msg.txt` with `%s` is refused by commit-msg" % ln.split(":")[0],
                 r.returncode != 0 and "attributes the work" in r.stderr, r.stderr)
        r = subprocess.run([sys.executable, os.path.join(HOOKS, "git_verify_guard.py")], input=J({
            "tool_name": "Bash", "tool_input": {"command": "git commit -m 'a change\n\n%s'" % ln}, "cwd": repo}),
            capture_output=True, text=True)
        scenario("B7: `git commit -m` with `%s` is refused before it runs" % ln.split(":")[0],
                 r.returncode == 2 and "rule 4" in r.stderr, r.stderr)
    open(msg, "w").write("a change\n\nCo-authored-by: Daniel Maitre <daniel@durham.ac.uk>\n")
    r = g("commit", "-qa", "-F", msg)
    scenario("B7: a real collaborator's Co-authored-by passes", r.returncode == 0, r.stderr)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "attribution.py"), "scan", "--since", "2026-10-01"],
                       capture_output=True, text=True, cwd=ROOT)
    scenario("B7: the history since 1 Oct scanned with the same pattern: %s" % r.stdout.strip().splitlines()[-1],
             r.returncode in (0, 1), r.stdout[-300:])


def guards_selftest():
    """B8: .claude/hooks/test_guards.py (the project's `site selftest`): a memory file is refused, every agent is on
    opus."""
    r = subprocess.run([sys.executable, os.path.join(HOOKS, "test_guards.py")], capture_output=True, text=True,
                       env=dict(os.environ, CLAUDE_PROJECT_DIR=ROOT))
    scenario("B8: test_guards.py passes, its memory-file and agent-model cases included",
             r.returncode == 0 and "MEMORY.md" in r.stdout and "model opus" in r.stdout, r.stdout[-600:])


def _site_modules():
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader("site_tool", os.path.expanduser("~/.local/bin/site"))
    spec = importlib.util.spec_from_loader("site_tool", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m, m._rx()


def run_end_markers():
    """The supervisor's misfires on the six gridpack probe jobs (their logs: tests/fixtures/hooks_20261029/
    gridpack_probe_logs.json, read with `site logs`): the end marker `RUN_SUMMARY \\S+ rc=0` missed the two-word
    `RUN_SUMMARY lhe_probe gridpack rc=0` (clean jobs sorted broke), and the RUN_SUMMARY reader took only the JSON form
    (false "no RUN_SUMMARY" alerts). Both against FA's live rules and the live site tool."""
    m, rx = _site_modules()
    logs = json.load(open(os.path.join(HF.DEFAULT_OUT, "gridpack_probe_logs.json")))
    import tomllib
    pr = tomllib.load(open(os.path.expanduser("~/.config/ccorch/projects.toml"), "rb"))["projects"]["FA"]
    rules = rx.rules_of(pr)
    old = dict(rules, end_markers=[r"RUN_SUMMARY \S+ rc=0\b" if p.startswith("RUN_SUMMARY") else p
                                   for p in rules["end_markers"]])
    now = {r: rx.sort_run_end(v["state"], v["job"], {"head": "", "tail": v["log_tail"]}, rules)[0]
           for r, v in logs.items()}
    then = {r: rx.sort_run_end(v["state"], v["job"], {"head": "", "tail": v["log_tail"]}, old)[0]
            for r, v in logs.items()}
    scenario("run ends: the six gridpack probe jobs (COMPLETED, `RUN_SUMMARY lhe_probe gridpack rc=0`) are sorted clean "
             "by FA's end markers; the old one-word marker sorted them broke",
             len(now) == 6 and set(now.values()) == {"clean"} and set(then.values()) == {"broke"}, (now, then))
    got = {}
    for r, v in logs.items():
        m.run_ssh = lambda sc, inner, timeout=90, _t=v["log_tail"]: (0, _t)
        got[r] = m._summary_line({"scheduler": "slurm"}, "FA", "/x", r.split("-")[-1])
    m.run_ssh = lambda sc, inner, timeout=90: (0, 'x\nRUN_SUMMARY {"rc": 0, "steps": 8000}\n')
    js = m._summary_line({"scheduler": "slurm"}, "FA", "/x", "1")
    scenario("run ends: the RUN_SUMMARY reader takes the k=v form of the six (rc=0 read, so no false 'no RUN_SUMMARY' "
             "alert) and still the JSON form", all(x and str(x.get("rc")) == "0" for x in got.values())
             and js == {"rc": 0, "steps": 8000}, (got, js))


# ------------------------------------------------------------------ (b) the GPU budget at the site's submission gate
SHIM = r'''#!/usr/bin/env python3
"""`site` for the tests: the live ~/.local/bin/site with its ssh replaced by a local bash in a scratch HOME (the
"site" is a directory of this machine), sync a no-op and `pick` a fixed plan. Everything else is the real code."""
import importlib.machinery, importlib.util, os, subprocess, sys
loader = importlib.machinery.SourceFileLoader("site_tool", "SITE_PATH")
spec = importlib.util.spec_from_loader("site_tool", loader)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)
FAKEBIN = os.environ["BUDGET_FAKEBIN"]
def run_ssh(sc, inner, timeout=120, **k):
    env = dict(os.environ, PATH=FAKEBIN + ":" + os.environ["PATH"])
    p = subprocess.run(["bash", "-c", inner], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                       timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")
m.run_ssh = run_ssh
m.cmd_sync = lambda a: None
if sys.argv[1:2] == ["pick"]:
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "ccin2p3"
    n = sys.argv[sys.argv.index("--jobs") + 1] if "--jobs" in sys.argv else "1"
    print("    %s %s jobs  (test plan)" % (only, n))
    sys.exit(0)
sys.argv[0] = "site"
m.main()
'''
FAKE_SBATCH = r'''#!/bin/sh
d=$(dirname "$0"); n=$(( $(cat "$d/n" 2>/dev/null || echo 1000) + 1 )); echo $n > "$d/n"
echo "sbatch $*" >> "$d/submitted.log"; echo "$n"
'''
FAKE_CONDOR = r'''#!/bin/sh
d=$(dirname "$0"); n=$(( $(cat "$d/n" 2>/dev/null || echo 1000) + 1 )); echo $n > "$d/n"
echo "condor_submit $*" >> "$d/submitted.log"; echo "Submitting job(s)."; echo "1 job(s) submitted to cluster $n."
'''


# the site tool under test: the live one, or a staged copy before it is applied (SITE_TOOL=...)
SITE_TOOL = os.environ.get("SITE_TOOL") or os.path.expanduser("~/.local/bin/site")


class SiteBox:
    """a scratch HOME with the ledger, FA registered, the three sites as local directories (each a view of this
    checkout), fake schedulers that only log, and the `site` shim first on PATH."""

    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="budget_")
        j = lambda *x: os.path.join(self.d, *x)
        self.home, self.bin, self.fake = j("home"), j("bin"), j("fakebin")
        for x in (self.bin, self.fake, os.path.join(self.home, ".config", "ccorch"),
                  os.path.join(self.home, ".local", "share", "ccorch", "launch", "FA")):
            os.makedirs(x)
        import tomllib
        real = tomllib.load(open(os.path.expanduser("~/.config/ccorch/sites.toml"), "rb"))
        txt = open(os.path.expanduser("~/.config/ccorch/sites.toml")).read()
        for name, sc in real["sites"].items():
            root = j("remote", name)
            os.makedirs(root)
            view = os.path.join(root, "Foundational_Amplitudes")
            os.makedirs(os.path.join(view, ".venv", "bin"))
            for e in os.listdir(ROOT):
                if e not in (".venv", ".venv-cpu"):
                    os.symlink(os.path.join(ROOT, e), os.path.join(view, e))
            open(os.path.join(view, ".venv", "bin", "activate"), "w").write(
                'export PATH="%s:$PATH"\n' % os.path.dirname(sys.executable))
            txt = txt.replace('root         = "%s"' % sc["root"], 'root         = "%s"' % root)
            if sc.get("submit_root"):
                os.makedirs(j("afs", name))
                txt = txt.replace('submit_root  = "%s"' % sc["submit_root"], 'submit_root  = "%s"' % j("afs", name))
        open(os.path.join(self.home, ".config", "ccorch", "sites.toml"), "w").write(txt)
        open(os.path.join(self.home, ".config", "ccorch", "projects.toml"), "w").write(
            '[projects.FA]\nlocal = "%s"\nbranch = "trunk"\nremote_name = "Foundational_Amplitudes"\nledger = true\n'
            % ROOT)
        open(os.path.join(self.bin, "site"), "w").write(SHIM.replace("#!/usr/bin/env python3", "#!" + sys.executable)
                                                .replace("SITE_PATH", SITE_TOOL))
        for name, body in (("sbatch", FAKE_SBATCH), ("condor_submit", FAKE_CONDOR), ("squeue", "#!/bin/sh\n"),
                           ("scontrol", "#!/bin/sh\n"), ("sacct", "#!/bin/sh\n")):
            open(os.path.join(self.fake, name), "w").write(body)
        for p in [os.path.join(self.bin, "site")] + [os.path.join(self.fake, x) for x in os.listdir(self.fake)]:
            os.chmod(p, 0o755)
        self.env = dict(os.environ, HOME=self.home, BUDGET_FAKEBIN=self.fake,
                        PATH=self.bin + ":" + os.environ["PATH"])
        self.env.pop("CCORCH_WORK", None)
        self.site("items")                                     # creates the ledger's tables

    def site(self, *a, env=None):
        return subprocess.run(["site"] + list(a), capture_output=True, text=True, env=dict(self.env, **(env or {})),
                              cwd=ROOT, timeout=600)

    def item(self, iid, est=None, match="*"):
        c = sqlite3.connect(os.path.join(self.home, ".local", "share", "ccorch", "runs.db"))
        c.execute("INSERT INTO items (project,id,title,decision,scope,est_gpu_hours,match,status,created,updated) "
                  "VALUES ('FA',?,?,'D1','test',?,?,'active',1,1)", (iid, iid, est, match))
        c.commit()

    def submitted(self):
        p = os.path.join(self.fake, "submitted.log")
        return open(p).read().splitlines() if os.path.exists(p) else []

    def window(self, site=None):
        c = sqlite3.connect(os.path.join(self.home, ".local", "share", "ccorch", "runs.db"))
        return [dict(zip(("site", "items", "gpu_h"), r)) for r in
                c.execute("SELECT site, items, gpu_h FROM gpu_window ORDER BY rowid")]

    def site_window(self):
        p = os.path.join(self.home, ".ccorch", "gpu_window.jsonl")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def budget_gate_reading():
    """the gate's reading of a job's GPU-hours (site_gate.py, wall time x GPUs x jobs), on the forms our scripts and
    the schedulers use."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("site_gate", os.path.join(os.path.dirname(os.path.realpath(
        SITE_TOOL)), "site_gate.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    d = tempfile.mkdtemp(prefix="gate_")
    sc = os.path.join(d, "job.sh")
    open(sc, "w").write("#!/bin/bash\n#SBATCH --time=02:00:00\n#SBATCH --gres=gpu:1\necho x\n#SBATCH --gres=gpu:8\n")
    sub = os.path.join(d, "j.sub")
    cases = [
        ("sbatch script header (2 h, gres gpu:1; a #SBATCH after the first command is ignored)", "sbatch", [sc], 2.0),
        ("command line beats the header (--time=30, --gpus=2)", "sbatch", ["--time=30", "--gpus=2", sc], 1.0),
        ("array 0-15%4", "sbatch", ["--array=0-15%4", sc], 32.0),
        ("array 1,3,5-9:2 -t 1-00:00:00", "sbatch", ["-a", "1,3,5-9:2", "-t", "1-00:00:00", sc], 24.0 * 5),
        ("--gpus-per-node=4 --nodes=2 --time=1:00:00", "sbatch", ["--gpus-per-node=4", "--nodes=2", "--time=1:00:00",
                                                                  "--wrap", "x"], 8.0),
        ("typed gres gpu:a100:2 with --nice and --dependency", "sbatch", ["--nice=100", "--dependency=afterok:1",
                                                                         "--gres=gpu:a100:2", sc], 4.0),
        ("--test-only submits nothing", "sbatch", ["--test-only", sc], 0.0),
        ("CPU job", "sbatch", ["--time=10:00:00", "--wrap", "true"], 0.0),
    ]
    for name, cmd, args, want in cases:
        got = g.gpu_hours(cmd, args, {})[0]
        scenario("(b) gate reading: %s -> %g GPU-h" % (name, want), abs(got - want) < 1e-6, got)
    for name, text, want in [
            ("queue 5, workday", 'request_gpus = 1\n+JobFlavour = "workday"\nqueue 5\n', 40.0),
            ("queue in (a, b, c), +MaxRuntime 7200, 2 GPUs", "request_gpus = 2\n+MaxRuntime = 7200\nqueue x in (a, b, c)\n",
             12.0),
            ("queue x from ( three lines ), espresso default", "request_gpus = 1\nqueue x from (\n a\n b\n c\n)\n", 1.0),
            ("no request_gpus", "request_cpus = 4\nqueue 100\n", 0.0)]:
        open(sub, "w").write(text)
        got = g.gpu_hours("condor_submit", [sub], {})[0]
        scenario("(b) gate reading, condor: %s -> %g GPU-h" % (name, want), abs(got - want) < 1e-6, got)
    open(sc, "w").write("#!/bin/bash\n#SBATCH --tres-per-task=gres/gpu:1\n#SBATCH --time=10:00:00\n")
    got = g.gpu_hours("sbatch", [sc], {})[0]
    scenario("(b) gate reading: --tres-per-task=gres/gpu:1, 10 h -> 10 GPU-h (review finding 4)", got == 10.0, got)
    open(sc, "w").write("#!/bin/bash\n#SBATCH --gpus-per-socket=1\n#SBATCH --time=10:00:00\n")
    try:
        g.gpu_hours("sbatch", [sc], {})
        ok = False
    except g.Unreadable:
        ok = True
    scenario("(b) gate reading: a GPU asked in a form not counted (--gpus-per-socket) raises, never 0", ok)
    # a DAG as generate_sweep.write_dag writes it for a chained lxplus sweep: JOB lines over per-trial submit files
    for k in range(4):
        open(os.path.join(d, "t%d.sub" % k), "w").write('request_gpus = 1\n+JobFlavour = "longlunch"\nqueue 1\n')
    dag = os.path.join(d, "sweep_0000.dag")
    open(dag, "w").write("".join("JOB t%d t%d.sub\n" % (k, k) for k in range(4)) + "PARENT t0 CHILD t1\n")
    got = g.gpu_hours("condor_submit_dag", ["-batch-name", "s_0000", dag], {})[0]
    scenario("(b) gate reading: condor_submit_dag of 4 nodes x 1 GPU x longlunch -> 8 GPU-h (review finding 1)",
             got == 8.0, got)
    names = g.job_names("sbatch", ["--nice=5", os.path.join(d, "sweeps", "tp3_x", "jobs", "trial_0001.sh")])
    scenario("(b) gate attribution: under --item '*' a sweep's trial counts for the item whose glob matches its sweep; "
             "no match counts for all; one item named counts for it",
             g.counted_items(["a", "tp3"], {"a": ["het*"], "tp3": ["tp3_*"]}, names) == ["tp3"]
             and g.counted_items(["a", "tp3"], {"a": ["het*"], "tp3": ["tp3_*"]}, ["other"]) == ["a", "tp3"]
             and g.counted_items(["a"], {}, names) == ["a"], names)
    try:
        g.gpu_hours("sbatch", ["--time=UNLIMITED", "--gpus=1", "--wrap", "x"], {})
        ok = False
    except g.Unreadable:
        ok = True
    scenario("(b) gate reading: an unreadable GPU job's time raises (the gate then refuses it)", ok)


def budget_gate():
    """Joaquin 2026-10-30 ("I see no reason to keep doing this wrongly"): the GPU budget at the site's submission gate,
    which every sbatch / condor_submit of a `site run` or `site submit` passes, so launch scripts, sweep_manager and
    generators count too. Live site tool and gate (~/.local/bin/site, site_gate.py); the sites are local directories,
    the schedulers fakes that only log."""
    # 1. a launch script submitting 8 jobs: scripts/launch_het_shared.sh (8 cells, job_het_delta.sh: 1 GPU, 3 h)
    b = SiteBox()
    b.item("het_test")
    jobs = os.path.join(b.d, "jobs8.txt")
    open(jobs, "w").write("".join("3 cellA%d cellB%d\n" % (k, k) for k in range(8)))
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "launch_het_shared.sh")], capture_output=True, text=True,
                       env=dict(b.env, JOBS=jobs, ITEM="het_test"), cwd=ROOT, timeout=900)
    sub, win = b.submitted(), b.window()
    scenario("(b) a launch script submitting 8 jobs (launch_het_shared.sh, 8 x 1 GPU x 3 h = 24 GPU-h): all 8 pass the "
             "gate and each is counted (site-side window and the laptop's gpu_window)",
             r.returncode == 0 and "submitted 8, left 0" in r.stdout and len(sub) == 8
             and len(win) == 8 and abs(sum(w["gpu_h"] for w in win) - 24) < 1e-6 and len(b.site_window()) == 8,
             (r.returncode, r.stdout[-1500:], r.stderr[-800:], sub, win))
    jobs2 = os.path.join(b.d, "jobs8b.txt")
    open(jobs2, "w").write("".join("3 cellC%d cellD%d\n" % (k, k) for k in range(8)))
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "launch_het_shared.sh")], capture_output=True, text=True,
                       env=dict(b.env, JOBS=jobs2, ITEM="het_test"), cwd=ROOT, timeout=900)
    scenario("(b) the same launch script again within the hour (8 more): 2 pass (30 GPU-h), the 3rd is refused at 33 "
             "GPU-h by the site's gate, the script keeps the 6 for a later pass (exit 75)",
             r.returncode == 75 and "submitted 2, left 6" in r.stdout and len(b.submitted()) == 10
             and "site gate: sbatch refused" in r.stdout + r.stderr and "33.0 GPU-h for het_test" in r.stdout + r.stderr,
             (r.returncode, r.stdout[-2000:], r.stderr[-800:]))

    # 2. a sweep_manager submission through `site run`: 10 trials x 1 GPU x 4 h
    def sweep(name, n=10, time_="04:00:00"):
        d = os.path.join(b.d, "sweeps", name, "jobs")
        os.makedirs(d)
        for k in range(n):
            open(os.path.join(d, "trial_%04d.sh" % k), "w").write(
                "#!/bin/bash\n#SBATCH --job-name=%s_%d\n#SBATCH --time=%s\n#SBATCH --gres=gpu:1\n"
                "#SBATCH --cpus-per-task=4\necho train\n" % (name, k, time_))
        return os.path.dirname(d)
    b = SiteBox()
    b.item("hpo_test", match="hpo_test")
    sd = sweep("hpo_test")
    r = b.site("run", "--item", "hpo_test", "ccin2p3", "FA", "--", "python", "sweep/sweep_manager.py", "submit", sd,
               "--seq-batches", "1")
    sub = b.submitted()
    scenario("(b) a sweep_manager submission (`site run --item ... sweep_manager.py submit`, 10 trials x 1 GPU x 4 h = "
             "40 GPU-h, no estimate recorded): 7 go in (28 GPU-h), the 8th is refused by the gate (32 > 30), and the "
             "laptop's ledger records the 7", len(sub) == 7 and "site gate: sbatch refused" in r.stdout
             and "32.0 GPU-h for hpo_test" in r.stdout and len(b.window()) == 7 and r.returncode != 0,
             (r.returncode, r.stdout[-2500:], r.stderr[-800:], sub))
    b.item("hpo_cov", est=45, match="hpo_cov")
    sd = sweep("hpo_cov")
    r = b.site("run", "--item", "hpo_cov", "ccin2p3", "FA", "--", "python", "sweep/sweep_manager.py", "submit", sd,
               "--seq-batches", "1")
    scenario("(b) the same sweep under an item whose recorded estimate is 45 GPU-h: all 10 trials go in (40 GPU-h, "
             "over 30 but covered)", r.returncode == 0 and len(b.submitted()) == 17 and "refused" not in r.stdout,
             (r.returncode, r.stdout[-2000:], r.stderr[-800:]))

    # 3. a single small job: scripts/job_probe.sh (1 GPU, 10 min)
    b = SiteBox()
    b.item("probe_test")
    r = b.site("submit", "ccin2p3", "FA", "scripts/job_probe.sh", "--item", "probe_test", "--est-gpu-hours", "0.2")
    win = b.window()
    scenario("(b) a single small job (`site submit` of job_probe.sh, 1 GPU x 10 min): passes the gate, its run is "
             "registered, 0.17 GPU-h counted", r.returncode == 0 and "run FA-ccin2p3-" in r.stdout and len(win) == 1
             and abs(win[0]["gpu_h"] - 1 / 6.0) < 1e-3, (r.returncode, r.stdout[-1500:], r.stderr[-800:], win))
    r = b.site("run", "--item", "probe_test", "ccin2p3", "FA", "--", "sbatch", "--parsable", "--partition=cpu",
               "--time=10:00:00", "--wrap", "true")
    scenario("(b) a CPU job (sbatch --wrap with no GPU, 10 h) passes and counts 0", r.returncode == 0
             and len(b.window()) == 1, (r.returncode, r.stdout[-800:]))
    r = b.site("run", "ccin2p3", "FA", "--", "true")
    os.makedirs(os.path.join(b.home, ".ccorch", "bin"), exist_ok=True)
    r = subprocess.run(["bash", "-c", "export PATH=$HOME/.ccorch/bin:%s:$PATH; sbatch --gres=gpu:1 x.sh" % b.fake],
                       capture_output=True, text=True, env=b.env)
    scenario("(b) a submission on the site with no work item named is still refused by the wrapper",
             r.returncode != 0 and "no work item named" in r.stderr, (r.returncode, r.stderr))

    # 4. over the limit, covered by the item's estimate; and the same without one, Jean Zay, another site
    b = SiteBox()
    b.item("big_est", est=40)
    b.item("big_none")
    rcs = {}
    for iid in ("big_est", "big_none"):
        rcs[iid] = [b.site("submit", "ccin2p3", "FA", "scripts/job_het_delta.sh", "--item", iid, "--est-gpu-hours",
                           "12", "--hdr=--time=12:00:00").returncode for _ in range(3)]
    scenario("(b) over the limit, covered by the item's estimate: 3 x 12 GPU-h = 36 for an item recorded at 40 all go "
             "in; for an item with no estimate the third (36 > 30) is refused",
             rcs["big_est"] == [0, 0, 0] and rcs["big_none"][:2] == [0, 0] and rcs["big_none"][2] != 0, rcs)
    b = SiteBox()
    b.item("jz_test")
    r = [b.site("submit", "jeanzay", "FA", "scripts/job_het_delta.sh", "--item", "jz_test", "--est-gpu-hours", "3")
         for _ in range(4)]
    scenario("(b) Jean Zay: 4 x 3 GPU-h, the 4th (12 > 10 there) is refused", [x.returncode for x in r][:3] == [0, 0, 0] and r[3].returncode != 0
             and "limit 10 there" in r[3].stdout + r[3].stderr, [(x.returncode, x.stdout[-400:], x.stderr[-400:]) for x in r])
    b = SiteBox()
    b.item("x_site")
    for _ in range(8):
        b.site("submit", "ccin2p3", "FA", "scripts/job_het_delta.sh", "--item", "x_site", "--est-gpu-hours", "3")
    r = b.site("submit", "lxplus", "FA", "scripts/job_het_delta.sh", "--item", "x_site", "--est-gpu-hours", "8")
    scenario("(b) the window spans sites: 24 GPU-h on CC-IN2P3, then a condor job on lxplus (1 GPU, workday = 8 h) is "
             "refused at 32 (24 elsewhere)", r.returncode != 0 and "24.0 elsewhere" in r.stdout + r.stderr
             and len(b.submitted()) == 8, (r.returncode, r.stdout[-1200:], b.submitted()[-2:]))
    # review finding 2: the CC feeder's upkeep (`site run --item '*'`) counts its trials for the sweep's own item only
    b = SiteBox()
    b.item("tp3", match="tp3_*")
    b.item("other", match="het_*")
    sd = sweep("tp3_feed", n=8, time_="04:00:00")
    r = b.site("run", "--item", "*", "ccin2p3", "FA", "--", "python", "sweep/sweep_manager.py", "submit", sd,
               "--seq-batches", "1")
    r2 = b.site("submit", "ccin2p3", "FA", "scripts/job_het_delta.sh", "--item", "other", "--est-gpu-hours", "3")
    win = b.window()
    scenario("(b) upkeep under --item '*': a sweep's trials count for its own item (tp3, 7 of 8 in at 28 GPU-h), and "
             "a submission of another item in the same hour is not charged with them", len(b.submitted()) == 8
             and r2.returncode == 0 and all(json.loads(w["items"]) == ["tp3"] for w in win[:7])
             and json.loads(win[-1]["items"]) == ["other"], (r.stdout[-800:], r2.stdout[-400:], r2.stderr[-400:], win))
    # review finding 3: a work command whose gate cannot be installed submits nothing, whatever its words
    b = SiteBox()
    b.item("w")
    open(os.path.join(b.home, ".ccorch"), "w").write("not a directory")
    mark = os.path.join(b.d, "ran")
    r = b.site("run", "--item", "w", "ccin2p3", "FA", "--", "touch", mark)
    scenario("(b) a `site run --item` whose gate cannot be written (HOME/.ccorch unwritable) is refused before the "
             "command runs, even with no submit word in it", r.returncode != 0 and not os.path.exists(mark)
             and "could not be installed" in r.stdout + r.stderr, (r.returncode, r.stdout[-600:], r.stderr[-300:]))
    b = SiteBox()
    b.item("x_lx")
    r = b.site("submit", "lxplus", "FA", "scripts/job_het_delta.sh", "--item", "x_lx", "--est-gpu-hours", "8")
    scenario("(b) condor on lxplus: the gate reads the submit file (request_gpus 1, +JobFlavour workday: 8 GPU-h)",
             r.returncode == 0 and [w["gpu_h"] for w in b.window()] == [8.0], (r.returncode, r.stdout[-1200:],
                                                                                b.window()))


GROUPS = {"fixtures": fixtures, "incident_tool": incident_tool, "dispatcher": dispatcher,
          "switch_a": switch_a, "scenarios_via_dispatch": scenarios_via_dispatch,
          "stop_events_replay": stop_events_replay, "stop_acked_runs": stop_acked_runs,
          "stop_conditions": stop_conditions, "stop_opener": stop_opener, "submit_replay": submit_replay,
          "submit_budget": submit_budget, "submit_scenarios": submit_scenarios, "review_merged": review_merged,
          "review_modes": review_modes, "review_paths": review_paths, "review_before_merge": review_before_merge,
          "attribution": attribution, "guards_selftest": guards_selftest, "run_end_markers": run_end_markers,
          "budget_gate_reading": budget_gate_reading, "budget_gate": budget_gate}


def main(argv):
    for name in argv or list(GROUPS):
        GROUPS[name]()
    print("%d/%d scenarios pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if RESULTS and all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
