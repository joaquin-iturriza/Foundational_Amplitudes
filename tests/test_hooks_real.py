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
import os, sys
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_calls.log"), "a").write(repr(sys.argv[1:]) + "\n")
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
    b = Box()
    b.say("lock guard: only repo-reviewer advances its pillar")
    b.tool_use(*cmd)
    rc1, _ = b.run("lock_guard.py", "PreToolUse", cmd)
    b.tool_use(*cmd)
    rc2, out2 = b.run("lock_guard.py", "PreToolUse", cmd)
    scenario("A3: a PreToolUse refusal repeated on the identical command twice in a row: disabled, the call goes "
             "through", rc1 == 2 and rc2 == 0 and "lock_guard.py" in b.disabled(), (rc1, rc2, out2))
    b = Box()
    b.say("lock guard: refused")
    b.run("lock_guard.py", "PreToolUse", cmd)
    b.tool_use("Read", {"file_path": "/x"})
    b.tool_use(*cmd)
    rc2, _ = b.run("lock_guard.py", "PreToolUse", cmd)
    scenario("A3: the same PreToolUse refusal with another tool call between is no loop", rc2 == 2 and not b.disabled(),
             b.disabled())
    b = Box()
    b.say("lock guard: refused")
    b.run("lock_guard.py", "PreToolUse", cmd)
    rc2, _ = b.run("lock_guard.py", "PreToolUse", ("Bash", {"command": "other"}))
    scenario("A3: a PreToolUse refusal of a different command is no loop", rc2 == 2 and not b.disabled(), b.disabled())
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
        same = len(old) == len(new) and all(o[0] == n[0] and o[1] == n[1] and o[3] == n[3] and n[2].startswith(prefix)
                                            and o[2].replace('"', "").endswith("/.claude/hooks/" + n[2][len(prefix):])
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


GROUPS = {"fixtures": fixtures, "incident_tool": incident_tool, "dispatcher": dispatcher,
          "switch_a": switch_a, "scenarios_via_dispatch": scenarios_via_dispatch}


def main(argv):
    for name in argv or list(GROUPS):
        GROUPS[name]()
    print("%d/%d scenarios pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if RESULTS and all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
