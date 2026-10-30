#!/usr/bin/env python3
"""The hook dispatcher (agent redesign, step A3): every registration runs through it.

    python3 <hooks dir>/dispatch.py <hook> [args...]        (hook input on stdin, as Claude Code sends it)

It runs <hooks dir>/<hook> with the same input, arguments, environment and working directory and passes its stdout,
stderr and exit code through unchanged, except:
  - a hook listed in <hooks dir>/disabled.json (by its name, or its name with these arguments) is not run: exit 0 and
    a one-line note on stderr;
  - a hook script that does not exist: exit 0 and a note (a missing guard never blocks a turn);
  - a loop: the same Stop (or SubagentStop) refusal (hook, sha1 of its message) twice in a row in one session or
    subagent, with no tool call in between and the same state (git HEAD, sha1 of plan/plan.yaml, the project's open
    event ids): the turn cannot end. The hook is disabled through the incident record (disabled.json, incidents.jsonl,
    an ALERT event) with why = the repeated message, and the turn ends (exit 0);
  - a tool-call guard (PreToolUse, PostToolUse...) refusing the identical call twice in a row the same way is the guard
    doing its job, not a loop (the agent can do something else; letting it through would make every guard
    bypassable by asking twice): the first repeat is recorded as an incident with one ALERT, the guard stays on and
    the refusal stands.
An incident's ALERT is for the main session, never addressed to the user: it carries what a fix needs (the hook, the
event, the tool call refused, the refusal's text, how many times in a row, the session and agent) and says what to
do: fix the hook at once when the fix is obvious and test it on the recorded refusal (`tools/hook_incident.py --replay
<id>` runs the hook again on the recorded input), or ack saying the guard was right; it reaches the user only through
`site ack E<n> --needs-user "<what>"` (a call that is the user's, or a fix that is not obvious).
A refusal is exit 2, or a JSON stdout with "decision": "block" or a PreToolUse permissionDecision "deny". "No other tool
call in between" is read from the transcript (the subagent's own when the input names one) from where it ended at the
previous refusal: every tool call written since must be the refused call itself. Without session_id or
transcript_path no loop is ever declared. A crash of the dispatcher before the hook ran exits 1, which Claude Code treats as non-blocking; once the hook
has refused, an error of the loop bookkeeping is caught and the refusal goes through unchanged. Output is passed
through byte for byte.

The incident record lives here too, so the user-level copy (~/.claude/hooks/dispatch.py, the same file) needs nothing
from a project: disable(), report(), resolve(), incidents(), last_refusal(). tools/hook_incident.py is its command line.
State: <hooks dir>/.dispatch_state.json (gitignored), DISPATCH_STATE overrides the path (tests)."""
import datetime as dt, fcntl, hashlib, json, os, re, sqlite3, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get("DISPATCH_SITE", "site")
RUNS_DB = os.environ.get("DISPATCH_RUNS_DB", os.path.expanduser("~/.local/share/ccorch/runs.db"))
PROJECTS = os.environ.get("DISPATCH_PROJECTS", os.path.expanduser("~/.config/ccorch/projects.toml"))


def _now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha(s):
    return hashlib.sha1(s.encode("utf-8", "replace") if isinstance(s, str) else s).hexdigest()


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, obj):
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


class _Lock:
    def __init__(self, path):
        self.f = open(path + ".lock", "a")

    def __enter__(self):
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


# ------------------------------------------------------------------ the incident record
def project_name(project_dir):
    """the projects.toml name whose `local` is this directory, else None (no event then)."""
    if not project_dir:
        return None
    try:
        import tomllib
        with open(PROJECTS, "rb") as f:
            projs = tomllib.load(f).get("projects", {})
    except Exception:
        return None
    real = os.path.realpath(project_dir)
    for name, p in projs.items():
        if os.path.realpath(os.path.expanduser(p.get("local", ""))) == real:
            return name
    return None


def disabled(hooks_dir=HERE):
    return _load(os.path.join(hooks_dir, "disabled.json"), {})


def _alert(msg, key, project_dir):
    proj = project_name(project_dir or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    if not proj:
        return None
    try:
        r = subprocess.run([SITE, "event", "--project", proj, "--kind", "ALERT", "--key", key, "--msg", msg],
                           capture_output=True, text=True, timeout=60)
        return (r.stdout.strip() or r.stderr.strip())[:200] if r.returncode == 0 else \
            "site event failed (exit %d): %s" % (r.returncode, (r.stderr or r.stdout).strip()[:200])
    except Exception as e:
        return "site event failed: %s" % e


def _one_line(x, n):
    x = re.sub(r"\s+", " ", str(x or "")).strip()
    return x if len(x) <= n else x[:n - 3] + "..."


def _call_of(inp):
    """the tool call a guard refused, as (name, input), from the hook input; None for a Stop."""
    if isinstance(inp, dict) and inp.get("tool_name"):
        return inp.get("tool_name"), inp.get("tool_input")
    return None


def incident_msg(rec, user_level=False):
    """the ALERT's text: everything the main session needs to fix the hook, and what to do with it."""
    call = rec.get("call")
    tool = ("%s %s" % (call[0], _one_line(json.dumps(call[1], sort_keys=True), 600))) if call else \
        "none (a %s refusal: the turn could not end)" % (rec.get("event") or "manual")
    state = "disabled, the call went through" if rec.get("disabled") else "still on, still refusing"
    tool_flag = " --user" if user_level else ""
    return ("HOOK INCIDENT %s, for the main session to fix (not addressed to the user): %s [%s], %s; refused %s "
            "time(s) in a row in session %s%s. Why: %s. Call refused: %s. Refusal: %s. "
            "Do: if the fix is obvious, fix the hook now, test it on this refusal (python3 tools/hook_incident.py%s "
            "--replay %s), commit%s, and ack with what changed and the replay's verdict; if the guard was right, ack "
            "saying so and correct the caller. Only a call that is the user's or a fix that is not obvious: "
            "site ack E<n> --needs-user \"<what he must decide>\". Full record: python3 tools/hook_incident.py%s "
            "--show %s" % (
                rec["id"], rec["hook"], rec.get("event") or "manual", state,
                rec.get("count") if rec.get("count") is not None else "?", rec.get("session") or "?",
                (" agent %s" % rec["agent"]) if rec.get("agent") else "", _one_line(rec.get("why"), 300), tool,
                _one_line(rec.get("last_refusal") or "not recorded", 800), tool_flag, rec["id"],
                (", then --resolve '%s'" % rec["hook"]) if rec.get("disabled") else "", tool_flag, rec["id"]))


def _record(hook, why, hooks_dir, session, last_refusal, project_dir, inp, count, disabled_now):
    inp = inp if isinstance(inp, dict) else None
    try:                                         # bounded as in the state: a huge Write is kept by its call, not input
        if inp is not None and len(json.dumps(inp)) > 20000:
            inp = dict(inp, tool_input={"truncated": True, "text": json.dumps(inp.get("tool_input"))[:20000]})
    except (TypeError, ValueError):
        inp = None
    t = _now()
    rec = {"id": "I-" + _sha("%s|%s|%s|%s" % (hook, t, session, os.getpid()))[:8], "hook": hook, "why": why,
           "time": t, "session": session, "agent": (inp or {}).get("agent_id"),
           "event": (inp or {}).get("hook_event_name"), "call": _call_of(inp), "last_refusal": last_refusal,
           "count": count, "disabled": disabled_now, "kept": not disabled_now, "input": inp,
           "project_dir": project_dir}
    rec["event_id"] = _alert(incident_msg(rec, os.path.realpath(hooks_dir) == os.path.realpath(
        os.path.expanduser("~/.claude/hooks"))), "hook-incident:" + rec["id"], project_dir)
    with open(os.path.join(hooks_dir, "incidents.jsonl"), "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def disable(hook, why, hooks_dir=HERE, session=None, last_refusal=None, project_dir=None, inp=None, count=None):
    """hook into disabled.json, one line into incidents.jsonl, one ALERT event for the main session. A hook already
    disabled: nothing new (no second event), returns False."""
    path = os.path.join(hooks_dir, "disabled.json")
    with _Lock(path):
        d = disabled(hooks_dir)
        if hook in d:
            return False
        rec = _record(hook, why, hooks_dir, session, last_refusal, project_dir, inp, count, True)
        d[hook] = {"why": why, "time": rec["time"], "session": session, "incident": rec["id"]}
        _save(path, d)
        return rec


def report(hook, why, hooks_dir=HERE, session=None, last_refusal=None, project_dir=None, inp=None, count=None):
    """a tool-call guard that refused the same call twice: one line into incidents.jsonl and one ALERT for the main
    session; the guard stays on and keeps refusing."""
    with _Lock(os.path.join(hooks_dir, "disabled.json")):
        return _record(hook, why, hooks_dir, session, last_refusal, project_dir, inp, count, False)


def last_refusal(hook, state_path=None, hooks_dir=HERE):
    """the newest refusal of this hook the dispatcher recorded (any session): {msg, input, count, session, time}, or
    None. A hand-opened incident (hook_incident.py --why) carries it, so its ALERT has the call and the text too."""
    st = _load(state_path or os.environ.get("DISPATCH_STATE") or os.path.join(hooks_dir, ".dispatch_state.json"), {})
    best = None
    for k, v in st.items():
        if k.split("|", 2)[-1] == hook and v.get("msg") is not None and (best is None or v["time"] > best["time"]):
            best = dict(v, session=k.split("|", 1)[0])
    return best


def resolve(hook, hooks_dir=HERE, note=None):
    path = os.path.join(hooks_dir, "disabled.json")
    with _Lock(path):
        d = disabled(hooks_dir)
        if hook not in d:
            return False
        d.pop(hook)
        _save(path, d)
        with open(os.path.join(hooks_dir, "incidents.jsonl"), "a") as f:
            f.write(json.dumps({"hook": hook, "resolved": _now(), "note": note}, sort_keys=True) + "\n")
        return True


def incidents(hooks_dir=HERE):
    out = []
    try:
        for line in open(os.path.join(hooks_dir, "incidents.jsonl")):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


# ------------------------------------------------------------------ refusals and loops
def refusal(rc, out, err):
    """the refusal's message, or None when the hook let the call or the turn end through."""
    if rc == 2:
        return err.strip() or out.strip() or "exit 2"
    s = out.strip()
    if s.startswith("{"):
        try:
            j = json.loads(s)
        except ValueError:
            return None
        if j.get("decision") == "block":
            return str(j.get("reason") or "block")
        hso = j.get("hookSpecificOutput") or {}
        if hso.get("permissionDecision") == "deny":
            return str(hso.get("permissionDecisionReason") or "deny")
    return None


def transcript_of(inp):
    t = inp.get("transcript_path")
    if not t:
        return None
    if inp.get("agent_id") and inp.get("session_id"):
        sub = os.path.join(os.path.dirname(t), inp["session_id"], "subagents", "agent-%s.jsonl" % inp["agent_id"])
        if os.path.exists(sub):
            return sub
    return t


def tool_calls_since(path, offset):
    """(end offset, [(name, sha of input)] of the tool calls written after offset)."""
    calls = []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if offset > end:
                offset = 0
            f.seek(offset)
            for line in f:
                if b'"tool_use"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") != "assistant":
                    continue
                for b in (r.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        calls.append((b.get("name"), _sha(json.dumps(b.get("input"), sort_keys=True))))
            return end, calls
    except OSError:
        return offset, calls


def world(cwd):
    """what a repeated refusal must not have changed: git HEAD, the plan's sha1, the open events."""
    w = {}
    try:
        w["head"] = subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True,
                                   timeout=10).stdout.strip()
    except Exception:
        w["head"] = None
    try:
        top = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        with open(os.path.join(top, "plan", "plan.yaml"), "rb") as f:
            w["plan"] = _sha(f.read())
    except Exception:
        w["plan"] = None
    proj = project_name(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    try:
        c = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True, timeout=10)
        w["events"] = [r[0] for r in c.execute(
            "SELECT id FROM events WHERE acked IS NULL AND (project=? OR project IS NULL) ORDER BY id", (proj,))]
        c.close()
    except Exception:
        w["events"] = None
    return w


def check_loop(hook, inp, msg, state_path, cwd):
    """record this refusal; (verdict, count): verdict "new" when it repeats the previous one of the same hook in the
    same session or agent for the first time, "again" for a further repeat, False otherwise; count: how many times in
    a row this hook refused the same thing in this session or agent (the refused call and input are kept in the state
    for a hand-opened incident)."""
    sess, tpath = inp.get("session_id"), transcript_of(inp)
    if not sess or not tpath:
        return False, None
    event = inp.get("hook_event_name") or ""
    tool = (inp.get("tool_name"), _sha(json.dumps(inp.get("tool_input"), sort_keys=True))) if inp.get("tool_name") else None
    fp = _sha(json.dumps([hook, event, _sha(msg), tool]))
    key = "%s|%s|%s" % (sess, inp.get("agent_id") or "", hook)
    w = world(cwd)
    with _Lock(state_path):
        st = _load(state_path, {})
        prev = st.get(key)
        if prev:
            end, calls = tool_calls_since(tpath, prev["offset"])
        else:                                                     # the first: where the transcript ends now
            end, calls = (os.path.getsize(tpath) if os.path.exists(tpath) else 0), []
        loop = bool(prev and prev["fp"] == fp and prev["world"] == w
                    and all(tool is not None and tuple(c) == tool for c in calls))
        # once reported, a repeat of the same refusal stays reported while its fingerprint holds, even when the state
        # moved between (the ALERT the report opens is itself a change of the open events)
        seen = bool(prev and prev["fp"] == fp and prev.get("reported"))
        count = (prev.get("count") or 1) + 1 if prev and prev["fp"] == fp else 1
        try:
            raw = json.dumps(inp, sort_keys=True)
        except (TypeError, ValueError):
            raw = None
        st[key] = {"fp": fp, "offset": end, "world": w, "time": _now(), "reported": loop or seen, "count": count,
                   "msg": msg[:4000], "input": inp if raw is not None and len(raw) <= 20000 else None}
        if len(st) > 500:                                        # keep the newest
            for k in sorted(st, key=lambda k: st[k].get("time", ""))[:len(st) - 500]:
                st.pop(k)
        _save(state_path, st)
    return (("again" if seen else "new"), count) if loop else (False, count)


def command_for(path, args):
    if path.endswith(".py"):
        return [sys.executable, path] + args
    if path.endswith(".sh"):
        return ["bash", path] + args
    return [path] + args


def main(argv):
    if not argv:
        sys.stderr.write("dispatch: usage: dispatch.py <hook> [args...]\n")
        return 1
    name, args = argv[0], argv[1:]
    spec = " ".join([name] + args)
    raw = sys.stdin.buffer.read()
    off = disabled(HERE)
    if spec in off or name in off:
        sys.stderr.write("dispatch: %s is disabled (%s); not run\n" % (spec, (off.get(spec) or off.get(name))["why"][:120]))
        return 0
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        sys.stderr.write("dispatch: no hook %s in %s; not run\n" % (name, HERE))
        return 0
    r = subprocess.run(command_for(path, args), input=raw, capture_output=True)
    if refusal(r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")) is not None:
        # the hook has refused: from here an error of the dispatcher's own bookkeeping must never turn the refusal into
        # a pass, so it is caught and the hook's verdict goes through unchanged
        try:
            if loop_through(spec, raw, r):
                return 0
        except Exception as e:
            sys.stderr.buffer.write(("dispatch: loop check failed (%r); the hook's verdict stands\n" % (e,)).encode())
    sys.stdout.buffer.write(r.stdout)                                  # byte for byte
    sys.stderr.buffer.write(r.stderr)
    return r.returncode


STOP_EVENTS = ("Stop", "SubagentStop")


def loop_through(spec, raw, r):
    """True when the call goes through: a Stop (or SubagentStop) refusal that repeats the previous one (check_loop) is
    a turn that cannot end, so the hook is disabled and the turn ends. A tool-call guard's repeat is the guard doing
    its job (the agent can do something else): it is recorded once as an incident and the refusal stands."""
    msg = refusal(r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"))
    try:
        inp = json.loads(raw or b"{}")
    except ValueError:
        return False
    if not isinstance(inp, dict):
        return False
    state = os.environ.get("DISPATCH_STATE") or os.path.join(HERE, ".dispatch_state.json")
    cwd = inp.get("cwd") or os.getcwd()
    loop, count = check_loop(spec, inp, msg, state, cwd)
    if not loop:
        return False
    why = "the same refusal twice with nothing done between: " + re.sub(r"\s+", " ", msg)[:400]
    kw = dict(session=inp.get("session_id"), last_refusal=msg[:4000],
              project_dir=os.environ.get("CLAUDE_PROJECT_DIR") or cwd, inp=inp, count=count)
    if (inp.get("hook_event_name") or "") not in STOP_EVENTS:
        if loop == "new":
            report(spec, why, HERE, **kw)
        return False
    disable(spec, why, HERE, **kw)
    sys.stderr.write("dispatch: %s refused the same thing twice with nothing done between; it is disabled and an "
                     "incident is open for the main session to fix (python3 tools/hook_incident.py --list)\n" % spec)
    return True

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException as e:                                     # never block on the dispatcher's own error
        sys.stderr.write("dispatch: internal error %r\n" % (e,))
        sys.exit(1)
