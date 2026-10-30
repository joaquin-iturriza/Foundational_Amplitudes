#!/usr/bin/env python3
"""Hook incidents: a hook that misbehaves, with what is needed to fix it on record (agent redesign, step A2).

    python3 tools/hook_incident.py <hook> --why "<why>" [--user]   disable it: the dispatcher skips it from now on
    python3 tools/hook_incident.py --show <id> [--user]           the whole record of one incident
    python3 tools/hook_incident.py --replay <id> [--user]         run the hook again on the refusal it recorded
                                                                  (for real: a hook with side effects acts again)
    python3 tools/hook_incident.py --resolve <hook> [--note N] [--user]   enable it again
    python3 tools/hook_incident.py --list [--user]                disabled hooks and the incident log

<hook> is the name the registration gives the dispatcher (plan_check.py, or with its arguments, "review_backlog.sh
check"). Disabling writes it to .claude/hooks/disabled.json, appends the incident to .claude/hooks/incidents.jsonl and
opens one ALERT event for the project (`site event`); a hook already disabled adds nothing. The incident carries the
hook's last refusal the dispatcher saw (the tool call refused, the refusal's text, how many times in a row, the
session), so the ALERT has what a fix needs.

An incident goes to the main session, never straight to the user: the main session fixes the hook at once when the
fix is obvious and tests the fix with --replay (exit 0: the recorded call passes now; 2: still refused, with the text),
or acks saying the guard was right. Only a call that is the user's, or a fix that is not obvious, reaches the user,
through `site ack E<n> --needs-user "<what>"`. --user acts on the user-level hooks (~/.claude/hooks). The logic is
the dispatcher's own (dispatch.py), so its loop detection and this command write the same record."""
import argparse, importlib.util, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dispatch(user, hooks_dir=None):
    d = hooks_dir or (os.path.expanduser("~/.claude/hooks") if user else os.path.join(ROOT, ".claude", "hooks"))
    spec = importlib.util.spec_from_file_location("dispatch", os.path.join(d, "dispatch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, d


def _find(m, d, iid):
    for r in m.incidents(d):
        if r.get("id") == iid:
            return r
    return None


def replay(m, d, rec):
    """(exit code, text): the hook run again, from the hooks directory as it is now, on the recorded input."""
    inp = rec.get("input")
    if not isinstance(inp, dict):
        return 1, "incident %s has no recorded hook input (opened by hand with no refusal seen); nothing to replay" \
            % rec["id"]
    name, *args = rec["hook"].split()
    path = os.path.join(d, name)
    if not os.path.isfile(path):
        return 1, "no hook %s in %s" % (name, d)
    proj = rec.get("project_dir") or ROOT
    cwd = inp.get("cwd") if os.path.isdir(inp.get("cwd") or "") else (proj if os.path.isdir(proj) else ROOT)
    r = subprocess.run(m.command_for(path, args), input=json.dumps(inp).encode(), capture_output=True, cwd=cwd,
                       env=dict(os.environ, CLAUDE_PROJECT_DIR=proj))
    out, err = r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
    msg = m.refusal(r.returncode, out, err)
    if msg is not None:
        return 2, "replay of %s on %s: still refused%s:\n%s" % (
            rec["id"], rec["hook"], " (the same text)" if msg.strip() == (rec.get("last_refusal") or "").strip()
            else " (another text)", msg.strip())
    return 0, "replay of %s on %s: passes now (exit %d)%s" % (rec["id"], rec["hook"], r.returncode,
                                                             ("\n" + err.strip()) if err.strip() else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hook", nargs="?")
    ap.add_argument("--why")
    ap.add_argument("--resolve", metavar="HOOK")
    ap.add_argument("--note")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="ID")
    ap.add_argument("--replay", metavar="ID")
    ap.add_argument("--user", action="store_true")
    ap.add_argument("--hooks-dir", help=argparse.SUPPRESS)          # tests: a copy of the hooks directory
    ap.add_argument("--session", default=os.environ.get("CLAUDE_SESSION_ID"))
    a = ap.parse_args(argv)
    m, d = _dispatch(a.user, a.hooks_dir)
    if a.list:
        off = m.disabled(d)
        print("disabled (%d):" % len(off))
        for h, v in sorted(off.items()):
            print("  %-28s since %s: %s" % (h, v.get("time"), v.get("why")))
        log = m.incidents(d)
        print("incidents (%d, last 10):" % len(log))
        for r in log[-10:]:
            print("  " + json.dumps({k: r[k] for k in ("id", "hook", "time", "resolved", "count", "why", "note",
                                                      "event_id", "event") if r.get(k)}))
        return 0
    if a.show or a.replay:
        rec = _find(m, d, a.show or a.replay)
        if not rec:
            print("hook_incident: no incident %s in %s" % (a.show or a.replay, os.path.join(d, "incidents.jsonl")))
            return 1
        if a.show:
            print(json.dumps(rec, indent=1, sort_keys=True))
            return 0
        rc, text = replay(m, d, rec)
        print(text)
        return rc
    if a.resolve:
        if m.resolve(a.resolve, d, a.note):
            print("hook_incident: %s enabled again" % a.resolve)
            return 0
        print("hook_incident: %s is not disabled" % a.resolve)
        return 1
    if not a.hook or not a.why:
        ap.error("<hook> --why WHY, --show ID, --replay ID, --resolve HOOK or --list")
    if not os.path.isfile(os.path.join(d, a.hook.split()[0])):
        ap.error("no hook %s in %s" % (a.hook.split()[0], d))
    last = m.last_refusal(a.hook, hooks_dir=d) or {}
    rec = m.disable(a.hook, a.why, d, session=a.session or last.get("session"), last_refusal=last.get("msg"),
                    project_dir=os.environ.get("CLAUDE_PROJECT_DIR") or ROOT, inp=last.get("input"),
                    count=last.get("count"))
    if rec:
        print("hook_incident: %s disabled; incident %s, an ALERT for the main session is open (--replay %s tests a "
              "fix on its last refusal, --resolve to enable it again)" % (a.hook, rec["id"], rec["id"]))
    else:
        print("hook_incident: %s was already disabled; nothing added" % a.hook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
