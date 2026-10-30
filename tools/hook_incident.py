#!/usr/bin/env python3
"""Disable a hook that misbehaves, with the reason on record (agent redesign, step A2).

    python3 tools/hook_incident.py <hook> --why "<why>" [--user]   disable it: the dispatcher skips it from now on
    python3 tools/hook_incident.py --resolve <hook> [--note N] [--user]   enable it again
    python3 tools/hook_incident.py --list [--user]                 disabled hooks and the incident log

<hook> is the name the registration gives the dispatcher (plan_check.py, or with its arguments, "review_backlog.sh
check"). Disabling writes it to .claude/hooks/disabled.json, appends {hook, why, time, session, last refusal} to
.claude/hooks/incidents.jsonl and opens one ALERT event for the project (`site event`); a hook already disabled adds
nothing. --user acts on the user-level hooks (~/.claude/hooks). The logic is the dispatcher's own (dispatch.py), so the
loop detection and this command write the same record."""
import argparse, importlib.util, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dispatch(user, hooks_dir=None):
    d = hooks_dir or (os.path.expanduser("~/.claude/hooks") if user else os.path.join(ROOT, ".claude", "hooks"))
    spec = importlib.util.spec_from_file_location("dispatch", os.path.join(d, "dispatch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hook", nargs="?")
    ap.add_argument("--why")
    ap.add_argument("--resolve", metavar="HOOK")
    ap.add_argument("--note")
    ap.add_argument("--list", action="store_true")
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
            print("  " + json.dumps({k: r[k] for k in ("hook", "time", "resolved", "why", "note", "event") if r.get(k)}))
        return 0
    if a.resolve:
        if m.resolve(a.resolve, d, a.note):
            print("hook_incident: %s enabled again" % a.resolve)
            return 0
        print("hook_incident: %s is not disabled" % a.resolve)
        return 1
    if not a.hook or not a.why:
        ap.error("<hook> --why WHY, --resolve HOOK or --list")
    if not os.path.isfile(os.path.join(d, a.hook.split()[0])):
        ap.error("no hook %s in %s" % (a.hook.split()[0], d))
    if m.disable(a.hook, a.why, d, session=a.session, project_dir=os.environ.get("CLAUDE_PROJECT_DIR") or ROOT):
        print("hook_incident: %s disabled; an ALERT is open (--resolve to enable it again)" % a.hook)
    else:
        print("hook_incident: %s was already disabled; nothing added" % a.hook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
