#!/usr/bin/env python3
"""Stop guard (setup redesign, D121): the one check that work is moving. It replaces the idle-work parts of
ledger_guard, plan_guard, continuous_guard and slurm_waiter_guard. The turn cannot end while
  - `plan next` lists a step not launched: a run of an entry ended, an entry has nothing in flight, or the user wrote
    after a question (their answer has not reached its entry);
  - an entry waits on the user but no message sent to them carries its question (from 15 min before the ask on);
  - a question already answered is asked again in a message sent after the answer;
  - an entry is held by anything but a hold or forbid in force that names it;
  - a step recorded in the state file is not a real step agent of its entry (re-validated on every read); a quote is
    not the words its decision recorded about that entry, at that time, after what it answers; a running entry has
    no approval in force naming it; a done entry no longer passes what plan done checked (rerun daily);
  - numbers committed to docs/results.tex on trunk since the last verified commit fail tools/verify_claims.py (the
    gate on the numbers, a check of state; the git hooks only catch a problem earlier);
  - a site event is open.
An item check that cannot reach its site counts as in flight and is reported as a note, never as a block.
It fails closed: an error in the check blocks too. Env for tests: see tools/plan.py, and PLAN_EVENTS_CMD."""
import datetime as dt, json, os, re, shlex, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools"))


def open_events():
    import plan
    r = subprocess.run(shlex.split(plan.ENV.get("PLAN_EVENTS_CMD", "site events --project FA")),
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError("site events failed: " + (r.stderr or r.stdout)[-300:])
    return [l.strip() for l in r.stdout.splitlines() if re.match(r"\s*E\d+\s", l)]


def problems():
    import plan
    d, st = plan.load()
    msgs = plan.messages(plan.oldest_open(d))          # only from the oldest open question or answer on
    out = []
    for e, why, launched in plan.next_steps(d, st, msgs=msgs):
        if launched:
            continue
        out.append("%s: %s%s" % (e["id"], why, "; launch its step agent (plan next --prompts, then plan step %s "
                                 "--agent <subagent id>)" % e["id"] if e.get("status") == "running" else ""))
    sent = [(ts, text) for ts, role, text in msgs if role == "assistant"]
    for e in d["entries"]:
        if e.get("status") == "waiting_user":
            t0 = plan.parse(e.get("waiting_since")) - dt.timedelta(minutes=plan.ASK_MARGIN_MIN)
            if not any(ts >= t0 and plan.overlap(e.get("ask"), text) for ts, text in sent):
                out.append("%s waits on the user, but no message sent to them asks: %s" % (e["id"], e.get("ask")))
        if e.get("status") == "running":
            why = plan.approval_problem(e)
            if why:
                out.append("%s runs, but %s" % (e["id"], why))
        for q in e.get("quotes") or []:
            why = plan.quote_problem(q, e["id"])
            if why:
                out.append("%s: a quote on it is not the user's recorded words: %s" % (e["id"], why))
        if e.get("status") == "held":
            dd = plan.decision(e.get("hold_decision"))
            if not dd or not dd["in_force"] or dd["kind"] not in ("hold", "forbid") or not plan.covers_name(dd, e["id"]):
                out.append("%s is held by %s, which is not a hold in force that names it" % (
                    e["id"], e.get("hold_decision")))
    for e, x in plan.answered_asks(d):
        t1 = plan.parse(x["answered_at"])
        again = [ts for ts, text in sent if ts > t1 and any(
            q.strip().endswith("?") and plan.overlap(x["ask"], q) for q in re.findall(r"[^.!?\n]*[.!?]?", text))]
        if again:
            out.append("a question answered on %s (%s, %s) was asked again at %s" % (
                x["answered_at"][:16], e["id"], x.get("decision"), again[-1].isoformat()[:16]))
    out += ["a step recorded on %s is not valid (%s); record steps only through plan step" % (eid, why)
            for eid, stp, why in plan.bad_steps(st)]
    out += plan.done_problems(d, st)
    out += plan.claims_problems()
    out += ["open event: " + ev for ev in open_events()]
    return out


def main():
    try:
        sys.stdin.read()
        bad = problems()
    except Exception as ex:                       # fail closed: a guard that errors is a guard that is off
        sys.stderr.write("plan check failed (%r); fix it before ending the turn\n" % ex)
        return 2
    import plan
    if bad:
        sys.stderr.write("PLAN CHECK: the turn cannot end yet.\n" + "\n".join("  " + b for b in bad + [
            "note: " + n for n in plan.NOTES]) + "\n")
        return 2
    if plan.NOTES:                                # seen, never blocking: an item check that could not reach its site
        print(json.dumps({"systemMessage": "plan check notes:\n" + "\n".join("  " + n for n in plan.NOTES)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
