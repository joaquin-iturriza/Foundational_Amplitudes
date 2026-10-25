#!/usr/bin/env python3
"""Stop guard (setup redesign, D121): the one check that work is moving. It replaces the idle-work parts of
ledger_guard, plan_guard, continuous_guard and slurm_waiter_guard. The turn cannot end while
  - `plan next` lists a step not launched: a run of an entry ended, an entry has nothing in flight, or the user wrote
    after a question (their answer has not reached its entry);
  - an entry waits on the user but no message sent to them carries its question (from 15 min before the ask on);
  - an entry is held by anything but a hold or forbid in force that names it;
  - a running entry has no approval in force naming it;
  - once, at one stop (checks of history): a step recorded in the state file is not a real step agent of its entry; a
    quote is not the words its decision recorded about that entry (covers as recorded), at that time, after what it
    answers; an answered question was asked again in a message since the previous stop (quoted text left out);
  - never blocking, a note: a done entry no longer passes what plan done checked (plan next suggests plan reopen);
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


BUDGET_S = 270            # all checks that reach out end within this; the hook is registered with 300 s
BUDGET_TEST = "PLAN_BUDGET"   # a shorter budget, honoured only under PLAN_TEST=1 (tests)
QUOTED = re.compile(r"\u201c[^\u201d]*\u201d|\"[^\"\n]*\"|`[^`\n]*`|^\s*>.*$", re.M)


def own_questions(text):
    """The sentences a message itself asks: quoted text (a question cited in a status message) is left out."""
    return [q.strip() for q in re.findall(r"[^.!?\n]*[.!?]?", QUOTED.sub(" ", text)) if q.strip().endswith("?")]


def problems():
    """(blocks, once): blocks stand until fixed through the plan CLI or a commit; `once` are checks of history
    (a quote, a recorded step, an answered question asked again), reported at one stop and then not again."""
    import plan, time
    plan.DEADLINE[0] = time.monotonic() + float(plan.ENV.get(BUDGET_TEST, BUDGET_S))
    d, st = plan.load()
    msgs = plan.messages(plan.oldest_open(d, st))     # from the oldest open question, or the previous re-ask scan
    out, once = [], []
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
                once.append("%s: a quote on it is not the user's recorded words: %s" % (e["id"], why))
        if e.get("status") == "held":
            dd = plan.decision(e.get("hold_decision"))
            if not dd or not dd["in_force"] or dd["kind"] not in ("hold", "forbid") or not plan.covers_name(dd, e["id"]):
                out.append("%s is held by %s, which is not a hold in force that names it" % (
                    e["id"], e.get("hold_decision")))
    mark = plan.asks_mark(st)                         # only messages since the previous scan
    for e, x in plan.answered_asks(d):
        t1 = max(plan.parse(x["answered_at"]), mark)
        again = [ts for ts, text in sent if ts > t1 and any(plan.overlap(x["ask"], q) for q in own_questions(text))]
        if again:
            once.append("a question answered on %s (%s, %s) was asked again at %s" % (
                x["answered_at"][:16], e["id"], x.get("decision"), again[-1].isoformat()[:16]))
    once += ["a step recorded on %s is not valid (%s); it does not count, record steps only through plan step" % (
        eid, why) for eid, stp, why in plan.bad_steps(st)]
    for e, why in plan.done_problems(d, st):
        plan.NOTES.append("%s is marked done but %s: plan reopen %s --why ..., or fix it (see plan next)" % (
            e["id"], why, e["id"]))
    out += plan.claims_problems()
    out += ["open event: " + ev for ev in open_events()]
    return out, once, st


def report_once(st, once):
    """The `once` messages not reported at an earlier stop; records them and the end of this re-ask scan."""
    import plan
    seen = list(st.get("_reported") or [])
    new = [m for m in once if m not in seen]
    plan._set_state("_reported", (seen + new)[-500:])
    plan._set_state("_asks_scanned", plan.now().isoformat())
    return new


def main():
    try:
        sys.stdin.read()
        bad, once, st = problems()
        import plan
        if os.path.exists(plan.PLAN_FILE):
            bad += report_once(st, once)
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
