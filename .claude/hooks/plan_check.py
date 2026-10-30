#!/usr/bin/env python3
"""Stop guard (setup redesign, D121): the one check that work is moving. It replaces the idle-work parts of
ledger_guard, plan_guard, continuous_guard and slurm_waiter_guard. The turn cannot end while
  - `plan next` lists a step not launched for a running entry (each of its independent next steps counted on its
    own): a run of it ended, or it has nothing in flight, or
    its latest design has had no verdict for 30 min with its step agent gone (the fallback: the step agent runs
    the design reviewer itself). Never demanded, a note for plan next: a step whose recorded step agent is still
    working (whenever it started), and a step the entry's open question holds (plan ask --blocks-steps, or the step
    numbers its --blocks text names). A refusal is only for what the turn can do (the user, 2026-10-29: "if it's not
    possible to do in any way, then it shouldn't be a hook, or at least not a hook that prevents the end of the
    turn");
  - never blocking: the user wrote after a question an entry waits on. Whether it answers is the main session's call
    through `plan next`, which lists it on every wake until plan answer or plan unanswered: a block put a non-answer on
    the user's screen at every message (the user, 2026-10-26: "Me not answering a question from you it's not you not
    answering a question from me, why would it need to retrigger you");
  - an entry has an open question (waiting, or running with the question blocking part of it) but no message sent to
    the user carries it (from 15 min before the ask on);
  - an entry is held by anything but a hold or forbid in force that names it;
  - a running entry has no approval in force naming it;
  - once, at one stop (checks of history): a step recorded in the state file is not a real step agent of its entry; a
    quote is not the words its decision recorded about that entry (covers as recorded), at that time, after what it
    answers; an answered question was asked again in a message since the previous stop (quoted text left out);
  - never blocking, a note: a done entry no longer passes what plan done checked (plan next suggests plan reopen);
  - numbers committed to docs/results.tex on trunk since the last verified commit fail tools/verify_claims.py (the
    gate on the numbers, a check of state; the git hooks only catch a problem earlier);
  - a site event is open;
  - never blocking, a note: every live entry waits on the user (waiting_user with holds_all, or running with an open question whose `blocks`
    names every next step) while no `site supervise` process for FA runs (/proc, as the turn ends): nothing would
    move until the user came back (2026-10-29, 7 hours idle after the supervisor was stopped);
  - a sweep on a site, live or just submitted, that no entry records and that fits several running entries (one that
    fits a single entry is recorded on it here, with a note: plan.reconcile_sweeps).
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


def supervisor_running():
    """A live `site supervise` process for FA (or for every project): /proc/<pid>/cmdline, argv split on NUL. Tests
    point PLAN_PROC at a fake /proc."""
    import plan, glob
    for f in glob.glob(os.path.join(plan.ENV.get("PLAN_PROC", "/proc"), "[0-9]*", "cmdline")):
        try:
            argv = [x for x in open(f, "rb").read().decode(errors="replace").split("\0") if x]
        except OSError:
            continue
        i = next((i for i, x in enumerate(argv) if os.path.basename(x) == "site"), None)
        if i is None or "supervise" not in argv[i + 1:]:
            continue
        rest = argv[i + 1:]
        proj = next((rest[j + 1] for j, x in enumerate(rest[:-1]) if x == "--project"), None) or \
            next((x.split("=", 1)[1] for x in rest if x.startswith("--project=")), None)
        if proj in (None, plan.PROJECT):
            return True
    return False


def waits_on_user(e):
    """The entry cannot move without the user: waiting_user (holds_all), or running with an open question whose
    `blocks` names every one of its next steps."""
    import plan
    if e.get("status") == "waiting_user":
        return True
    if e.get("status") != "running" or not plan.open_question(e) or not (e.get("blocks") or e.get("blocks_steps")):
        return False
    return set(plan.step_keys(e)) <= plan.blocked_steps(e)


def problems():
    """(blocks, once): blocks stand until fixed through the plan CLI or a commit; `once` are checks of history
    (a quote, a recorded step, an answered question asked again), reported at one stop and then not again."""
    import plan, time
    plan.DEADLINE[0] = time.monotonic() + float(plan.ENV.get(BUDGET_TEST, BUDGET_S))
    unrecorded = plan.reconcile_sweeps()              # first: a sweep it records on an entry counts below
    d, st = plan.load()
    msgs = plan.messages(plan.oldest_open(d, st))     # from the oldest open question, or the previous re-ask scan
    out, once = list(unrecorded), []
    if plan.owner_mode():                             # switch-over D: one owner per entry, no demand per step or run
        out += owner_problems(d, st, msgs)
    # each unlaunched next step counts on its own: an entry with independent next steps lists every free one
    for e, why, launched, k in ([] if plan.owner_mode() else plan.next_steps_keyed(d, st, msgs=msgs)):
        if launched or e.get("status") == "waiting_user":     # the user wrote after a question: plan next only
            continue
        if why.startswith("design to review"):
            out.append("%s: %s (plan next --prompts gives its prompt; subagent_type design-reviewer, then plan review "
                       "%s --agent <subagent id>)" % (e["id"], why, e["id"]))
            continue
        out.append("%s%s: %s%s" % (e["id"], "" if k is None else " step %d" % k, why,
                                   "; launch its step agent (plan next --prompts, then plan step %s --agent <subagent "
                                   "id>%s)" % (e["id"], "" if k is None else " --next %d" % k)
                                   if e.get("status") == "running" else ""))
    sent = [(ts, text) for ts, role, text in msgs if role == "assistant"]
    for e in d["entries"]:
        if plan.open_question(e):                   # waiting, or running with a question open on part of it
            t0 = plan.parse(e.get("waiting_since")) - dt.timedelta(minutes=plan.ASK_MARGIN_MIN)
            if not any(ts >= t0 and plan.overlap(e.get("ask"), text) for ts, text in sent):
                out.append("%s has a question open for the user, but no message sent to them asks: %s" % (
                    e["id"], e.get("ask")))
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
    live = [e for e in d["entries"] if e.get("status") in ("running", "waiting_user")]
    # Everything waits on the user: no turn can move it, and restarting the watch is not the turn's to do (it can go
    # against the user's stop, 2026-10-29). A note for plan next, never a refusal.
    if live and all(waits_on_user(e) for e in live) and not supervisor_running():
        plan.NOTES.append("every live entry waits on the user (%s) and `site supervise` is not running: nothing moves until "
                   "they answer. Start it in the background (`site supervise --project FA`), and take what does not "
                   "need the answer: a question a decision in force settles is withdrawn (plan withdraw), a part that "
                   "can move is a next step of its own (plan add-next)" % ", ".join(e["id"] for e in live))
    out += plan.claims_problems()
    # an event on an entry whose step agent is still working is that agent's to ack: blocking on it only bounces the
    # main session while the agent reads the logs (2026-10-26, eight failed regeneration parts)
    if plan.owner_mode():                             # a running entry's events are its owner's (owner_problems)
        working = {e["id"] for e in d["entries"] if e.get("status") == "running"}
        refuse, wait = split_events(open_events(), working)
        wait = []
    else:
        working = {e["id"] for e in d["entries"] if plan.step_working(st, e["id"], plan.now())}
        refuse, wait = split_events(open_events(), working)
    out += refuse
    plan.NOTES.extend(wait)
    return out, once, st


def owner_problems(d, st, msgs):
    """Switch-over D (agent redesign D6): a running entry with something to do and no live or stopped owner is a
    refusal (launch); an open event of an entry whose owner is stopped, neither resumed nor launched this turn, is a
    refusal (resume); a design without a verdict, as before. A working owner, an owner stopped while its jobs run and
    an owner past the context threshold are notes."""
    import plan
    out = []
    for e, act, agent, why in plan.owner_actions(d, st, msgs=msgs, events=plan.events_by_item(strict=True)):
        if act == "close":
            plan.NOTES.append("%s: close its owner %s: %s" % (e["id"], agent, why))
        elif act == "resume":
            out.append("%s: resume its owner %s (SendMessage): %s" % (e["id"], agent, why))
        elif act == "launch":
            out.append("%s: no live owner: %s" % (e["id"], why))
        else:
            out.append("%s: %s (plan next --prompts gives its prompt; subagent_type design-reviewer, then plan review "
                       "%s --agent <subagent id>)" % (e["id"], why, e["id"]))
    return out


def split_events(lines, working):
    """(refusals, notes) of the open events (lines of `site events`): an event of an entry whose step agent is
    launched and working is that agent's to ack, a note (waits on the agent); any other open event is the turn's."""
    refuse, wait = [], []
    for ev in lines:
        w = ev.split()
        if len(w) > 5 and w[5] in working:
            wait.append("event %s is the working step agent's of %s to ack (a note, never a refusal)" % (w[0], w[5]))
        else:
            refuse.append("open event: " + ev)
    return refuse, wait


def stop_conditions(inp, refused):
    """(refusals, notes) of the Stop conditions taken over from the standalone hooks (stop_conditions.py); each is
    skipped while its own hook is still registered for Stop (before switch-over B), so nothing is asked twice. The
    reply check (a model call) runs only when nothing else refuses the turn."""
    import plan
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stop_conditions as sc
    repo = plan.ENV.get("PLAN_ROOT") if plan.ENV.get("PLAN_TEST") == "1" else sc.REPO
    bad, notes = [], []
    parts = [("figure_pair_guard.sh", lambda: sc.figure_pairs(repo)),
             ("context_guard.py", lambda: sc.context(inp, repo)),
             (None, lambda: sc.ledger(repo, plan.SITE) if (plan.ENV.get("PLAN_TEST") == "1" or sc.ledger_deferred())
              else ([], [])),                       # LEDGER_MARK: ledger: site guard --stop
             (None, lambda: sc.opener(inp, OPENER_STATE)),
             (None, lambda: sc.incidents(os.path.join(repo, ".claude", "hooks")))]
    for hook, f in parts:
        if hook and sc.registered_at_stop(hook, repo):
            continue
        b, n = f()
        bad += b
        notes += n
    if not refused and not bad and not sc.registered_at_stop("reply_check.py", repo):
        b, n = sc.reply(inp)
        bad += b
    return bad, notes


OPENER_STATE = {}


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
        try:
            inp = json.loads(sys.stdin.read() or "{}")
        except ValueError:
            inp = {}
        inp = inp if isinstance(inp, dict) else {}
        bad, once, st = problems()
        import plan
        if os.path.exists(plan.PLAN_FILE):
            bad += report_once(st, once)
        OPENER_STATE.update(st.get("_opener") or {})
        more, notes = stop_conditions(inp, bad)
        bad += more
        plan.NOTES.extend(notes)
        if OPENER_STATE:
            plan._set_state("_opener", dict(OPENER_STATE))
    except Exception as ex:                       # fail closed: a guard that errors is a guard that is off
        sys.stderr.write("plan check failed (%r); fix it before ending the turn\n" % ex)
        return 2
    import plan
    if bad:
        if plan.NOTES:                            # notes go to plan next, never to the user's screen with a block
            plan._set_state("_stop_notes", list(plan.NOTES))
        import re
        def short(b):
            m = re.search(r"runs ended: ((?:FA-\S+, )+FA-\S+?);", b)
            if m and m.group(1).count(",") >= 3:
                runs = m.group(1).split(", ")
                b = b.replace(m.group(1), ", ".join(runs[:3]) + ", and %d more" % (len(runs) - 3))
            return b if len(b) <= 400 else b[:400] + "..."
        ev = [b for b in bad if b.startswith("open event:")]
        shown = [short(b) for b in bad if not b.startswith("open event:")] + ev[:5]
        if len(ev) > 5:
            shown.append("open event: ... and %d more (site events --project FA)" % (len(ev) - 5))
        sys.stderr.write("PLAN CHECK: the turn cannot end yet.\n" + "\n".join("  " + b for b in shown) + "\n")
        return 2
    # Notes never reach the user: a passing check prints nothing (a systemMessage is shown on their screen; the user,
    # 2026-10-25 and 2026-10-26: "Why are you back again with this?"). They are kept for `plan next`, which prints them
    # to the main session once.
    if plan.NOTES:
        plan._set_state("_stop_notes", list(plan.NOTES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
