#!/usr/bin/env python3
"""plan.user_said finds the user's own messages that may already answer a question, and plan ask / plan add --ask
refuse until --searched names each by its time. Tonight's case (2026-11-10): the sweep template's fidelity question was
asked again although Joaquin had written on 2026-10-23 'I never asked to run a multifidelity sweep'.

Run:  python3 tests/test_plan_user_said.py   (pytest also collects it)"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402

ASK = ("Scope: the sweep template's fidelity_schedule.t_steps is [1000, 10000, 40000] (multi-fidelity) while the HPO "
       "rule says single-fidelity, one value: which holds, the rule (cut the template to one length) or the template "
       "(change the rule)?")
SAID = ("Again so many things that are baffling. What do you mean \"Sweeps run by drive.py from the laptop\" which "
        "laptop? What do you mean by \"multi fidelity sweeps\", as far as I know, I never asked to run a multifidelity "
        "sweep")
FILLER = ["Please check the run on the cluster and tell me the loss of process %d at the best checkpoint." % i
          for i in range(150)]


def user(ts, text):
    return {"type": "user", "timestamp": ts, "entrypoint": "cli", "message": {"role": "user", "content": text}}


def world(msgs):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "main.jsonl"), "w") as f:
        for m in msgs:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")   # site._interactive reads "entrypoint":"cli"
    old = plan.ENV.get("PLAN_TRANSCRIPTS")
    plan.ENV["PLAN_TRANSCRIPTS"] = d
    return d, old


def restore(d, old):
    shutil.rmtree(d)
    if old is None:
        plan.ENV.pop("PLAN_TRANSCRIPTS", None)
    else:
        plan.ENV["PLAN_TRANSCRIPTS"] = old


def corpus():
    msgs = [user("2026-10-20T10:%02d:00Z" % (i % 60), t) for i, t in enumerate(FILLER)]
    msgs.append(user("2026-10-23T02:12:30Z", SAID))
    msgs.append({"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-10-23T02:12:20Z",
                 "content": SAID})                       # the same message, queued mid-turn first
    return msgs


def refused(searched):
    try:
        plan._said_ok(ASK, searched)
    except SystemExit as e:
        return str(e)
    return None


def test_finds_the_earlier_answer():
    d, old = world(corpus())
    try:
        hits = plan.user_said(ASK)
        assert len(hits) == 1, hits                       # the queued copy is the same message
        ts, text, terms = hits[0]
        assert plan._said_stamp(ts) == "2026-10-23T02:12" and text == SAID
        assert {"fidelity", "multi"} <= set(terms)
    finally:
        restore(d, old)


def test_ask_refused_until_named():
    d, old = world(corpus())
    try:
        why = refused("CLAUDE.md HPO rule, fa-sweeps-and-hpo rule 6, D241; none settles which side is right")
        assert why and "2026-10-23T02:12" in why          # tonight's --searched: refused, the message shown
        assert refused("his 2026-10-23T02:12 message says he never asked for multi-fidelity; it does not name a length") is None
        assert refused("his message of 2026-10-23 02:12 ...") is None
    finally:
        restore(d, old)


def test_unrelated_question_passes():
    d, old = world(corpus())
    try:
        q = "Should the warm start grow to width 16 at k=533 or stay at 8?"
        assert plan.user_said(q) == []
        plan._said_ok(q, "anything")                       # no SystemExit
    finally:
        restore(d, old)


if __name__ == "__main__":
    test_finds_the_earlier_answer()
    test_ask_refused_until_named()
    test_unrelated_question_passes()
    print("ok")
