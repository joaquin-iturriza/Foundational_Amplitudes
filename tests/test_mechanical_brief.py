#!/usr/bin/env python3
"""brief_guard on mechanical-worker briefs (D213): `plan brief <entry> --mechanical "<step>"` builds a brief with the
short mechanical rules block of plan/mechanical_prompt.txt, which brief_guard accepts for mechanical-worker only.

  K1  mechanical-worker with the --mechanical brief: passes
  K2  mechanical-worker with no rules block: refused
  K3  mechanical-worker whose step cuts the work ("only 2 of the jobs"): refused
  K4  compact-worker with the mechanical block only: refused (the owner block is still required)
  K5  mechanical-worker launched with model opus: refused (the definition's model check is unchanged)
  K6  mechanical-worker with an owner's full brief: passes

Run:  python3 tests/test_mechanical_brief.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.path.join(ROOT, ".claude", "hooks", "brief_guard.py")


def plan(*a):
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plan.py")] + list(a), capture_output=True,
                          text=True, check=True, cwd=ROOT).stdout


def guard(prompt, typ="mechanical-worker", model=None):
    ti = {"subagent_type": typ, "prompt": prompt, "description": "step"}
    if model:
        ti["model"] = model
    r = subprocess.run([sys.executable, GUARD], input=json.dumps({"tool_name": "Agent", "tool_input": ti}),
                       capture_output=True, text=True, env=dict(os.environ, CLAUDE_PROJECT_DIR=ROOT))
    return r.returncode, r.stderr


def test_all():
    mech = plan("brief", "agent_redesign", "--mechanical", "Resubmit run FA-ccin2p3-0000000a with --hdr=--time=02:00:00.")
    assert "=== MECHANICAL RULES" in mech and "--item agent_redesign" in mech, mech[:300]
    rc, err = guard(mech)
    assert rc == 0, err
    print("K1 ok: the --mechanical brief passes")
    rc, err = guard("Resubmit run FA-ccin2p3-0000000a.")
    assert rc == 2 and "mechanical rules block" in err, err
    print("K2 ok: no block, refused")
    rc, err = guard(plan("brief", "agent_redesign", "--mechanical", "Resubmit only 2 of the jobs."))
    assert rc == 2 and "cuts the work" in err, err
    print("K3 ok: a cut in the step, refused")
    rc, err = guard(mech, typ="compact-worker")
    assert rc == 2 and "lacks the rules block" in err, err
    print("K4 ok: compact-worker still needs the owner block")
    rc, err = guard(mech, model="opus")
    assert rc == 2 and "differs from" in err, err
    print("K5 ok: model opus on mechanical-worker, refused")
    rc, err = guard(plan("brief", "agent_redesign"))
    assert rc == 0, err
    print("K6 ok: an owner's full brief also passes")


if __name__ == "__main__":
    test_all()
    print("all ok")
