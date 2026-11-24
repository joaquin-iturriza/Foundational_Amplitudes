"""D399 (1): plan_check counts an event's routing to an agent, not a bare `site ack` by the main session.

    .venv-cpu/bin/python tests/test_plan_check_event_routing.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, ".claude", "hooks")]
import plan_check

KINDS = {"E1": "RUN", "E2": "RUN", "E3": "ALERT", "E4": "GAP"}


def transcript(tools):
    recs = [{"type": "user", "uuid": "u1", "message": {"content": "go"}}]
    for name, inp in tools:
        recs.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": name,
                                                                   "input": inp}]}})
    recs.append({"type": "assistant", "message": {"content": [{"type": "text", "text": "Joaquin, done."}]}})
    p = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    open(p, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
    return {"transcript_path": p}


def test_bare_ack_of_log_events_refused():
    got = plan_check.bare_acks(transcript([("Bash", {"command": "site ack E1 E2 E3 --note 'read the log: OOM'"})]), KINDS)
    assert len(got) == 1 and "E1 (RUN), E2 (RUN) acked by you" in got[0], got


def test_routed_ack_passes():
    t = transcript([("SendMessage", {"to": "a0123456789abcdef", "message": "E1 and E2: read their logs"}),
                    ("Bash", {"command": "cd ~/work/FA && site ack E1 E2 --note 'sent to its owner'"})])
    assert plan_check.bare_acks(t, KINDS) == []
    t = transcript([("Agent", {"subagent_type": "mechanical-worker", "prompt": "Read the log of E4 and ack it"}),
                    ("Bash", {"command": "site ack E4 --note x"})])
    assert plan_check.bare_acks(t, KINDS) == []


def test_other_kinds_and_no_ack_pass():
    assert plan_check.bare_acks(transcript([("Bash", {"command": "site ack E3 --note 'site up again'"})]), KINDS) == []
    assert plan_check.bare_acks(transcript([("Bash", {"command": "site events --project FA"})]), KINDS) == []
    assert plan_check.bare_acks(transcript([("Bash", {"command": "git commit -m 'site ack E1 later'"})]), KINDS) == []


def test_open_event_routing():
    out, notes = plan_check.route_events(["open event: E5 11-11 20:00 FA RUN - FA-x ended FAILED", "open event: E6 x",
                                          "open event: E7 11-11 20:00 FA READY kit 1 result",
                                          "open event: E8 11-11 20:00 FA ALERT - site down"], {"E6"})
    assert notes == ["E6 handed to an agent this turn (a note until it acks)"], notes
    assert "mechanical-worker whose brief names E5" in out[0] and "owner analyses" in out[1] \
        and "explain it and ack it" in out[2] and len(out) == 3, out


def test_routed_ids():
    t = transcript([("SendMessage", {"to": "a", "message": "E12, then E13"}), ("Agent", {"prompt": "fix E7"})])
    assert plan_check.routed_ids(plan_check._turn(t)[0]) == {"E12", "E13", "E7"}


def test_site_called_down_needs_a_retry():
    def say(tools, text):
        t = transcript(tools)
        recs = [json.loads(l) for l in open(t["transcript_path"])]
        recs[-1]["message"]["content"][0]["text"] = text
        open(t["transcript_path"], "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
        return plan_check.site_down(t)
    assert say([], "Joaquin, lxplus needs 2FA, reconnect it.")
    assert say([("Bash", {"command": "site run ccin2p3 FA -- hostname"})], "Joaquin, lxplus is down.")
    assert not say([("Bash", {"command": "site run lxplus FA -- hostname"})], "Joaquin, lxplus is down.")
    assert not say([("Bash", {"command": "site list"}), ("Bash", {"command": "site list"})], "Joaquin, lxplus is down.")
    assert not say([], "Joaquin, lxplus is up; jobs finished.")
    assert say([], "Joaquin, lxplus is down.")
    assert not say([], "lxplus should only be reported down after a real ssh retry")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
