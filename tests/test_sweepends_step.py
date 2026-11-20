"""plan sweepends reports a step's sweeps only once every sweep its wait names has ended."""
import contextlib, io, json, sys, types
sys.path.insert(0, "tools")
import plan


def rows(ended_names, monkeypatch):
    sw = [{"site": "s", "name": n, "at": None} for n in ("a", "b")]
    e = {"id": "e", "step_waits": {"2": "sweeps s:a s:b"}}
    monkeypatch.setattr(plan, "load", lambda: ({"entries": [e]}, {}))
    monkeypatch.setattr(plan, "sweeps_of", lambda st, eid: sw)
    monkeypatch.setattr(plan, "want_of", lambda st, ids: None)
    monkeypatch.setattr(plan, "site_sweeps", lambda *a, **k: {})
    monkeypatch.setattr(plan, "sweep_label", lambda n, s, f: "%s (x)" % n)
    monkeypatch.setattr(plan, "sweep_status", lambda *a: ([("%s (x)" % n, None, None) for n in ended_names], [], []))
    monkeypatch.setattr(plan, "owner_of", lambda e: None)
    monkeypatch.setattr(plan, "owner_state", lambda e: None)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        plan.cmd_sweepends(types.SimpleNamespace(read=None))
    return json.loads(out.getvalue())


def test_one_of_two_ended_no_step(monkeypatch):
    assert all(not r["steps"] for r in rows(["a"], monkeypatch))


def test_both_ended_step_lists_both(monkeypatch):
    r = rows(["a", "b"], monkeypatch)
    assert [x["steps"] for x in r] == [["2"], ["2"]] and r[0]["step_sweeps"] == {"2": [["s", "a"], ["s", "b"]]}
