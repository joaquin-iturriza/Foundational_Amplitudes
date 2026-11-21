"""plan sweepends reports a step's sweeps only once every sweep its wait names has ended."""
import contextlib, io, json, sys, types
sys.path.insert(0, "tools")
import plan


def rows(ended_names, monkeypatch, sw=None, wait="sweeps s:a s:b"):
    """ended_names: names (site s) or (site, name) pairs whose sweep ended."""
    sw = sw or [{"site": "s", "name": n, "at": None} for n in ("a", "b")]
    ended = [(n if isinstance(n, tuple) else ("s", n)) for n in ended_names]
    e = {"id": "e", "step_waits": {"2": wait}}
    monkeypatch.setattr(plan, "load", lambda: ({"entries": [e]}, {}))
    monkeypatch.setattr(plan, "sweeps_of", lambda st, eid: sw)
    monkeypatch.setattr(plan, "want_of", lambda st, ids: None)
    monkeypatch.setattr(plan, "site_sweeps", lambda *a, **k: {})
    monkeypatch.setattr(plan, "sweep_label", lambda n, s, f: "%s (%s, 8 of 8 trials submitted, 8 results)" % (n, s))
    monkeypatch.setattr(plan, "sweep_status", lambda *a: ([("%s (%s, 8 of 8 trials submitted, 8 results)" % (n, s), None, None)
                                                         for s, n in ended], [], []))
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


def test_same_name_live_on_another_site_keeps_the_wait(monkeypatch):
    """A search continued on lxplus under its CC name: the ended CC copy does not close a wait on the lxplus one."""
    sw = [{"site": "cc", "name": "a", "at": None}, {"site": "lx", "name": "a", "at": None}]
    assert all(not r["steps"] for r in rows([("cc", "a")], monkeypatch, sw, "sweeps lx:a"))
    r = rows([("cc", "a"), ("lx", "a")], monkeypatch, sw, "sweeps lx:a")
    assert [x["steps"] for x in r if x["site"] == "lx"] == [["2"]]
