"""drive.py first trial per site: a search's first trial on a site goes alone; the rest wait for TRAINING_STARTED in its
log; a first trial that ends without it closes that site to the search for this driver run, with an ALERT.
The case: 2026-11-12, 42-trial waves sent whole failed on a stale checkout (~11:00) and a stale Born record (~16:00)."""
import sys, os, types
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import drive


class FakeSweep(drive.Sweep):
    def __init__(self, want=3):
        self.name, self.hours, self.n_trials, self.n_startup, self.waves = "s", 1.0, 10, 3, None
        self.sampler = types.SimpleNamespace(surrogate="x", objective_scale=1)
        self.st, self.commit, self.gate, self.want, self.sent = {"trials": {}, "prebuild": {}}, "c0", {}, want, []

    def save(self):
        pass

    def collect(self, a):
        pass

    def may_issue(self, parallel):
        return self.want

    def data_ready(self, s, a):
        return True

    def submit(self, s, a, t0=None):
        rid = "FA-%s-%d" % (s, len(self.sent))
        self.sent.append(s)
        self.st["trials"][str(len(self.sent))] = dict(idx=len(self.sent), rid=rid, site=s, status="submitted")
        return rid


def _run(monkeypatch, sw, order, logs="", sites=("ccin2p3",)):
    calls = []

    def fake_site(*a, **k):
        calls.append(a)
        return logs if a[0] == "logs" else ""
    monkeypatch.setattr(drive, "site", fake_site)
    monkeypatch.setattr(drive, "Sweep", lambda c, a: sw)
    monkeypatch.setattr(drive, "plan", lambda k, sweeps, a, record=True: (list(order), []))
    monkeypatch.setattr(drive.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    monkeypatch.setattr(sys, "argv", ["drive.py", "--config", "x.yaml", "--item", "i", "--once", "--sites"] + list(sites))
    drive.main()
    return calls


def test_first_trial_goes_alone_then_rest_after_marker(monkeypatch):
    sw = FakeSweep()
    _run(monkeypatch, sw, ["ccin2p3"] * 3)
    assert sw.sent == ["ccin2p3"] and sw.gate["ccin2p3"]["state"] == "probe"
    # next pass: still queued -> held, logs not read
    calls = _run(monkeypatch, sw, ["ccin2p3"] * 3)
    assert sw.sent == ["ccin2p3"] and not [c for c in calls if c[0] == "logs"]
    # running, log past data load and model build -> the rest go
    sw.st["trials"]["1"]["sched_state"] = "RUNNING"
    _run(monkeypatch, sw, ["ccin2p3"] * 3, logs="init_data ...\nTRAINING_STARTED\nstep 1")
    assert sw.gate["ccin2p3"]["state"] == "ok" and sw.sent == ["ccin2p3"] * 4


def test_running_without_marker_keeps_holding(monkeypatch):
    sw = FakeSweep()
    _run(monkeypatch, sw, ["ccin2p3"] * 3)
    sw.st["trials"]["1"]["sched_state"] = "RUNNING"
    _run(monkeypatch, sw, ["ccin2p3"] * 3, logs="Loading pools ...")
    assert sw.sent == ["ccin2p3"] and sw.gate["ccin2p3"]["state"] == "probe"


def test_failed_before_training_closes_site_with_alert(monkeypatch):
    # the 16:00 case: the data load assertion on target_over_born, on one of two sites
    sw = FakeSweep()
    _run(monkeypatch, sw, ["ccin2p3", "lxplus", "ccin2p3"], sites=("ccin2p3", "lxplus"))
    assert sw.sent == ["ccin2p3", "lxplus"]
    t = sw.st["trials"]["1"]
    t.update(status="failed", sched_state="FAILED", cause="AssertionError: target_over_born: Born record mismatch")
    calls = _run(monkeypatch, sw, ["ccin2p3", "ccin2p3", "lxplus"], logs="Traceback ...\nAssertionError: ...",
                 sites=("ccin2p3", "lxplus"))
    assert sw.gate["ccin2p3"]["state"] == "failed"
    ev = [c for c in calls if c[0] == "event"]
    assert len(ev) == 1 and "target_over_born" in ev[0][ev[0].index("--msg") + 1]
    assert sw.sent == ["ccin2p3", "lxplus"]               # nothing more to ccin2p3; lxplus still probing
    assert "failed" not in str(sw.st["gate"])            # a restart after a data fix tries again


def test_failed_on_every_site_stops(monkeypatch):
    sw = FakeSweep()
    _run(monkeypatch, sw, ["ccin2p3"] * 3)
    sw.st["trials"]["1"].update(status="lost", sched_state="FAILED", error="no RESULT_JSON")
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, sw, ["ccin2p3"] * 3, logs="FileNotFoundError: sweep/sweep_config_x.yaml")
    assert "failed before training on every site" in str(e.value)


def test_diverged_after_training_is_ok(monkeypatch):
    sw = FakeSweep()
    _run(monkeypatch, sw, ["ccin2p3"] * 3)
    sw.st["trials"]["1"].update(status="failed", sched_state="COMPLETED", error="diverged")
    _run(monkeypatch, sw, ["ccin2p3"] * 3, logs="TRAINING_STARTED\n... NaN, diverged")
    assert sw.gate["ccin2p3"]["state"] == "ok" and len(sw.sent) == 4


def test_repick_never_moves_a_probe_nor_into_an_unproven_site():
    moved = []
    sw = FakeSweep()
    sw.gate = {"ccin2p3": {"state": "probe", "rid": "FA-ccin2p3-0"}}
    sw.st["trials"] = {"0": dict(idx=0, rid="FA-ccin2p3-0", site="ccin2p3", status="submitted", sched_state="PENDING"),
                       "1": dict(idx=1, rid="FA-ccin2p3-1", site="ccin2p3", status="submitted", sched_state="PENDING")}
    sw.move = lambda t, dest, a: moved.append((t["rid"], dest)) or True
    row = lambda s, **kw: dict(site=s, state="UP", wait_now=kw.get("w", 0) * 3600, slots_now=kw.get("slots", 0))
    rows = [row("ccin2p3", w=10), row("lxplus", slots=5)]
    drive.repick([sw], None, rows, now=1e9)
    assert moved == []                                   # lxplus not proven for this search
    sw.gate["lxplus"] = {"state": "ok"}
    drive.repick([sw], None, rows, now=1e9)
    assert moved == [("FA-ccin2p3-1", "lxplus")]         # the probe itself stays


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
