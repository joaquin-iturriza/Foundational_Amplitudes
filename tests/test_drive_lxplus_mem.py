"""drive.lxplus_mem_ok: lxplus gives 3000 MB per CPU, so a request above 3000*cpus MB is refused there."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.drive import lxplus_mem_ok  # noqa: E402


def test_lxplus_mem_ok():
    assert not lxplus_mem_ok("32G", 4)
    assert lxplus_mem_ok("12G", 8)
    assert lxplus_mem_ok("6000M", 2)
    assert not lxplus_mem_ok("6001", 2)
    assert not lxplus_mem_ok("lots", 4)


def _plan(monkeypatch, plan, lx_fit, mem="20G"):
    import json, types
    import sweep.drive as dr
    monkeypatch.setattr(dr, "site", lambda *a, **k: json.dumps({"plan": plan, "rows": [
        {"site": "lxplus", "slots_now": lx_fit}, {"site": "ccin2p3", "slots_now": 9}]}))
    monkeypatch.setattr(dr, "log", lambda *a: None)
    s0 = types.SimpleNamespace(mem=mem, cpus=4, hours=14.6, gpu_min_mem="32G")
    a = types.SimpleNamespace(sites=["lxplus", "ccin2p3"], allow_jeanzay=False)
    return dr.plan(3, [s0], a)[0]


def test_raised_request_keeps_lxplus_up_to_its_fitting_slots(monkeypatch):
    assert sorted(_plan(monkeypatch, {"lxplus": 2, "ccin2p3": 1}, 1)) == ["ccin2p3", "ccin2p3", "lxplus"]
    assert sorted(_plan(monkeypatch, {"lxplus": 2, "ccin2p3": 1}, 5)) == ["ccin2p3", "lxplus", "lxplus"]
    assert _plan(monkeypatch, {"lxplus": 3}, 1) == ["lxplus"]          # no other site: the rest waits a round
    assert sorted(_plan(monkeypatch, {"lxplus": 2, "ccin2p3": 1}, 0, mem="12000M")) == ["ccin2p3", "lxplus", "lxplus"]  # not raised: pick's plan as is


def test_sync_site_retries_a_transient_failure(monkeypatch):
    import sweep.drive as dr
    n = []
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)

    def flaky(*a, **k):
        n.append(1)
        if len(n) < 3:
            raise RuntimeError("ssh")
        return "ok"
    monkeypatch.setattr(dr, "site", flaky)
    assert dr.sync_site("lxplus") == "ok" and len(n) == 3


if __name__ == "__main__":
    test_lxplus_mem_ok()
    print("ok")
