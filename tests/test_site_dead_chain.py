"""The supervisor's dead dependency chain: a job PENDING behind a smoke that FAILED (2026-11-12, FA-ccin2p3-4ab99f50 and
three more waited from 05:45) is found from squeue's "%r|%E" line and the ledger's state of its parent. The code under
test is ~/.local/bin/site (_dead_dependency, _dead_chain).

    python3 tests/test_site_dead_chain.py
"""
import _tmpdir  # noqa: F401
import sqlite3, sys, tempfile
import test_run_done_waiter as T

m = T.load_site(tempfile.mkdtemp(prefix="deadchain_"))
SC = {"scheduler": "slurm"}


def chain(squeue_line, parent_state):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE runs (id TEXT, site TEXT, sched_id TEXT, state TEXT)")
    if parent_state:
        c.execute("INSERT INTO runs VALUES ('FA-ccin2p3-smoke','ccin2p3','111',?)", (parent_state,))
    m.run_ssh = lambda sc, inner, timeout=0: (0, squeue_line + "\n")
    return m._dead_chain(c, SC, "ccin2p3", "222")


def test_failed_parent():
    assert "ended FAILED" in chain("Dependency|afterok:111(unfulfilled)", "FAILED")
    assert "ended CANCELLED" in chain("Dependency|afterok:111(unfulfilled)", "CANCELLED")


def test_never_satisfied():
    assert "DependencyNeverSatisfied" in chain("DependencyNeverSatisfied|afterok:111(failed)", None)


def test_live_chain_is_not_dead():
    assert chain("Dependency|afterok:111(unfulfilled)", "RUNNING") is None
    assert chain("Dependency|afterok:111(unfulfilled)", "PENDING") is None
    assert chain("Resources|", None) is None


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
