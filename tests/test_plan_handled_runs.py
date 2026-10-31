"""plan.acked_runs: a run is handled when an acked event of any kind names it (message or note, full id or its 8-hex
suffix), when the registry has it CANCELLED, or when a finding or dead end of the entry records its cancel
(2026-10-31: catalog_coverage's five cancelled runs, acked in one combined event E3164, were demanded again)."""
import os, sqlite3, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(dbpath):
    os.environ.update(PLAN_RUNS_DB=dbpath, PLAN_TEST="1", PLAN_ROOT=os.path.dirname(dbpath))
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib, plan
    return importlib.reload(plan)


def test_handled():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "runs.db")
    c = sqlite3.connect(p)
    c.execute("create table events (id, ts, project, item, kind, key, msg, acked, ack_note, mute_until, tell)")
    c.execute("create table runs (id, state)")
    c.execute("insert into events values (3164, 0, 'FA', 'x', 'RUN', 'cancelled_cc_20261031', "
              "'runs cancelled on purpose, read: FA-ccin2p3-cfae8890 FA-ccin2p3-db4afd3f', 1, 'read', null, null)")
    c.execute("insert into events values (3165, 0, 'FA', 'x', 'RUN', 'FA-ccin2p3-0000aaaa:end', 'x', null, null, null, null)")
    c.executemany("insert into runs values (?, ?)", [("FA-ccin2p3-c3ab6791", "CANCELLED"),
                                                     ("FA-ccin2p3-0000aaaa", "FAILED"), ("FA-ccin2p3-560f6685", "FAILED"),
                                                     ("FA-ccin2p3-27e731b4", "COMPLETED")])
    c.commit()
    plan = load(p)
    e = {"findings": [{"text": "560f6685 cancelled unstarted (scancel 1)"}],
         "dead_ends": [{"tried": "27e731b4 read", "why": "finished"}]}
    ids = ["FA-ccin2p3-" + x for x in ("cfae8890", "db4afd3f", "c3ab6791", "560f6685", "27e731b4", "0000aaaa")]
    got = plan.acked_runs(ids, e)
    assert got == {"FA-ccin2p3-cfae8890", "FA-ccin2p3-db4afd3f", "FA-ccin2p3-c3ab6791", "FA-ccin2p3-560f6685"}, got
    assert plan.acked_runs(ids) == {"FA-ccin2p3-cfae8890", "FA-ccin2p3-db4afd3f", "FA-ccin2p3-c3ab6791"}


if __name__ == "__main__":
    test_handled()
    print("PASS test_handled")
