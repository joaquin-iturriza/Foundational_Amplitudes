"""drive.py re-pick: a trial queued on one site moves only when another allowed site ends it > MOVE_H hours sooner and
in under 1/MOVE_X of the time, by site pick's model."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import drive


def row(site, wait_h=0.0, slots=0, mine_q=0, mine_run=0, med_run_h=None, state="UP", **kw):
    return dict(site=site, state=state, wait_now=wait_h * 3600, slots_now=slots, mine_q=mine_q, mine_run=mine_run,
                med_run=med_run_h * 3600 if med_run_h else None, **kw)


def test_moves_to_free_site():
    # CC queue full (Slurm says 10 h), lxplus has a free GPU: 12 h stay vs 2 h there
    rows = [row("ccin2p3", wait_h=10, mine_q=5), row("lxplus", slots=3)]
    assert drive.repick_dest("ccin2p3", rows, 2.0) == "lxplus"


def test_keeps_when_margin_small():
    # 3 h stay vs 2 h elsewhere: 1 h sooner but not 2x
    rows = [row("ccin2p3", wait_h=1), row("lxplus", slots=1)]
    assert drive.repick_dest("ccin2p3", rows, 2.0) is None
    # 2x but under 1 h sooner: 0.8 h vs 0.3 h
    rows = [row("ccin2p3", wait_h=0.5), row("lxplus", slots=1)]
    assert drive.repick_dest("ccin2p3", rows, 0.3) is None


def test_keeps_when_free_slot_at_home():
    rows = [row("ccin2p3", slots=1, mine_q=4), row("lxplus", slots=8)]
    assert drive.repick_dest("ccin2p3", rows, 2.0) is None


def test_destination_queue_counts():
    # lxplus free of slots, 6 of mine queued, 1 running at 2 h each: 0.5 + 6*2 + 2 = 14.5 h, worse than staying (6 h)
    rows = [row("ccin2p3", wait_h=4, mine_q=10), row("lxplus", wait_h=0.5, mine_q=6, mine_run=1, med_run_h=2)]
    assert drive.repick_dest("ccin2p3", rows, 2.0) is None


def test_dead_home_and_closed_destinations():
    # home takes no work (cap 0, CC's MaxJobs=0): any site that does wins
    rows = [row("ccin2p3", wait_h=0, cap=0), row("lxplus", wait_h=1)]
    assert drive.repick_dest("ccin2p3", rows, 2.0) == "lxplus"
    # destinations that take no work: kept (held Jean Zay without --allow-jeanzay, a down site, a skipped site)
    rows = [row("ccin2p3", wait_h=20), row("jeanzay", slots=9, state="HELD"), row("lxplus", slots=9, state="DOWN"),
            row("x", slots=9, skip="0 GPUs fit 48G/4cpu")]
    assert drive.repick_dest("ccin2p3", rows, 2.0) is None
    # home not probed (outside --sites): kept
    assert drive.repick_dest("jeanzay", [row("lxplus", slots=9)], 2.0) is None


def test_repick_moves_queued_only_and_fills_destination():
    moved = []
    t = lambda i, st: dict(idx=i, rid="FA-ccin2p3-%d" % i, site="ccin2p3", status="submitted", sched_state=st)
    trials = [t(0, "PENDING"), t(1, "RUNNING"), t(2, "PENDING")]
    sw = types.SimpleNamespace(hours=2.0, in_flight=lambda: trials,
                               move=lambda tr, dest, a: moved.append((tr["idx"], dest)) or True)
    # lxplus has one free GPU and nothing queued: the first queued trial takes it, the second then sees 1 queued there
    # ahead of it at 2 h (0 + 1*2 + 2 = 4 h vs 12 h stay: still moves); the running one never moves
    rows = [row("ccin2p3", wait_h=10), row("lxplus", slots=1)]
    drive.repick([sw], None, rows)
    assert moved == [(0, "lxplus"), (2, "lxplus")]
    assert rows[1]["mine_q"] == 2 and rows[1]["slots_now"] == 0


def test_cancel_cmd():
    assert drive.cancel_cmd("ccin2p3", "123") == "scancel 123"
    assert drive.cancel_cmd("lxplus", "456") == "condor_rm 456"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
