"""plan_check's 'a question open for the user, but no message sent to them asks it': a message that names the entry's
step or shares half the question's terms counts, a question never sent still refuses."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import plan  # noqa: E402

ASK = "Should the multichannel arm train on the whole set of 1e5 events per process or only on 1e4, given the budget?"
E = {"id": "multichannel", "status": "running", "ask": ASK, "waiting_since": "2026-11-05T10:00:00+00:00",
     "next": ["a", "b"], "blocks_steps": [2]}


def test_verbatim_passes():
    assert plan.ask_matches(E, "Joaquin, " + ASK)


def test_reworded_before_recording_passes():
    msg = ("Joaquin, one call on multichannel: for the arm, whole set (1e5 events per process) or 1e4 per process? "
           "The budget differs.")
    assert plan.ask_matches(E, msg)
    assert plan.ask_matches(E, "Joaquin, multichannel step 2 needs your pick between the two data sizes.")


def test_never_sent_refuses():
    assert not plan.ask_matches(E, "Joaquin, the warm start grow runs finished; nothing needs you.")
    assert not plan.ask_matches(E, "Joaquin, multichannel step 2 finished its probe: 3.1 s per event, nothing needs you.")
    assert not plan.ask_matches(E, "Joaquin, digest: the multichannel arm trained on the whole set, 1e5 events per "
                                   "process, within budget; the 1e4 cells done.")
    assert not plan.ask_matches(E, "Joaquin, multichannel step 7 finished its probe.")   # a step it does not hold


def test_margin():
    assert plan.ASK_MARGIN_MIN == 30


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")
