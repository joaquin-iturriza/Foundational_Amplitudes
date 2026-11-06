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


MULTI = {"id": "catalog_coverage", "status": "running", "waiting_since": "2026-11-06T05:23:32+00:00",
         "next": ["a"], "blocks_steps": [1],
         "ask": "Two calls: (1) the scaling grid costs 449 GPU-h: run the whole grid, only the 2x row (the six "
                "under-one-epoch points redone at 1.27 epochs, 39 GPU-h), or that row at the original settings? "
                "(2) will you allow the resubmission of the 12 sign-output jobs at 3:30 (28 GPU-h of work), which the "
                "automatic permission check refused for the agent?"}
SENT = ("1. For the scaling grid, should we run the whole grid (449 GPU-h), only the 2x row with the long sweep's "
        "settings (39 GPU-h), or the 2x row at the original settings?\n2. Will you allow the 12 sign-output jobs to be "
        "resubmitted at 3:30 (28 GPU-h of work)? The automatic permission check refused that for the agent.")


def test_multipart_asked_part_by_part():
    assert plan.ask_matches(MULTI, "Joaquin, two calls.\n" + SENT)


def test_multipart_one_part_missing_refuses():
    assert not plan.ask_matches(MULTI, "Joaquin, " + SENT.split("\n")[0])
    assert not plan.ask_matches(MULTI, "Joaquin, " + SENT.split("\n")[1])
    assert not plan.ask_matches(MULTI, "Joaquin, the grid costs 449 GPU-h and the sign-output jobs need 3:30; nothing asked.")


def test_margin():
    assert plan.ASK_MARGIN_MIN == 30


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")
