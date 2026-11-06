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


WS = {"id": "warm_start", "status": "running", "waiting_since": "2026-11-06T13:51:00+00:00", "next": ["a"],
      "ask": "Two calls on the warm start, both scope changes (yours): (1) The one-loop magnitude-target change (het) is "
             "not decided at any k because its from-scratch pair's two seeds disagree (R_seed 1.28, so the 1.35-bar "
             "interval is widened by ln 1.28 = 0.25 on each side). More from-scratch seeds of v4_het_control_magnitude "
             "would narrow that term. My estimate: n=3 seeds widens by x1.23, n=4 by x1.19. So 4 more seeds (44 to 47) "
             "would very likely make k=533 a MATCH, a YES at 1/8 of a pretraining. Cost: 2.35 h per run on a V100, so "
             "4 runs = 9.4 GPU-h. Do you want the 4 seeds (9.4 GPU-h) for the magnitude target, 2 seeds, or none? "
             "(2) Asked 06:48, still open: should the tuned width growth also run at 1066 and 2132 steps, like the "
             "untuned growth, or only at 533? (blocks: nothing; searched first: CLAUDE.md scope rule)"}
WS_MSG = ("Joaquin, the two warm start questions, put as questions: (1) Should I run 4 more from-scratch seeds (44 to "
          "47) of the one-loop magnitude-target change, at 9.4 GPU-h, to try to decide it at an eighth of a "
          "pretraining? (2) Should the tuned width growth also run at 1066 and 2132 steps, or only at 533?")


def test_warm_start_short_words_pass():
    assert plan.ask_matches(WS, WS_MSG)


def test_warm_start_unrelated_or_one_part_refuses():
    assert not plan.ask_matches(WS, "Joaquin, should I rerun the multichannel sweep at a lower learning rate, or keep the current one?")
    assert not plan.ask_matches(WS, "Joaquin, the warm start runs finished; nothing needs you.")
    assert not plan.ask_matches(WS, "Joaquin, (1) Should I run 4 more from-scratch seeds (44 to 47) of the one-loop magnitude-target change, at 9.4 GPU-h?")


def test_margin():
    assert plan.ASK_MARGIN_MIN == 30


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")
