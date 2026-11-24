"""plan ask / plan add refuse an ask that does not state first one reason it is the user's call (scope change, GPU
budget above 30 GPU-h, permission-classifier refusal quoting the call, a hold of his). Tonight's three real asks."""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402

SIGN_PLAIN = ("Will you allow resubmitting the 12 sign-output jobs of catalog_coverage (28 GPU-h), a repair of "
              "approved work?")
SIGN_CLASSIFIER = ("Permission classifier refusal: `site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --item "
                   "catalog_coverage --env SHORT_TAG=signout -- v4_het_control_signhead` was refused; will you "
                   "allow the resubmission of the 12 sign-output jobs (28 GPU-h)?")
CACHES = ("Will you allow the deletion of two of our own growth data caches on CC-IN2P3, about 18 GB each and "
          "rebuildable, or remove them yourself?")
CACHES_CLASSIFIER = "Permission classifier refusal: `site run ccin2p3 FA -- rm -rf amp_mmap_cache/27a1422795e9` - " + CACHES
MARGIN = ("Scope: the wall-time margin of the sign-output jobs is above 2x at budget 750 GPU-h; do you want the "
          "arms cut to the shorter horizon or the margin kept?")


def test_sign_jobs_pass_only_with_the_classifier_refusal():
    assert plan.ask_reason(SIGN_PLAIN) is None
    assert plan.ask_reason(SIGN_CLASSIFIER) == "classifier"
    assert plan.ask_reason("Permission classifier refusal: it refused.") is None      # no plain words after it


def test_cache_deletion_refused_even_with_a_classifier_quote():
    for ask in (CACHES, CACHES_CLASSIFIER):
        try:
            plan._ask_ok(ask)
        except SystemExit as e:
            assert "own cheap-to-rebuild debris" in str(e)
        else:
            raise AssertionError(ask)


def test_margin_at_budget_750_passes_as_scope():
    assert plan.ask_reason(MARGIN) == "scope"
    plan._ask_ok(MARGIN)


def test_gpu_and_hold_reasons():
    assert plan.ask_reason("GPU budget: 45 GPU-h with no decision covering it; go?") == "gpu"
    assert plan.ask_reason("GPU budget: a big run; go?") is None
    assert plan.ask_reason("Your hold: warm_start held at its design (items warm_start); release?") == "hold"


def test_raw_command_in_an_ask_is_refused():
    for ask in ("Permission classifier refusal: `python3 - <<'X' ... X` was refused, will you allow it?",
                "Permission classifier refusal: it ran EOF and was refused, will you allow it?",
                "Permission classifier refusal: an edit\nwas refused, will you allow it?"):
        try:
            plan._ask_ok(ask)
        except SystemExit as e:
            assert "plain words" in str(e)
        else:
            raise AssertionError(ask)
    plan._ask_ok("Permission classifier refusal: an edit of the gate was refused, will you make it yourself?")


def test_ask_age_shown():
    """D534: an open question carries its age (the zng stage-2 ask sat 94 h after one relay, 2026-11-20..24)."""
    import datetime as dt
    for since in ((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=94)).isoformat(),
                  (dt.datetime.utcnow() - dt.timedelta(hours=94)).strftime("%Y-%m-%dT%H:%M:%SZ")):
        assert 93.9 < plan.ask_age_h({"waiting_since": since}) < 94.1, since


def test_part_reask():
    """Re-asking the unanswered part of an answered two-part ask passes; the whole ask again does not."""
    old = ("GPU budget: two questions. (1) Proposed diff to CLAUDE.md budget rule: replace the usual-size exception "
           "with a named-work bound of 100 GPU-h. Yes or no? (2) Permission classifier refusal: an edit of site_gate.py "
           "so the gate charges an lxplus job its own time limit instead of its HTCondor flavour. Will you make that "
           "edit yourself, or add a permission rule?")
    new = ("Permission classifier refusal: an edit of site_gate.py so the gate charges an lxplus job its own time "
           "limit instead of its HTCondor flavour. Will you make that edit yourself, or add a permission rule?")
    assert plan.part_reask(old, new)
    assert not plan.part_reask(old, old)
    assert not plan.part_reask(new, new)
