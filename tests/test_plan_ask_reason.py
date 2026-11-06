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
    assert plan.ask_reason("Permission classifier refusal: it refused.") is None      # the call is not quoted


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
