"""A short asking sentence ("Do you want it?") counts with the two sentences before it (real messages of 2026-11-18,
11-16, 11-20); a status that never asks and a different question stay unmatched. Not fixed: pretrain_data_saving 11-12."""
import _tmpdir  # noqa: F401
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402

CASES = json.load(open(os.path.join(ROOT, "tests", "data_plan_ask_short.json")))


def test_short_ask_cases():
    for c in CASES:
        got = plan.ask_matches_set({"id": c["entry"], "ask": c["ask"]}, c["sent"])
        assert got == c["should_match"], (c["entry"], c["should_match"])
