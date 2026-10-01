"""Print a run's off-shellness column stats [mean, std] as JSON: from its data_stats.json, or for a run older than
that record from a build on its own pools (tools/rebuild_run.own_offshell_stats; CPU, no training). A fine-tune of
such a parent passes them as fine_tune.parent_offshell_stats.
    python tools/offshell_stats.py runs/<run>
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild_run import own_offshell_stats  # noqa: E402

run = sys.argv[1]
st = json.load(open(os.path.join(run, "data_stats.json"))).get("offshell_stats") or own_offshell_stats(run)
print("OFFSHELL_STATS " + json.dumps(st))
