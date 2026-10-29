"""One tested version of the readers, pickers and statistics analysis scripts kept rewriting (the user, 2026-10-29:
one script per family of figures, existing code reused; the duplicate counts are in
analysis/ponytail_ab/duplicates.json). Import from here, never copy:
    sys.path.insert(0, ROOT)            # the repository root
    from analysis.common import best_trial, curve_stats, ...
`python3 tools/readers_index.py` lists every public function with its signature."""
from analysis.common.readers import (best_checkpoint, best_trial, curve_stats, find_pool, gmean, ratio_stats,  # noqa
                                     read_errors, read_jsonl, run_metrics, split_pool, sweep_trials, trial_mse)
