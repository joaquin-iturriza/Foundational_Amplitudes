"""Whether a pp read's prediction file (tools/zero_shot_parent.py --save-dir) is complete: it opens, and holds the
truth and a prediction of the same nonzero length (as analysis/catalog_v3/pp_compare.py collect requires). Used by
scripts/job_catalog_v3_pp_score.sh for MISSING_ONLY and its completeness check; a file cut short by a full disk fails."""
import os

import numpy as np


def ok(path):
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path)
        return "t" in z.files and "q" in z.files and len(z["t"]) > 0 and len(z["q"]) == len(z["t"])
    except Exception:
        return False
