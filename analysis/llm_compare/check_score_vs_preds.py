"""Cross-check of tools/llm_score.py on a run's own training recipe (seen): its pooled RMSE of log|M|^2 against the one
read from the run's saved test predictions (<run>/<stage>/preds_test.npz, written by the run's own test stage), through
the experiment's own inverse transform (raw_pred, raw_truth: |M|^2 with the target factor put back), and through the
standardized pred/truth times the run's sigma. Runs on the site holding the run (analysis/common site_python).
    python analysis/llm_compare/check_score_vs_preds.py <site run dir> [--stage sigma_stage] [--site ccin2p3]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.common.readers import site_python  # noqa: E402

REMOTE = '''import json, sys, numpy as np
run, stage = sys.argv[1], sys.argv[2]
z = np.load(f"{run}/{stage}/preds_test.npz")
sd = json.load(open(f"{run}/llm_score_seen.json"))
s = float(np.ravel(sd["prepd_std"])[0])
d_raw = np.log(z["raw_pred"].ravel()) - np.log(z["raw_truth"].ravel())
d_std = (z["pred"].ravel().astype(np.float64) - z["truth"].ravel()) * s
print("CHK " + json.dumps({"events": int(d_raw.size), "rmse_raw_log": float(np.sqrt(np.mean(d_raw ** 2))),
      "rmse_std_times_sigma": float(np.sqrt(np.mean(d_std ** 2))), "sigma": s,
      "llm_score_seen": sd["pooled_rmse_logm2"], "llm_score_events": sd["events"]}))'''

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stage", default="sigma_stage")
ap.add_argument("--site", default="ccin2p3")
a = ap.parse_args()
print(site_python(REMOTE, [a.run, a.stage], "CHK", site=a.site))
