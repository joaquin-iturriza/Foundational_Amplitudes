"""dancat_het_pre's 8 trials read as a plain-mean MSE (llm_compare's design check): per trial, the mean over the 1352
training processes of the validation MSE of log|M|^2 at the best checkpoint, sum_p MSE_p sigma_p^2 / 1352, from the final
evaluation's 'MSE (prepd) val_<p>' lines (each process's own standardized units) and data_stats.json's prepd_std.
A shared z divides this by one constant, so it ranks the trials as a shared-target plain-mean MSE does.
Reads on CC (analysis.common.site_python); prints one line per trial, ordered by lr.
    python analysis/catalog_daniel/dancat_plain_mse.py
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from analysis.common import site_python  # noqa: E402

REMOTE = r'''
import glob, json, re
out = {}
for d in sorted(glob.glob("runs/dancat_het_pre/trial_*")):
    sd = json.load(open(d + "/data_stats.json"))["prepd_std"]
    order = json.load(open(d + "/plots_0/per_process_metrics.json"))["dataset_order"]
    v = {}
    for l in open(d + "/out_0.log"):
        m = re.search(r"MSE \(prepd\) val_(\S+) \S+: (\S+)", l)
        if m:
            v[m.group(1)] = float(m.group(2))
    assert len(v) == len(order), (d, len(v), len(order))
    out[d[-4:]] = sum(v[n] * sd[i] ** 2 for i, n in enumerate(order)) / len(order)
print("MARK " + json.dumps(out))
'''

summary = os.path.join(ROOT, "sweeps_local", "dancat_het_pre", "summary.txt")
lr = {int(l.split()[0][3:]): float(re.search(r"lr=(\S+)", l).group(1))
      for l in open(summary) if l.strip().startswith("hp_")}
for k, v in sorted(site_python(REMOTE, [], "MARK").items(), key=lambda kv: lr[int(kv[0])]):
    print(f"hp{k}  lr {lr[int(k)]:.3g}  plain-mean val MSE of log|M|^2 {v:.4g}")
