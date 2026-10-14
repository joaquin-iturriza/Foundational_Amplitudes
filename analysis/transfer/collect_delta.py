"""Pack the test-split Delta = (A_theta - A)/A of every tools/eval_delta.py output on this site into one npz,
keyed <sweep>__<trial>, and print it base64 after a __B64__ line (site run brings it home; delta_plots.py reads
analysis/transfer/delta_data/<site>.npz). Login node, CPU only.
    python analysis/transfer/collect_delta.py [$SCRATCH/zz_delta]
"""
import base64, glob, io, os, sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["SCRATCH"], "zz_delta")
out = {}
for f in sorted(glob.glob(os.path.join(root, "*", "preds_test.npz"))):
    with np.load(f) as z:
        t, p = z["raw_truth"].reshape(-1).astype(np.float64), z["raw_pred"].reshape(-1).astype(np.float64)
    out[os.path.basename(os.path.dirname(f))] = ((p - t) / t).astype(np.float32)
    print(f"{os.path.basename(os.path.dirname(f))} {len(t)} median|Delta|={np.median(np.abs(out[os.path.basename(os.path.dirname(f))])):.3g}", file=sys.stderr)
buf = io.BytesIO()
np.savez_compressed(buf, **out)
print("__B64__")
print(base64.b64encode(buf.getvalue()).decode())
