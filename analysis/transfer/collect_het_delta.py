"""Pack the het retrains of the Delta cells (D47, scripts/job_het_delta.sh) on this site into one npz and print it
base64 after a __B64__ line (site run brings it home as analysis/transfer/het_delta_data/<site>.npz). Per cell, keyed
<sweep>__<trial> as delta_plots.py keys the MSE cells: the test-split Delta = (A_theta - A)/A of the sigma stage's mu
(preds_test.npz, raw |M|^2) and the pulls (mu - y)/sigma on the standardized target (sigma_test.npz); "_meta" holds per
cell the rebuilt validation NLL. Login node, CPU only.
    python analysis/transfer/collect_het_delta.py [$SCRATCH/zz_delta_het]
"""
import base64, glob, io, json, os, sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["SCRATCH"], "zz_delta_het")
out, meta = {}, {}
for d in sorted(glob.glob(os.path.join(root, "*__sigma_stage"))):
    key = os.path.basename(d)[: -len("__sigma_stage")]
    try:
        with np.load(os.path.join(d, "preds_test.npz")) as z:
            t, p = z["raw_truth"].reshape(-1).astype(np.float64), z["raw_pred"].reshape(-1).astype(np.float64)
        with np.load(os.path.join(d, "sigma_test.npz")) as z:
            mu, y, sg = (z[k].reshape(-1).astype(np.float64) for k in ("pred", "truth", "sigma"))
    except (OSError, KeyError) as e:
        print(f"{key} skipped: {e}", file=sys.stderr)
        continue
    out[key + "__delta"] = ((p - t) / t).astype(np.float32)
    out[key + "__pull"] = ((mu - y) / sg).astype(np.float32)
    try:
        meta[key] = json.load(open(os.path.join(d, "eval_delta.json")))["val_loss_rebuilt"]
    except (OSError, KeyError, ValueError):
        pass
    print(f"{key} {len(t)} median|Delta|={np.median(np.abs(out[key + '__delta'])):.3g} "
          f"pull std={out[key + '__pull'].std():.2f}", file=sys.stderr)
out["_meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
buf = io.BytesIO()
np.savez_compressed(buf, **out)
print("__B64__")
print(base64.b64encode(buf.getvalue()).decode())
