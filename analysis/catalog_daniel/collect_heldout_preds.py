"""Pack the per-event held-out predictions of scripts/job_catalog_heldout_preds.sh (D74) into one npz and print it base64
after a __B64__ line (brought home as analysis/catalog_daniel/data/heldout_preds.npz). Per probe: <probe>__t (truth
log|M|^2), <probe>__q_aff (after the best affine map) and, for a relabel scored with its twin's statistics,
<probe>__q. Login node, CPU only.
    python analysis/catalog_daniel/collect_heldout_preds.py [$SCRATCH/zz_catalog_heldout_preds]
"""
import base64, glob, io, os, sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["SCRATCH"], "zz_catalog_heldout_preds")
out = {}
for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
    probe = os.path.basename(f)[:-4]
    with np.load(f) as z:
        for k in ("t", "q_aff", "q_own", "q", "h", "mu_sd", "logF", "s", "sig"):
            if k in z.files and z[k].size:
                out[f"{probe}__{k}"] = z[k].astype(np.float32)
print(len({k.split("__")[0] for k in out}), "probes", file=sys.stderr)
buf = io.BytesIO(); np.savez_compressed(buf, **out)
print("__B64__"); print(base64.b64encode(buf.getvalue()).decode())
