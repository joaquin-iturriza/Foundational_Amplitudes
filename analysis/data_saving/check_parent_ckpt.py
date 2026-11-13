"""Gate before fine-tuning from a new parent: load two parents' best checkpoints on CPU (a .pt or its .pt.gz, as
base_experiment loads them) and check the new one has the reference's tensors, same names and shapes.
    python analysis/data_saving/check_parent_ckpt.py <reference run dir> <new run dir>
"""
import gzip, io, os, sys
import torch


def load(d):
    p = os.path.join(d, "models", "model_run0_best.pt")
    raw = gzip.open(p + ".gz").read() if not os.path.exists(p) else open(p, "rb").read()
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck.state_dict()
    return {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"PARENT_CKPT tensors {len(a)} / {len(b)}, same names and shapes: {a == b}")
sys.exit(0 if a == b else 1)
