#!/usr/bin/env python3
"""Every module's output in the evaluator's forward pass of the first --n events of a dc_bbx_hh val pool, to find where
a CPU read and a GPU read of the same events part (the CPU read sat 0.194 low in log|M|^2).

    python evaluate_daniels_catalog/layer_dump.py --pool <dc_bbx_hh_..._val_lhe_amplitudes.npy> --out dump.npz \
        [--no-xformers] [--n 2000]
    python evaluate_daniels_catalog/layer_dump.py --compare a.npz b.npz

--no-tf32 turns TF32 matmuls off on a GPU. --no-xformers hides xformers, so a GPU runs the dense masked attention the CPU runs. The dump holds, per module call in
call order ("<k>:<class>"), the first 20000 entries of its output (float32) and the whole output's mean and rms;
--compare prints, call by call, the largest difference over the largest value, and the first call where it passes 1e-3.
"""
import argparse, os, sys

import numpy as np

K = 20000

ap = argparse.ArgumentParser()
ap.add_argument("--pool")
ap.add_argument("--out")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--no-xformers", action="store_true")
ap.add_argument("--no-tf32", action="store_true", help="matmul and cudnn TF32 off (the model config turns it on)")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()

if a.compare:
    A, B = (np.load(f) for f in a.compare)
    keys = [k for k in A.files if k.startswith("out|")]
    first = None
    for k in keys:
        x, y = A[k].astype(np.float64), B[k].astype(np.float64) if k in B.files else None
        if y is None or x.shape != y.shape:
            print(f"{k}: shape {x.shape} vs {None if y is None else y.shape}")
            continue
        rel = np.abs(x - y).max() / max(np.abs(x).max(), 1e-30)
        print(f"{k[4:]:40s} rel {rel:.2e}  mean {A['mean|' + k[4:]]:+.5g} {B['mean|' + k[4:]]:+.5g}")
        if first is None and rel > 1e-3:
            first = k[4:]
    print("FIRST_DIFF", first)
    sys.exit(0)

if a.no_xformers:
    sys.modules["xformers"] = sys.modules["xformers.ops"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import evaluate  # noqa: E402
import experiment  # noqa: E402
import base_experiment  # noqa: E402

if a.no_tf32:
    _backend = base_experiment.BaseExperiment._init_backend

    def _backend_no_tf32(self, *args, **kw):
        r = _backend(self, *args, **kw)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        return r
    base_experiment.BaseExperiment._init_backend = _backend_no_tf32

rec, calls = {}, [0]


def hook(m, inp, out):
    t = out[0] if isinstance(out, (tuple, list)) and out else out
    if not (isinstance(t, torch.Tensor) and t.is_floating_point()):
        return
    k = f"{calls[0]:04d}:{type(m).__name__}"
    calls[0] += 1
    t = t.detach().double()
    rec["out|" + k] = t.reshape(-1)[:K].float().cpu().numpy()
    rec["mean|" + k] = float(t.mean())
    rec["rms|" + k] = float(t.pow(2).mean().sqrt())


real = experiment.AmplitudeExperiment._forward_lloca


def forward_dumped(self, data, *args, **kw):
    if calls[0]:                                   # the first batch only (--n events fit one eval batch)
        return real(self, data, *args, **kw)
    h = torch.nn.modules.module.register_module_forward_hook(hook)
    try:
        return real(self, data, *args, **kw)
    finally:
        h.remove()


experiment.AmplitudeExperiment._forward_lloca = forward_dumped
z = evaluate.score("dc_bbx_hh", os.path.abspath(a.pool), 10000, a.n, os.path.join(evaluate.HERE, "model"))
q, t = z["q"].astype(np.float64), z["t"].astype(np.float64)
import models.attention_lloca_mup as att  # noqa: E402
dev = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
print(f"LAYER_DUMP xformers {att._XFORMERS_AVAILABLE} device {dev} tf32 matmul {torch.backends.cuda.matmul.allow_tf32} "
      f"cudnn {torch.backends.cudnn.allow_tf32} calls {calls[0]} "
      f"mse {np.mean((q - t) ** 2):.4f} mean_q-t {np.mean(q - t):+.4f}")
np.savez(a.out, q=q, t=t, **rec)
