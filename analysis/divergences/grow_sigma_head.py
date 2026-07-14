#!/usr/bin/env python
"""Grow a 1-channel MSE checkpoint into a 2-channel HETEROSC one, PRESERVING mu.

Stage 1->2 of the two-stage recipe. The inverse of slice_mu_head.py (which drops the sigma
row to make a HETEROSC ckpt loadable by an MSE net); here we ADD a sigma row to an MSE ckpt.

Why not fine_tune.reset_output_head? That drops the whole readout on a shape mismatch
(base_experiment._load_pretrained_weights), i.e. it would reset the MU row too and throw away
the converged MSE head. We need mu bit-identical and only sigma fresh.

  linear_out.weight : (1, H) -> (2, H)   row 0 = MSE mu row (verbatim), row 1 = ZEROS
  linear_out.bias   : (1,)   -> (2,)     [0] = MSE mu bias,  [1] = SIGMA_B0

Zero weight + bias SIGMA_B0 means sigma starts as the CONSTANT softplus(SIGMA_B0), and its
x-dependence has to be learned from zero (the weight row still gets gradient via h). We seed
SIGMA_B0 so that constant is ~1e-2, the RMS residual of the converged MSE model in
preprocessed (standardized log-amplitude) space -- i.e. sigma starts globally calibrated and
only has to learn WHERE the error is bigger/smaller. Starting instead at the default
softplus(0)=0.69 would put sigma ~70x above the true error and waste the run climbing down.

CPU-only (pure state_dict surgery). Usage:
  python grow_sigma_head.py <mse_ckpt.pt[.gz]> <out.pt> [sigma0]
"""
import gzip
import math
import os
import sys

import torch

SIGMA0_DEFAULT = 1e-2   # ~ RMS residual of the converged MSE uug model (mu-MSE ~ 1e-4)


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sigma0 = float(sys.argv[3]) if len(sys.argv) > 3 else SIGMA0_DEFAULT

    # softplus(b) = sigma0  ->  b = ln(exp(sigma0) - 1)
    b0 = math.log(math.expm1(sigma0))

    ck = _load(src)
    sd = ck["model"]

    wkey = [k for k in sd if k.endswith("linear_out.weight")]
    bkey = [k for k in sd if k.endswith("linear_out.bias")]
    assert len(wkey) == 1 and len(bkey) == 1, f"expected one readout, got {wkey} {bkey}"
    wkey, bkey = wkey[0], bkey[0]

    w, b = sd[wkey], sd[bkey]
    assert w.shape[0] == 1, f"{wkey} has {w.shape[0]} rows — not a 1-ch MSE checkpoint"

    sd[wkey] = torch.cat([w, torch.zeros_like(w)], dim=0)          # (2, H)
    sd[bkey] = torch.cat([b, torch.full_like(b, b0)], dim=0)       # (2,)

    # Optimizer/EMA state refer to the old 1-ch shapes; stage 2 starts a fresh optimizer
    # anyway (_load_pretrained_weights loads weights only), so drop them rather than ship
    # tensors that silently mismatch.
    out = {"model": sd}

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    print(f"grown {wkey}: {tuple(w.shape)} -> {tuple(sd[wkey].shape)}  (mu row verbatim)")
    print(f"      {bkey}: mu={b.item():.4g}  sigma_bias={b0:.4g} -> sigma_init=softplus={sigma0:.3g}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
