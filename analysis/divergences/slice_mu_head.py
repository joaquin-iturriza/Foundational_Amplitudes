#!/usr/bin/env python
"""Slice the mu rows out of a HETEROSC (2-ch) checkpoint to make a valid MSE (1-ch) one.

Isolation test for the coupled-finetune residual: the coupled uug finetune starts from the
HETEROSC foundation, the 8.5e-5 MSE baseline starts from the MSE foundation — so the loss
and the foundation are confounded. To isolate the FOUNDATION we want an MSE finetune that
starts from the HETEROSC foundation's body.

Using fine_tune.reset_output_head for the 2ch->1ch load would zero the head, which the MSE
baseline never had to do (its head transferred intact) — that would be a head-reset confound.
Instead we keep the mu head: the net emits [mu, sigma] via one linear_out with out=2*out_shape,
where mu = out[..., :out_shape] (row 0) and sigma = out[..., -out_shape:] (row 1). So dropping
row 1 yields exactly the 1-ch MSE readout, body AND mu-head preserved. CPU only.
"""
import argparse
import gzip
import io
import os

import torch


def _load(p):
    if not p.endswith(".gz") and not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, map_location="cpu", weights_only=False)
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HETEROSC 2-ch checkpoint (.pt or .pt.gz)")
    ap.add_argument("--dst", required=True, help="output 1-ch (mu-only) checkpoint (.pt)")
    ap.add_argument("--out_shape", type=int, default=1, help="real target dim (mu channels)")
    ap.add_argument("--head", default="net.net.linear_out", help="readout module prefix")
    args = ap.parse_args()

    ck = _load(args.src)
    sd = ck["model"]
    k_w, k_b = f"{args.head}.weight", f"{args.head}.bias"
    n = args.out_shape

    w = sd[k_w]
    if w.shape[0] != 2 * n:
        raise SystemExit(f"{k_w} has {w.shape[0]} rows, expected 2*out_shape={2*n} (not HETEROSC?)")
    sd[k_w] = w[:n].clone()                     # rows [0:n] = mu; drop [-n:] = sigma
    if k_b in sd:
        sd[k_b] = sd[k_b][:n].clone()
    print(f"{k_w}: {tuple(w.shape)} -> {tuple(sd[k_w].shape)}")
    if k_b in sd:
        print(f"{k_b}: -> {tuple(sd[k_b].shape)}")

    # model weights only: _load_pretrained_weights reads ck['model'] and starts optim fresh
    torch.save({"model": sd, "step": ck.get("step")}, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
