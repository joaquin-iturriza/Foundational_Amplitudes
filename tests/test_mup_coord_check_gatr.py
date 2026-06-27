"""μP coordinate check for the L-GATr / L-GATr-slim μP backbones.

Trains the bare nets (no amplitude wrapper / no data needed) at a range of widths
for a few steps on a *fixed* synthetic batch, and records the average per-coordinate
magnitude (L1) of the readout-input activations. μP guarantees these stay O(1) in
width (flat lines vs width); standard parametrization (SP) drifts (sloped lines).

Run on Jean Zay in the env with the forked lloca + lgatr installed (and make sure the
old local ``lgatr/`` directory does not shadow the pip package):

    python test_mup_coord_check_gatr.py --net slim   # or: full
    python test_mup_coord_check_gatr.py --net full --mode sp

A PASS/FAIL verdict (|slope of log L1 vs log width|) is printed; PASS means flat.
"""
import argparse

import numpy as np
import torch

import os, sys
# tests/ -> repo root on sys.path so project-root modules import when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lgatr_slim_mup import MuPLGATrSlim
from models.lgatr_mup import MuPLGATr
from lloca.mup import MuAdam, finalize


def build(net_kind, width, mode):
    if net_kind == "slim":
        net = MuPLGATrSlim(
            in_v_channels=1, out_v_channels=0, hidden_v_channels=width,
            in_s_channels=4, out_s_channels=1, hidden_s_channels=2 * width,
            num_blocks=4, num_heads=2, parametrization=mode,
        )
        probe = net.linear_out  # readout
    else:
        net = MuPLGATr(
            in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=width,
            in_s_channels=4, out_s_channels=1, hidden_s_channels=width,
            num_blocks=4, num_heads=2, parametrization=mode,
        )
        probe = net.net.linear_out  # readout EquiLinear
    return net, probe


def run(net_kind, mode, widths, steps, lr, seed, device):
    torch.manual_seed(seed)
    if net_kind == "slim":
        vec = torch.randn(1, 16, 1, 4, device=device)
        sca = torch.randn(1, 16, 4, device=device)
    else:
        from lgatr.interface import embed_vector
        vec = embed_vector(torch.randn(1, 16, 1, 4, device=device))  # (1,16,1,16)
        sca = torch.randn(1, 16, 4, device=device)
    target = torch.randn(1, 16, 1, device=device)

    mags = []
    for w in widths:
        torch.manual_seed(seed)
        net, probe = build(net_kind, w, mode)
        net = net.to(device)
        finalize(net)
        opt = MuAdam(net.parameters(), lr=lr) if mode == "mup" else \
            torch.optim.Adam(net.parameters(), lr=lr)

        captured = {}
        h = probe.register_forward_hook(
            lambda m, inp, out: captured.__setitem__("a", inp[0].detach())
        )
        for _ in range(steps):
            opt.zero_grad()
            out = net(vec, sca) if net_kind == "slim" else net(vec, scalars=sca)
            out_s = out[1] if net_kind == "slim" else out[0][..., 0:1, 0]
            loss = ((out_s.reshape(1, 16, 1) - target) ** 2).mean()
            loss.backward()
            opt.step()
        h.remove()
        mags.append(captured["a"].abs().mean().item())
    return np.array(mags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=["slim", "full"], default="slim")
    ap.add_argument("--mode", choices=["mup", "sp"], default="mup")
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mags = run(args.net, args.mode, args.widths, args.steps, args.lr, args.seed, device)
    log_w = np.log(np.array(args.widths, float))
    slope = np.polyfit(log_w, np.log(mags), 1)[0]
    print(f"net={args.net} mode={args.mode}")
    for w, mag in zip(args.widths, mags):
        print(f"  width {w:4d}: readout-input L1 = {mag:.4f}")
    print(f"  slope(log L1 vs log width) = {slope:+.3f}")
    if args.mode == "mup":
        print("  VERDICT:", "PASS (flat -> μP works)" if abs(slope) < 0.25
              else "FAIL (not flat)")
    else:
        print("  (SP control: expect a clearly non-zero slope)")


if __name__ == "__main__":
    main()
