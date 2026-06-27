"""μP coordinate check for the LLoCa MuP transformer.

This is the *definitive* test of whether μP is wired correctly. It trains the
real `LLOCAMuPTransformer` (wrapped exactly as in training) at a range of widths
(num_heads) for a few steps on a *fixed* synthetic batch, and records the
average per-coordinate magnitude (L1) of every block's activations.

μP guarantees that, at any fixed training step, these activation coordinates are
O(1) in width — i.e. roughly *flat* as width grows. Standard parametrization (SP)
instead shows activations drifting (typically growing) with width.

So:
  * `--mode mup`  (default) should give near-horizontal lines  -> μP works.
  * `--mode sp`   (control) should give sloped lines           -> contrast.

We reuse the exact construction path of `base_experiment.init_model`:
  base width  = num_heads 2   (matches base_cfg in base_experiment.py)
  delta width = num_heads 8   (matches delta_cfg)
and the same MuAdam optimizer + MuReadout. Nothing about μP scaling is
re-implemented here — we only *probe* the real model.

Run on Jean Zay (needs the `foundational` env, torch + mup + lloca):

    conda activate /lustre/fswork/.../conda/envs/foundational
    python test_mup_coord_check.py --mode mup  --out coord_mup.csv
    python test_mup_coord_check.py --mode sp   --out coord_sp.csv

A PASS/FAIL verdict (slope of log L1 vs log width) is printed at the end, and a
PDF (coord_<mode>.pdf) is written next to the CSV.
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

import mup
from mup import set_base_shapes, MuAdam
from lloca.mup import finalize as mup_finalize

import os, sys
# tests/ -> repo root on sys.path so project-root modules import when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lloca import LLOCAMuPTransformer
from wrappers import AmplitudeLLoCaWrapper
from particle_ids import GLOBAL_PROPERTY_MATRIX

# Modules whose output activations we probe. We hook the residual-stream feed
# (linear_in), every transformer block (the residual stream after each block),
# and the readout input/output.
ATTN_REPS = "8x0n+2x1n"
NUM_BLOCKS = 8
D_PARTICLE_HIDDEN = 16
N_ORDER_FEATURES = 2          # default amp_orders is [n_loops, alpha_s_power]
NUM_SCALARS = D_PARTICLE_HIDDEN + N_ORDER_FEATURES        # 18
IN_CHANNELS = NUM_SCALARS + 4                             # 22

BASE_NH = 2     # must match base_experiment.init_model base_cfg
DELTA_NH = 8    # must match base_experiment.init_model delta_cfg


def build_model(num_heads, parametrization="mup"):
    # With parametrization="mup" the LLoCa transformer backbone computes its own μP
    # base shapes inside __init__ (width axis = num_heads, base/delta = 2/8 by default);
    # see lloca.mup. parametrization="sp" builds the same architecture without μP so we
    # can use it as the standard-parametrization control below.
    net = LLOCAMuPTransformer(
        num_scalars=NUM_SCALARS,
        hidden_channels_mlp=128,
        num_layers_mlp=2,
        in_channels=IN_CHANNELS,
        attn_reps=ATTN_REPS,
        out_channels=1,
        num_blocks=NUM_BLOCKS,
        num_heads=num_heads,
        parametrization=parametrization,
    )
    model = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=D_PARTICLE_HIDDEN)
    model.setup_particle_features(use_pids=False, property_matrix=GLOBAL_PROPERTY_MATRIX)
    return model


def set_shapes(model, mode):
    """Finalize μP infshapes on `model`.

    mode == 'mup': the backbone already set its base shapes during construction; we
                   only finalize the parameters living outside it (framesnet, particle
                   encoder), marking them as standard parametrization.
    mode == 'sp' : control — base shapes == model's own shapes, so every width_mult == 1.
                   MuAdam/MuReadout then become no-ops, i.e. ordinary training. The model
                   must have been built with parametrization='sp' (no μP rescale).
    """
    if mode == "sp":
        # base == model => all dims "finite-relative-to-themselves" => width_mult 1.
        set_base_shapes(model, model, rescale_params=False)
        return
    mup_finalize(model)


def make_optimizer(model, lr, mode):
    if mode == "sp":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    return MuAdam(
        [{"params": model.parameters(), "lr": lr}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


def make_batch(n_events=16, seed=0, device="cpu"):
    """Fixed synthetic batch in the exact flat/sparse layout the wrapper expects.

    Four-momenta must be *physical* (metric (+,-,-,-), index 0 = energy): the
    Lorentz framesnet computes gamma = E / mass, so a spacelike/lightlike vector
    (mass -> 0) gives gamma -> inf -> NaN. We build genuine massive on-shell
    momenta: spatial p ~ N(0, 9), mass in [0.5, 1.5], E = sqrt(|p|^2 + m^2).
    """
    g = torch.Generator().manual_seed(seed)
    n_species = GLOBAL_PROPERTY_MATRIX.shape[0]
    counts = torch.randint(3, 6, (n_events,), generator=g)        # 3..5 particles/event
    N = int(counts.sum())
    ptr = torch.zeros(n_events + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, 0)
    p_spatial = 3.0 * torch.randn(N, 3, generator=g)
    mass = 0.5 + torch.rand(N, 1, generator=g)                    # mass in [0.5, 1.5]
    energy = (p_spatial.pow(2).sum(-1, keepdim=True) + mass.pow(2)).sqrt()
    fourmomenta = torch.cat([energy, p_spatial], dim=-1)          # (N, 4), timelike, E>0
    ptypes = torch.randint(0, n_species, (N,), generator=g)
    order_labels = torch.zeros(n_events, N_ORDER_FEATURES)
    target = torch.randn(n_events, 1, generator=g)
    return dict(
        fourmomenta=fourmomenta.to(device),
        particle_type_indices=ptypes.to(device),
        mean=0.0, std=1.0,
        ptr=ptr.to(device),
        order_labels=order_labels.to(device),
        target=target.to(device),
    )


def hooked_modules(model):
    """(name, module) pairs whose output L1 we record.

    model              = AmplitudeLLoCaWrapper
    model.net          = LLOCAMuPTransformer
    model.net.net      = MuPTransformer (holds linear_in / blocks / linear_out)
    """
    backbone = model.net.net
    mods = [("linear_in", backbone.linear_in)]
    for i, blk in enumerate(backbone.blocks):
        mods.append((f"block{i}", blk))
    mods.append(("readout", backbone.linear_out))
    return mods


def l1(t):
    return t.abs().mean().item()


def run(mode, widths, nsteps, lr, seed, device):
    batch = make_batch(seed=seed, device=device)
    records = []
    for w in widths:
        torch.manual_seed(seed)
        model = build_model(w, parametrization=mode).to(device)
        set_shapes(model, mode)
        opt = make_optimizer(model, lr, mode)
        model.train()

        for step in range(1, nsteps + 1):
            handles, captured = [], {}

            def mk_hook(name):
                def hook(_m, _inp, out):
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    captured[name] = l1(o)
                return hook

            for name, m in hooked_modules(model):
                handles.append(m.register_forward_hook(mk_hook(name)))

            out = model(
                batch["fourmomenta"], batch["particle_type_indices"],
                batch["mean"], batch["std"], batch["ptr"],
                order_labels=batch["order_labels"],
            )
            loss = torch.nn.functional.mse_loss(out, batch["target"])
            opt.zero_grad(); loss.backward(); opt.step()

            for h in handles:
                h.remove()
            for name, v in captured.items():
                records.append(dict(width=w, module=name, t=step, l1=v))
        print(f"[{mode}] width={w:>3}  done (loss={loss.item():.4g})")
    return pd.DataFrame(records)


def verdict(df, nsteps):
    """Fit slope of log10(l1) vs log2(width) at the final step for each module.
    μP -> |slope| small (flat). Report worst (largest |slope|) hidden module."""
    last = df[df.t == nsteps]
    worst_name, worst_slope = None, 0.0
    for name, sub in last.groupby("module"):
        sub = sub[sub.l1 > 0]
        if sub.width.nunique() < 2:
            continue
        x = np.log2(sub.width.values.astype(float))
        y = np.log10(sub.l1.values.astype(float))
        slope = np.polyfit(x, y, 1)[0]   # decades of L1 per doubling of width
        if abs(slope) > abs(worst_slope):
            worst_slope, worst_name = slope, name
    return worst_name, worst_slope


def plot(df, mode, nsteps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = df[(df.t == nsteps) & np.isfinite(df.l1) & (df.l1 > 0)]
    if last.empty:
        print("WARNING: no finite positive activations to plot (model produced "
              "NaN/inf — check the forward pass).")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, sub in last.groupby("module"):
        sub = sub.sort_values("width")
        ax.plot(sub.width, sub.l1, marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width (num_heads)"); ax.set_ylabel(f"mean |activation| @ step {nsteps}")
    ax.set_title(f"μP coord check — mode={mode}\n(flat = μP correct)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mup", "sp"], default="mup")
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--nsteps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={args.mode}  widths={args.widths}  nsteps={args.nsteps}  lr={args.lr}")

    df = run(args.mode, args.widths, args.nsteps, args.lr, args.seed, device)
    out = args.out or f"coord_{args.mode}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    plot(df, args.mode, args.nsteps, os.path.splitext(out)[0] + ".pdf")

    name, slope = verdict(df, args.nsteps)
    print("\n=== verdict ===")
    print(f"worst-drifting module: {name}  slope={slope:+.3f} decades / width-doubling")
    if args.mode == "mup":
        ok = abs(slope) < 0.15
        print("μP COORD CHECK:", "PASS ✅ (activations ~flat in width)" if ok
              else "FAIL ❌ (activations drift with width — μP not working)")
    else:
        print("(SP control: expect a clearly non-zero slope here for contrast)")


if __name__ == "__main__":
    main()
