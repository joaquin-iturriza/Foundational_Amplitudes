"""
VRAM scaling test: peak VRAM as a function of (num_heads, batch_size, n_particles).

Includes a real AdamW optimizer.step() so optimizer state buffers (2x model size)
are fully allocated — giving realistic peak VRAM as seen during actual training.

Run on Jean-Zay with a GPU allocated:
    python test_vram_scaling.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vram_scaling_output.txt"), "w")
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = _Tee(sys.__stdout__, _log)

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from hydra.utils import instantiate
from omegaconf import OmegaConf
from mup import set_base_shapes
from torch_ema import ExponentialMovingAverage

N_SCALARS = 18    # d_particle_hidden(16) + n_order_features(2)
DEVICE    = torch.device("cuda")
LIMIT_16  = 16 * 1024   # MB
LIMIT_32  = 32 * 1024   # MB


def make_model_cfg(num_heads):
    return OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars":         N_SCALARS,
        "hidden_channels_mlp": 128,
        "num_layers_mlp":      2,
        "in_channels":         N_SCALARS + 4,
        "attn_reps":           "8x0n+2x1n",
        "out_channels":        1,
        "num_blocks":          8,
        "num_heads":           num_heads,
    })


def make_batch(n_particles, batch_size):
    N_total     = batch_size * n_particles
    p_xyz       = torch.randn(N_total, 3, device=DEVICE)
    mass        = torch.rand(N_total, 1, device=DEVICE) + 0.1
    E           = (p_xyz.pow(2).sum(-1, keepdim=True) + mass.pow(2)).sqrt()
    fourmomenta = torch.cat([E, p_xyz], dim=-1)
    particle_type = torch.randn(N_total, N_SCALARS, device=DEVICE)
    ptr = torch.arange(0, N_total + 1, n_particles, dtype=torch.long, device=DEVICE)
    return fourmomenta, particle_type, ptr


def measure(model, optimizer, n_particles, batch_size, ema=None):
    """Forward + backward + optimizer step (+ EMA update); returns peak VRAM in MB."""
    fourmomenta, particle_type, ptr = make_batch(n_particles, batch_size)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    optimizer.zero_grad()
    out  = model(fourmomenta, particle_type, mean=0.0, std=1.0, ptr=ptr)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    if ema is not None:
        ema.update()

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def fit_and_table(data, num_heads, model_mb):
    """
    Fit peak = a*bs*n + b*bs + c from measurements.
    Print n_max table for 16 GB and 32 GB limits.
    Returns (a, b, c).
    """
    arr = np.array(data)
    bs_arr, n_arr, peak_arr = arr[:,0], arr[:,1], arr[:,2]
    X = np.column_stack([bs_arr * n_arr, bs_arr, np.ones(len(arr))])
    coeffs, _, _, _ = np.linalg.lstsq(X, peak_arr, rcond=None)
    a, b, c = coeffs
    pred     = X @ coeffs
    residual = np.std(peak_arr - pred)

    optim_mb = 2 * model_mb   # AdamW m + v buffers
    print(f"\n  Fit: peak ≈ {a:.4f}·bs·n + {b:.4f}·bs + {c:.0f} MB  (residual={residual:.0f} MB)")
    print(f"  Memory breakdown — params: {model_mb:.1f} MB | optim: {optim_mb:.1f} MB | total fixed: {model_mb+optim_mb:.1f} MB")

    print(f"\n  {'bs':>8}  {'max n (16GB)':>14}  {'max n (32GB)':>14}")
    print(f"  {'-'*40}")
    for bs in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        def n_max(limit, bs=bs):
            return (limit - b * bs - c) / (a * bs)
        print(f"  {bs:>8}  {n_max(LIMIT_16):>14.1f}  {n_max(LIMIT_32):>14.1f}")

    return a, b, c


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"Total VRAM: {total_mb:.0f} MB\n")

    # Grid: enough points to fit 3 coefficients robustly per width
    batch_sizes    = [256, 512, 1024, 2048, 4096, 8192]
    n_particles    = [2, 4, 6, 8, 10, 12]
    num_heads_list = [2, 4, 8, 16, 32, 64]

    all_fits = {}   # (num_heads, use_ema) -> (a, b, c, model_mb)
    all_data = {}   # (num_heads, use_ema) -> list of (bs, n, peak_mb)

    for use_ema in [False, True]:
        for num_heads in num_heads_list:
            ema_label = "EMA" if use_ema else "no EMA"
            print(f"\n{'='*60}")
            print(f"num_heads={num_heads}  hidden_channels={16*num_heads}  {ema_label}")
            print(f"{'='*60}")

            model = instantiate(make_model_cfg(num_heads)).to(DEVICE)
            set_base_shapes(model, model)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            ema = ExponentialMovingAverage(model.parameters(), decay=0.99) if use_ema else None

            model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

            # Warm up: first step allocates optimizer m/v buffers and EMA shadow copy
            print(f"  Warming up...")
            fourmomenta, particle_type, ptr = make_batch(4, 256)
            optimizer.zero_grad()
            model(fourmomenta, particle_type, mean=0.0, std=1.0, ptr=ptr).sum().backward()
            optimizer.step()
            if ema is not None:
                ema.update()
            torch.cuda.empty_cache()

            print(f"  {'bs':>6}  {'n':>3}  {'peak MB':>10}")
            print(f"  {'-'*24}")

            data = []
            for bs in batch_sizes:
                for n in n_particles:
                    try:
                        peak = measure(model, optimizer, n, bs, ema=ema)
                        data.append((bs, n, peak))
                        print(f"  {bs:>6}  {n:>3}  {peak:>10.1f}")
                    except torch.cuda.OutOfMemoryError:
                        print(f"  {bs:>6}  {n:>3}  {'OOM':>10}")
                        torch.cuda.empty_cache()

            a, b, c = fit_and_table(data, num_heads, model_mb)
            all_fits[(num_heads, use_ema)] = (a, b, c, model_mb)
            all_data[(num_heads, use_ema)] = data

            del model, optimizer, ema
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary 1: how model memory and activation coefficient scale with width
    # -----------------------------------------------------------------------
    for use_ema in [False, True]:
        ema_label = "EMA" if use_ema else "no EMA"
        print(f"\n\n{'='*72}")
        print(f"Width scaling summary — {ema_label}")
        print(f"{'='*72}")
        heads_done = [nh for nh in num_heads_list if (nh, use_ema) in all_fits]
        print(f"  {'num_heads':>10}  {'hidden_ch':>10}  {'params MB':>10}  "
              f"{'optim MB':>10}  {'total fixed':>12}  {'a (per bs·n)':>14}")
        print(f"  {'-'*72}")
        for nh in heads_done:
            a, b, c, model_mb = all_fits[(nh, use_ema)]
            print(f"  {nh:>10}  {16*nh:>10}  {model_mb:>10.1f}  "
                  f"{2*model_mb:>10.1f}  {3*model_mb:>12.1f}  {a:>14.4f}")

    # -----------------------------------------------------------------------
    # Summary 2: for each (bs, n), predicted VRAM per num_heads
    # -----------------------------------------------------------------------
    for use_ema in [False, True]:
        ema_label = "EMA" if use_ema else "no EMA"
        heads_done = [nh for nh in num_heads_list if (nh, use_ema) in all_fits]
        heads_str  = "  ".join(f"{nh:>6}" for nh in heads_done)
        print(f"\n\n{'='*72}")
        print(f"Predicted peak VRAM (GB) per num_heads — {ema_label}")
        print(f"  bs={{}}, n={{}} columns = num_heads: {heads_str}")
        print(f"{'='*72}")
        print(f"  {'bs':>6}  {'n':>4}  " + "  ".join(f"{'nh='+str(nh):>6}" for nh in heads_done))
        print(f"  {'-'*60}")
        for bs in [512, 1024, 2048, 4096, 8192]:
            for n in [4, 6, 8, 10]:
                peaks = []
                for nh in heads_done:
                    a, b, c, _ = all_fits[(nh, use_ema)]
                    p = a * bs * n + b * bs + c
                    flag = " ✓" if p <= LIMIT_16 else ("  " if p <= LIMIT_32 else " !")
                    peaks.append(f"{p/1024:>5.1f}{flag}")
                print(f"  {bs:>6}  {n:>4}  " + "   ".join(peaks))
        print(f"\n  ✓ = fits 16GB   (blank) = fits 32GB only   ! = OOM on 32GB")

    make_plots(all_fits, all_data, num_heads_list, batch_sizes, n_particles,
               torch.cuda.get_device_name(0))


def make_plots(all_fits, all_data, num_heads_list, batch_sizes, n_particles, device_name):
    COLORS_N  = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_particles)))
    COLORS_BS = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))
    COLORS_NH = plt.cm.cividis(np.linspace(0.1, 0.9, len(num_heads_list)))

    def lookup(data, bs, n):
        for b, nn, p in data:
            if b == bs and nn == n:
                return p
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f"VRAM scaling  ({device_name})", fontsize=13)

    # ── Panel 0,0: VRAM vs batch_size, vary n, fixed nh=16, no EMA ──────────
    ax = axes[0, 0]
    data = all_data.get((16, False), [])
    a, b, c, _ = all_fits.get((16, False), (0, 0, 0, 0))
    bs_range = np.linspace(min(batch_sizes), max(batch_sizes), 200)
    for i, n in enumerate(n_particles):
        pts = [(bs, p) for bs, nn, p in data if nn == n]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, color=COLORS_N[i], zorder=3)
            ax.plot(bs_range, (a * n + b) * bs_range + c, color=COLORS_N[i],
                    label=f'n={n}', linewidth=1.5)
    ax.set(title='VRAM vs batch size  (num_heads=16, no EMA)',
           xlabel='batch size', ylabel='peak VRAM (MB)', xscale='log', yscale='log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.axhline(16*1024, color='grey', ls='--', lw=1, label='16 GB')
    ax.axhline(32*1024, color='black', ls='--', lw=1, label='32 GB')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # ── Panel 0,1: VRAM vs n_particles, vary bs, fixed nh=16, no EMA ────────
    ax = axes[0, 1]
    n_range = np.linspace(min(n_particles), max(n_particles), 100)
    for i, bs in enumerate(batch_sizes):
        pts = [(n, p) for bss, n, p in data if bss == bs]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, color=COLORS_BS[i], zorder=3)
            ax.plot(n_range, a * bs * n_range + b * bs + c, color=COLORS_BS[i],
                    label=f'bs={bs}', linewidth=1.5)
    ax.set(title='VRAM vs n particles  (num_heads=16, no EMA)',
           xlabel='n particles per event', ylabel='peak VRAM (MB)')
    ax.axhline(16*1024, color='grey', ls='--', lw=1, label='16 GB')
    ax.axhline(32*1024, color='black', ls='--', lw=1, label='32 GB')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # ── Panel 1,0: max affordable n vs batch_size per num_heads (16 GB) ──────
    ax = axes[1, 0]
    heads_done = [nh for nh in num_heads_list if (nh, False) in all_fits]
    for i, nh in enumerate(heads_done):
        a, b, c, _ = all_fits[(nh, False)]
        bss = np.array([256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
        n_max = (16*1024 - b*bss - c) / (a*bss)
        ax.plot(bss, n_max, marker='o', label=f'nh={nh}', color=COLORS_NH[i])
    ax.set(title='Max n particles before 16 GB  (no EMA)',
           xlabel='batch size', ylabel='max n particles', xscale='log', ylim=(0, None))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.axhline(6, color='tomato', ls=':', lw=1.5, label='current max (n=6)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 1,1: max affordable n vs batch_size per num_heads (32 GB) ──────
    ax = axes[1, 1]
    for i, nh in enumerate(heads_done):
        a, b, c, _ = all_fits[(nh, False)]
        bss = np.array([256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
        n_max = (32*1024 - b*bss - c) / (a*bss)
        ax.plot(bss, n_max, marker='o', label=f'nh={nh}', color=COLORS_NH[i])
    ax.set(title='Max n particles before 32 GB  (no EMA)',
           xlabel='batch size', ylabel='max n particles', xscale='log', ylim=(0, None))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.axhline(6, color='tomato', ls=':', lw=1.5, label='current max (n=6)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2,0: fixed memory vs num_heads (params, optim, total) ──────────
    ax = axes[2, 0]
    nh_arr = np.array(heads_done, dtype=float)
    params_mb = np.array([all_fits[(nh, False)][3] for nh in heads_done])
    ax.plot(nh_arr, params_mb,        marker='o', label='params',          color='steelblue')
    ax.plot(nh_arr, 2*params_mb,      marker='s', label='optim (AdamW)',   color='orange')
    ax.plot(nh_arr, 3*params_mb,      marker='^', label='total fixed',     color='green')
    ax.plot(nh_arr, 3*params_mb + params_mb, marker='D', label='+ EMA shadow', color='purple', ls='--')
    # reference quadratic line
    ref = params_mb[0] * (nh_arr / nh_arr[0])**2
    ax.plot(nh_arr, ref, ls=':', color='grey', label='∝ nh² reference')
    ax.set(title='Fixed memory vs num_heads', xlabel='num_heads',
           ylabel='memory (MB)', xscale='log', yscale='log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2,1: activation coefficient `a` vs num_heads ───────────────────
    ax = axes[2, 1]
    a_vals = np.array([all_fits[(nh, False)][0] for nh in heads_done])
    ax.plot(nh_arr, a_vals, marker='o', color='steelblue', label='measured a')
    # reference linear line
    ref_lin = a_vals[0] * (nh_arr / nh_arr[0])
    ax.plot(nh_arr, ref_lin, ls=':', color='grey', label='∝ nh reference')
    ax.set(title='Activation cost coefficient `a` vs num_heads\n'
                 '(peak ≈ a·bs·n + …)', xlabel='num_heads',
           ylabel='a  [MB / (bs · n)]', xscale='log', yscale='log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vram_plots.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"\nSaved VRAM plots to {out}")


if __name__ == "__main__":
    main()
