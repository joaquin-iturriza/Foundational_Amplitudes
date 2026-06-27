"""
Iteration time scaling: step time and throughput as a function of
(num_heads, batch_size, n_particles).

Measures N_ITERS steps after N_WARMUP warmup steps, reports mean ± std,
and saves plots to iteration_time_plots.pdf.

Run on Jean-Zay with a GPU allocated:
    python test_iteration_time.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iteration_time_output.txt"), "w")
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = _Tee(sys.__stdout__, _log)

import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from hydra.utils import instantiate
from omegaconf import OmegaConf
from mup import set_base_shapes

N_SCALARS = 18
DEVICE    = torch.device("cuda")
N_WARMUP  = 5
N_ITERS   = 20

# Sweep axes
ALL_HEADS = [4, 8, 16, 32, 64]
ALL_BS    = [256, 512, 1024, 2048, 4096, 8192]
ALL_N     = [2, 4, 6, 8, 10]


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
    N = batch_size * n_particles
    p = torch.randn(N, 3, device=DEVICE)
    m = torch.rand(N, 1, device=DEVICE) + 0.1
    E = (p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt()
    fourmomenta   = torch.cat([E, p], dim=-1)
    particle_type = torch.randn(N, N_SCALARS, device=DEVICE)
    ptr = torch.arange(0, N + 1, n_particles, dtype=torch.long, device=DEVICE)
    return fourmomenta, particle_type, ptr


def time_step(model, optimizer, n, bs):
    """Returns mean step time in ms, or None on OOM."""
    # warmup
    for _ in range(N_WARMUP):
        try:
            fourmomenta, particle_type, ptr = make_batch(n, bs)
            optimizer.zero_grad()
            model(fourmomenta, particle_type, mean=0.0, std=1.0, ptr=ptr).sum().backward()
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    torch.cuda.synchronize()
    times = []
    for _ in range(N_ITERS):
        try:
            fourmomenta, particle_type, ptr = make_batch(n, bs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad()
            model(fourmomenta, particle_type, mean=0.0, std=1.0, ptr=ptr).sum().backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)   # ms
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    return np.mean(times), np.std(times)


# ── collect all measurements ────────────────────────────────────────────────
# results[num_heads][bs][n] = (mean_ms, std_ms)  or  None
results = {nh: {bs: {} for bs in ALL_BS} for nh in ALL_HEADS}

for num_heads in ALL_HEADS:
    print(f"\n=== num_heads={num_heads} ===")
    model     = instantiate(make_model_cfg(num_heads)).to(DEVICE)
    set_base_shapes(model, model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # warm up optimizer buffers
    fourmomenta, pt, ptr = make_batch(4, 256)
    optimizer.zero_grad()
    model(fourmomenta, pt, mean=0.0, std=1.0, ptr=ptr).sum().backward()
    optimizer.step()
    torch.cuda.empty_cache()

    print(f"  {'bs':>6}  {'n':>4}  {'mean ms':>10}  {'std ms':>8}  {'ksamples/s':>12}")
    for bs in ALL_BS:
        for n in ALL_N:
            r = time_step(model, optimizer, n, bs)
            results[num_heads][bs][n] = r
            if r is not None:
                mean_ms, std_ms = r
                throughput = bs / (mean_ms / 1000) / 1000   # ksamples/s
                print(f"  {bs:>6}  {n:>4}  {mean_ms:>10.2f}  {std_ms:>8.2f}  {throughput:>12.1f}")
            else:
                print(f"  {bs:>6}  {n:>4}  {'OOM':>10}")

    del model, optimizer
    torch.cuda.empty_cache()


# ── plotting ─────────────────────────────────────────────────────────────────
COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(ALL_HEADS), len(ALL_N), len(ALL_BS))))

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle(f"Iteration time & throughput  ({torch.cuda.get_device_name(0)})", fontsize=13)

def get_times(results, num_heads, bs_list, n):
    """Collect (bs, mean_ms) pairs where result is not None."""
    xs, ys, yerrs = [], [], []
    for bs in bs_list:
        r = results[num_heads][bs].get(n)
        if r is not None:
            xs.append(bs); ys.append(r[0]); yerrs.append(r[1])
    return np.array(xs), np.array(ys), np.array(yerrs)

def get_times_n(results, num_heads, bs, n_list):
    xs, ys, yerrs = [], [], []
    for n in n_list:
        r = results[num_heads][bs].get(n)
        if r is not None:
            xs.append(n); ys.append(r[0]); yerrs.append(r[1])
    return np.array(xs), np.array(ys), np.array(yerrs)

# ── Row 0: time vs BS and throughput vs BS  (vary n, fixed num_heads=16) ────
ax = axes[0, 0]
for i, n in enumerate(ALL_N):
    xs, ys, yerrs = get_times(results, 16, ALL_BS, n)
    if len(xs): ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'n={n}', color=COLORS[i])
ax.set(title='Time vs batch size  (num_heads=16)', xlabel='batch size',
       ylabel='step time (ms)', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='n particles'); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for i, n in enumerate(ALL_N):
    xs, ys, _ = get_times(results, 16, ALL_BS, n)
    if len(xs):
        throughput = xs / (ys / 1000) / 1000
        ax.plot(xs, throughput, marker='o', label=f'n={n}', color=COLORS[i])
ax.set(title='Throughput vs batch size  (num_heads=16)', xlabel='batch size',
       ylabel='throughput (k samples/s)', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='n particles'); ax.grid(True, alpha=0.3)

# ── Row 1: time vs BS and throughput vs BS  (vary num_heads, fixed n=6) ─────
ax = axes[1, 0]
for i, nh in enumerate(ALL_HEADS):
    xs, ys, yerrs = get_times(results, nh, ALL_BS, n=6)
    if len(xs): ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'nh={nh}', color=COLORS[i])
ax.set(title='Time vs batch size  (n=6)', xlabel='batch size',
       ylabel='step time (ms)', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='num_heads'); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for i, nh in enumerate(ALL_HEADS):
    xs, ys, _ = get_times(results, nh, ALL_BS, n=6)
    if len(xs):
        throughput = xs / (ys / 1000) / 1000
        ax.plot(xs, throughput, marker='o', label=f'nh={nh}', color=COLORS[i])
ax.set(title='Throughput vs batch size  (n=6)', xlabel='batch size',
       ylabel='throughput (k samples/s)', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='num_heads'); ax.grid(True, alpha=0.3)

# ── Row 2: time vs n_particles and time vs num_heads ─────────────────────────
ax = axes[2, 0]
for i, bs in enumerate(ALL_BS):
    xs, ys, yerrs = get_times_n(results, 16, bs, ALL_N)
    if len(xs): ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='Time vs n particles  (num_heads=16)', xlabel='n particles per event',
       ylabel='step time (ms)')
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

ax = axes[2, 1]
for i, (bs, n) in enumerate([(512, 6), (1024, 6), (2048, 6), (4096, 6), (8192, 6)]):
    xs, ys, yerrs = [], [], []
    for nh in ALL_HEADS:
        r = results[nh][bs].get(n)
        if r is not None:
            xs.append(nh); ys.append(r[0]); yerrs.append(r[1])
    if xs:
        ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='Time vs num_heads  (n=6)', xlabel='num_heads',
       ylabel='step time (ms)', yscale='log')
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iteration_time_plots.pdf")
plt.savefig(out_path, bbox_inches='tight')
print(f"\nSaved plots to {out_path}")
