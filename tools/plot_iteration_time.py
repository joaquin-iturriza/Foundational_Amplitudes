"""
Regenerate iteration time plots from iteration_time_output.txt.
No GPU required — parses the saved text output.

Usage:
    python plot_iteration_time.py
"""
import re, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
TXT  = os.path.join(HERE, "iteration_time_output.txt")
OUT  = os.path.join(HERE, "iteration_time_plots.pdf")

# ── parse ──────────────────────────────────────────────────────────────────────
# results[num_heads][bs][n] = (mean_ms, std_ms) or None
results = {}

header_re = re.compile(r"=== num_heads=(\d+) ===")
row_re    = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
oom_re    = re.compile(r"^\s+(\d+)\s+(\d+)\s+OOM\s*$")

current_nh = None
with open(TXT) as f:
    for line in f:
        m = header_re.search(line)
        if m:
            current_nh = int(m.group(1))
            results[current_nh] = {}
            continue
        if current_nh is None:
            continue
        m = row_re.match(line)
        if m:
            bs, n = int(m.group(1)), int(m.group(2))
            mean_ms, std_ms = float(m.group(3)), float(m.group(4))
            results[current_nh].setdefault(bs, {})[n] = (mean_ms, std_ms)
            continue
        m = oom_re.match(line)
        if m:
            bs, n = int(m.group(1)), int(m.group(2))
            results[current_nh].setdefault(bs, {})[n] = None

ALL_HEADS = sorted(results.keys())
ALL_BS    = sorted({bs for nh in results for bs in results[nh]})
ALL_N     = sorted({n for nh in results for bs in results[nh] for n in results[nh][bs]
                    if n != 2})   # drop n=2

# ── helpers ────────────────────────────────────────────────────────────────────
def get_by_bs(nh, n, bs_list):
    xs, ys, yerrs = [], [], []
    for bs in bs_list:
        r = results[nh].get(bs, {}).get(n)
        if r is not None:
            xs.append(bs); ys.append(r[0]); yerrs.append(r[1])
    return np.array(xs), np.array(ys), np.array(yerrs)

def get_by_n(nh, bs, n_list):
    xs, ys, yerrs = [], [], []
    for n in n_list:
        r = results[nh].get(bs, {}).get(n)
        if r is not None:
            xs.append(n); ys.append(r[0]); yerrs.append(r[1])
    return np.array(xs), np.array(ys), np.array(yerrs)

def get_by_nh(n, bs, nh_list):
    xs, ys, yerrs = [], [], []
    for nh in nh_list:
        r = results[nh].get(bs, {}).get(n)
        if r is not None:
            xs.append(nh); ys.append(r[0]); yerrs.append(r[1])
    return np.array(xs), np.array(ys), np.array(yerrs)

COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(ALL_HEADS), len(ALL_N), len(ALL_BS))))

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Iteration time & throughput  (Tesla V100-SXM2-16GB)", fontsize=13)

# ── Row 0: time vs BS and throughput vs BS  (vary n, fixed nh=16) ─────────────
ax = axes[0, 0]
for i, n in enumerate(ALL_N):
    xs, ys, yerrs = get_by_bs(16, n, ALL_BS)
    if len(xs):
        ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'n={n}', color=COLORS[i])
ax.set(title='Time vs batch size  (num_heads=16)', xlabel='batch size',
       ylabel='step time (ms)', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='n particles'); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for i, n in enumerate(ALL_N):
    xs, ys, _ = get_by_bs(16, n, ALL_BS)
    if len(xs):
        throughput = xs / (ys / 1000) / 1000
        ax.plot(xs, throughput, marker='o', label=f'n={n}', color=COLORS[i])
ax.set(title='Throughput vs batch size  (num_heads=16)', xlabel='batch size',
       ylabel='throughput (k samples/s)', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='n particles'); ax.grid(True, alpha=0.3)

# ── Row 1: time vs BS and throughput vs BS  (vary nh, fixed n=6) ──────────────
ax = axes[1, 0]
for i, nh in enumerate(ALL_HEADS):
    xs, ys, yerrs = get_by_bs(nh, 6, ALL_BS)
    if len(xs):
        ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'nh={nh}', color=COLORS[i])
ax.set(title='Time vs batch size  (n=6)', xlabel='batch size',
       ylabel='step time (ms)', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='num_heads'); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for i, nh in enumerate(ALL_HEADS):
    xs, ys, _ = get_by_bs(nh, 6, ALL_BS)
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
    xs, ys, yerrs = get_by_n(16, bs, ALL_N)
    if len(xs):
        ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='Time vs n particles  (num_heads=16)', xlabel='n particles per event',
       ylabel='step time (ms)')
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

ax = axes[2, 1]
for i, bs in enumerate([512, 1024, 2048, 4096, 8192]):
    xs, ys, yerrs = get_by_nh(6, bs, ALL_HEADS)
    if len(xs):
        ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='Time vs num_heads  (n=6)', xlabel='num_heads',
       ylabel='step time (ms)', yscale='log')
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

plt.tight_layout()

# ── Figure 2: compute efficiency (TFLOPs/s) ────────────────────────────────────
# MLP-dominated FLOPs estimate per step (forward + backward ≈ 3× forward):
#   8 blocks × (QKV 6d² + out_proj 2d² + MLP 16d²) × bs×n × 3
#   = 8 × 24 × 3 × d² × bs × n = 576 × (16·n_h)² × bs × n
NUM_BLOCKS = 8
FLOPS_PER_TOKEN_PER_D2 = 576   # 8 blocks × 24 × 3 (fwd+bwd)

def tflops(nh, bs, n, mean_ms):
    d = 16 * nh
    flops = FLOPS_PER_TOKEN_PER_D2 * d**2 * bs * n
    return flops / (mean_ms * 1e-3) / 1e12   # TFLOPs/s

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Compute efficiency  (TFLOPs/s, MLP-dominated estimate)", fontsize=13)

# ── (0,0): TFLOPs/s vs BS, vary n_h, fixed n=6 ────────────────────────────────
ax = axes2[0, 0]
for i, nh in enumerate(ALL_HEADS):
    xs, ys_ms, _ = get_by_bs(nh, 6, ALL_BS)
    if len(xs):
        tf = [tflops(nh, bs, 6, ms) for bs, ms in zip(xs, ys_ms)]
        ax.plot(xs, tf, marker='o', label=f'nh={nh}', color=COLORS[i])
ax.set(title='TFLOPs/s vs batch size  (n=6)', xlabel='batch size',
       ylabel='TFLOPs/s', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='num_heads'); ax.grid(True, alpha=0.3)

# ── (0,1): TFLOPs/s vs BS, vary n, fixed n_h=16 ───────────────────────────────
ax = axes2[0, 1]
for i, n in enumerate(ALL_N):
    xs, ys_ms, _ = get_by_bs(16, n, ALL_BS)
    if len(xs):
        tf = [tflops(16, bs, n, ms) for bs, ms in zip(xs, ys_ms)]
        ax.plot(xs, tf, marker='o', label=f'n={n}', color=COLORS[i])
ax.set(title='TFLOPs/s vs batch size  (num_heads=16)', xlabel='batch size',
       ylabel='TFLOPs/s', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='n particles'); ax.grid(True, alpha=0.3)

# ── (1,0): TFLOPs/s vs n_h, vary bs, fixed n=6 ────────────────────────────────
ax = axes2[1, 0]
for i, bs in enumerate(ALL_BS):
    xs, ys_ms, _ = get_by_nh(6, bs, ALL_HEADS)
    if len(xs):
        tf = [tflops(nh, bs, 6, ms) for nh, ms in zip(xs, ys_ms)]
        ax.plot(xs, tf, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='TFLOPs/s vs num_heads  (n=6)', xlabel='num_heads',
       ylabel='TFLOPs/s', xscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

# ── (1,1): TFLOPs/s vs n, vary bs, fixed n_h=16 ──────────────────────────────
ax = axes2[1, 1]
for i, bs in enumerate(ALL_BS):
    xs, ys_ms, _ = get_by_n(16, bs, ALL_N)
    if len(xs):
        tf = [tflops(16, bs, n, ms) for n, ms in zip(xs, ys_ms)]
        ax.plot(xs, tf, marker='o', label=f'bs={bs}', color=COLORS[i])
ax.set(title='TFLOPs/s vs n particles  (num_heads=16)', xlabel='n particles per event',
       ylabel='TFLOPs/s')
ax.legend(title='batch size'); ax.grid(True, alpha=0.3)

fig2.tight_layout()

with PdfPages(OUT) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    pdf.savefig(fig2, bbox_inches='tight')
print(f"Saved to {OUT}")
