"""
Regenerate VRAM scaling plots from vram_scaling_output.txt.
No GPU required — parses the saved text output.

Usage:
    python plot_vram.py
"""
import re, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
TXT  = os.path.join(HERE, "vram_scaling_output.txt")
OUT  = os.path.join(HERE, "vram_plots.pdf")

LIMIT_16 = 16 * 1024
LIMIT_32 = 32 * 1024

# ── parse ──────────────────────────────────────────────────────────────────────
all_fits = {}   # (num_heads, use_ema) -> (a, b, c, model_mb)
all_data = {}   # (num_heads, use_ema) -> [(bs, n, peak_mb), ...]

header_re = re.compile(r"num_heads=(\d+)\s+hidden_channels=\d+\s+(no EMA|EMA)")
meas_re   = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$")
fit_re    = re.compile(r"Fit: peak ≈ ([\d.]+)·bs·n \+ ([-\d.]+)·bs \+ ([\d.]+) MB")
mem_re    = re.compile(r"params: ([\d.]+) MB")

current_key  = None
current_data = []

with open(TXT) as f:
    for line in f:
        m = header_re.search(line)
        if m:
            current_key  = (int(m.group(1)), m.group(2) == "EMA")
            current_data = []
            all_data[current_key] = current_data
            continue

        if current_key is None:
            continue

        m = meas_re.match(line)
        if m:
            current_data.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
            continue

        m = fit_re.search(line)
        if m:
            a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
            # model_mb filled when we see the memory breakdown line
            all_fits[current_key] = [a, b, c, None]
            continue

        m = mem_re.search(line)
        if m and current_key in all_fits and all_fits[current_key][3] is None:
            all_fits[current_key][3] = float(m.group(1))

# convert lists to tuples
all_fits = {k: tuple(v) for k, v in all_fits.items()}

num_heads_list = sorted({k[0] for k in all_fits})
batch_sizes    = [256, 512, 1024, 2048, 4096, 8192]
n_particles    = [2, 4, 6, 8, 10, 12]

device_name = "Tesla V100-SXM2-16GB"

# ── plot ───────────────────────────────────────────────────────────────────────
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

# ── Panel 0,0: VRAM vs batch_size, vary n, fixed nh=16, no EMA ────────────────
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
ax.axhline(LIMIT_16, color='grey', ls='--', lw=1, label='16 GB')
ax.axhline(LIMIT_32, color='black', ls='--', lw=1, label='32 GB')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

# ── Panel 0,1: VRAM vs n_particles, vary bs, fixed nh=16, no EMA ──────────────
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
ax.axhline(LIMIT_16, color='grey', ls='--', lw=1, label='16 GB')
ax.axhline(LIMIT_32, color='black', ls='--', lw=1, label='32 GB')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

# ── Panel 1,0: max affordable n vs batch_size per num_heads (16 GB) ───────────
ax = axes[1, 0]
heads_done = [nh for nh in num_heads_list if (nh, False) in all_fits]
bss = np.array([256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
for i, nh in enumerate(heads_done):
    a, b, c, _ = all_fits[(nh, False)]
    n_max = (LIMIT_16 - b * bss - c) / (a * bss)
    n_max = np.clip(n_max, 1e-1, None)
    ax.plot(bss, n_max, marker='o', label=f'nh={nh}', color=COLORS_NH[i])
ax.set(title='Max n particles before 16 GB  (no EMA)',
       xlabel='batch size', ylabel='max n particles', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.axhline(6, color='tomato', ls=':', lw=1.5, label='current max (n=6)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 1,1: max affordable n vs batch_size per num_heads (32 GB) ───────────
ax = axes[1, 1]
for i, nh in enumerate(heads_done):
    a, b, c, _ = all_fits[(nh, False)]
    n_max = (LIMIT_32 - b * bss - c) / (a * bss)
    n_max = np.clip(n_max, 1e-1, None)
    ax.plot(bss, n_max, marker='o', label=f'nh={nh}', color=COLORS_NH[i])
ax.set(title='Max n particles before 32 GB  (no EMA)',
       xlabel='batch size', ylabel='max n particles', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.axhline(6, color='tomato', ls=':', lw=1.5, label='current max (n=6)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 2,0: fixed memory vs num_heads ──────────────────────────────────────
ax = axes[2, 0]
nh_arr    = np.array(heads_done, dtype=float)
params_mb = np.array([all_fits[(nh, False)][3] for nh in heads_done])
ax.plot(nh_arr, params_mb,            marker='o', label='params',        color='steelblue')
ax.plot(nh_arr, 2 * params_mb,        marker='s', label='optim (AdamW)', color='orange')
ax.plot(nh_arr, 3 * params_mb,        marker='^', label='total fixed',   color='green')
ax.plot(nh_arr, 3 * params_mb + params_mb, marker='D', label='+ EMA shadow',
        color='purple', ls='--')
ref = params_mb[0] * (nh_arr / nh_arr[0]) ** 2
ax.plot(nh_arr, ref, ls=':', color='grey', label='∝ nh² reference')
ax.set(title='Fixed memory vs num_heads', xlabel='num_heads',
       ylabel='memory (MB)', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 2,1: activation coefficient `a` vs num_heads ───────────────────────
ax = axes[2, 1]
a_vals  = np.array([all_fits[(nh, False)][0] for nh in heads_done])
ref_lin = a_vals[0] * (nh_arr / nh_arr[0])
ax.plot(nh_arr, a_vals,  marker='o', color='steelblue', label='measured a')
ax.plot(nh_arr, ref_lin, ls=':', color='grey', label='∝ nh reference')
ax.set(title='Activation cost coefficient `a` vs num_heads\n(peak ≈ a·bs·n + …)',
       xlabel='num_heads', ylabel='a  [MB / (bs · n)]', xscale='log', yscale='log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight')
print(f"Saved to {OUT}")
