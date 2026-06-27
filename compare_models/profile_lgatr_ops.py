"""Profile where time goes in full L-GATr and slim, forward+backward, on GPU.

Builds the real MuP nets at the measurement config, feeds synthetic data matching
the real per-step shapes (N tokens spread over tiny events, block-diagonal xformers
bias), and reports a per-op CUDA-time breakdown plus isolated microbenchmarks of the
geometric_product and equi_linear primitives.
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

torch.manual_seed(0)
dev = torch.device("cuda")
torch.set_default_dtype(torch.float32)

from models.attention_lloca_mup import build_block_diagonal_bias

# --- synthetic batch matching real workload: 1024 events, ~4.5 particles each ---
import numpy as np
n_events = 1024
counts = np.random.randint(4, 7, size=n_events)        # 4..6 particles/event
N = int(counts.sum())
ptr = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), device=dev)
seq_lens = tuple(int(c) for c in counts)
attn_bias = build_block_diagonal_bias(ptr, seq_lens)
print(f"N tokens = {N}, events = {n_events}, attn_bias = {type(attn_bias).__name__}")

def sync(): torch.cuda.synchronize()

def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters * 1e3   # ms

# ---------------- microbench primitives ----------------
from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.linear import equi_linear
C = 22
x_mv = torch.randn(1, N, C, 16, device=dev, requires_grad=True)
y_mv = torch.randn(1, N, C, 16, device=dev, requires_grad=True)
coeffs = torch.randn(C, C, 10, device=dev, requires_grad=True)

def gp_fwd(): return geometric_product(x_mv, y_mv)
def gp_fwdbwd():
    o = geometric_product(x_mv, y_mv); o.sum().backward()
def el_fwd(): return equi_linear(x_mv, coeffs)
def el_fwdbwd():
    o = equi_linear(x_mv, coeffs); o.sum().backward()

print(f"\n[micro] geometric_product  fwd: {bench(gp_fwd):.3f} ms   fwd+bwd: {bench(gp_fwdbwd):.3f} ms  (C={C}, N={N})")
print(f"[micro] equi_linear        fwd: {bench(el_fwd):.3f} ms   fwd+bwd: {bench(el_fwdbwd):.3f} ms  (C={C}, N={N})")

# ---------------- full net fwd+bwd + profiler ----------------
def build_full():
    from models.lgatr_mup import MuPLGATr
    return MuPLGATr(
        in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=22,
        in_s_channels=18, out_s_channels=None, hidden_s_channels=22,
        num_blocks=8, num_heads=2, parametrization="sp",  # std param avoids mup base-shape setup
    ).to(dev)

def build_slim():
    from models.lgatr_slim_mup import MuPLGATrSlim
    return MuPLGATrSlim(
        in_v_channels=1, out_v_channels=1, hidden_v_channels=52,
        in_s_channels=18, out_s_channels=1, hidden_s_channels=104,
        num_blocks=8, num_heads=2, parametrization="sp",
    ).to(dev)

def run_model(tag, model, mv_in, s_in):
    def step():
        out = model(mv_in, scalars=s_in, attn_bias=attn_bias)
        loss = (out[0] if isinstance(out, tuple) else out)
        loss = loss.float().pow(2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
    ms = bench(step, iters=30, warmup=10)
    print(f"\n[{tag}] full fwd+bwd step: {ms:.2f} ms")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10): step()
        sync()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))

# full
s_in = torch.randn(1, N, 18, device=dev)
mv_in = torch.randn(1, N, 1, 16, device=dev)
try:
    m = build_full()
    run_model("FULL", m, mv_in, s_in)
except Exception as e:
    import traceback; print("FULL failed:", e); traceback.print_exc()

# slim
try:
    m = build_slim()
    v_in = torch.randn(1, N, 1, 4, device=dev)
    run_model("SLIM", m, v_in, s_in)
except Exception as e:
    import traceback; print("SLIM failed:", e); traceback.print_exc()
print("\nPROFILE DONE")
