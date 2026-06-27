"""A/B test speedup interventions for full L-GATr (and slim) fwd+bwd on GPU.

Interventions:
  1. baseline eager fp32
  2. torch.compile (static shapes)
  3. torch.compile (dynamic=True)  -- real training has variable N tokens/batch
  4. fp16 autocast (eager)
  5. fp16 autocast + compile(dynamic)

Reports ms/step and, for each, the max abs/rel deviation of the output vs the fp32
eager baseline on identical inputs (compile must match ~exactly; fp16 will differ).
Variable-N robustness for compile is checked by running a second batch with a
different token count and confirming no error / no pathological recompile stall.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.manual_seed(0)
dev = torch.device("cuda")
torch.set_default_dtype(torch.float32)
from models.attention_lloca_mup import build_block_diagonal_bias

def make_batch(n_events=1024, lo=4, hi=7, seed=0):
    rng = np.random.RandomState(seed)
    counts = rng.randint(lo, hi, size=n_events)
    N = int(counts.sum())
    ptr = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), device=dev)
    seq_lens = tuple(int(c) for c in counts)
    bias = build_block_diagonal_bias(ptr, seq_lens)
    return N, bias

def sync(): torch.cuda.synchronize()

def bench(step, iters=30, warmup=10):
    for _ in range(warmup): step()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): step()
    sync(); return (time.perf_counter() - t0) / iters * 1e3

def build_full():
    from models.lgatr_mup import MuPLGATr
    return MuPLGATr(in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=22,
                    in_s_channels=18, out_s_channels=None, hidden_s_channels=22,
                    num_blocks=8, num_heads=2, parametrization="sp").to(dev)

N, bias = make_batch(seed=0)
N2, bias2 = make_batch(seed=1)   # different token count
print(f"N={N}, N2={N2}")

def make_inputs(N):
    return (torch.randn(1, N, 1, 16, device=dev), torch.randn(1, N, 18, device=dev))

mv_in, s_in = make_inputs(N)

def loss_of(out):
    o = out[0] if isinstance(out, tuple) else out
    return o.float().pow(2).mean()

torch.manual_seed(42)
model = build_full()
model.eval()  # eval to make output deterministic across calls (no dropout); grads still flow

# reference output (fp32 eager)
with torch.no_grad():
    ref = model(mv_in, scalars=s_in, attn_bias=bias)[0].clone()

def dev_vs_ref(out_mv):
    d = (out_mv - ref).abs()
    return d.max().item(), (d / (ref.abs() + 1e-6)).max().item()

def make_step(fn, autocast=False, dual=False):
    def step():
        if autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                out = fn(mv_in, scalars=s_in, attn_bias=bias)
                loss = loss_of(out)
        else:
            out = fn(mv_in, scalars=s_in, attn_bias=bias)
            loss = loss_of(out)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if dual:  # exercise a 2nd token count (recompile / dynamic-shape check)
            out2 = fn(torch.randn(1, N2, 1, 16, device=dev),
                      scalars=torch.randn(1, N2, 18, device=dev), attn_bias=bias2)
            loss_of(out2).backward()
            model.zero_grad(set_to_none=True)
    return step

# 1. baseline
ms = bench(make_step(model))
with torch.no_grad():
    o = model(mv_in, scalars=s_in, attn_bias=bias)[0]
print(f"\n[1] baseline eager fp32          : {ms:7.2f} ms/step")

# 2. compile static
try:
    cmodel = torch.compile(model)
    ms = bench(make_step(cmodel), iters=30, warmup=15)
    with torch.no_grad():
        o = cmodel(mv_in, scalars=s_in, attn_bias=bias)[0]
    md, mr = dev_vs_ref(o)
    print(f"[2] compile (static)             : {ms:7.2f} ms/step   maxabs={md:.2e} maxrel={mr:.2e}")
except Exception as e:
    import traceback; traceback.print_exc(); print("[2] compile static FAILED:", e)

# 3. compile dynamic
try:
    torch._dynamo.reset()
    dmodel = torch.compile(model, dynamic=True)
    ms = bench(make_step(dmodel, dual=True), iters=20, warmup=15)
    with torch.no_grad():
        o = dmodel(mv_in, scalars=s_in, attn_bias=bias)[0]
    md, mr = dev_vs_ref(o)
    print(f"[3] compile (dynamic, 2 sizes)   : {ms:7.2f} ms/step   maxabs={md:.2e} maxrel={mr:.2e}")
except Exception as e:
    import traceback; traceback.print_exc(); print("[3] compile dynamic FAILED:", e)

# 4. fp16 autocast eager
try:
    ms = bench(make_step(model, autocast=True))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = model(mv_in, scalars=s_in, attn_bias=bias)[0].float()
    md, mr = dev_vs_ref(o)
    print(f"[4] fp16 autocast (eager)        : {ms:7.2f} ms/step   maxabs={md:.2e} maxrel={mr:.2e}")
except Exception as e:
    import traceback; traceback.print_exc(); print("[4] fp16 autocast FAILED:", e)

# 5. fp16 autocast + compile dynamic
try:
    torch._dynamo.reset()
    dmodel = torch.compile(model, dynamic=True)
    ms = bench(make_step(dmodel, autocast=True), iters=20, warmup=15)
    print(f"[5] fp16 autocast + compile      : {ms:7.2f} ms/step")
except Exception as e:
    import traceback; traceback.print_exc(); print("[5] fp16+compile FAILED:", e)

print("\nOPTS DONE")
