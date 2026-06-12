"""
THROWAWAY benchmark: memory vs speed tradeoff of model.net.checkpoint_blocks.

For each num_heads, builds the model with checkpoint_blocks False and True, runs
fwd+bwd+optimizer.step(), and reports peak VRAM and mean step time both ways.

Run on Jean Zay with a GPU:
    python checkpoint_tradeoff.py
Delete when done.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")          # silence the use_reentrant checkpoint warning

import torch
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from mup import set_base_shapes

DEVICE    = torch.device("cuda")
N_SCALARS = 18
NUM_BLOCKS = 8
BS, N_PART = 4096, 6
HEADS     = [8, 16, 32]
N_WARMUP, N_ITERS = 5, 20


def make_model(num_heads, checkpoint_blocks):
    cfg = OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars": N_SCALARS, "hidden_channels_mlp": 128, "num_layers_mlp": 2,
        "in_channels": N_SCALARS + 4, "attn_reps": "8x0n+2x1n", "out_channels": 1,
        "num_blocks": NUM_BLOCKS, "num_heads": num_heads,
        "checkpoint_blocks": checkpoint_blocks,
    })
    m = instantiate(cfg).to(DEVICE)
    set_base_shapes(m, m)
    return m


def make_batch():
    N = BS * N_PART
    p = torch.randn(N, 3, device=DEVICE); mass = torch.rand(N, 1, device=DEVICE) + 0.1
    E = (p.pow(2).sum(-1, keepdim=True) + mass.pow(2)).sqrt()
    fm = torch.cat([E, p], dim=-1)
    pt = torch.randn(N, N_SCALARS, device=DEVICE)
    ptr = torch.arange(0, N + 1, N_PART, dtype=torch.long, device=DEVICE)
    return fm, pt, ptr


def step(model, opt):
    fm, pt, ptr = make_batch()
    opt.zero_grad()
    model(fm, pt, mean=0.0, std=1.0, ptr=ptr).sum().backward()
    opt.step()


def bench(num_heads, checkpoint_blocks):
    """Returns (step_ms, peak_MB) or (None, None) on OOM."""
    try:
        model = make_model(num_heads, checkpoint_blocks)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(N_WARMUP):
            step(model, opt)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            step(model, opt)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITERS * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**2
        del model, opt
        torch.cuda.empty_cache()
        return ms, peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


print(f"Device: {torch.cuda.get_device_name(0)}  |  BS={BS} n={N_PART} blocks={NUM_BLOCKS}\n")
print(f"  {'nh':>4} | {'mem off':>9} {'mem on':>9} {'mem saved':>10} | "
      f"{'time off':>9} {'time on':>9} {'time cost':>10}")
print(f"  {'-'*4}-+-{'-'*30}-+-{'-'*31}")
for nh in HEADS:
    ms_off, mem_off = bench(nh, False)
    ms_on,  mem_on  = bench(nh, True)

    def fmt_mem(x): return f"{x:.0f}MB" if x is not None else "OOM"
    def fmt_ms(x):  return f"{x:.1f}ms" if x is not None else "—"
    mem_saved = f"{100*(1-mem_on/mem_off):.0f}%" if (mem_off and mem_on) else "—"
    time_cost = f"+{100*(ms_on/ms_off-1):.0f}%" if (ms_off and ms_on) else "—"
    print(f"  {nh:>4} | {fmt_mem(mem_off):>9} {fmt_mem(mem_on):>9} {mem_saved:>10} | "
          f"{fmt_ms(ms_off):>9} {fmt_ms(ms_on):>9} {time_cost:>10}")

print("\nmem saved = how much peak VRAM checkpointing frees; "
      "time cost = extra step time from recomputing blocks in backward.")
