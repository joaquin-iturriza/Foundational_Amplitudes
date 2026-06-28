"""GPU micro-benchmark: batched DiagramEncoder.encode_all vs the per-process loop.

Times both paths on the full 25-process recipe set (the A/B workload) to confirm
the per-step optimization. CUDA-only (run via sbatch on a GPU node).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml

from diagram_graphs import load_diagram_registry, feature_dims, build_diagram_batch
from models.diagram_encoder import DiagramEncoder

assert torch.cuda.is_available(), "needs a GPU"
dev = torch.device("cuda")

procs = [p["name"] for p in yaml.safe_load(open("recipes/pretrain25_short.yaml"))["processes"]]
reg, n_prop = load_diagram_registry("data/diagrams", procs, spin_onehot=True,
                                    is_massless=True, standardize=True, k_pe=8, max_diagrams=512)
f_node, f_edge = feature_dims(n_prop)
enc = DiagramEncoder(f_node, f_edge, k_pe=8, d_model=64, n_heads=4, n_layers=3, d_out=32).to(dev)
pd_by_pid = [reg[n].to(dev) for n in procs]
batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
         for k, v in build_diagram_batch(pd_by_pid).items()}
total_D = sum(pd.n_diagrams for pd in pd_by_pid)
print(f"{len(procs)} processes, {total_D} total diagrams")


def loop_forward():
    E = torch.zeros(len(pd_by_pid), 32, device=dev)
    for pid in torch.unique(torch.arange(len(pd_by_pid), device=dev)).tolist():
        E = E.index_copy(0, torch.tensor([pid], device=dev), enc(pd_by_pid[pid]).unsqueeze(0))
    return E


def batched_forward():
    return enc.encode_all(batch)


def bench(fn, n=50, train=True):
    # warmup
    for _ in range(5):
        out = fn()
        if train:
            out.pow(2).sum().backward(); enc.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = fn()
        if train:
            out.pow(2).sum().backward(); enc.zero_grad()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000  # ms/call


for train in (False, True):
    tl = bench(loop_forward, train=train)
    tb = bench(batched_forward, train=train)
    tag = "fwd+bwd" if train else "fwd-only"
    print(f"[{tag}]  loop {tl:7.2f} ms   batched {tb:7.2f} ms   speedup {tl / tb:5.2f}x")
