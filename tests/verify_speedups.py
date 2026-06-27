"""
Verify the three per-step vectorization fixes: equivalence (new == old) and
wall-time (new faster). Run on Jean Zay with a GPU allocated:

    python verify_speedups.py

Fixes under test (each has an env-var A/B toggle, default = fast path):
  #1 attention mask built once/forward   LLOCA_ATTN_MASK = per_forward | per_block
  #2 per-process loss segment-mean       LLOCA_PROC_LOSS = vectorized | loop
  #3 L2/L1 regularization via _foreach    LLOCA_REG       = foreach    | loop
  #4 per-step host/device syncs           LLOCA_SYNC      = deferred   | blocking

#1 is tested against the real model. #2 and #3 reimplement both branches inline
on synthetic GPU tensors (the inline code mirrors experiment.py exactly), so we
don't need to build the full AmplitudeExperiment (data/Hydra) just to time them.
#4 uses the real model: it checks the seq_lens mask path is bit-identical to the
ptr path, then times a training step mirroring base_experiment._step's two sync
structures (4 separate syncs vs 1 fused sync + a CPU-side seq_lens).
"""
import os
import time

import torch
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from mup import set_base_shapes

DEVICE     = torch.device("cuda")
N_SCALARS  = 18
NUM_BLOCKS = 8
NUM_HEADS  = 4          # small model = where per-step overhead hurts most
BS         = 4096
N_PART     = 4
N_WARMUP   = 10
N_ITERS    = 50
torch.manual_seed(0)


def make_model():
    cfg = OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars":         N_SCALARS,
        "hidden_channels_mlp": 128,
        "num_layers_mlp":      2,
        "in_channels":         N_SCALARS + 4,
        "attn_reps":           "8x0n+2x1n",
        "out_channels":        1,
        "num_blocks":          NUM_BLOCKS,
        "num_heads":           NUM_HEADS,
    })
    model = instantiate(cfg).to(DEVICE)
    set_base_shapes(model, model)
    return model


def make_batch():
    N = BS * N_PART
    p = torch.randn(N, 3, device=DEVICE)
    m = torch.rand(N, 1, device=DEVICE) + 0.1
    E = (p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt()
    fourmomenta   = torch.cat([E, p], dim=-1)
    particle_type = torch.randn(N, N_SCALARS, device=DEVICE)
    ptr = torch.arange(0, N + 1, N_PART, dtype=torch.long, device=DEVICE)
    return fourmomenta, particle_type, ptr


def timeit(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1000  # ms/iter


# ───────────────────────── #1 attention mask ──────────────────────────────────
def test_attention():
    print("\n=== #1 attention mask (per_block vs per_forward) ===")
    model = make_model()
    fm, pt, ptr = make_batch()

    # equivalence: same input + weights, eval/no_grad → must match bit-for-bit
    model.eval()
    with torch.no_grad():
        # warm up: the framesnet initializes edge-standardization buffers on the
        # FIRST forward, so compare only after that one-time init has happened.
        os.environ["LLOCA_ATTN_MASK"] = "per_forward"
        model(fm, pt, mean=0.0, std=1.0, ptr=ptr)

        # determinism baseline: same mode twice → isolates kernel nondeterminism
        a = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
        b = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
        base = (a - b).abs().max().item()

        os.environ["LLOCA_ATTN_MASK"] = "per_block"
        out_old = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
        os.environ["LLOCA_ATTN_MASK"] = "per_forward"
        out_new = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
    cross = (out_new - out_old).abs().max().item()
    print(f"  determinism baseline (same mode, 2 calls) = {base:.3e}")
    print(f"  max|per_forward - per_block|              = {cross:.3e}")
    print(f"  -> {'OK: within kernel nondeterminism' if cross <= max(base*2, 1e-6) else 'MISMATCH: investigate'}")

    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        model(fm, pt, mean=0.0, std=1.0, ptr=ptr).sum().backward()
        opt.step()

    os.environ["LLOCA_ATTN_MASK"] = "per_block";   t_old = timeit(step)
    os.environ["LLOCA_ATTN_MASK"] = "per_forward"; t_new = timeit(step)
    print(f"  per_block   : {t_old:7.3f} ms/iter")
    print(f"  per_forward : {t_new:7.3f} ms/iter   ({t_old / t_new:.2f}x)")


# ───────────────────────── #2 per-process loss ────────────────────────────────
def test_proc_loss():
    print("\n=== #2 per-process loss (loop vs vectorized, MSE) ===")
    n_datasets = 10
    y_pred = torch.randn(BS, 1, device=DEVICE)
    y      = torch.randn(BS, 1, device=DEVICE)
    process_ids = torch.randint(0, n_datasets, (BS,), device=DEVICE)
    mse = torch.nn.MSELoss()

    def loop(agg):
        uniq = torch.unique(process_ids)
        per = [mse(y_pred[process_ids == p], y[process_ids == p]) for p in uniq]
        if agg == "geometric_mean" and len(per) > 1:
            return torch.stack(per).log().mean().exp()
        return torch.stack(per).mean()

    def vec(agg):
        per_event = ((y_pred - y) ** 2).flatten(1).mean(dim=1)
        sums   = per_event.new_zeros(n_datasets)
        counts = per_event.new_zeros(n_datasets)
        sums.index_add_(0, process_ids, per_event)
        counts.index_add_(0, process_ids, torch.ones_like(per_event))
        present   = counts > 0
        n_present = present.sum().clamp(min=1)
        proc_mean = sums / counts.clamp(min=1)
        if agg == "geometric_mean":
            log_pm = torch.where(present, proc_mean.clamp(min=1e-30).log(),
                                 torch.zeros_like(proc_mean))
            return (log_pm.sum() / n_present).exp()
        return proc_mean.sum() / n_present

    for agg in ("mean", "geometric_mean"):
        d = (loop(agg) - vec(agg)).abs().item()
        print(f"  [{agg:14s}] |loop - vec| = {d:.3e}  (expect ~0)")
    t_old = timeit(lambda: loop("mean"))
    t_new = timeit(lambda: vec("mean"))
    print(f"  loop       : {t_old:7.3f} ms/call")
    print(f"  vectorized : {t_new:7.3f} ms/call   ({t_old / t_new:.2f}x)")


# ───────────────────────── #3 regularization ──────────────────────────────────
def test_reg():
    print("\n=== #3 L2 regularization (loop vs _foreach) ===")
    model = make_model()
    params = list(model.parameters())
    print(f"  {len(params)} parameter tensors")

    loop = lambda: sum(p.pow(2.0).sum() for p in params)
    vec  = lambda: torch.stack(torch._foreach_norm(params)).square().sum()
    rel = (loop() - vec()).abs().item() / loop().abs().item()
    print(f"  relative diff = {rel:.3e}  (expect ~1e-6, norm-then-square rounding)")
    t_old = timeit(loop)
    t_new = timeit(vec)
    print(f"  loop     : {t_old:7.3f} ms/call")
    print(f"  _foreach : {t_new:7.3f} ms/call   ({t_old / t_new:.2f}x)")


# ───────────────────────── #4 per-step host/device syncs ──────────────────────
def test_sync():
    print("\n=== #4 per-step syncs (blocking vs deferred) ===")
    model = make_model()
    fm, pt, ptr = make_batch()
    seq_lens = tuple((ptr[1:] - ptr[:-1]).tolist())   # CPU lengths (test only)

    # equivalence: the seq_lens path uses the SAME cached BlockDiagonalMask as the
    # ptr path (same seq_lens key), so any difference is just xformers' forward
    # nondeterminism. Compare the cross-path diff against a same-path determinism
    # baseline (two identical calls), exactly like test_attention above.
    model.eval()
    with torch.no_grad():
        model(fm, pt, mean=0.0, std=1.0, ptr=ptr)            # one-time framesnet init
        a = model(fm, pt, mean=0.0, std=1.0, ptr=ptr, seq_lens=seq_lens)
        b = model(fm, pt, mean=0.0, std=1.0, ptr=ptr, seq_lens=seq_lens)
        base = (a - b).abs().max().item()
        out_ptr = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
        out_sl  = model(fm, pt, mean=0.0, std=1.0, ptr=ptr, seq_lens=seq_lens)
    cross = (out_ptr - out_sl).abs().max().item()
    print(f"  determinism baseline (same path, 2 calls) = {base:.3e}")
    print(f"  max|ptr-path - seq_lens-path|             = {cross:.3e}")
    print(f"  -> {'OK: within kernel nondeterminism' if cross <= max(base * 2, 1e-6) else 'MISMATCH: investigate'}")

    # timing: mirror base_experiment._step's two sync structures on the real model.
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    reg = lambda: torch.stack(torch._foreach_norm(list(model.parameters()))).square().sum()

    def step(blocking):
        opt.zero_grad()
        y = model(fm, pt, mean=0.0, std=1.0, ptr=ptr,
                  seq_lens=None if blocking else seq_lens)
        loss = y.pow(2).mean()
        r = 1e-4 * reg()
        if blocking:
            _ = loss.item()                              # #2 pre-backward sync
            assert torch.isfinite(loss + r).all()        # #3 isfinite guard sync
        loss = loss + r
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                            error_if_nonfinite=False)
        if blocking:
            gn = gn.cpu().item()                         # #4 grad-norm sync
            _lv = loss.item()                            # #5 duplicate loss sync
        else:
            _lv, gn = torch.stack([loss.detach().float(), gn.float()]).tolist()  # 1 fused sync
        opt.step()

    t_old = timeit(lambda: step(True))
    t_new = timeit(lambda: step(False))
    print(f"  blocking (4 syncs) : {t_old:7.3f} ms/iter")
    print(f"  deferred (1 sync)  : {t_new:7.3f} ms/iter   ({t_old / t_new:.2f}x)")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  |  "
          f"BS={BS} n={N_PART} nh={NUM_HEADS} blocks={NUM_BLOCKS}")
    test_attention()
    test_proc_loss()
    test_reg()
    test_sync()
    print("\nDone. Equivalence diffs should be ~0 (≈1e-6 for L2 reg); "
          "the new path should be ≥ as fast on each.")
