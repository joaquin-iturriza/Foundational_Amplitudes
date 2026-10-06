#!/usr/bin/env python
"""Per-process micro-batches with gradient accumulation (training.per_process_accumulation).

CPU test on a tiny synthetic setup: three processes of different multiplicity, the real
AmplitudeExperiment loss path (_forward_lloca, _aggregate_per_process_loss, the
regularization, base_experiment._step) and a small per-event model with the LLoCa call
signature (the μP LLoCa net needs xformers/CUDA; the accumulation only relies on the model
treating events independently, which the block-diagonal attention guarantees). Checks
  (a) the accumulated gradient equals the mixed-batch gradient on the same events, for
      loss_aggregation mean, geometric_mean (tau 0 and > 0) and excess (beta 0 and > 0),
      L2 regularization on;
  (b) one optimizer, scheduler and EMA step per iteration, regularization counted once;
  (c) with the flag off the training step is bit-identical to the original step;
  plus the loader: grouped single-process micro-batches, distinct events, each process's
  stream a reshuffled permutation of its own training events.
Heavy imports missing on a CPU laptop (hydra, lgatr, lloca, mup, torch_ema, ...) are
stubbed for this test only; on a site the real modules are used.
Run: python tests/test_per_process_accumulation.py
"""
import os, sys, importlib.machinery, logging.handlers
from unittest import mock
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _ in range(60):
    try:
        import experiment
        break
    except ModuleNotFoundError as e:       # test-only stub of a GPU/cluster dependency
        stub = mock.MagicMock(name=e.name)
        stub.__spec__ = importlib.machinery.ModuleSpec(e.name, None)   # torch._dynamo probes specs
        sys.modules[e.name] = stub
import base_experiment
from omegaconf import OmegaConf
from dataset import AmplitudeDataset, PerProcessMicroBatchSampler

torch.set_default_dtype(torch.float64)
DT = torch.float64
NPART = (4, 5, 6)          # particles per event of each process
NEV = (40, 30, 5)          # training events per process (process 2 smaller than a micro-batch)


class ToyNet(torch.nn.Module):
    """Per-event model with the LLoCa wrapper's call signature: per-particle MLP, mean pool
    over the event's particles (ptr), coupling-order features, readout."""

    def __init__(self, n_tok=8, n_ord=4):
        super().__init__()
        self.emb = torch.nn.Embedding(n_tok, 6)
        self.phi = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16))
        self.rho = torch.nn.Sequential(torch.nn.Linear(16 + n_ord, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def forward(self, particles, tokens, mean, std, ptr, order_labels, seq_lens=None, process_ids=None):
        x = torch.cat([(particles - mean) / std, self.emb(tokens)], dim=-1)
        h = self.phi(x)
        B = len(ptr) - 1
        ev = torch.repeat_interleave(torch.arange(B), ptr[1:] - ptr[:-1])
        pooled = torch.zeros(B, h.shape[1]).index_add_(0, ev, h) / (ptr[1:] - ptr[:-1]).unsqueeze(1)
        return self.rho(torch.cat([pooled, order_labels], dim=-1))


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    parts, toks, amps, ords, pids = [], [], [], [], []
    for p, (n, k) in enumerate(zip(NPART, NEV)):
        for _ in range(k):
            parts.append(rng.normal(size=(n, 4)))
            toks.append(rng.integers(0, 8, size=n))
            # process-dependent scale, so the per-process losses differ by orders of magnitude
            amps.append([rng.normal() * 10.0 ** (p - 1)])
            ords.append([0, 0, p, 2])
            pids.append(p)
    order = rng.permutation(len(pids))     # mixed, as the experiment's shuffled arrays
    parts = [parts[i] for i in order]; toks = [toks[i] for i in order]
    amps = np.asarray(amps)[order]; ords = np.asarray(ords, float)[order]; pids = np.asarray(pids)[order]
    lens = np.array([len(x) for x in parts]); starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return dict(particles_flat=np.concatenate(parts).astype(np.float64),
                tokens_flat=np.concatenate(toks).astype(np.int64),
                offsets=np.stack([starts, starts + lens], 1), amplitudes=amps,
                order_labels=ords, process_ids=pids)


def make_exp(agg="geometric_mean", tau=0.0, beta=0.0, lam=1e-3, accum=False, micro=None, seed=0):
    exp = object.__new__(experiment.AmplitudeExperiment)
    exp.cfg = OmegaConf.create({
        "seed": 0,
        "training": dict(loss="MSE", loss_aggregation=agg, loss_aggregation_tau=tau, excess_beta=beta,
                         regularization="L2", regularization_lambda=lam, clip_grad_value=None,
                         clip_grad_norm=1e9, max_grad_norm=None, scheduler="CosineAnnealingLR",
                         optimizer="AdamW", log_every_n_steps=0, batchsize=60, num_workers=0,
                         per_process_accumulation=accum, accumulation_microbatch=micro,
                         use_balanced_sampler=False),
        "plotting": {"plot_without_regularization": True},
        "evaluation": {"batchsize": 60, "train_subsample": None},
        "data": {"dataset": ["pa", "pb", "pc"], "train_test_val": [0.8, 0.1, 0.1]},
        "use_mlflow": False,
    })
    torch.manual_seed(seed)
    exp.model = ToyNet()
    exp.device = torch.device("cpu"); exp.dtype = DT
    exp.n_datasets = 3; exp.modelname = "LLOCAMuPTransformer"
    exp.mom_mean = torch.zeros(1, 4); exp.mom_std = torch.ones(1, 4)
    exp.ewc = None; exp.ema = None
    exp._per_process_accum = accum
    exp._init_loss(); exp._init_regularization()
    exp.optimizer = torch.optim.AdamW(exp.model.parameters(), lr=1e-2)
    exp.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(exp.optimizer, T_max=100)
    exp._consec_skips, exp._diverged, exp._diverge_patience = 0, False, 50
    exp.train_loss, exp.train_lr, exp.train_grad_norm, exp.train_loss_no_reg, exp.train_mse = [], [], [], [], []
    exp.training_start_time = 0.0
    exp._excess_ref_cache = torch.tensor([3e-2, 1.0, 50.0])     # excess reference, per process
    exp.model.train()
    return exp


def dataset(d):
    return AmplitudeDataset(d["particles_flat"], d["offsets"], d["amplitudes"], d["tokens_flat"],
                            d["order_labels"], DT, process_ids=d["process_ids"])


def grouped_and_mixed(d, micro=7, seed=3):
    """One grouped batch from the sampler, and the same events as one mixed (shuffled) batch."""
    ds = dataset(d)
    idx = next(iter(PerProcessMicroBatchSampler(d["process_ids"], micro, seed=seed)))
    mixed = list(np.random.default_rng(seed).permutation(idx))
    return ds.__getitems__(idx)[0].data, ds.__getitems__(mixed)[0].data


def grads(model):
    return torch.cat([q.grad.reshape(-1) for q in model.parameters()])


def test_gradient_equivalence():
    d = make_data()
    grouped, mixed = grouped_and_mixed(d)
    worst = 0.0
    for agg, tau, beta in [("mean", 0, 0), ("geometric_mean", 0, 0), ("geometric_mean", 1e-2, 0),
                           ("excess", 0, 0), ("excess", 0, 0.5)]:
        a = make_exp(agg, tau, beta)
        loss, lnr, _ = a._batch_loss(mixed)
        a.optimizer.zero_grad(); loss.backward()
        g_mixed = grads(a.model)
        b = make_exp(agg, tau, beta, accum=True)
        loss_b, lnr_b, _ = b._per_process_step_grads(grouped)
        g_acc = grads(b.model)
        rel = ((g_acc - g_mixed).abs().max() / g_mixed.abs().max()).item()
        dl = abs(loss_b.item() - loss.item()) / abs(loss.item())
        dn = abs(lnr_b.item() - lnr.item()) / abs(lnr.item())
        print(f"[accum] {agg:15s} tau={tau:g} beta={beta:g}: loss {loss.item():.6e} vs {loss_b.item():.6e} "
              f"(rel {dl:.1e}), no-reg rel {dn:.1e}, max|dgrad|/max|grad| = {rel:.2e}")
        assert rel < 1e-12 and dl < 1e-12 and dn < 1e-12, (agg, tau, beta, rel, dl, dn)
        worst = max(worst, rel)
    # the mean (not the sum) over processes: the sum would be P x the mixed gradient
    return worst


def test_one_step_per_iteration_and_reg_once():
    d = make_data()
    grouped, _ = grouped_and_mixed(d)
    lam = 0.37
    e = make_exp("geometric_mean", lam=lam, accum=True)
    calls = {"opt": 0, "sched": 0, "ema": 0, "reg": 0}
    e.optimizer.register_step_post_hook(lambda *a: calls.__setitem__("opt", calls["opt"] + 1))
    real_sched = e.scheduler.step
    e.scheduler.step = lambda *a, **k: (calls.__setitem__("sched", calls["sched"] + 1), real_sched())
    e.ema = mock.Mock(); e.ema.update.side_effect = lambda: calls.__setitem__("ema", calls["ema"] + 1)
    real_reg = e.regularization
    e.regularization = lambda m: (calls.__setitem__("reg", calls["reg"] + 1), real_reg(m))[1]
    for step in range(3):
        base_experiment.BaseExperiment._step(e, grouped, step)
    print(f"[accum] 3 iterations: optimizer.step {calls['opt']}, scheduler.step {calls['sched']}, "
          f"ema.update {calls['ema']}, regularization evaluated {calls['reg']} times")
    assert calls == {"opt": 3, "sched": 3, "ema": 3, "reg": 3}, calls
    # regularization enters the loss once: loss - loss_no_reg = lambda * R, not P * lambda * R
    e2 = make_exp("geometric_mean", lam=lam, accum=True)
    loss, lnr, _ = e2._per_process_step_grads(grouped)
    R = real_reg(e2.model).item()
    print(f"[accum] loss - loss_no_reg = {loss.item() - lnr.item():.12e}, lambda*R = {lam * R:.12e}")
    assert abs((loss.item() - lnr.item()) - lam * R) < 1e-12 * lam * R
    # and its gradient once: accumulated grad - (grad without reg) = 2 lambda theta
    e3 = make_exp("geometric_mean", lam=0.0, accum=True)
    e3._per_process_step_grads(grouped)
    theta = torch.cat([q.detach().reshape(-1) for q in e2.model.parameters()])
    diff = (grads(e2.model) - grads(e3.model) - 2 * lam * theta).abs().max().item()
    print(f"[accum] reg gradient counted once: max|g_reg - 2 lambda theta| = {diff:.2e}")
    assert diff < 1e-12


def _original_step(exp, data):
    """The training step before per_process_accumulation existed (MSE, no EWC, deferred sync)."""
    loss, loss_no_reg, _ = exp._batch_loss(data)
    exp.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(exp.model.parameters(), exp.cfg.training.clip_grad_norm,
                                   error_if_nonfinite=False)
    exp.optimizer.step()
    exp.scheduler.step()


def test_default_off_bit_identical():
    d = make_data()
    _, mixed = grouped_and_mixed(d)
    a, b = make_exp("geometric_mean"), make_exp("geometric_mean")
    b._per_process_step_grads = mock.Mock(side_effect=AssertionError("accumulation path taken"))
    for step in range(4):
        _original_step(a, mixed)
        base_experiment.BaseExperiment._step(b, mixed, step)
    same = all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))
    print(f"[accum] flag off: parameters after 4 steps bit-identical to the original step: {same}")
    assert same


def test_loader_streams():
    d = make_data()
    e = make_exp(accum=True, micro=7)
    n = len(d["process_ids"])
    e.N_events = n
    e._role_counts = (n, 0, 0)     # all train; val/test empty are not needed by the train loader
    e.particles_flat, e.tokens_flat, e.offsets = d["particles_flat"], d["tokens_flat"], d["offsets"]
    e.all_amplitudes, e.all_order_labels, e.all_process_ids = d["amplitudes"], d["order_labels"], d["process_ids"]
    e._init_dataloader()
    assert isinstance(e.train_loader.batch_sampler, PerProcessMicroBatchSampler)
    s = e.train_loader.batch_sampler
    assert s.m == {0: 7, 1: 7, 2: 5}, s.m
    seen = {p: [] for p in range(3)}
    it = iter(e.train_loader)
    for _ in range(10):
        batch = next(it)
        mbs = e._split_by_process(batch)
        assert [int(mb[5][0]) for mb in mbs] == [0, 1, 2]
        for mb in mbs:
            p = int(mb[5][0])
            assert (mb[5] == p).all() and len(mb[1]) == s.m[p]
            assert mb[4][0] == 0 and mb[4][-1] == len(mb[0])
            key = [tuple(np.round(r, 12)) for r in mb[1].numpy()]
            assert len(set(key)) == len(key), "repeated event in a micro-batch"
            seen[p].extend(key)
    # process 0 (40 events, 7 per step): every 5 steps one pass over 35 distinct events
    first = seen[0][:35]
    assert len(set(first)) == 35
    # process 2 (5 events, micro capped at 5): each step is a permutation of all of them
    assert all(len(set(seen[2][i:i + 5])) == 5 for i in range(0, 50, 5))
    print(f"[accum] loader: micro-batch sizes {s.m}, {s.events_per_step} events/update, "
          f"grouped single-process micro-batches, no repeats, streams reshuffled per epoch")


if __name__ == "__main__":
    test_gradient_equivalence()
    test_one_step_per_iteration_and_reg_once()
    test_default_off_bit_identical()
    test_loader_streams()
    print("[accum] ok")
