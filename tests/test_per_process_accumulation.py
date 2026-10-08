#!/usr/bin/env python
"""Per-process micro-batches with gradient accumulation (training.per_process_accumulation).

Two modes. all_processes (true): one micro-batch from every process per step, tests (a)-(c)
and the loader test below. round_robin (arXiv:2606.23791's batching): G single-process
micro-batches per step, processes in round-robin order; tests test_rr_*: the sampler's order
and coverage (interleaved and random), each micro-batch single-process, the stream continuing
across loader re-iterations, and the accumulated gradient equal to the gradient of the
per-event mean loss over the union of the G micro-batches (float64, 1e-12).

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
from dataset import AmplitudeDataset, PerProcessMicroBatchSampler, RoundRobinMicroBatchSampler

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


def make_exp(agg="geometric_mean", tau=0.0, beta=0.0, lam=1e-3, accum=False, micro=None, seed=0,
             G=None, order="interleaved"):
    exp = object.__new__(experiment.AmplitudeExperiment)
    exp.cfg = OmegaConf.create({
        "seed": 0,
        "training": dict(loss="MSE", loss_aggregation=agg, loss_aggregation_tau=tau, excess_beta=beta,
                         regularization="L2", regularization_lambda=lam, clip_grad_value=None,
                         clip_grad_norm=1e9, max_grad_norm=None, scheduler="CosineAnnealingLR",
                         optimizer="AdamW", log_every_n_steps=0, batchsize=60, num_workers=0,
                         per_process_accumulation=accum, accumulation_microbatch=micro,
                         accumulation_steps=G, accumulation_order=order,
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
    exp._accum_mode = experiment.AmplitudeExperiment._accumulation_mode(accum)
    exp._per_process_accum = exp._accum_mode is not None
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


def _runs(idx, pids):
    """(process, length) of the contiguous single-process runs of one yielded step."""
    p = pids[np.asarray(idx)]
    cut = np.flatnonzero(np.diff(p)) + 1
    return [(int(r[0]), len(r)) for r in np.split(p, cut)]


def test_rr_sampler_order_and_coverage():
    sizes = (50, 40, 30, 25, 20)
    pids = np.random.default_rng(1).permutation(np.repeat(np.arange(5), sizes))
    B, G = 4, 3
    s = RoundRobinMicroBatchSampler(pids, B, G, seed=0, min_batches=4)
    steps = []
    for _ in range(3):                         # three loader epochs of 4 steps: the cycle continues
        steps.extend(list(iter(s)))
    slots = []
    for idx in steps:
        assert len(idx) == G * B and len(set(idx)) == len(idx)
        runs = _runs(idx, pids)
        assert all(n == B for _, n in runs), runs          # G distinct consecutive processes here
        slots.extend(p for p, _ in runs)
    assert slots == [k % 5 for k in range(len(slots))], slots[:15]
    # equal steps per process: every 5 steps (G cycles of P) each process has G micro-batches
    for k in range(0, len(steps) - 4, 5):
        c = np.bincount(slots[k * G:(k + 5) * G], minlength=5)
        assert (c == G).all(), c
    # each process's stream: distinct events until its pool is used up
    for p in range(5):
        ev = np.concatenate([np.asarray(idx)[pids[np.asarray(idx)] == p] for idx in steps])
        n_full = (sizes[p] // B) * B
        assert len(set(ev[:n_full].tolist())) == n_full
    print(f"[rr] interleaved: {len(steps)} steps of G={G} x B={B}, process slots "
          f"{slots[:10]}..., cycle continues across loader epochs, equal micro-batches per process")
    # G > P wraps around the cycle; P = 1 is plain accumulation over one process
    p3 = np.repeat(np.arange(3), 20)
    s2 = RoundRobinMicroBatchSampler(p3, 4, 4, seed=0, min_batches=3)
    sl = [p for idx in s2 for p, _ in _runs(idx, p3)]
    assert sl == [k % 3 for k in range(12)], sl
    p1 = np.zeros(40, int)
    s1 = RoundRobinMicroBatchSampler(p1, 4, 3, seed=0, min_batches=2)
    assert all(_runs(idx, p1) == [(0, 12)] for idx in s1)
    # random order: each cycle a permutation of K = min_p n_p // B slots per process
    s3 = RoundRobinMicroBatchSampler(pids, B, G, order="random", seed=0, min_batches=50)
    assert s3.K == 20 // B
    sl3 = []
    for idx in s3:
        for p, n in _runs(idx, pids):
            assert n % B == 0
            sl3.extend([p] * (n // B))
    cyc = 5 * s3.K
    for k in range(len(sl3) // cyc):
        c = np.bincount(sl3[k * cyc:(k + 1) * cyc], minlength=5)
        assert (c == s3.K).all(), c
    assert sl3[:cyc] != sorted(sl3[:cyc])
    print(f"[rr] random: cycles of {cyc} slots, {s3.K} per process, shuffled; G>P wraps, P=1 ok")


def test_rr_gradient_is_per_event_mean():
    d = make_data()
    ds = dataset(d)
    worst = 0.0
    # micro 5: every micro-batch full (process 2 has 5 events); micro 7: process 2 capped at 5.
    # G=4 > P=3, so a step holds one process twice (two runs, or one run of 2B in random order)
    for micro, order in [(5, "interleaved"), (7, "interleaved"), (5, "random")]:
        e = make_exp("geometric_mean", accum="round_robin", micro=micro, G=4, order=order)
        e._accum_sampler = RoundRobinMicroBatchSampler(d["process_ids"], micro, 4, order=order, seed=2)
        it = iter(e._accum_sampler)
        for _ in range(3):
            idx = next(it)
            data = ds.__getitems__(idx)[0].data
            mbs = e._split_microbatches(data)
            assert len(mbs) == 4, [len(mb[1]) for mb in mbs]
            for mb in mbs:
                p = int(mb[5][0])
                assert (mb[5] == p).all() and len(mb[1]) == e._accum_sampler.m[p]
                assert mb[4][0] == 0 and mb[4][-1] == len(mb[0])
            # reference: the per-event mean loss over the union of the micro-batches, + lambda R
            ref = make_exp("geometric_mean", accum="round_robin", micro=micro, G=4, order=order)
            ref.model.load_state_dict(e.model.state_dict())
            y_pred, y, _, sigma, _ = ref._forward_lloca(data)
            L = ref._per_event_loss(y_pred, y, sigma=sigma).mean()
            L_tot = L + ref.regularization_lambda * ref.regularization(ref.model)
            ref.optimizer.zero_grad(); L_tot.backward()
            g_ref = grads(ref.model)
            loss, lnr, _ = e._round_robin_step_grads(data)
            g = grads(e.model)
            rel = ((g - g_ref).abs().max() / g_ref.abs().max()).item()
            dl = abs(loss.item() - L_tot.item()) / abs(L_tot.item())
            dn = abs(lnr.item() - L.item()) / abs(L.item())
            assert rel < 1e-12 and dl < 1e-12 and dn < 1e-12, (micro, order, rel, dl, dn)
            worst = max(worst, rel)
            if micro == 5:
                # their trainer: each micro-batch's mean loss weighted 1/G (equal sizes: the same)
                ref.optimizer.zero_grad()
                tg = 0
                for mb in mbs:
                    yp, yy, pp, sg, _ = ref._forward_lloca(mb)
                    tg = tg + ref._per_event_loss(yp, yy, sigma=sg).mean() / 4
                (tg + ref.regularization_lambda * ref.regularization(ref.model)).backward()
                r2 = ((g - grads(ref.model)).abs().max() / g_ref.abs().max()).item()
                assert r2 < 1e-12, r2
            e.optimizer.step()                # move on, so the three steps see different weights
    print(f"[rr] accumulated gradient vs per-event mean over the union (G=4, B=5/7, interleaved "
          f"and random, loss_aggregation=geometric_mean ignored): max rel {worst:.2e}; equal to "
          f"the 1/G-weighted micro-batch means when the micro-batches are full")


def test_rr_loader_and_step():
    d = make_data()
    e = make_exp(accum="round_robin", micro=5, G=2)
    n = len(d["process_ids"])
    e.N_events = n
    e._role_counts = (n, 0, 0)
    e.particles_flat, e.tokens_flat, e.offsets = d["particles_flat"], d["tokens_flat"], d["offsets"]
    e.all_amplitudes, e.all_order_labels, e.all_process_ids = d["amplitudes"], d["order_labels"], d["process_ids"]
    e._init_dataloader()
    assert e._accum_mode == "round_robin"
    assert isinstance(e.train_loader.batch_sampler, RoundRobinMicroBatchSampler)
    assert e.cfg.training.accumulation_steps == 2 and e.cfg.training.accumulation_microbatch == 5
    it = iter(e.train_loader)
    procs = []
    calls = {"opt": 0}
    e.optimizer.register_step_post_hook(lambda *a: calls.__setitem__("opt", calls["opt"] + 1))
    for step in range(6):
        batch = next(it)
        procs.extend(int(mb[5][0]) for mb in e._split_microbatches(batch))
        base_experiment.BaseExperiment._step(e, batch, step)
    assert procs == [k % 3 for k in range(12)], procs
    assert calls["opt"] == 6
    # defaults: G = 4, B = batchsize // G
    e2 = make_exp(accum="round_robin")
    for k in ("N_events", "_role_counts", "particles_flat", "tokens_flat", "offsets",
              "all_amplitudes", "all_order_labels", "all_process_ids"):
        setattr(e2, k, getattr(e, k))
    e2._init_dataloader()
    assert e2.cfg.training.accumulation_steps == 4
    assert e2.cfg.training.accumulation_microbatch == e2.cfg.training.batchsize // 4
    for v, m in [(False, None), ("false", None), (None, None), (True, "all_processes"),
                 ("true", "all_processes"), ("all_processes", "all_processes"), ("round_robin", "round_robin")]:
        assert experiment.AmplitudeExperiment._accumulation_mode(v) == m, (v, m)
    print(f"[rr] loader: processes per step {procs[:6]}..., one optimizer step per iteration; "
          f"defaults G=4, B=batchsize//4")


if __name__ == "__main__":
    test_gradient_equivalence()
    test_one_step_per_iteration_and_reg_once()
    test_default_off_bit_identical()
    test_loader_streams()
    test_rr_sampler_order_and_coverage()
    test_rr_gradient_is_per_event_mean()
    test_rr_loader_and_step()
    print("[accum] ok")
