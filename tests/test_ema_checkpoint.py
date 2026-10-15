#!/usr/bin/env python
"""EMA checkpoint handling (D63; docs/results.tex sec:ema-ckpt).

With EMA on, validation scores the EMA average, so the best checkpoint must carry those weights. CPU test on a tiny
model with a real torch_ema ExponentialMovingAverage, through the real BaseExperiment methods (_save_model,
_load_model_weights, _load_previous_best_model, _load_pretrained_weights) and tools/rebuild_run.load_best_weights:
  (a) new format: "model" is the EMA shadow (and differs from the live weights), "model_live" the live weights,
      "model_is_ema" the marker; EMA off writes exactly the old keys with "model" the live weights;
  (b) resume: _load_model_weights restores live weights and EMA state; the warm-start entry (_ckpt_live_state) is the
      live weights for both formats; reloading the current checkpoint after the best one (_evals) continues training
      bit-identically;
  (c) post-training evaluation: after _load_previous_best_model, ema.average_parameters() gives the best
      checkpoint's EMA weights, not the final shadow;
  (d) fine-tune: _load_pretrained_weights starts from the parent's EMA weights, for new and old checkpoints, with and
      without reset_output_head, the old format's shadow mapped past a buffer; EMA-off parents load "model" as is;
  (e) rebuild_run.load_best_weights gives the EMA weights for both formats.
torch_ema must be importable (requirements.txt); other heavy imports missing on a CPU laptop are stubbed.
Run: python tests/test_ema_checkpoint.py
On a site, the same checks on real runs' μP models (scripts/job_ema_ckpt_check.sh; the full stack, data included):
    python tests/test_ema_checkpoint.py --real --out-dir $SCRATCH/zz_ema_ckpt <run_dir> [...]
for each run: the old checkpoint's EMA weights as _ckpt_scored_state maps them (by name, past the buffers) equal
rebuild_run.load_best_weights' (by model.parameters() order), and _load_pretrained_weights loads them; then a
new-format save of that model round-trips (save, best reload + average, fine-tune load, load_best_weights).
"""
import os, sys, tempfile, importlib.machinery, types, logging.handlers  # logger.py uses logging.handlers
from unittest import mock
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage   # real: the test is about its state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
for _ in range(80):
    try:
        import base_experiment
        import rebuild_run
        break
    except ModuleNotFoundError as e:       # test-only stub of a GPU/cluster dependency
        stub = mock.MagicMock(name=e.name)
        stub.__spec__ = importlib.machinery.ModuleSpec(e.name, None)
        sys.modules[e.name] = stub
from base_experiment import BaseExperiment, _ckpt_live_state

torch.set_default_dtype(torch.float32)


class Core(nn.Module):
    def __init__(self, out=1):
        super().__init__()
        self.linear_in = nn.Linear(3, 8)
        self.register_buffer("scale", torch.tensor([2.0]))       # a buffer between parameters
        self.mid = nn.Linear(8, 8)
        self.linear_out = nn.Linear(8, out)

    def forward(self, x):
        return self.linear_out(torch.tanh(self.mid(torch.tanh(self.linear_in(x)) * self.scale)))


class Tiny(nn.Module):
    """model.net.net.linear_out, as _load_pretrained_weights(reset_output_head=True) expects."""
    def __init__(self, out=1):
        super().__init__()
        self.net = nn.Module()
        self.net.net = Core(out)

    def forward(self, x):
        return self.net.net(x)


def make_exp(run_dir, ema=True, seed=0, out=1):
    torch.manual_seed(seed)
    e = BaseExperiment.__new__(BaseExperiment)
    e.cfg = types.SimpleNamespace(save=True, run_dir=run_dir, run_idx=0)
    e.device = torch.device("cpu")
    e.model = Tiny(out)
    e.ema = ExponentialMovingAverage(e.model.parameters(), decay=0.9) if ema else None
    e.optimizer = torch.optim.Adam(e.model.parameters(), lr=1e-2)
    e.scheduler = None
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    return e


def step(e, n=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        x = torch.randn(16, 3, generator=g)
        loss = (e.model(x) ** 2).mean()
        e.optimizer.zero_grad(); loss.backward(); e.optimizer.step()
        if e.ema is not None:
            e.ema.update()


def save_old_format(e, filename):
    """The pre-D63 _save_model, verbatim: state_dict() inside the context, saved after it."""
    model_path = os.path.join(e.cfg.run_dir, "models", filename)
    if e.ema is not None:
        with e.ema.average_parameters():
            model_state = e.model.state_dict()
    else:
        model_state = e.model.state_dict()
    torch.save({"model": model_state, "optimizer": e.optimizer.state_dict(), "scheduler": None,
                "ema": e.ema.state_dict() if e.ema is not None else None, "step": 0}, model_path)


def live(e):
    return [p.detach().clone() for p in e.model.parameters()]


def shadow(e):
    return [s.detach().clone() for s in e.ema.shadow_params]


def averaged(e):
    with e.ema.average_parameters():
        return live(e)


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def sd_params(sd, model):
    return [sd[n] for n, _ in model.named_parameters()]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_save_format():
    with tempfile.TemporaryDirectory() as d:
        e = make_exp(d); step(e, 5)
        e._save_model(4, "model_run0_best.pt")
        ck = load(os.path.join(d, "models", "model_run0_best.pt"))
        assert ck["model_is_ema"] is True
        assert same(sd_params(ck["model"], e.model), shadow(e)), "\"model\" is not the EMA shadow"
        assert not same(sd_params(ck["model"], e.model), live(e)), "EMA shadow equals the live weights: weak test"
        assert same(sd_params(ck["model_live"], e.model), live(e)), "\"model_live\" is not the live weights"
        assert same(ck["ema"]["shadow_params"], shadow(e))
        assert torch.equal(ck["model"]["net.net.scale"], e.model.net.net.scale)
        assert same(live(e), [p for p in e.model.parameters()]), "saving moved the live weights"
    with tempfile.TemporaryDirectory() as d:
        e = make_exp(d, ema=False); step(e, 5)
        e._save_model(4, "model_run0_best.pt")
        ck = load(os.path.join(d, "models", "model_run0_best.pt"))
        assert set(ck) == {"model", "optimizer", "scheduler", "ema", "step"} and ck["ema"] is None
        assert same(sd_params(ck["model"], e.model), live(e))
    print("[ema] (a) save: \"model\" = EMA shadow != live, \"model_live\" = live; EMA off unchanged")


def test_resume_and_evals():
    for fmt in ("new", "old"):
        with tempfile.TemporaryDirectory() as d:
            e = make_exp(d); step(e, 5, seed=1)
            save = (lambda f: e._save_model(4, f)) if fmt == "new" else (lambda f: save_old_format(e, f))
            save("model_run0_best.pt")
            best_live, best_shadow = live(e), shadow(e)
            if fmt == "old":    # the pre-D63 bug this replaces: "model" held the live weights
                ck = load(os.path.join(d, "models", "model_run0_best.pt"))
                assert same(sd_params(ck["model"], e.model), best_live) and "model_live" not in ck
            step(e, 5, seed=2)
            save("model_run0_current.pt")
            cur_live, cur_shadow, cur_n = live(e), shadow(e), e.ema.num_updates
            assert not same(best_shadow, cur_shadow)

            # warm start reads _ckpt_live_state: the live weights in both formats
            ck = load(os.path.join(d, "models", "model_run0_current.pt"))
            assert same(sd_params(_ckpt_live_state(ck), e.model), cur_live)

            # (c) post-training evaluation: best checkpoint, then evaluate()'s average_parameters()
            e._load_previous_best_model()
            assert same(live(e), best_live) and same(shadow(e), best_shadow)
            assert same(averaged(e), best_shadow), "evaluation does not score the best checkpoint's EMA"
            assert same(live(e), best_live), "average_parameters did not restore"

            # (b) _evals restores the current checkpoint and training continues bit-identically
            e._load_previous_current_model()
            assert same(live(e), cur_live) and same(shadow(e), cur_shadow) and e.ema.num_updates == cur_n
            twin = make_exp(os.path.join(d, "twin"))          # the same history, never saved or reloaded
            step(twin, 5, seed=1); step(twin, 5, seed=2)
            step(e, 3, seed=3); step(twin, 3, seed=3)
            assert same(live(e), live(twin)) and same(shadow(e), shadow(twin)), "resume is not bit-identical"
    print("[ema] (b) resume restores live + EMA (both formats), training continues bit-identically; "
          "(c) evaluation scores the best checkpoint's EMA")


def test_finetune_and_rebuild():
    for fmt in ("new", "old"):
        with tempfile.TemporaryDirectory() as d:
            parent = make_exp(d); step(parent, 6)
            if fmt == "new":
                parent._save_model(5, "model_run0_best.pt")
            else:
                save_old_format(parent, "model_run0_best.pt")
            want = shadow(parent)
            path = os.path.join(d, "models", "model_run0_best.pt")

            child = make_exp(os.path.join(d, "child"), seed=7)
            child._load_pretrained_weights(path)
            assert same(live(child), want), f"{fmt}: fine-tune does not start from the parent's EMA weights"
            assert torch.equal(child.model.net.net.scale, parent.model.net.net.scale)

            # reset_output_head with a 2-channel head: the body from the EMA weights, the head zeroed
            child2 = make_exp(os.path.join(d, "child2"), seed=8, out=2)
            child2._load_pretrained_weights(path, reset_output_head=True)
            names = [n for n, _ in child2.model.named_parameters()]
            for n, p, w in zip(names, child2.model.parameters(), want):
                if "linear_out" in n:
                    assert torch.count_nonzero(p) == 0
                else:
                    assert torch.equal(p, w), f"{fmt}: {n} not the parent's EMA weight"

            m = Tiny()
            assert rebuild_run.load_best_weights(m, d) is True
            assert same(live_m(m), want), f"{fmt}: rebuild_run.load_best_weights is not the EMA weights"
    with tempfile.TemporaryDirectory() as d:
        parent = make_exp(d, ema=False); step(parent, 6)
        parent._save_model(5, "model_run0_best.pt")
        child = make_exp(os.path.join(d, "child"), seed=7)
        child._load_pretrained_weights(os.path.join(d, "models", "model_run0_best.pt"))
        assert same(live(child), live(parent))
        m = Tiny()
        assert rebuild_run.load_best_weights(m, d) is False and same(live_m(m), live(parent))
    print("[ema] (d) fine-tunes start from the parent's EMA weights (new and old format, reset head); "
          "(e) rebuild_run.load_best_weights both formats")


def live_m(m):
    return [p.detach().clone() for p in m.parameters()]


def real(run_dir, out_dir):
    """The checks on a real run's model (site only). Prints one EMA_CKPT_REAL line per run."""
    import json, eval_delta
    from base_experiment import _ckpt_scored_state
    exp, cfg = eval_delta.build(run_dir, out_dir)
    model = exp.model
    names = [n for n, _ in model.named_parameters()]
    buffers = {n for n, _ in model.named_buffers()}
    ck = rebuild_run._load_best_ckpt(run_dir)
    sd, source = _ckpt_scored_state(ck, buffers)
    ema_applied = rebuild_run.load_best_weights(model, run_dir)
    ref = live_m(model)
    rec = {"run_dir": run_dir, "ema_applied": ema_applied, "source": source, "n_params": len(names),
           "n_buffers": len(buffers), "old_format": "model_live" not in ck}
    assert source == ("shadow" if ema_applied else "model"), rec
    assert same([sd[n].to(p.device, p.dtype) for n, p in zip(names, ref)], ref), f"{run_dir}: name map != order map"

    # fine-tune load of the old checkpoint
    path = os.path.join(run_dir, "models", "model_run0_best.pt")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    exp._load_pretrained_weights(path)
    assert same(live_m(model), ref), f"{run_dir}: fine-tune load is not the EMA weights"

    # new-format round trip on this model: live = ref + noise, one EMA update, save, reload
    torch.manual_seed(0)
    exp.ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    exp.ema.update()
    exp.optimizer, exp.scheduler = torch.optim.Adam(model.parameters()), None
    exp.cfg.save, exp.cfg.run_dir, exp.cfg.run_idx = True, out_dir, 0
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    want_live, want_ema = live_m(model), shadow(exp)
    assert not same(want_live, want_ema)
    exp._save_model(0, "model_run0_best.pt")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    exp._load_previous_best_model()
    assert same(live_m(model), want_live) and same(averaged(exp), want_ema), f"{run_dir}: best reload"
    exp._load_pretrained_weights(os.path.join(out_dir, "models", "model_run0_best.pt"))
    assert same(live_m(model), want_ema), f"{run_dir}: fine-tune load of the new format"
    m = model
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    assert rebuild_run.load_best_weights(m, out_dir) is True and same(live_m(m), want_ema)
    os.remove(os.path.join(out_dir, "models", "model_run0_best.pt"))
    print("EMA_CKPT_REAL " + json.dumps(rec), flush=True)


if __name__ == "__main__" and "--real" in sys.argv:
    import argparse, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    import siteconf
    bad = 0
    for rd in a.run_dirs:
        rd = os.path.normpath(rd if os.path.isabs(rd) else os.path.join(siteconf.PROJECT_DIR, rd))
        try:
            real(rd, os.path.join(a.out_dir, "__".join(rd.split(os.sep)[-2:])))
        except Exception:
            traceback.print_exc(); bad += 1
            print("EMA_CKPT_REAL_FAIL " + rd, flush=True)
    print(f"EMA_CKPT_REAL_DONE {len(a.run_dirs) - bad}/{len(a.run_dirs)}", flush=True)
    sys.exit(1 if bad else 0)
elif __name__ == "__main__":
    test_save_format()
    test_resume_and_evals()
    test_finetune_and_rebuild()
    print("[ema] ok")
