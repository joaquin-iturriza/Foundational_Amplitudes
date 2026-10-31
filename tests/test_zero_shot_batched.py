"""tools/zero_shot_parent.py reads the parent's checkpoint once per call (rebuild_run.cached_parent_loads): every
probe's two loads (init_model's fine-tune load through base_experiment._torch_load, load_best_weights through
_load_best_ckpt) return a deep copy of that one read, so each probe's model holds the same tensors as with a read per
probe, and a probe that edits its state dict leaves the next one unchanged. CPU, a small module as the parent.
    .venv-cpu/bin/pytest tests/test_zero_shot_batched.py"""
import os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import base_experiment  # noqa: E402
import rebuild_run  # noqa: E402


def _parent(tmp_path):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
    os.makedirs(tmp_path / "models")
    p = str(tmp_path / "models" / "model_run0_best.pt")
    torch.save({"model": m.state_dict(), "model_is_ema": True}, p)
    return m, p


def _fresh():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))


def test_probes_score_identically_with_one_read(tmp_path, monkeypatch):
    parent, ck = _parent(tmp_path)
    x = torch.randn(64, 4)
    # a read per probe (before): each probe's model and output
    before = []
    for _ in range(3):
        m = _fresh(); rebuild_run.load_best_weights(m, str(tmp_path)); before.append(m(x).detach())
    monkeypatch.setattr(base_experiment, "_torch_load", base_experiment._torch_load)
    monkeypatch.setattr(rebuild_run, "_load_best_ckpt", rebuild_run._load_best_ckpt)
    cache = rebuild_run.cached_parent_loads(str(tmp_path / "models"))
    after = []
    for i in range(3):
        m = _fresh()
        ck_ft = base_experiment._torch_load(ck, map_location="cpu", weights_only=False)   # init_model's load
        m.load_state_dict(ck_ft["model"])
        # a probe editing its state dict as the HETEROSC sigma row does (a new tensor assigned; the cache shares tensors)
        ck_ft["model"]["2.weight"] = torch.zeros_like(ck_ft["model"]["2.weight"])
        assert not torch.equal(base_experiment._torch_load(ck, map_location="cpu", weights_only=False)["model"]["2.weight"],
                               ck_ft["model"]["2.weight"])     # the next probe's copy keeps the parent's
        assert rebuild_run.load_best_weights(m, str(tmp_path)) is True
        after.append(m(x).detach())
        if i == 0:
            os.rename(ck, ck + ".moved")              # later probes never touch the disk
    assert len(cache) == 2                            # one read per loader, kept for the call
    for b, a in zip(before, after):
        assert torch.equal(a, b)
    assert torch.equal(after[0], parent(x).detach())


def test_other_paths_load_as_before(tmp_path, monkeypatch):
    _parent(tmp_path)
    other = tmp_path / "models_old"; os.makedirs(other)
    torch.save({"k": torch.ones(2)}, str(other / "c.pt"))
    monkeypatch.setattr(base_experiment, "_torch_load", base_experiment._torch_load)
    monkeypatch.setattr(rebuild_run, "_load_best_ckpt", rebuild_run._load_best_ckpt)
    cache = rebuild_run.cached_parent_loads(str(tmp_path / "models"))
    assert torch.equal(base_experiment._torch_load(str(other / "c.pt"))["k"], torch.ones(2))
    assert cache == {}


def test_reader_uses_the_cache_and_no_module_listing():
    zs = open(os.path.join(ROOT, "tools", "zero_shot_parent.py")).read()
    assert 'cached_parent_loads(os.path.join(parent, "models"))' in zs
    assert "cfg.log_module_names = False" in zs and 'rec["s_stages"]' in zs
    be = open(os.path.join(ROOT, "base_experiment.py")).read()
    assert 'if self.cfg.get("log_module_names", True):\n            list_all_layer_names(self.model)' in be
