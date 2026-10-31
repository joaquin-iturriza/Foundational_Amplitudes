"""The zero-shot reader's parent-checkpoint cache (tools/rebuild_run.cached_parent_loads): every call returns the one
read's containers copied and its tensors shared (container_copy), so a probe's edit of its state dict (an assigned
tensor, as the HETEROSC sigma row is) never reaches the next probe, the state dict keeps its _metadata, and the
weights loaded equal a plain torch.load's. CPU only.
    python3 tests/test_parent_load_cache.py"""
import os, shutil, sys, tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

os.environ["ZS_PARENT_COPY"] = "shared"
import rebuild_run  # noqa: E402


def test_cache():
    d = tempfile.mkdtemp(prefix="parent_cache_")
    try:
        os.makedirs(os.path.join(d, "models"))
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
        opt = torch.optim.Adam(m.parameters())
        m(torch.randn(2, 4)).sum().backward(); opt.step()
        ck = {"model": m.state_dict(), "optimizer": opt.state_dict(), "ema": {"shadow_params": [p.detach().clone() for p in m.parameters()]},
              "model_is_ema": False}
        torch.save(ck, os.path.join(d, "models", "model_run0_best.pt"))
        cache = rebuild_run.cached_parent_loads(os.path.join(d, "models"))
        a = rebuild_run._load_best_ckpt(d)
        b = rebuild_run._load_best_ckpt(d)
        assert len(cache) == 1, cache.keys()
        assert a is not b and a["model"] is not b["model"] and a["ema"]["shadow_params"] is not b["ema"]["shadow_params"]
        assert a["model"]["0.weight"] is b["model"]["0.weight"]               # tensors shared, not cloned
        assert type(a["model"]) is type(ck["model"]) and a["model"]._metadata == ck["model"]._metadata
        assert a["model"]._metadata is not b["model"]._metadata
        a["model"]["0.weight"] = torch.zeros(3, 4)                            # a probe's edit (assignment)
        a["model"].pop("1.bias")
        c = rebuild_run._load_best_ckpt(d)
        assert torch.equal(c["model"]["0.weight"], ck["model"]["0.weight"]) and "1.bias" in c["model"]
        m2 = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
        m2.load_state_dict(c["model"])
        plain = torch.load(os.path.join(d, "models", "model_run0_best.pt"))
        for k, v in m2.state_dict().items():
            assert torch.equal(v, plain["model"][k]), k
        assert rebuild_run.load_best_weights(m2, d) is True                  # the shadow applied
        for p, s in zip(m2.parameters(), plain["ema"]["shadow_params"]):     # the EMA shadow copied in, not edited
            assert torch.equal(p.detach(), s)
        assert torch.equal(rebuild_run._load_best_ckpt(d)["ema"]["shadow_params"][0], plain["ema"]["shadow_params"][0])
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("ok parent load cache")


if __name__ == "__main__":
    test_cache()
