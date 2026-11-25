"""llm_compare step 20: perevent_level.py's search pieces (spaces, sampler loop, model builders, content signature)."""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
import perevent_level as pl  # noqa: E402
from dyhpo_sampler import DyHPOSampler  # noqa: E402


def test_content_sig():
    assert pl.content_sig("g g > h h z") == "h h z"
    assert pl.content_sig("u d~ > h ta+ vt") == "h ta v"
    assert pl.content_sig("g s > s g") == "(jets)"


def test_net_defaults_are_step17():
    n = pl.Net()
    assert n.hp == dict(width=64, depth=2, l2=1e-2 / 200, lr=3e-3, batch=4096) and n.seeds == 3


def test_search_loop(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4)); y = X @ [1.0, -2.0, 0.5, 0.0] + 0.1 * rng.normal(size=3000)
    for model, space in pl.SPACE.items():
        S = DyHPOSampler(space, {"t_steps": [1]}, n_candidates=50, seed=42, output_path=str(tmp_path / model),
                         n_startup=3, surrogate="gp")
        seen = []
        for _ in range(5):   # 3 random start-up trials, then 2 from the GP
            idx, hp, ts = S.suggest()
            assert set(hp) == {e["name"] for e in space}
            seen.append(idx)
            S.observe(idx, ts, float(sum(abs(float(v)) for v in hp.values())))
        assert len(set(seen)) == 5
        if model == "mlp":
            m = pl.Net(**{**hp, "width": 8, "batch": 1024}, seeds=1).fit(X, y)
        else:
            m = pl.HistGradientBoostingRegressor(max_iter=int(hp["max_iter"]), learning_rate=float(hp["learning_rate"]),
                                                 max_leaf_nodes=int(hp["max_leaf_nodes"]),
                                                 l2_regularization=float(hp["l2_regularization"])).fit(X, y)
        assert np.mean((m.predict(X) - y) ** 2) < np.var(y)


if __name__ == "__main__":
    import tempfile, pathlib
    test_content_sig(); test_net_defaults_are_step17(); test_search_loop(pathlib.Path(tempfile.mkdtemp())); print("ok")
