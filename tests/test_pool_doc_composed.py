"""keyed_aug.pool_identity, the memmap cache key's pool identity (data_mmap format 3): a pool composed at load is
identified by its rows' digest, a stored pool by its recipe_id, size and the bytes at both ends; neither by its path
or mtime, so every job of the same data opens one cache wherever its pools lie."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import json, os, shutil, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyed_aug import pool_identity


def _composed(d, rows, sha="ab" * 32, asha="cd" * 32):
    p = os.path.join(d, "v4_x_train_smixd25_amplitudes.npy")
    np.save(p, rows); np.save(p[:-4] + ".alphas.npy", rows[:, :1])
    json.dump({"recipe_id": "r1", "derived_from": {"composed_at_load": True, "rows_sha256": sha,
                                                   "alphas_sha256": asha}}, open(p + ".recipe.json", "w"))
    return p


def test_composed_pool_identity_stable_across_jobs():
    rows = np.arange(12.0).reshape(4, 3)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pa = _composed(a, rows)
        time.sleep(0.01)
        pb = _composed(b, rows)
        assert pool_identity(pa) == pool_identity(pb) == ["r1", os.path.getsize(pa), "rows:" + "ab" * 32]
        assert pool_identity(pa[:-4] + ".alphas.npy") == pool_identity(pb[:-4] + ".alphas.npy")
        pc = _composed(b, rows, sha="ef" * 32)          # other content, other identity
        assert pool_identity(pc) != pool_identity(pa)


def test_stored_pool_identity_is_content():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        p = os.path.join(a, "pool.npy")
        np.save(p, np.random.default_rng(0).normal(size=(50000, 9)))
        json.dump({"recipe_id": "rid"}, open(p + ".recipe.json", "w"))
        q = os.path.join(b, "pool.npy")
        shutil.copy(p, q); shutil.copy(p + ".recipe.json", q + ".recipe.json")
        time.sleep(0.01); os.utime(q, None)
        assert pool_identity(p) == pool_identity(q) and pool_identity(p)[0] == "rid"
        x = np.load(q); x[-1, -1] += 1; np.save(q, x)                # the last row changed
        assert pool_identity(p) != pool_identity(q)
        x = np.load(p)[:-1]; np.save(q, x)                           # one row fewer (the header's shape)
        assert pool_identity(p) != pool_identity(q)


def test_composed_alphas_key_ignores_node_rounding():
    """A CPU prebuild and its GPU run compose the same rows but their alpha_s may differ by an ulp (numpy's SIMD log on
    another node, 2026-11-12): the memmap key must not see it; a stored pool's alpha_s stays keyed by its bytes."""
    from omegaconf import OmegaConf
    import data_mmap as dm
    cfg = OmegaConf.create({"data": {"particle_order": "none"}})
    rows = np.arange(12.0).reshape(4, 3)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        key = lambda p: dm._hash(dm.base_key_doc(cfg, ["x"], ["train"], {"train": {"x": p}}))
        pa = _composed(a, rows)
        assert key(pa) != key(_composed(b, rows, sha="ef" * 32))           # other rows, other key
        pb = _composed(b, rows)
        al = np.load(pb[:-4] + ".alphas.npy"); np.save(pb[:-4] + ".alphas.npy", np.nextafter(al, 1.0))
        assert key(pa) == key(pb)
        for d in (a, b):                                                   # stored pools: alpha_s bytes count
            os.remove(os.path.join(d, "v4_x_train_smixd25_amplitudes.npy.recipe.json"))
        assert key(pa) != key(pb)


def test_features_key_ignores_mass_rounding():
    """The derived-inputs key holds the internal masses; a table mass is 10**log10(m), 172.49999999999991 on one node
    and 172.50000000000009 on another (CC, 2026-11-12): the key must be the same, and differ for another mass."""
    import types
    from omegaconf import OmegaConf
    import data_mmap as dm
    def key(masses):
        e = types.SimpleNamespace(cfg=OmegaConf.create({"data": {"dimensionless_inputs": True}}),
                                  _internal_mass_pdgs=[23, 6], _internal_mass_by_proc=masses)
        return dm._hash(dm.features_key_doc(e, ["x"], True, []))
    assert key([[91.18799999999999, 172.49999999999991]]) == key([[91.18799999999999, 172.50000000000009]])
    assert key([[91.188, 172.5]]) != key([[91.188, 172.6]])
    assert dm._key_floats(None) is None


if __name__ == "__main__":
    test_features_key_ignores_mass_rounding()
    test_composed_pool_identity_stable_across_jobs(); test_stored_pool_identity_is_content()
    test_composed_alphas_key_ignores_node_rounding(); print("ok")
