import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sweep"))
import prune


def test_caches_keeps_pools(tmp_path, monkeypatch):
    monkeypatch.setattr(prune.siteconf, "SCRATCH", str(tmp_path))
    d = tmp_path / "amp_data_cache"
    d.mkdir()
    old = time.time() - 86400 * 365
    names = [p + s for p in ("v4_", "pp13v4_", "pp13_", "dsh_", "ee4_") for s in ("a.npy", "a.recipe.json")]
    for n in names + ["other.npy"]:
        (d / n).write_text("x")
        os.utime(d / n, (old, old))
    prune.caches(30, True)
    assert all((d / n).exists() for n in names)
    assert not (d / "other.npy").exists()
