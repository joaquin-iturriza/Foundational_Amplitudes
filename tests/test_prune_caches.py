import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sweep"))
import prune


def test_caches_keeps_pools(tmp_path, monkeypatch):
    monkeypatch.setattr(prune.siteconf, "SCRATCH", str(tmp_path))
    d = tmp_path / "amp_data_cache"
    d.mkdir()
    old = time.time() - 86400 * 365
    names = [p + s for p in ("v4_", "pp13v4_", "pp13_", "dsh_", "ee4_") for s in ("a.npy", "a.recipe.json")]
    kept = ["dc_bb_bb_13000-13000GeV_train_lhe_amplitudes.npy", "dc_bb_bb_13000-13000GeV_train_lhe_amplitudes.alphas.npy"]
    for n in names + kept + ["other.npy", "dc_notaprocess_x.npy"]:
        (d / n).write_text("x")
        os.utime(d / n, (old, old))
    prune.caches(30, True)
    assert all((d / n).exists() for n in names + kept)   # a recipe (catalog_daniel_all) names dc_bb_bb
    assert not (d / "other.npy").exists() and not (d / "dc_notaprocess_x.npy").exists()


def test_open_entry_sweeps_protected(tmp_path, monkeypatch, capsys):
    plan = tmp_path / "plan.yaml"
    plan.write_text("entries:\n- {id: a, status: running, match: 'llmcmp_*,x_*'}\n- {id: b, status: done, match: 'old_*'}\n")
    holds = prune.open_globs(str(plan))
    assert holds == [("a", "llmcmp_*"), ("a", "x_*")]
    monkeypatch.setattr(prune.siteconf, "PROJECT_DIR", str(tmp_path))
    for sw in ("llmcmp_a", "old_a"):
        (tmp_path / "runs" / sw / "trial_0001").mkdir(parents=True)
    assert prune.plan_sweep("llmcmp_a", {}, set(), False, holds=holds) is None
    assert "PROTECTED runs/llmcmp_a/trial_0001 (open plan entry a)" in capsys.readouterr().out
    assert prune.plan_sweep("old_a", {}, set(), False, holds=holds) is not None
