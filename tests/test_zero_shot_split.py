"""tools/zero_shot_parent.py --split: the role a probe read scores (rebuild_run.scored_split). The default stays val, so
every read taken so far reproduces; test scores the test loader, checks the test pool's physics and takes the test
events' target factors. CPU, no model built.
    .venv-cpu/bin/pytest tests/test_zero_shot_split.py"""
import os, subprocess, sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from rebuild_run import scored_split, recipe_role_counts, stand_in_role, stand_in_split_set  # noqa: E402


def _exp():
    return SimpleNamespace(val_loader="VAL", test_loader="TEST",
                           _pool_paths={"val": {"p": "p_val.npy"}, "test": {"p": "p_test.npy"}},
                           _split_indices={"train": [0, 1], "val": [2, 3], "test": [4, 5]})


def test_val_and_test_roles():
    e = _exp()
    assert scored_split(e, "val", "p") == ("VAL", "p_val.npy", [2, 3])
    assert scored_split(e, "test", "p") == ("TEST", "p_test.npy", [4, 5])
    assert scored_split(e, "test", "missing")[1] is None          # no pool path: the physics check reports it
    try:
        scored_split(e, "train", "p")
    except AssertionError:
        pass
    else:
        raise AssertionError("the train role is no held-out read")


def test_cli_default_is_val():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "zero_shot_parent.py"), "--help"],
                         capture_output=True, text=True, env={**os.environ, "COLUMNS": "200"})
    assert out.returncode == 0, out.stderr
    assert "--split {val,test}" in out.stdout and "default, every read so far" in out.stdout


def test_reader_and_job_use_the_split():
    zs = open(os.path.join(ROOT, "tools", "zero_shot_parent.py")).read()
    assert "scored_split(exp, a.split" in zs and "exp.val_loader" not in zs and '_split_indices["val"]' not in zs
    job = open(os.path.join(ROOT, "scripts", "job_catalog_v3_short.sh")).read()
    assert '--split "$SPLIT"' in job and 'SPLIT=val;' in job and '[ "$A" = PP4 ]' in job


def test_probe_without_train_pool_reads_a_role_it_has(tmp_path=None):
    """An EE4 probe (recipes/catalog_v4_ee_test.yaml: n_train 0, n_val 0, n_test 2000) has every role read from its test
    pool, so the experiment builds (it failed in require_cache: 'dataset for (ee4_epem_aabbx, train, 0 ev) not
    prebuilt'); a probe with all its roles reads its own; a probe with none asks its own role (and fails as before)."""
    import tempfile, yaml
    d = tmp_path or tempfile.mkdtemp()
    rec = os.path.join(str(d), "r.yaml")
    yaml.safe_dump({"processes": [
        {"name": "ee4_x", "sqrts": [500, 3000], "n_train": 0, "n_val": 0, "n_test": 2000},
        {"name": "pp_y", "sqrts": [13000, 13000], "n_train": 10000, "n_val": 2000, "n_test": 2000},
        {"name": "none_z", "sqrts": [1, 2], "n_train": 0, "n_val": 0, "n_test": 0}]}, open(rec, "w"))
    c = recipe_role_counts(rec)
    assert c["ee4_x"] == {"train": 0, "val": 0, "test": 2000}
    calls = []

    def real(specs, role, seed, dest_dir=None, require_cache=False):
        assert require_cache and len(specs) == 1
        calls.append((specs[0]["process"], role, specs[0]["n_events"]))
        return {specs[0]["process"]: f"{specs[0]['process']}_{role}_{specs[0]['n_events']}.npy"}
    for scored in ("test", "val"):
        split = stand_in_split_set(real, c, scored)
        spec = lambda name, role: {"process": name, "sqrts_min": 1, "sqrts_max": 2, "n_events": c[name][role]}
        for role in ("train", "val", "test"):
            calls.clear()
            got = split([spec("ee4_x", role), spec("pp_y", role)], role=role, seed=42, require_cache=True)
            assert got == {"ee4_x": "ee4_x_test_2000.npy", "pp_y": f"pp_y_{role}_{c['pp_y'][role]}.npy"}, (role, got)
        assert stand_in_role(c, "ee4_x", "train", scored) == ("test", 2000)
        assert stand_in_role(c, "none_z", "train", scored) == ("train", 0)
        assert stand_in_role(c, "pp_y", "train", scored) == ("train", 10000)
    # the reader installs it per recipe and records the role the probe's own statistics came from
    zs = open(os.path.join(ROOT, "tools", "zero_shot_parent.py")).read()
    assert "datagen.ensure_split_set = stand_in_split_set(" in zs and '"own_stats_role": own_stats_role' in zs
    # on the real EE4 recipe: every probe has only its test pool, so all its roles read it
    c4 = recipe_role_counts(os.path.join(ROOT, "recipes", "catalog_v4_ee_test.yaml"))
    assert len(c4) == 170 and all(stand_in_role(c4, n, r, "test")[0] == "test" for n in c4 for r in ("train", "val"))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
