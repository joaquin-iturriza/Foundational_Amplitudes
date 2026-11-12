"""cache_census --quick: duplicates are pool files sharing one recipe id; one process in two samplings is a variant."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_quick_duplicates_by_recipe_id():
    with tempfile.TemporaryDirectory() as t:
        c = os.path.join(t, "amp_data_cache")
        os.makedirs(os.path.join(c, ".chunks_r1"))
        for name, rid in (("y_1-2GeV_train_smix", "r1"), ("y_1-2GeV_train_smixq", "r2"), ("z_1-2GeV_train_smix", "r1")):
            np.save(os.path.join(c, name + "_amplitudes.npy"), np.zeros(10))
            json.dump({"recipe_id": rid}, open(os.path.join(c, name + "_amplitudes.npy.recipe.json"), "w"))
        np.save(os.path.join(c, "x_1-3GeV_train_amplitudes.npy"), np.zeros(10))
        out = subprocess.run([sys.executable, "analysis/agent_redesign/cache_census.py", "--quick", "--out", os.devnull],
                             cwd=ROOT, env=dict(os.environ, SCRATCH=t, DATA_DIR=t), capture_output=True, text=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["recipe_ids_in_several_files"] == 1                  # y smix and z smix: one content, two files
    assert r["process_roles_with_variants"] == 1                  # y in smix and smixq: a variant, not a duplicate
    assert r["pool_files_without_recipe_json"] == 1               # x
    assert r["chunk_dirs_pool_built"] == 1                        # .chunks_r1: its pool is built


if __name__ == "__main__":
    test_quick_duplicates_by_recipe_id(); print("ok")
