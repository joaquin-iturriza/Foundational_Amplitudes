"""store_tag: a recipe entry's pool gets its own file name, its recipe id unchanged (the data-saving parents' 1e4
top train pools beside catalog_v4's 1e5 ones)."""
import copy

import yaml

import mg5_pipeline_final as mg


def test_store_tag_moves_path_not_id():
    spec = yaml.safe_load(open("recipes/catalog_v4_datasaving_h3_train.yaml"))
    e = next(p for p in spec["processes"] if p["name"] == "v4_bb_abtwm")
    assert e["store_tag"] == "n1e4"
    saved = copy.deepcopy(mg.PROCESSES.get("v4_bb_abtwm"))
    try:
        mg.register_recipe_processes([dict(e, store_tag=None)], default_sampling=spec["sampling"])
        r0 = mg.variable_energy_recipe("v4_bb_abtwm", 273, 6000, 10000, role="train", seed=42)
        mg.register_recipe_processes([e], default_sampling=spec["sampling"])
        r1 = mg.variable_energy_recipe("v4_bb_abtwm", 273, 6000, 10000, role="train", seed=42)
    finally:
        mg.PROCESSES["v4_bb_abtwm"] = saved
    assert mg.recipe_id(r0) == mg.recipe_id(r1) == "7cd06c31f3083d67"   # the id r5's parent read (job 63543636 log)
    p0, p1 = mg.recipe_output_path(r0, "/c"), mg.recipe_output_path(r1, "/c")
    assert p1 == p0.replace("_smix_amplitudes", "_smix_n1e4_amplitudes") and p1 != p0


if __name__ == "__main__":
    test_store_tag_moves_path_not_id(); print("ok")
