"""catalog_v4's one-loop sign extension (plan v3_prep_explore): the class choice (recipes/gen_catalog_v4.signext_trees)
against recipes/catalog_v4_roles.json, the written entries against their trees, and the training-recipe builder
(recipes/make_catalog_v4_train_signext.build). CPU only, no torch, no MadGraph.

    python tests/test_v4_signext.py
"""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recipes"))

import gen_catalog_v4 as G  # noqa: E402
from make_catalog_v4_train_signext import build, MAX_FAIL_FRAC  # noqa: E402


def test_classes_are_training_and_outside_the_one_loop_layer():
    roles = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_roles.json")))
    trees, audit = G.signext_trees()
    assert len(trees) == 123 == audit["n"]
    loop_cls = {v["class"] for v in roles["loops"].values()}
    for n in trees:
        assert roles["trees"][n]["role"] == "train" and n not in loop_cls
    # a held-out tree class or a one-loop class planted in the roles is refused
    bad = json.loads(json.dumps(roles)); bad["trees"][trees[0]]["role"] = "heldout"
    assert trees[0] not in G.signext_trees(roles=bad)[0]


def test_written_entries_match_their_trees():
    P = G.load(G.OUT_PROC)["processes"]
    L = G.load(G.OUT_LOOPS_SIGNEXT)
    gen = yaml.safe_load(open(G.OUT_GEN_SIGNEXT))
    trees, _ = G.signext_trees(P)
    assert sorted(L["virt"]) == trees and sorted(L["processes"]) == [n + "_nlo" for n in trees]
    for n in trees:
        v, e = L["virt"][n], L["processes"][n + "_nlo"]
        assert v["mg5"] == P[n]["mg5_generate"][0] + " [virt=QCD]" and "QED<=" in v["mg5"]
        assert v["model"] == "loop_sm-no_b_mass" and v["param_card_patches"]["WT"] == 1.4915
        assert e["alphas_power"] == int(P[n].get("alphas_power", 0)) + 1 and e["virt_base"] == n
        assert e["fiducial_cuts"] == P[n]["fiducial_cuts"] and e["sqrts_measure"] == "log"
    assert all(p["n_train"] == 10000 and p["n_val"] == 1000 and p["n_test"] == 1000 for p in gen["processes"])
    assert gen["sampling"] == yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_gen_stage2.yaml")))["sampling"]
    # no name collides with catalog_v4's own one-loop layer
    assert not set(L["processes"]) & set(G.load(G.OUT_LOOPS)["processes"])


def test_builder_keeps_mixed_sign_unflagged_certified_pools():
    base = {"sampling": {"mode": "mixture"}, "processes": [{"name": "t1"}, {"name": "t2"}]}
    ext = [{"name": f"x{i}_nlo"} for i in range(5)]
    au = {"x0_nlo": {"flags": [], "train": {"negative": 0.3}},      # kept
          "x1_nlo": {"flags": [], "train": {"negative": 0.0}},      # one sign
          "x2_nlo": {"flags": ["val missing"], "train": {"negative": 0.5}},
          "x3_nlo": {"flags": [], "train": {"negative": 1.0}},      # all negative: one sign
          "x4_nlo": {"flags": [], "train": {"negative": 0.2}}}
    out, s = build(base, ext, au, {"x4_nlo": "pole certification FAIL"})
    assert [p["name"] for p in out["processes"]] == ["t1", "t2", "x0_nlo"]
    assert out["sampling"] == base["sampling"] and out["processes"][:2] == base["processes"]
    assert s["kept"] == ["x0_nlo"] and s["one_sign"] == ["x1_nlo", "x3_nlo"] and s["flagged"] == ["x2_nlo"]
    assert MAX_FAIL_FRAC == 0.2


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
