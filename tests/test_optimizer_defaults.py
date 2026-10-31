"""CPU check (no torch) of the switch to Muon (the user, 2026-10-26, D155): the canonical sweep template pretrains and
fine-tunes with MuonAdamW at the Muon search's centre (lr window [c/3, 3c] around 5e-3, muon_weight_decay in
[1e-3, 0.3]); tools/catalog_ft.py fine-tunes with MuonAdamW unless told otherwise; config/default.yaml keeps AdamW so
sweeps and reruns written before the switch keep their optimizer, and nothing forces a fine-tune onto AdamW.
    python tests/test_optimizer_defaults.py        (or pytest tests/test_optimizer_defaults.py)
"""
import math
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_template_muon():
    t = yaml.safe_load(open(os.path.join(ROOT, "sweep", "sweep_config_jeanzay_template.yaml")))
    assert t["fixed_params"]["training.optimizer"] == "MuonAdamW"
    ss = {s["name"]: s for s in t["search_space"]}
    lr = ss["training.lr"]
    assert math.isclose(math.sqrt(lr["low"] * lr["high"]), 5e-3, rel_tol=1e-3)     # centred on the search's best
    assert math.isclose(lr["high"] / lr["low"], 9.0, rel_tol=1e-3)                 # +/- 1/2 decade, [c/3, 3c]
    wd = ss["training.muon_weight_decay"]
    assert (wd["type"], wd["low"], wd["high"]) == ("float_log", 1e-3, 0.3)
    assert wd["low"] <= 0.0217 <= wd["high"]


def test_catalog_ft_default_muon():
    import tools.catalog_ft as cf
    src = open(cf.__file__).read()
    assert 'ap.add_argument("--optimizer", default="MuonAdamW"' in src
    cmd = cf.command("runs/x", "MuonAdamW", "r.yaml", 10, 176, "rd", "rd_result.json", "e")
    assert "training.optimizer=MuonAdamW" in cmd


def test_catalog_ft_parent_orders_flag(tmp_path=None):
    import json, tempfile
    import tools.catalog_ft as cf
    d = tempfile.mkdtemp()
    for src, want in (("table", False), ("diagrams_highest_alphas", True)):
        json.dump({"target_physics_orders_source": src}, open(os.path.join(d, "data_stats.json"), "w"))
        flag, s, _how = cf.parent_orders_flag(d)
        assert (flag, s) == (want, src)
        cmd = cf.command("runs/x", "MuonAdamW", "r.yaml", 10, 176, "rd", "rd.json", "e", orders_from_diagrams=flag)
        assert f"++data.orders_from_diagrams={str(want).lower()}" in cmd
    json.dump({"target_physics_orders_source": "diagrams_alphas_only"}, open(os.path.join(d, "data_stats.json"), "w"))
    try:
        cf.parent_orders_flag(d)
        raise AssertionError("alpha_s-only parent not refused")
    except ValueError:
        pass


def test_default_yaml_unchanged_and_no_forced_adamw():
    d = yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))
    assert d["training"]["optimizer"] == "AdamW"           # old sweeps / reruns keep what they ran with
    assert "optimizer" not in d["fine_tune"]               # a fine-tune takes training.optimizer (its parent's config)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
