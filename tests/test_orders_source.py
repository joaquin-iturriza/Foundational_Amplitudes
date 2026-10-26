#!/usr/bin/env python
"""The dimensionless target's coupling monomial follows data.orders_from_diagrams, and the frozen data stats record
where it came from (plan catalog_coverage, the repo review of the coupling-order change).

1. data.orders_from_diagrams=false reproduces the target as it was before the diagrams were read (the catalog table's
   powers, a mixed entry refused) exactly, on stubs: the method of the commit before the change (6dee967a^, read with
   git) and a hand formula, on zp_uux_Y1g (the table says alpha_s^0, its diagrams alpha_s^1) with a hand-made sidecar
   that would change the monomial if it were read; with the switch on the sidecar's monomial is used.
2. A table-mixed entry (uubar_ddbar, [0, 0, 2, 2]) is refused with the switch off, as before, and divided by its
   highest alpha_s power with it on.
3. _save_data_stats writes target_physics_orders_source; _check_frozen_orders_source refuses a mismatch while
   target_physics_orders is on (stats without the key count as the table, which wrote them), ignores it when off, and
   init_data calls it on every reload of frozen stats.

Needs the project's env (torch, omegaconf): run on a site,  python tests/test_orders_source.py
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
from experiment import AmplitudeExperiment  # noqa: E402

OLD_COMMIT = "6dee967a^"     # the last experiment.py with the table's monomial only


def _sidecar(d, name, orders):
    dg = [{"diagram_type": "tree", "nodes": [{"kind": "vertex", "orders": {"QCD": a, "QED": b}}, {"kind": "external"}]}
          for a, b in orders]
    json.dump({"subprocesses": [{"diagrams": dg}]}, open(os.path.join(d, f"{name}.diagrams.json"), "w"))


def _stub(cls, name, ddir, from_diagrams):
    exp = cls.__new__(cls)
    exp.cfg = OmegaConf.create({
        "data": {"dataset": [name], "amp_orders": [mg.order_vector(mg.PROCESSES[name])],
                 "target_physics_factor": True, "target_physics_orders": True, "target_physics_dim_only": True,
                 "orders_from_diagrams": from_diagrams},
        "model": {"diagrams_dir": ddir}})
    exp._phys_logc = {name: 0.0}
    exp._phys_ls0 = float(np.log(289.0))
    return exp


def _events(nfin, n=64, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.uniform(50.0, 500.0, n)
    mom = rng.normal(size=(n, nfin + 2, 4)) * 30.0
    mom[:, 0] = np.stack([E, 0 * E, 0 * E, E], 1)
    mom[:, 1] = np.stack([E, 0 * E, 0 * E, -E], 1)
    return mom, rng.uniform(0.09, 0.15, n)


def _old_class():
    """AmplitudeExperiment of the commit before the change, or None where git history is not at hand."""
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{OLD_COMMIT}:experiment.py"], check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"[old] git show {OLD_COMMIT} failed ({err}); the hand formula only")
        return None
    path = os.path.join(tempfile.mkdtemp(), "experiment_before_diagrams.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("experiment_before_diagrams", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AmplitudeExperiment


def test_switch_off_is_the_table_target():
    name = "zp_uux_Y1g"
    o = mg.order_vector(mg.PROCESSES[name])
    assert o[2:] == [0, 2], o
    mom, als = _events(2)
    with tempfile.TemporaryDirectory() as d:
        _sidecar(d, name, [(1, 0)])                          # the diagrams: alpha_s^1, no QED vertex
        new_off = _stub(AmplitudeExperiment, name, d, False)._physics_logF(name, mom, als)
        new_on = _stub(AmplitudeExperiment, name, d, True)._physics_logF(name, mom, als)
        q = mom[:, 0] + mom[:, 1]
        lref = np.log(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))
        hand = -((4 - 4) * (lref - np.log(289.0)) + o[2] * np.log(4 * np.pi * als)
                 + o[3] * np.log(4 * np.pi / 132.507))
        assert np.allclose(new_off, hand, rtol=0, atol=1e-12), np.abs(new_off - hand).max()
        assert np.allclose(new_on, -np.log(4 * np.pi * als), rtol=0, atol=1e-12)
        assert not np.allclose(new_on, new_off)
        Old = _old_class()
        if Old is not None:
            old = _stub(Old, name, d, False)._physics_logF(name, mom, als)
            assert np.array_equal(new_off, old), np.abs(new_off - old).max()
            print(f"[off] orders_from_diagrams=false equals {OLD_COMMIT}'s target bit for bit on {name}")
    print("[off] the switch off ignores the sidecar (table alpha_s^0 alpha^2), on reads it (alpha_s^1)")


def test_mixed_entry():
    name = "uubar_ddbar"
    mom, als = _events(2, seed=1)
    with tempfile.TemporaryDirectory() as d:
        _sidecar(d, name, [(0, 2), (2, 0), (1, 1)])
        try:
            _stub(AmplitudeExperiment, name, d, False)._physics_logF(name, mom, als)
            raise RuntimeError("a table-mixed entry was accepted with orders_from_diagrams=false")
        except AssertionError as err:
            assert "coupling homogeneity is not exact" in str(err), err
        on = _stub(AmplitudeExperiment, name, d, True)._physics_logF(name, mom, als)
        assert np.allclose(on, -2 * np.log(4 * np.pi * als), rtol=0, atol=1e-12)
    print("[mixed] refused with the switch off (as before), alpha_s^2 alpha^0 with it on")


def test_frozen_stats_record_and_check():
    exp = AmplitudeExperiment.__new__(AmplitudeExperiment)
    exp.mom_mean, exp.mom_std, exp._amp_trafos_pp = [0.0], [1.0], None
    exp.prepd_mean, exp.prepd_std = [0.0], [1.0]
    with tempfile.TemporaryDirectory() as d:
        for flag, src in ((True, "diagrams_highest_alphas"), (False, "table")):
            exp.cfg = OmegaConf.create({"data": {"target_physics_orders": True, "orders_from_diagrams": flag}})
            p = os.path.join(d, "data_stats.json")
            exp._save_data_stats(p, 1.0, ["log"], False)
            stats = json.load(open(p))
            assert stats["target_physics_orders_source"] == src, stats
            exp._check_frozen_orders_source(stats)                        # its own stats pass
            other = not flag
            exp.cfg = OmegaConf.create({"data": {"target_physics_orders": True, "orders_from_diagrams": other}})
            try:
                exp._check_frozen_orders_source(stats)
                raise RuntimeError(f"stats fitted with {src} accepted by a run with orders_from_diagrams={other}")
            except AssertionError as err:
                assert "target_physics_orders_source" in str(err), err
            exp.cfg = OmegaConf.create({"data": {"target_physics_orders": False, "orders_from_diagrams": other}})
            exp._check_frozen_orders_source(stats)                        # the monomial is not in the target
    exp.cfg = OmegaConf.create({"data": {"target_physics_orders": True, "orders_from_diagrams": False}})
    exp._check_frozen_orders_source({})                                   # written before the key: the table's
    exp.cfg = OmegaConf.create({"data": {"target_physics_orders": True}})
    try:
        exp._check_frozen_orders_source({})
        raise RuntimeError("stats without the key (table) accepted by a diagrams run")
    except AssertionError:
        pass
    src = open(os.path.join(ROOT, "experiment.py")).read()
    body = src[src.index("Loaded frozen data stats") - 6000: src.index("Loaded frozen data stats")]
    assert re.search(r"self\._check_frozen_orders_source\(stats\)", body), "init_data does not check the source"
    print("[stats] the source is recorded, a mismatch refused, stats without it read as the table")


if __name__ == "__main__":
    test_switch_off_is_the_table_target()
    test_mixed_entry()
    test_frozen_stats_record_and_check()
    print("all ok")
