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
4. Stats written before the record (the dimensionless runs of 6dee967a..77b43ae3, whose dumped config has no
   data.orders_from_diagrams: the key was read with a default and never dumped) are placed by the run that wrote
   them: the commit in its out_0.log against the history (before 6dee967a the table, 6dee967a and 5e4040a9 the
   alpha_s power alone, from 1d9d6a29 the current source), else its config.yaml (the key true: the diagrams; absent:
   the table). A parent of today's catalog comparison (git_hash 1d9d6a29) passes a diagrams run's check.
5. tools/catalog_ft.py gives the fine-tune the parent's data.orders_from_diagrams (recorded or inferred source).

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
    assert re.search(r"self\._check_frozen_orders_source\(stats\b", body), "init_data does not check the source"
    print("[stats] the source is recorded, a mismatch refused, stats without it read as the table")


def _parent(d, cfg_data=None, git_hash=None):
    """A parent run folder as a run leaves it: config.yaml (data block), out_0.log with the start line."""
    os.makedirs(d, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"data": cfg_data or {"target_physics_orders": True}}),
                   os.path.join(d, "config.yaml"))
    if git_hash is not None:
        with open(os.path.join(d, "out_0.log"), "w") as f:
            f.write("[2026-10-26 16:10:00 I] ### Starting experiment catalog_v4_short/run (mlflowid=0) (jobid=1) "
                    f"(git_hash={git_hash} ###\n")
    return d


def _git_has(commit):
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", commit + "^{commit}"],
                          capture_output=True).returncode == 0


def test_stats_without_record_inferred_from_parent():
    infer = AmplitudeExperiment._infer_orders_source
    exp = AmplitudeExperiment.__new__(AmplitudeExperiment)
    diag_run = OmegaConf.create({"data": {"target_physics_orders": True}})                 # the default: diagrams
    table_run = OmegaConf.create({"data": {"target_physics_orders": True, "orders_from_diagrams": False}})
    with tempfile.TemporaryDirectory() as d:
        # the config fallback (no log): the key true -> the diagrams, absent -> the table
        on = _parent(os.path.join(d, "cfg_on"), {"target_physics_orders": True, "orders_from_diagrams": True})
        off = _parent(os.path.join(d, "cfg_absent"))
        assert infer(on)[0] == "diagrams_highest_alphas", infer(on)
        assert infer(off)[0] == "table", infer(off)
        exp.cfg = diag_run
        exp._check_frozen_orders_source({}, on)
        try:
            exp._check_frozen_orders_source({}, off)
            raise RuntimeError("a parent without the key (table) accepted by a diagrams run")
        except AssertionError as err:
            assert "target_physics_orders_source=table" in str(err), err
        exp.cfg = table_run
        exp._check_frozen_orders_source({}, off)
        # a commit this checkout does not know: the config decides
        unk = _parent(os.path.join(d, "unknown_commit"), {"orders_from_diagrams": True}, git_hash="f" * 40)
        assert infer(unk)[0] == "diagrams_highest_alphas", infer(unk)
        print("[infer] config.yaml: data.orders_from_diagrams true -> diagrams, absent -> table; unknown commit -> config")
        # the log's commit, placed in the history (needs the checkout's git history, as on a site)
        cases = (("3694e90eba4830d3d4aca8a7fdac6b0791b76186", "table"),                    # 6dee967a^
                 ("6dee967a49596d36a16055653cdf5a7df05f520b", "diagrams_alphas_only"),
                 ("5e4040a9724f562b260cab781c039b531dc3e5bf", "diagrams_alphas_only"),
                 ("1d9d6a29464211df804574c3aeafea1cbbde11f8", "diagrams_highest_alphas"))  # today's comparison
        if not all(_git_has(c) for c, _ in cases):
            print("[infer] git history not at hand: the commit cases skipped")
            return
        for c, want in cases:
            run = _parent(os.path.join(d, c[:8]), git_hash=c)       # its config has no key, like the real parents
            got = infer(run)
            assert got[0] == want, (c[:8], got)
        today = os.path.join(d, "1d9d6a29")
        exp.cfg = diag_run
        exp._check_frozen_orders_source({}, today)                  # fine-tune / warm start from it: accepted
        exp.cfg = table_run
        try:
            exp._check_frozen_orders_source({}, today)
            raise RuntimeError("a diagrams parent accepted by a table run")
        except AssertionError:
            pass
        exp.cfg = diag_run
        try:
            exp._check_frozen_orders_source({}, os.path.join(d, "6dee967a"))
            raise RuntimeError("an alpha_s-only parent accepted by a current diagrams run")
        except AssertionError:
            pass
    src = open(os.path.join(ROOT, "experiment.py")).read()
    assert re.search(r"self\._check_frozen_orders_source\(stats, os\.path\.dirname\(os\.path\.abspath\(stats_src\)\)\)",
                     src), "init_data does not pass the stats' run folder"
    print("[infer] the log's commit: before 6dee967a table, 6dee967a/5e4040a9 alpha_s only, 1d9d6a29 the diagrams; "
          "a parent at 1d9d6a29 passes a diagrams run")


def test_catalog_ft_follows_parent():
    """5. tools/catalog_ft.py sets data.orders_from_diagrams from the parent: the source recorded in its
    data_stats.json, else inferred (_infer_orders_source); a table parent (before 6dee967a) gets false on the run.py
    command line, a current one true, an alpha_s-only one (6dee967a..5e4040a9) is refused."""
    spec = importlib.util.spec_from_file_location("catalog_ft", os.path.join(ROOT, "tools", "catalog_ft.py"))
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    with tempfile.TemporaryDirectory() as d:
        rec = _parent(os.path.join(d, "recorded"))
        json.dump({"target_physics_orders_source": "table"}, open(os.path.join(rec, "data_stats.json"), "w"))
        assert ft.parent_orders_flag(rec)[:2] == (False, "table"), ft.parent_orders_flag(rec)
        cfg_on = _parent(os.path.join(d, "cfg_on"), {"orders_from_diagrams": True})
        json.dump({}, open(os.path.join(cfg_on, "data_stats.json"), "w"))       # stats written before the record
        assert ft.parent_orders_flag(cfg_on)[:2] == (True, "diagrams_highest_alphas"), ft.parent_orders_flag(cfg_on)
        cfg_off = _parent(os.path.join(d, "cfg_absent"))
        assert ft.parent_orders_flag(cfg_off)[:2] == (False, "table"), ft.parent_orders_flag(cfg_off)
        if _git_has("6dee967a49596d36a16055653cdf5a7df05f520b"):
            old = _parent(os.path.join(d, "alphas_only"), git_hash="6dee967a49596d36a16055653cdf5a7df05f520b")
            try:
                ft.parent_orders_flag(old)
                raise RuntimeError("an alpha_s-only parent accepted")
            except ValueError:
                pass
        for flag in (True, False):
            cmd = ft.command(rec, "AdamW", "r.yaml", 10, 20, "rd", "res.json", "e", orders_from_diagrams=flag)
            assert f"++data.orders_from_diagrams={str(flag).lower()}" in cmd, cmd
    print("[catalog_ft] the fine-tune's data.orders_from_diagrams follows the parent's source (recorded, inferred)")


if __name__ == "__main__":
    test_switch_off_is_the_table_target()
    test_mixed_entry()
    test_frozen_stats_record_and_check()
    test_stats_without_record_inferred_from_parent()
    test_catalog_ft_follows_parent()
    print("all ok")
