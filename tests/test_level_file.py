"""data.level_file (llm_compare step 16): each process's target mean from a frozen per-process level file, sigma the
affine_multiplicity_std fit's; a parent standardized by it gives a new process its own entry of the same file.
experiment.py's _resolve_amp_preprocessing is lifted by source as in test_parent_target_stats.py.

    .venv-cpu/bin/python -m pytest -q tests/test_level_file.py
"""
import json
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parent_target_stats as tpt  # noqa: E402
from multiplicity_extrap import fit_affine_multiplicity, parent_target_standardization  # noqa: E402

LEVELS = os.path.join(ROOT, "analysis", "catalog_daniel", "data", "level_file_B_diag.json")


def _method():
    fn = tpt._method()
    fn.__globals__.update(os=os, json=json, __file__=os.path.join(ROOT, "experiment.py"))
    return fn


def _run(e, amps, N):
    names = list(amps)
    _method()(e, names, ["train", "val"], lambda r, n: amps[n][r], None, lambda n: N[n], True,
              None, None, None, None)


def _amps():
    rng = np.random.default_rng(0)
    return {nm: {r: np.exp(rng.normal(mu, sd, 300)) for r in ("train", "val")}
            for nm, mu, sd in (("p1", -12.0, 2.0), ("p2", -20.0, 3.0), ("p3", -25.0, 1.5))}


def test_level_file_covers_the_llm_catalogue():
    lv = json.load(open(LEVELS))["levels"]
    for r in ("catalog_daniel_train", "catalog_daniel_heldout"):
        names = [p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"{r}.yaml")))["processes"]]
        assert names and not [n for n in names if n not in lv], r
    assert all(-80 < v < 40 for v in lv.values())   # mean log|M|^2 in nat: finite and physical


def test_level_file_means_and_affine_sigma(tmp_path):
    amps, N = _amps(), {"p1": 4, "p2": 5, "p3": 5}
    lf = tmp_path / "levels.json"
    lf.write_text(json.dumps({"levels": {"p3": -24.0, "p1": -11.0, "p2": -19.5, "other": 0.0}}))
    e = tpt._exp(); e.cfg.data.level_file = str(lf)
    _run(e, amps, N)
    assert e.prepd_mean == [-11.0, -19.5, -24.0]                       # from the file, in names order
    n = np.array([300.0] * 3)
    mu = [float(np.mean(np.log(amps[k]["train"]))) for k in amps]
    sd = [float(np.std(np.log(amps[k]["train"]))) for k in amps]
    sig = fit_affine_multiplicity(n, [4, 5, 5], np.array(mu), np.array(sd))[2]
    assert np.allclose(e.prepd_std, [sig] * 3, rtol=1e-6)              # sigma = affine_multiplicity_std's
    assert e._level_file["sigma"] == e.prepd_std[0] and e._level_file["levels"]["p2"] == -19.5

    lf.write_text(json.dumps({"levels": {"p1": -11.0}}))
    e = tpt._exp(); e.cfg.data.level_file = str(lf)
    with pytest.raises(AssertionError):                                # a process without a level is refused
        _run(e, amps, N)
    e = tpt._exp(); e.cfg.data.level_file = str(lf); e.cfg.data.affine_multiplicity_std = True
    with pytest.raises(AssertionError):                                # one of level_file and mu_N, never both
        _run(e, amps, N)


def test_parent_level_file_standardization():
    stats = {"shared_standardization": True, "affine_multiplicity": {"a": 1.0, "b": -2.0, "sigma": 9.0},
             "level_file": {"path": "x.json", "sigma": 4.5, "levels": {"p1": -3.0, "p2": 7.5}}}
    m, s, kind = parent_target_standardization(stats, [4, 5], names=["p2", "p1"])
    assert kind == "level_file" and m == [7.5, -3.0] and s == [4.5, 4.5]   # the level file wins over mu_N
    with pytest.raises(AssertionError):
        parent_target_standardization(stats, [4], names=["p3"])            # a process without a level is refused
    with pytest.raises(AssertionError):
        parent_target_standardization(stats, [4])                          # names are needed
    # the parent_shared reload of a level-file parent: a new process takes its own level, the entry carried on
    e = tpt._exp(stats)
    _run(e, {"p1": _amps()["p1"]}, {"p1": 6})
    assert e.prepd_mean == [-3.0] and e.prepd_std == [4.5] and e._level_file == stats["level_file"]
    src = open(os.path.join(ROOT, "experiment.py")).read()
    assert '{"level_file": self._level_file}' in src                    # and written back into data_stats.json
    del stats["level_file"]
    m, s, kind = parent_target_standardization(stats, [4, 5])
    assert kind == "affine_multiplicity" and m == [-7.0, -9.0] and s == [9.0, 9.0]   # unchanged without it


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
