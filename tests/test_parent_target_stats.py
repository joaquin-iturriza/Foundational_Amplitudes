"""fine_tune.target_stats=parent_shared (experiment._resolve_amp_preprocessing, multiplicity_extrap.
parent_target_standardization): a fine-tune onto a process its parent never saw keeps the parent's target
standardization, so the standardized target its loss sees is the one the zero-shot read (tools/zero_shot_parent.py)
scores the parent's output against: at step 0 the fine-tune's val loss times the parent's sigma^2 is the zero-shot
MSE in log|M|^2. The method is lifted out of experiment.py by source (experiment.py imports lloca, absent on the laptop)
and run on stubs; needs numpy, torch (CPU), scipy and scikit-learn (preprocessing.py), e.g. .venv-driver.

    .venv-driver/bin/python tests/test_parent_target_stats.py
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import multiplicity_extrap  # noqa: E402
import preprocessing  # noqa: E402
from multiplicity_extrap import parent_target_standardization  # noqa: E402


class _A(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class _Log:
    def info(self, *a, **k):
        pass

    warning = info


def _method():
    src = open(os.path.join(ROOT, "experiment.py")).read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "AmplitudeExperiment")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_amp_preprocessing")
    g = {"np": np, "LOGGER": _Log(), "resolve_amp_trafos": preprocessing.resolve_amp_trafos,
         "preprocess_amplitude": preprocessing.preprocess_amplitude,
         "fit_affine_multiplicity": multiplicity_extrap.fit_affine_multiplicity,
         "parent_target_standardization": parent_target_standardization}
    exec(compile(ast.Module([fn], []), "experiment.py", "exec"), g)
    return g["_resolve_amp_preprocessing"]


def _exp(parent_stats=None):
    e = _A(cfg=_A(data=_A(amp_trafos=["log", "standardization"], shared_standardization=True,
                          affine_multiplicity_std=False, signed_abslog=False, preprocess_per_dataset=True),
                  training=_A(sign_head=False, loss="MSE"), model=_A()),
           modelname="LLOCAMuPTransformer", _signedlog_quantile=lambda: 0.01, _signedlog_scale_shift=lambda: 0.0)
    if parent_stats is not None:
        e._ft_parent_target_stats = parent_stats
    return e


def _resolve(e, amps, N=5):
    _method()(e, ["probe"], ["train", "val"], lambda r, n: amps[r], amps["train"], lambda n: N, True,
              None, None, None, None)
    tr = e._amp_trafos_pp[0]
    z = {r: np.asarray(preprocessing.preprocess_amplitude(a, trafos=tr, mean=e.prepd_mean[0], std=e.prepd_std[0])[0],
                       dtype=np.float64).reshape(-1) for r, a in amps.items()}
    return tr, z


def _amps(seed=0):
    rng = np.random.default_rng(seed)
    return {r: np.exp(rng.normal(-12.0, 2.5, n)) for r, n in (("train", 300), ("val", 400))}


def test_parent_shared_target_is_the_zero_shot_reads():
    amps = _amps()
    mu_c, sd_c = -6.0, 4.0                                   # the parent's pooled statistics, far from the probe's own
    pstats = {"shared_standardization": True, "prepd_mean": [mu_c] * 7, "prepd_std": [sd_c] * 7,
              "amp_trafos_pp": [["log", "standardization"]] * 7}
    tr_own, z_own = _resolve(_exp(), amps)                   # the zero-shot read's truth: the probe's own fit, inverted
    own = _exp(); _resolve(own, amps)
    t = own.prepd_mean[0] + own.prepd_std[0] * z_own["val"]  # log|M|^2
    assert np.allclose(t, np.log(amps["val"].reshape(-1)), atol=1e-6)

    e = _exp(pstats)
    tr, z = _resolve(e, amps)
    assert list(tr) == list(tr_own)                          # the probe's own transform
    assert e.prepd_mean == [mu_c] and e.prepd_std == [sd_c]  # the parent's standardization
    assert np.allclose(z["val"], (t - mu_c) / sd_c, atol=1e-6)

    # any parent output h: the fine-tune's step-0 loss in log units is the zero-shot MSE
    h = np.random.default_rng(1).normal(0.3, 0.8, len(t))
    zero_shot = np.mean((mu_c + sd_c * h - t) ** 2)          # tools/zero_shot_parent.py, mse_logm2_real
    ft_step0 = np.mean((h - z["val"]) ** 2) * sd_c ** 2      # tools/catalog_ft.py, mse_log = val_loss * prepd_std^2
    assert abs(ft_step0 / zero_shot - 1) < 1e-6
    # the refit fine-tune's start is the output read in the probe's own units instead (q_own)
    q_own = own.prepd_mean[0] + own.prepd_std[0] * h
    assert abs(np.mean((h - z_own["val"]) ** 2) * own.prepd_std[0] ** 2 / np.mean((q_own - t) ** 2) - 1) < 1e-6


def test_affine_and_per_process_parents():
    am = {"affine_multiplicity": {"a": 1.0, "b": -2.0, "sigma": 3.0}, "shared_standardization": True,
          "prepd_mean": [0.0], "prepd_std": [1.0]}
    assert parent_target_standardization(am, [5, 6]) == ([-9.0, -11.0], [3.0, 3.0], "affine_multiplicity")
    e = _exp(am)
    _resolve(e, _amps(), N=6)
    assert e.prepd_mean == [-11.0] and e.prepd_std == [3.0] and e._affine_multiplicity == am["affine_multiplicity"]
    try:
        parent_target_standardization({"prepd_mean": [0.0, 1.0], "prepd_std": [1.0, 1.0]}, [5])
    except ValueError:
        pass
    else:
        raise AssertionError("a per-process parent has no statistics for a new process")


def test_tools_share_the_helper():
    zs = open(os.path.join(ROOT, "tools", "zero_shot_parent.py")).read()
    ft = open(os.path.join(ROOT, "tools", "catalog_ft.py")).read()
    assert "parent_target_standardization(pstats, [N])" in zs
    assert '"parent_shared" if a.keep_parent_stats' in ft and "fine_tune.target_stats={target_stats}" in ft


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
