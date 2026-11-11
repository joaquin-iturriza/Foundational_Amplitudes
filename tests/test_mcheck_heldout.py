"""CPU checks of multichannel_check's held-out pp read (plan catalog_coverage step 58, design-reviewer FIX 2026-11-06):
pp_sources names the probe's train+test as the table and the probe's val plus the class's dsh_ pools as held out;
_pp reads only pools that exist, splits the out-of-support share into sqrt(s_hat) outside the window and cuts failed in
the partonic frame; heldout_read counts held-out in-support events outside the kept range and gives 3/N only at 0.
    .venv-cpu/bin/python -m pytest tests/test_mcheck_heldout.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v4"))
import multichannel_check as mc  # noqa: E402


def test_sources():
    s = mc.pp_sources("v4_uu_bbxuu", "pp13v4_uu_bbxuu")
    assert s["table"] == ["pp13v4_uu_bbxuu_13000-13000GeV_train_lhe_amplitudes.npy",
                          "pp13v4_uu_bbxuu_13000-13000GeV_test_lhe_amplitudes.npy"]
    assert s["heldout"][0] == "pp13v4_uu_bbxuu_13000-13000GeV_val_lhe_amplitudes.npy"
    assert s["heldout"][1:] == [f"dsh_uu_bbxuu_13000-13000GeV_{r}_lhe_amplitudes.npy" for r in ("train", "val", "test")]


def _events(n, rng, sq, eta):
    """u u~ > e+ e- at partonic sqrt(s) sq, final rapidity eta in the partonic frame, the pair boosted along z by 1"""
    E = sq / 2
    pz = E * np.tanh(eta); pt = np.sqrt(E ** 2 - pz ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    P = np.zeros((n, 4, 4))
    P[:, 0] = np.stack([E, 0 * E, 0 * E, E], 1); P[:, 1] = np.stack([E, 0 * E, 0 * E, -E], 1)
    P[:, 2] = np.stack([E, pt * np.cos(phi), pt * np.sin(phi), pz], 1)
    P[:, 3] = np.stack([E, -pt * np.cos(phi), -pt * np.sin(phi), -pz], 1)
    b = np.tanh(1.0); g = 1 / np.sqrt(1 - b * b)
    E2 = g * (P[..., 0] + b * P[..., 3]); z2 = g * (P[..., 3] + b * P[..., 0])
    P[..., 0], P[..., 3] = E2, z2
    pdg = np.tile([2, -2, -11, 11], (n, 1))
    return np.concatenate([P.reshape(n, 16), pdg, np.full((n, 1), 1e-3)], 1)


def test_pp_read_and_split(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    ok = _events(60, rng, np.full(60, 500.0), np.full(60, 1.0))          # in the window, central
    low = _events(30, rng, np.full(30, 50.0), np.full(30, 1.0))          # below the window
    fwd = _events(10, rng, np.full(10, 500.0), np.full(10, 5.5))         # partonic |eta| 5.5 > 5
    np.save(tmp_path / "a_val.npy", np.concatenate([ok, low, fwd]))
    monkeypatch.setattr(mc, "_find", lambda stem: str(tmp_path / stem) if (tmp_path / stem).exists() else None)
    cfg = {"pdg_ids": [2, -2, -11, 11], "m_finals": [0.0, 0.0],
           "fiducial_cuts": {"pt_min": 10.0, "cos_max": float(np.tanh(5.0)), "dr_min": 0.4, "m_min": -1.0}}
    v = mc._pp(["a_val.npy", "missing.npy"], cfg, 100.0, 6000.0, "t")
    assert v["files"] == ["a_val.npy"] and len(v["lm"]) == 100 and int(v["sup"].sum()) == 60
    assert np.isclose(v["outside_window"], 0.30) and np.isclose(v["fail_cuts_in_window"], 0.10)
    assert mc._pp(["missing.npy"], cfg, 100.0, 6000.0, "t") is None


def test_heldout_read():
    src = {"table": {"n_in_support": 100, "n_in_support_outside_kept": 0, "by_construction": True},
           "heldout": {"n_in_support": 1500, "n_in_support_outside_kept": 0}}
    h = mc.heldout_read({"sources": src})
    assert h["heldout_outside_kept"] == 0 and h["heldout_upper95"] == pytest.approx(3 / 1500)
    src["heldout"]["n_in_support_outside_kept"] = 4
    h = mc.heldout_read({"sources": src})
    assert h["heldout_upper95"] is None and h["heldout_share"] == pytest.approx(4 / 1500)
    assert mc.heldout_read({}) == {"pp_sources": {}}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
