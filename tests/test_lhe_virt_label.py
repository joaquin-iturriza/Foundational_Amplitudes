"""The lhe label rule 'virt' (mg5_pipeline_final._lhe_virt_block): a MadLoop-unusable point is replaced by the next
spare event of the same collider energy, the block keeps its size and energy shares, the labels stay signed, and a
pool above 2% unusable or out of spare events is refused. MadLoop is replaced by a stub (lhe_label_virt); CPU.
    .venv-cpu/bin/pytest tests/test_lhe_virt_label.py"""
import os, sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402

NAME = "hd_epem_bbx_nlo"


def _events(n, energies, seed=0):
    rng = np.random.default_rng(seed)
    E = np.asarray(energies)[np.arange(n) % len(energies)]
    m = np.zeros((n, 4, 4))
    m[:, 0] = np.stack([E / 2, 0 * E, 0 * E, E / 2], 1)
    m[:, 1] = np.stack([E / 2, 0 * E, 0 * E, -E / 2], 1)
    c = rng.uniform(-1, 1, n)
    s = np.sqrt(1 - c ** 2)
    m[:, 2] = np.stack([E / 2, E / 2 * s, 0 * E, E / 2 * c], 1)
    m[:, 3] = np.stack([E / 2, -E / 2 * s, 0 * E, -E / 2 * c], 1)
    return m


def _stub(bad_rows_of):
    """lhe_label_virt stand-in: value = signed function of the event, unusable where bad_rows_of(m) says."""
    def f(nlo, m, alphas_mz=0.118, workers=None):
        v = m[:, 2, 3] - 0.1 * m[:, 2, 0]           # signed
        ok = ~bad_rows_of(m)
        return v, ok, np.abs(v) + 1.0, np.full(len(m), alphas_mz)
    return f


def _cfg():
    return mg.PROCESSES[NAME]


def test_replaced_same_energy(monkeypatch):
    E = [240.0, 3000.0]
    mom = _events(400, E, seed=3)
    aq = np.arange(400, dtype=float)
    need = 100
    # the block's most forward event is unusable (one in a hundred), and any spare as forward as it
    cos = mom[:, 2, 3] / mom[:, 2, 0]
    thr = np.sort(cos[:need])[-2]
    monkeypatch.setattr(mg, "lhe_label_virt", _stub(lambda m: m[:, 2, 3] / m[:, 2, 0] > thr))
    spare = [need]
    m, a_up, amps, born, a, info = mg._lhe_virt_block(NAME, _cfg(), mom[:need], aq[:need], mom, aq, spare, need)
    assert m.shape == (need, 4, 4)
    assert info["n_unusable"] >= 1 and spare[0] > need
    assert not np.any(m[:, 2, 3] / m[:, 2, 0] > thr)
    # energy shares unchanged
    e0 = np.round(mom[:need, 0, 0] * 2, 3); e1 = np.round(m[:, 0, 0] * 2, 3)
    assert np.array_equal(e0, e1)
    assert (amps < 0).any() and (amps > 0).any()     # signed labels kept
    assert np.allclose(a, 0.118)
    # usable rows untouched
    good = ~(cos[:need] > thr)
    assert np.array_equal(m[good], mom[:need][good]) and np.array_equal(a_up[good], aq[:need][good])


def test_refused_above_two_percent(monkeypatch):
    mom = _events(300, [500.0])
    monkeypatch.setattr(mg, "lhe_label_virt", _stub(lambda m: m[:, 2, 3] / m[:, 2, 0] > 0.9))
    with pytest.raises(RuntimeError, match="not trustworthy"):
        mg._lhe_virt_block(NAME, _cfg(), mom[:100], np.zeros(100), mom, np.zeros(300), [100], 100)


def test_refused_without_spares(monkeypatch):
    mom = _events(100, [500.0])
    monkeypatch.setattr(mg, "lhe_label_virt", _stub(lambda m: np.arange(len(m)) == 0))
    with pytest.raises(RuntimeError, match="no spare events"):
        mg._lhe_virt_block(NAME, _cfg(), mom, np.zeros(100), mom, np.zeros(100), [100], 100)


def test_stripped_virtual_alpha_s_fixed():
    with pytest.raises(ValueError, match="0.118"):
        mg.lhe_label_virt("v4_epem_bbx_nlo", _events(2, [500.0]), alphas_mz=0.13)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_virt_entry_parent_physics_is_the_labellers():
    """A 'virt'-labelled H5 pool is labelled by its parent's own one-loop standalone, so its physics check against the
    parent passes without comparing the Born tree standalone (which only draws kinematics: sm-no_b_mass, MadEvent's
    alpha_s), the refusal every hd_test_h5 part hit on 2026-10-31."""
    import mg5_pipeline_final as mg
    names = [n for n, c in mg.PROCESSES.items() if (c.get("lhe") or {}).get("label_alphas") == "virt"]
    assert names, "no virt-labelled entry registered"
    for n in names:
        chk = mg.lhe_parent_physics_check(n)
        assert chk["status"] == "ok" and chk["labels_from"] == mg.PROCESSES[n]["parent"], (n, chk)


def test_virt_assembly_builds_no_born_standalone():
    """Assembly of a 'virt' pool never builds the Born tree standalone: g g > z h ([noborn=QCD]) has none, and its
    hd_test_h5 assembly died on 'No compilable backend' (2026-10-31)."""
    import mg5_pipeline_final as mg
    name = next(n for n, c in mg.PROCESSES.items()
                if (c.get("lhe") or {}).get("label_alphas") == "virt" and "zh_loop" in n)
    saved = (mg.lhe_part_paths, mg._lhe_check_recipes, mg.lhe_prepare, mg._lhe_finish)
    try:
        mg.lhe_part_paths = lambda process, items: []
        mg._lhe_check_recipes = lambda process, items: (0, 0, 0)

        def no_prepare(*a, **k):
            raise AssertionError("lhe_prepare called for a virt pool")
        mg.lhe_prepare = no_prepare
        seen = {}

        def finish(process, recs, items, moms, aqcds, runs, banner, secs, sa, *a, **k):
            seen["sa"] = sa
            return "finished"
        mg._lhe_finish = finish
        import tempfile
        try:
            mg.lhe_assemble(name, [], work_root=tempfile.mkdtemp())
        except (ValueError, IndexError, KeyError):
            pass                                  # empty part list past the standalone step: not this test's concern
    finally:
        mg.lhe_part_paths, mg._lhe_check_recipes, mg.lhe_prepare, mg._lhe_finish = saved
    assert seen.get("sa", None) is None


if __name__ == "__main__":
    test_virt_entry_parent_physics_is_the_labellers()
    test_virt_assembly_builds_no_born_standalone()
    print("ok")
