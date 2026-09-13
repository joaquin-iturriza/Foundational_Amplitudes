#!/usr/bin/env python
"""Beam-order / labelling guard for the matrix-element backends.

Every labelling path maps stored rows to MadGraph slots through
mg5_pipeline_final.row_to_slot_perm. The stored convention is: rows follow the
catalog's pdg_ids, the two beams first, beam- (row 0) along +z. MadGraph wants the
legs in generate-string order (e.g. `e+ e- > u u~` puts e+ in slot 0), so `ee_*`
entries need the beams swapped and `u u~ > ...` entries do not. Before this guard
the C++ driver fed the rows unpermuted, so every LO dataset built on it carried
the theta -> pi - theta image of its own beam labels, while matrix2py/MadLoop
(hard-coded swap) were right for e+e- and would have been wrong for quarks.

Part 1 (any machine): the permutation builder on the whole catalog.
Part 2 (cluster, needs the compiled ee_uu standalone): physics check on the real
C++ driver -- above the Z pole the u quark prefers the e- direction
(A_FB(u) ~ +0.7 at 200 GeV), so |M|^2(u along e-) > |M|^2(u along e+).

Run:  python tests/test_beam_order.py          (part 2 is skipped if no standalone)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402


def test_perm_builder():
    n_ee = n_q = 0
    for name, cfg in mg.PROCESSES.items():
        if "pdg_ids" not in cfg or "mg5_generate" not in cfg:
            continue
        perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
        assert sorted(perm) == list(range(len(cfg["pdg_ids"]))), name
        ini = mg.generate_slot_pdgs(cfg["mg5_generate"][0])[0]
        assert [cfg["pdg_ids"][j] for j in perm[:2]] == ini, name
        if cfg["pdg_ids"][:2] == [11, -11]:
            assert perm[:2] == [1, 0], (name, perm); n_ee += 1
        else:
            n_q += 1
    # identical initial particles and a coupling-order suffix
    assert mg.row_to_slot_perm([2, 2, 2, 2], ["generate u u > u u QED<=2"]) == [0, 1, 2, 3]
    # a mislabelled catalog entry must be refused, never silently permuted
    try:
        mg.row_to_slot_perm([11, -11, 1, -1], ["generate e+ e- > u u~"])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched pdg_ids accepted")
    print(f"[perm] ok: {n_ee} e+e- entries swap the beams, {n_q} others keep slot order")


def _event_2to2(sqrts, cos_t, pdg=(11, -11, 2, -2)):
    E = sqrts / 2.0
    st = np.sqrt(1.0 - cos_t ** 2)
    P = np.array([[E, 0.0, 0.0, E], [E, 0.0, 0.0, -E],
                  [E, E * st, 0.0, E * cos_t], [E, -E * st, 0.0, -E * cos_t]])
    return P, np.array(pdg)


def test_cpp_driver_physics():
    sa = f"{mg.WORK_DIR}/ee_uu_standalone"
    sub = f"{sa}/SubProcesses/P1_Sigma_sm_epem_uux"
    if not os.path.isdir(sub):
        print("[cpp] skipped: no compiled ee_uu standalone at", sa)
        return
    driver = mg.compile_cpp_driver(sa, ["P1_Sigma_sm_epem_uux"], 4)
    perm = mg.row_to_slot_perm([11, -11, 2, -2], ["generate e+ e- > u u~"])
    fwd, bwd = _event_2to2(200.0, +0.6), _event_2to2(200.0, -0.6)
    with mg.CppDriverPipe(driver, sa) as pipe:
        m_fwd, m_bwd = pipe.compute([fwd, bwd], perm=perm, alphas=0.118)
        m_naive = pipe.compute([fwd], perm=None, alphas=0.118)[0]
        m_as = pipe.compute([fwd, fwd], perm=perm, alphas=[0.118, 0.100])
    print(f"[cpp] |M|^2(u along e-) = {m_fwd:.4e}   |M|^2(u along e+) = {m_bwd:.4e}"
          f"   unpermuted = {m_naive:.4e}")
    assert m_fwd > m_bwd, "A_FB(u) must be positive above the Z pole"
    assert np.isclose(m_naive, m_bwd, rtol=1e-12), "unpermuted rows must equal the mirrored point"
    assert np.isclose(m_as[0], m_as[1], rtol=1e-12), "EW process must not depend on alpha_s"
    print("[cpp] ok")


def test_stored_pools(recipe="recipes/pretrain8_short.yaml", n_check=5):
    """Recompute a few stored events of every built train pool of `recipe` with the
    pipeline's own backend (perm + alpha_s(sqrt s)) and require agreement: catches a
    stale driver binary, a wrong permutation, or a broken alpha_s protocol."""
    import yaml
    import datagen
    cache = datagen.train_cache_dir()
    procs = yaml.safe_load(open(os.path.join(ROOT, recipe)))["processes"]
    checked = 0
    for pr in procs:
        name = pr["name"]; cfg = mg.PROCESSES[name]
        lo, hi = pr["sqrts"]
        path = f"{cache}/{name}_{lo}-{hi}GeV_train_amplitudes.npy"
        if not os.path.exists(path):
            print(f"[pool] {name}: no built pool, skipped"); continue
        sa = f"{mg.WORK_DIR}/{mg.standalone_name(name)}_standalone"
        backend, dirs, driver, _ = mg.detect_compiled_backend(sa)
        if backend != "cpp" or driver is None:
            print(f"[pool] {name}: no v2 driver, skipped"); continue
        d = np.load(path)[:n_check]
        npart = cfg["nfinal"] + 2
        mom = d[:, :npart * 4].reshape(-1, npart, 4)
        pdg = d[0, npart * 4:npart * 4 + npart].astype(int)
        stored = d[:, -1]
        perm = mg.row_to_slot_perm(pdg, cfg["mg5_generate"])
        pin = mom[:, 0] + mom[:, 1]
        sqrts = np.sqrt(pin[:, 0] ** 2 - (pin[:, 1:] ** 2).sum(1))
        amz = float(cfg.get("alphas_mz", 0.118))
        events = [(mom[i], pdg) for i in range(len(d))]
        with mg.CppDriverPipe(driver, sa) as pipe:
            fresh = pipe.compute(events, perm=perm, alphas=mg.compute_alphas(sqrts, alphas_mz=amz))
        ok = np.allclose(fresh, stored, rtol=1e-9)
        print(f"[pool] {name:12s} max rel dev {np.max(np.abs(fresh / stored - 1)):.1e}  {'ok' if ok else 'MISMATCH'}")
        assert ok, name
        checked += 1
    print(f"[pool] {checked} pools agree with a fresh evaluation")


if __name__ == "__main__":
    test_perm_builder()
    test_cpp_driver_physics()
    test_stored_pools()
