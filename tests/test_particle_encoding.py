"""The particle property table describes the particles the generator produces.

Two things went wrong before catalog_v2 (docs/results.tex, catalog_v2 census): the
log10_mass_gev column carried PDG masses (d 4.7 MeV, s 93 MeV, b 4.18 GeV) although
the generator makes u, d, s, c, e, mu massless and b 4.7 GeV, and nothing else told d
from s or u from c, so dd->dd and ds->ds (or ud->ud and us->us, which differ by W
exchange) were one input with two targets. The table now carries the generator's
masses and a generation column; these tests pin both."""
import os, sys
import numpy as np
try:
    import pytest
except Exception:          # the venv's pytest is shadowed by a local `py` module; the runner below works without it
    pytest = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import particle_ids as P


def _row(pdg, names=None, **flags):
    mat, nm = P.build_property_matrix(**flags)
    return mat[P.GLOBAL_PDG_IDX[pdg]], nm


def test_light_fermions_are_massless_like_the_generator():
    m = P.PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
    for pdg in (1, 2, 3, 4, 11, 13, 12, 14, 16, 21, 22):
        assert P.PARTICLE_PROPERTIES[pdg][m] == P._MASSLESS, pdg
        assert P.PARTICLE_PROPERTIES[-pdg if pdg < 20 else pdg][m] == P._MASSLESS, pdg


def test_massive_entries_match_the_generator_card():
    m = P.PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
    want = {15: 1.777, 5: 4.7, 6: 172.5, 23: 91.188, 24: 80.419, 25: 125.0}
    for pdg, mass in want.items():
        assert abs(10 ** P.PARTICLE_PROPERTIES[pdg][m] - mass) < 1e-6 * mass, pdg
    assert P.GENERATOR_MASSES_GEV["t"] == 172.5 and P.GENERATOR_MASSES_GEV["b"] == 4.7


def test_locked_top_mass_agrees_with_the_pipeline():
    try:
        import mg5_pipeline_final as mg
    except Exception as e:      # the pipeline needs the cluster env; the table constant is pinned above
        print(f"  (skipped: pipeline not importable here: {e})"); return
    assert float(mg.LOCKED_MT) == P.GENERATOR_MASSES_GEV["t"]
    assert abs(mg._table_mass(6) - 172.5) < 1e-9 and mg._table_mass(1) == 0.0
    assert abs(mg._table_mass(5) - 4.7) < 1e-9


def test_generation_is_the_only_difference_between_flavour_twins():
    for a, b in ((1, 3), (2, 4), (11, 13), (12, 14), (-1, -3), (-2, -4)):
        ra, rb = P.PARTICLE_PROPERTIES[a], P.PARTICLE_PROPERTIES[b]
        diff = [n for n, x, y in zip(P.PARTICLE_FEATURE_NAMES, ra, rb) if abs(x - y) > 1e-9]
        assert diff == ["generation"], (a, b, diff)
    g = P.PARTICLE_FEATURE_NAMES.index("generation")
    assert [P.PARTICLE_PROPERTIES[p][g] for p in (2, 4, 6)] == [1.0, 2.0, 3.0]
    assert [P.PARTICLE_PROPERTIES[p][g] for p in (21, 22, 23, 24, 25)] == [0.0] * 5


def test_generation_onehot_columns():
    flags = dict(spin_onehot=True, color_onehot=True, generation_onehot=True,
                 is_massless=True, standardize=True)
    mat, names = P.build_property_matrix(**flags)
    gcols = [names.index(f"gen_is_{i}") for i in (1, 2, 3)]
    assert "generation" not in names
    for pdg, gen in ((1, 1), (3, 2), (5, 3), (13, 2), (-16, 3)):
        assert mat[P.GLOBAL_PDG_IDX[pdg]][gcols].tolist() == [float(gen == i) for i in (1, 2, 3)]
    assert np.all(mat[P.GLOBAL_PDG_IDX[21]][gcols] == 0) and np.all(mat[0] == 0)   # boson, padding
    d, s = mat[P.GLOBAL_PDG_IDX[1]], mat[P.GLOBAL_PDG_IDX[3]]
    assert [n for n, x, y in zip(names, d, s) if abs(x - y) > 1e-9] == ["gen_is_1", "gen_is_2"]
    # the mass-from-momenta spec still points at the right columns
    spec = P.mass_feature_spec(**flags)
    assert names[spec["mass_col"]] == "log10_mass_gev" and names[spec["is_massless_col"]] == "is_massless"


def test_generation_feature_off_is_the_old_encoding():
    """generation_feature=False drops the column: d and s (u and c) become identical
    inputs, which is exactly the collision the census found; used for A/B baselines."""
    flags = dict(spin_onehot=True, color_onehot=True, is_massless=True, standardize=True)
    mat, names = P.build_property_matrix(**flags, generation_feature=False)
    assert "generation" not in names and not any(n.startswith("gen_is_") for n in names)
    assert mat.shape[1] == len(names)
    for a, b in ((1, 3), (2, 4), (11, 13)):
        assert np.allclose(mat[P.GLOBAL_PDG_IDX[a]], mat[P.GLOBAL_PDG_IDX[b]]), (a, b)
    spec = P.mass_feature_spec(**flags, generation_feature=False)
    assert names[spec["mass_col"]] == "log10_mass_gev"


def test_frozen_standardization_matches_the_table():
    m2, n2 = P.add_is_massless_flag(P.GLOBAL_PROPERTY_MATRIX)
    real = np.zeros(len(m2), dtype=bool)
    real[1:len(P.SM_PDGS) + 1] = True        # the SM rows the constants were frozen over (non-SM rows appended, D49)
    for f, (mu, sd) in P._FROZEN_STD_STATS.items():
        col = m2[real, n2.index(f)]
        assert abs(col.mean() - mu) < 1e-5 and abs(col.std() - sd) < 1e-5, f


def test_non_sm_rows_are_appended_after_the_sm_ones():
    """D49: the non-SM rows come after every SM row, so an SM particle keeps its index and its encoded row (a
    checkpoint's property_matrix is a prefix of the current one) and the massless neutralisation ignores them."""
    n_sm = len(P.SM_PDGS)
    assert [P.GLOBAL_PDG_IDX[p] for p in P.SM_PDGS] == list(range(1, n_sm + 1))
    assert sorted(P.GLOBAL_PDG_IDX[p] for p in P.BSM_PARTICLE_PROPERTIES) == list(range(n_sm + 1, P.GLOBAL_N_ENTRIES))
    m = P.PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
    sm = np.array([P.PARTICLE_PROPERTIES[p][m] for p in P.SM_PDGS], dtype=np.float32)
    assert P.MASSLESS_NEUTRAL_LOG10 == sm[np.abs(sm - P._MASSLESS) >= 1e-6].mean()
    mat, names = P.build_property_matrix(is_massless=True)
    assert mat[P.GLOBAL_PDG_IDX[22], names.index("log10_mass_gev")] == P.MASSLESS_NEUTRAL_LOG10


def test_massless_b_is_its_own_species_for_catalog_v4_only():
    """catalog_v4 generates the b massless (sm-no_b_mass, beam_masses {}): its pools read the massless-b row, which
    differs from the b's only in the mass; catalog_v3's entries (b at 4.7 in beams and finals) keep the b's row."""
    m = P.PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
    for s in (1, -1):
        rb, r0 = P.PARTICLE_PROPERTIES[5 * s], P.PARTICLE_PROPERTIES[s * P.MASSLESS_B_PDG]
        assert [n for n, x, y in zip(P.PARTICLE_FEATURE_NAMES, rb, r0) if abs(x - y) > 1e-9] == ["log10_mass_gev"]
        assert r0[m] == P._MASSLESS
    ids = np.array([[5, -5, 5, -5, 21]])
    v4_final = {"pdg_ids": [11, -11, 5, -5, 21], "m_finals": [0.0, 0.0, 0.0], "beam_masses": {}}
    v4_beam = {"pdg_ids": [5, -5, 23], "m_finals": [91.188], "beam_masses": {}}
    v3_final = {"pdg_ids": [11, -11, 5, -5, 21], "m_finals": [4.7, 4.7, 0.0]}
    v3_beam = {"pdg_ids": [5, -5, 23], "m_finals": [91.188]}                       # no beam_masses: b on 4.7
    old_2to2 = {"pdg_ids": [11, -11, 5, -5], "m_final": 4.7}
    assert P.b_mass_of_process(v4_final) == 0.0 and P.b_mass_of_process(v4_beam) == 0.0
    assert P.b_mass_of_process(v3_final) == 4.7 and P.b_mass_of_process(v3_beam) == 4.7
    assert P.b_mass_of_process(old_2to2) == 4.7
    assert P.b_mass_of_process({"pdg_ids": [11, -11, 1, -1], "m_finals": [0.0, 0.0]}) is None
    mb = P.MASSLESS_B_PDG
    assert P.species_pdgs(ids, None, v4_final).tolist() == [[mb, -mb, mb, -mb, 21]]
    for cfg in (v3_final, v3_beam, old_2to2, None):
        assert np.asarray(P.species_pdgs(ids, None, cfg)).tolist() == ids.tolist()
    # with a heavy-quark top the two mappings compose
    hq = {"pdg_ids": [5, -5, 6, -6], "m_finals": [300.0, 300.0], "beam_masses": {}}
    out = P.species_pdgs(np.array([[5, -5, 6, -6]]), {"masses": {6: 300.0}}, hq)
    assert out.tolist() == [[mb, -mb, 6000300, -6000300]]
    # appended after every earlier row: catalog_v3 checkpoints' property matrices stay a prefix
    assert P.GLOBAL_PDG_IDX[P.MASSLESS_B_PDG] > P.GLOBAL_PDG_IDX[-6000600]
    assert P.GLOBAL_PDG_IDX[-P.MASSLESS_B_PDG] == P.GLOBAL_N_ENTRIES - 1


def test_massless_b_matches_the_catalog_v4_entries():
    try:
        import mg5_pipeline_final as mg
    except Exception as e:      # the pipeline needs the cluster env
        print(f"  (skipped: pipeline not importable here: {e})"); return
    n4 = n3 = 0
    for name, cfg in mg.PROCESSES.items():
        if cfg.get("_v4") and any(abs(int(p)) == 5 for p in cfg.get("pdg_ids", [])):
            assert P.b_mass_of_process(cfg) == 0.0, name; n4 += 1
        if cfg.get("_v3") and any(abs(int(p)) == 5 for p in cfg.get("pdg_ids", [])):
            assert P.b_mass_of_process(cfg) == 4.7, name; n3 += 1
    assert n4 > 0 and n3 > 0


if __name__ == "__main__":       # plain runner: python tests/test_particle_encoding.py
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  {name}: OK")
            except Exception:
                fails += 1; print(f"  {name}: FAIL"); traceback.print_exc()
    raise SystemExit(fails)
