#!/usr/bin/env python
"""The order features' coupling powers come from the generated diagrams, the source the dimensionless target reads
(plan catalog_coverage), and the pools' names and recipe ids do not move with them.

The catalog table (mg5_pipeline_final.order_vector, from alphas_power and QED<=N) is wrong for catalog_v4's b-quark
Higgs entries (massless b, no Yukawa: purely electroweak, the table claims alpha_s^2 or alpha_s^3) and for the two BSM
entries with a gluon (Z' + g, the 2HDM h2 h3 + g: the table has no alpha_s), in both catalogs. Expected values: the
diagram audit (tools/mixed_order_audit.py, analysis/catalog_v4/data/mixed_order_audit.json), each a single amplitude
QCD order k, so [0, 0, k, nfinal - k].

Part 1 (any machine, no torch): order_vector_from_diagrams on hand-made sidecars (pure, mixed, stripped virtual,
alphas_prefactor, explicit order) and recipe_id's independence of amp_orders.
Part 2 (a site: torch, the diagram sidecars in data/diagrams and the pools under DATA_DIR; FA_TEST_SITE=1): the
experiment's recipe path (_resolve_recipe_config) on the affected entries gives the diagram's orders, every other entry
of the four recipes keeps the table's unless its diagrams say otherwise (listed), and every affected pool (train, val,
test) is still a cache hit at the current code (datagen.ensure_dataset with require_cache: same path, same recipe_id).

Run:  python tests/test_order_features.py      (FA_TEST_SITE=1 on a site for part 2)
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402

# name: (nfinal, the single amplitude QCD order the diagrams have)
B_HIGGS_TRAIN = {
    "v4_bb_abbh": (4, 0), "v4_bb_bbgh": (4, 1), "v4_bb_bbh": (3, 0), "v4_bb_bbhh": (4, 0), "v4_bbx_abbxh": (4, 0),
    "v4_bbx_bbxgh": (4, 1), "v4_bbx_bbxh": (3, 0), "v4_bbx_ccxh": (3, 0), "v4_db_abdh": (4, 0), "v4_db_bdh": (3, 0),
    "v4_db_bdhh": (4, 0), "v4_dbx_abxdh": (4, 0), "v4_dbx_bxdgh": (4, 1), "v4_ddx_bbxh": (3, 0),
    "v4_ubx_abxhu": (4, 0),
}
B_HIGGS_HELDOUT = {
    "v4_bbx_accxh": (4, 0), "v4_bbx_addxh": (4, 0), "v4_bbx_bbxhh": (4, 0), "v4_bbx_ccxgh": (4, 1),
    "v4_bbx_ddxhh": (4, 0), "v4_ddx_bbxghh": (5, 1), "v4_dg_bbxdh": (4, 1),
}
BSM = {"v4_zp_uux_Y1g": (2, 1), "v4_thdm_uux_h2h3g": (3, 1), "zp_uux_Y1g": (2, 1), "thdm_uux_h2h3g": (3, 1)}
AFFECTED = {**B_HIGGS_TRAIN, **B_HIGGS_HELDOUT, **BSM}
RECIPES = ["recipes/catalog_v4_train.yaml", "recipes/catalog_v4_shared_heldout.yaml",
           "recipes/catalog_v3_train.yaml", "recipes/catalog_v3_shared_heldout.yaml"]


def _sidecar(d, name, orders, loop=False):
    """A minimal sidecar: one diagram per QCD order, its vertices carrying that order."""
    dg = [{"diagram_type": "loop" if loop else "tree",
           "nodes": [{"kind": "vertex", "orders": {"QCD": a + (2 if loop else 0)}}, {"kind": "external"}]}
          for a in orders]
    p = os.path.join(d, f"{name}.diagrams.json")
    json.dump({"subprocesses": [{"diagrams": dg}]}, open(p, "w"))
    return p


def test_from_diagrams():
    with tempfile.TemporaryDirectory() as d:
        pure = {"nfinal": 3, "alphas_power": 2, "mg5_generate": ["b b~ > b b~ h QED<=3"]}
        assert mg.order_vector(pure) == [0, 0, 2, 3]                       # the table's claim
        q = mg.sidecar_qcd_orders(_sidecar(d, "pure", [0]))
        assert q == [0] and mg.order_vector_from_diagrams(pure, q) == [0, 0, 0, 3]
        mixed = {"nfinal": 2, "alphas_power": 2, "mg5_generate": ["u u~ > d d~ QED<=2"]}
        q = mg.sidecar_qcd_orders(_sidecar(d, "mixed", [0, 2, 1]))
        assert q == [0, 1, 2] and mg.order_vector_from_diagrams(mixed, q) == [0, 0, 2, 2]   # both maxima
        # a stripped virtual: loop diagrams' QCD orders minus 2 are the Born's; the loop count is kept
        virt = {"nfinal": 2, "kind": "virt", "alphas_power": 1, "virt_base": None}
        q = mg.sidecar_qcd_orders(_sidecar(d, "virt", [0], loop=True))
        assert q == [0] and mg.order_vector_from_diagrams(virt, q, [1, 0, 0, 2]) == [1, 0, 0, 2]
        assert mg.order_vector_from_diagrams({**virt, "alphas_prefactor": True}, q, [1, 0, 1, 2]) == [1, 0, 1, 2]
        expl = {"nfinal": 2, "order": [1, 0, 2, 2]}
        assert mg.order_vector_from_diagrams(expl, [0]) == [1, 0, 2, 2]  # loop-induced: explicit order kept
        assert mg.order_vector_from_diagrams(pure, None) == [0, 0, 2, 3]  # no sidecar: the table
    # the affected entries' table values really are wrong, and the expected vectors are pure
    for n, (nf, k) in AFFECTED.items():
        pc = mg.PROCESSES[n]
        assert int(pc["nfinal"]) == nf, n
        assert mg.order_vector(pc)[2] != k, (n, mg.order_vector(pc))
    print("[from_diagrams] ok")


def test_recipe_id_ignores_orders():
    for n, (nf, k) in AFFECTED.items():
        pc = mg.PROCESSES[n]
        lo = max(25.0, 1.05 * sum(pc.get("m_finals", [0.0] * nf)))
        rec = mg.variable_energy_recipe(n, lo, 1000.0, 10000, role="val", seed=42)
        alt = dict(rec, amp_orders=[0, 0, k, nf - k])
        assert mg.recipe_id(rec) == mg.recipe_id(alt), n
        assert mg.recipe_output_path(rec, "/x") == mg.recipe_output_path(alt, "/x"), n
    print("[recipe_id] ok: amp_orders is not part of a pool's identity or path")


def test_site():
    import yaml
    from omegaconf import OmegaConf
    import datagen
    from experiment import AmplitudeExperiment
    os.chdir(ROOT)
    for rf in RECIPES:
        procs = yaml.safe_load(open(rf))["processes"]
        names = [p["name"] for p in procs]
        table = {}
        for p in procs:
            pc = mg.PROCESSES.get(p["name"]) or mg.PROCESSES.get(p.get("base", p["name"]))
            table[p["name"]] = mg.order_vector(pc)
        cfg = OmegaConf.create({"data": {"source": "recipes", "processes_file": os.path.join(ROOT, rf)},
                                "model": {"diagrams_dir": "data/diagrams"}})
        exp = AmplitudeExperiment.__new__(AmplitudeExperiment)
        exp.cfg = cfg
        exp._resolve_recipe_config()
        got = {n: [int(x) for x in o] for n, o in zip(cfg.data.dataset, cfg.data.amp_orders)}
        base = {p["name"]: p.get("base", p["name"]).split("__")[0] for p in procs}   # a decorated variant (__c30)
        hit = [n for n in names if base[n] in AFFECTED]
        for n in hit:
            nf, k = AFFECTED[base[n]]
            assert got[n] == [0, 0, k, nf - k], (n, got[n], table[n])
        # every other change must be a mixed entry's alpha maximum the diagrams pin below the QED<=N bound, never an
        # alpha_s power (the audit found the 17 + 8 above as the only alpha_s disagreements)
        other = {n: (table[n], got[n]) for n in names if base[n] not in AFFECTED and got[n] != table[n]}
        bad = {n: v for n, v in other.items() if v[0][:3] != v[1][:3]}
        assert not bad, f"{rf}: alpha_s power changed outside the expected entries: {bad}"
        print(f"[site] {os.path.basename(rf)}: {len(hit)} expected entries have the diagram's orders; "
              f"{len(other)} others change only their alpha maximum: {sorted(other.items())[:10]}")
        # pools: the same paths and recipe ids at the current code (a cache hit, nothing generated)
        for p in procs:
            if base[p["name"]] not in AFFECTED:
                continue
            smin, smax = p["sqrts"]
            for role in ("train", "val", "test"):
                out = datagen.ensure_dataset(p["name"], smin, smax, int(p[f"n_{role}"]), role, 42,
                                             require_cache=True)
                assert os.path.exists(out), out
        print(f"[site] {os.path.basename(rf)}: {3 * len(hit)} affected pools are cache hits (keys unchanged)")


if __name__ == "__main__":
    test_from_diagrams()
    test_recipe_id_ignores_orders()
    if os.environ.get("FA_TEST_SITE"):
        test_site()
    print("all ok")
