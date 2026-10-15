#!/usr/bin/env python
"""catalog_daniel (arXiv:2606.23791 regenerated): catalog counts, registry, lhe_pp identity, the LHE
reader, and that no existing recipe changed identity.

Part 1 (any machine, numpy + yaml): counts, registry, recipes, LHE parsing, existing recipe ids.
Part 2 (FA_TEST_MG5=1 and MadGraph at mg.MG5_BIN): one tiny MadEvent run per process end to end
through build_lhe_pp (dc_gg_gg at 1000 events, against the analytic gg -> gg |M|^2, and dc_gb_bbbx
with massive incoming b quarks), into a temporary directory.

Run:  python tests/test_catalog_daniel.py      (FA_TEST_MG5=1 for part 2)
"""
import gzip
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recipes"))
import mg5_pipeline_final as mg  # noqa: E402


def test_catalog_counts():
    import gen_catalog_daniel as g
    entries, stats, relabels = g.build()
    g.check(entries, stats, relabels)          # 1501 read, 1491 kept, 149 -> 139 held out, 347/345 initial b
    print("[counts] ok")


def test_registry():
    dc = {n: c for n, c in mg.PROCESSES.items() if c.get("kind") == "lhe_pp"}
    assert mg.CATALOG_DANIEL_ENTRIES == 1491 == len(dc)
    assert all(n.startswith("dc_") for n in dc)
    assert not any(n.startswith("dc_") for n, c in mg.PROCESSES.items() if c.get("kind") != "lhe_pp")
    for n, c in dc.items():
        perm = mg.row_to_slot_perm(c["pdg_ids"], c["mg5_generate"])
        assert perm == list(range(len(c["pdg_ids"]))), n          # rows are already in slot order
    c = mg.PROCESSES["dc_gb_bbbx"]
    assert mg.order_vector(c) == [0, 0, 3, 0] and c["pdg_ids"] == [21, 5, 5, 5, -5]
    print(f"[registry] ok: {len(dc)} lhe_pp entries")


def test_recipe_identity():
    r = {role: mg.variable_energy_recipe("dc_gb_bbbx", 13000, 13000, n, role=role, seed=42)
         for role, n in (("val", 10000), ("test", 10000), ("train", 100000))}
    assert [r[k]["lhe"]["block"] for k in ("val", "test", "train")] == [[0, 10000], [10000, 20000], [20000, 120000]]
    assert len({x["lhe"]["iseed"] for x in r.values()}) == 1 and r["val"]["mode"] == "lhe_pp"
    assert mg.recipe_output_path(r["train"], "/D") == "/D/dc_gb_bbbx_13000-13000GeV_train_lhe_amplitudes.npy"
    small = mg.variable_energy_recipe("dc_gb_bbbx", 13000, 13000, 5000, role="train", seed=42)
    assert small["lhe"]["block"] == [20000, 25000] and mg.recipe_id(small) != mg.recipe_id(r["train"])
    try:
        mg.variable_energy_recipe("dc_gb_bbbx", 13000, 13000, 10001, role="val", seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("a role larger than its block was accepted")
    # a run record never changes the identity
    assert mg.recipe_id(dict(r["val"], lhe_run={"xsec_pb": 1.0})) == mg.recipe_id(r["val"])
    print("[recipe] ok")


def test_existing_ids_unchanged():
    """A non-lhe entry gets no lhe fields. (The full fingerprint, recipe_id + path of every entry x
    role of recipes/*.yaml, 16326 of them, was compared before/after when lhe_pp was added.)"""
    r = mg.variable_energy_recipe("ee_uu", 25, 1000, 100000, role="train", seed=42)
    assert r["mode"] == "variable_energy" and "lhe" not in r
    assert "_lhe_" not in mg.recipe_output_path(r, "/D")
    print("[ids] ok")


_LHE = """<LesHouchesEvents version="3.0">
<init>
2212 2212 6.5e3 6.5e3 0 0 247000 247000 -4 1
 1.0e+00 1.0e-02 1.0e+00 1
</init>
<event>
 6  1 +1.0e+00 9.0e+01 7.5e-03 1.30e-01
       21 -1    0    0  501  502 +0.0e+00 +0.0e+00 +5.0e+01 5.0e+01 0.0e+00 0.0e+00 -1.0e+00
        5 -1    0    0  503    0 -0.0e+00 -0.0e+00 -6.0e+01 6.0e+01 4.7e+00 0.0e+00 1.0e+00
        5  1    1    2  501    0 +1.0e+00 +2.0e+00 +3.0e+00 1.0e+01 4.7e+00 0.0e+00 1.0e+00
       -5  1    1    2    0  502 +4.0e+00 +5.0e+00 +6.0e+00 2.0e+01 4.7e+00 0.0e+00 1.0e+00
       25  2    1    2    0    0 +0.0e+00 +0.0e+00 +0.0e+00 1.3e+02 1.25e+02 0.0e+00 0.0e+00
        5  1    1    2  503    0 +7.0e+00 +8.0e+00 +9.0e+00 3.0e+01 4.7e+00 0.0e+00 1.0e+00
<mgrwt>
<rscale>  2 0.9E+02</rscale>
</mgrwt>
</event>
<event>
 5  1 +1.0e+00 8.0e+01 7.5e-03 1.20e-01
        5 -1    0    0  503    0 +0.0e+00 +0.0e+00 +6.0e+01 6.0e+01 4.7e+00 0.0e+00 1.0e+00
       21 -1    0    0  501  502 -0.0e+00 -0.0e+00 -5.0e+01 5.0e+01 0.0e+00 0.0e+00 -1.0e+00
       -5  1    1    2    0  502 +1.0e+00 +1.0e+00 +1.0e+00 1.1e+01 4.7e+00 0.0e+00 1.0e+00
        5  1    1    2  501    0 +2.0e+00 +2.0e+00 +2.0e+00 2.2e+01 4.7e+00 0.0e+00 1.0e+00
        5  1    1    2  503    0 +3.0e+00 +3.0e+00 +3.0e+00 3.3e+01 4.7e+00 0.0e+00 1.0e+00
</event>
</LesHouchesEvents>
"""


def test_read_lhe():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.lhe.gz")
        with gzip.open(p, "wt") as f:
            f.write(_LHE)
        mom, a, q = mg.read_lhe_pp(p, [21, 5, 5, 5, -5])
    assert mom.shape == (2, 5, 4)
    assert np.allclose(a, [0.13, 0.12]) and np.allclose(q, [90.0, 80.0])
    # event 1: resonance (status 2) dropped, b's in LHE order, b~ last
    assert np.allclose(mom[0, :, 0], [50, 60, 10, 30, 20])
    # event 2: incoming listed b first -> reordered to g, b
    assert np.allclose(mom[1, :, 0], [50, 60, 22, 33, 11])
    print("[lhe] ok")


def _analytic_gg_gg(mom, alphas):
    """MadGraph's gg -> gg |M|^2: spin/colour-averaged (9/2) g^4 (3 - tu/s^2 - su/t^2 - st/u^2), with
    the 1/2 of the identical final gluons that MadGraph folds in."""
    p1, p2, p3 = mom[:, 0], mom[:, 1], mom[:, 2]
    dot = lambda a, b: a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)
    s, t = 2 * dot(p1, p2), -2 * dot(p1, p3)
    u = -s - t
    g4 = (4 * np.pi * alphas) ** 2
    return 0.5 * 4.5 * g4 * (3 - t * u / s ** 2 - s * u / t ** 2 - s * t / u ** 2)


def test_end_to_end():
    if os.environ.get("FA_TEST_MG5") != "1" or not os.path.exists(mg.MG5_BIN):
        print("[e2e] skipped (FA_TEST_MG5=1 and MadGraph needed)")
        return
    import copy
    with tempfile.TemporaryDirectory() as d:
        mg.WORK_DIR = os.path.join(d, "mg5work")
        mg.OUTPUT_DIR = d                  # write_recipe upserts its manifest here
        os.makedirs(mg.WORK_DIR)
        for base in ("dc_gg_gg", "dc_gb_bbbx"):
            name = "zz_e2e_" + base[3:]
            cfg = copy.deepcopy(mg.PROCESSES[base])
            cfg["lhe"].update(n_val=200, n_test=200, n_train=600)
            mg.PROCESSES[name] = cfg
            items = []
            for role, n in (("val", 200), ("test", 200), ("train", 600)):
                rec = mg.variable_energy_recipe(name, 13000, 13000, n, role=role, seed=42)
                items.append((rec, mg.recipe_output_path(rec, os.path.join(d, "pools"))))
            res = mg.build_lhe_pp(name, items, work_root=os.path.join(d, "lhe"))
            assert not os.path.exists(os.path.join(d, "lhe", f"{name}_madevent")), "MadEvent dir kept"
            data = {rec["role"]: np.load(p) for (rec, p) in items}
            npart = cfg["nfinal"] + 2
            assert [len(data[r]) for r in ("val", "test", "train")] == [200, 200, 600]
            allm = np.concatenate([data[r][:, :npart * 4] for r in data])
            assert len(np.unique(allm, axis=0)) == 1000, "roles overlap"
            for r, x in data.items():
                assert np.all(x[:, npart * 4:npart * 5] == np.asarray(cfg["pdg_ids"])), r
                assert np.all(np.isfinite(x[:, -1])) and np.all(x[:, -1] > 0), r
            full = res[0][1]
            print(f"[e2e] {base}: xsec {full['lhe_run']['xsec_pb']} pb, alpha_s {full['lhe_run']['alphas_range']}, "
                  f"|M|^2 {full['lhe_run']['amp_range']}, seconds {full['lhe_run']['seconds']}")
            if base == "dc_gg_gg":
                x = data["train"]
                mom = x[:, :16].reshape(-1, 4, 4)
                # the ME at a fixed alpha_s against the analytic formula: momenta rows, slots, card
                fixed = mg.label_lhe_pp(name, mom, np.full(len(mom), 0.118))
                dev = np.max(np.abs(fixed / _analytic_gg_gg(mom, 0.118) - 1))
                assert dev < 1e-6, dev
                # the stored label is at the event's own alpha_s (AQCDUP), inside the run's range
                a = np.sqrt(x[:, -1] / _analytic_gg_gg(mom, 1.0))
                lo, hi = res[2][1]["lhe_run"]["alphas_range"]           # the train pool's record
                assert abs(a.min() / lo - 1) < 1e-6 and abs(a.max() / hi - 1) < 1e-6 and a.std() > 0, (a.min(), a.max(), lo, hi)
                print(f"[e2e] gg->gg: max rel dev from analytic {dev:.1e}; implied alpha_s in [{a.min():.4f}, {a.max():.4f}]")
            del mg.PROCESSES[name]
    print("[e2e] ok")


if __name__ == "__main__":
    test_catalog_counts()
    test_registry()
    test_recipe_identity()
    test_existing_ids_unchanged()
    test_read_lhe()
    test_end_to_end()
