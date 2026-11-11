"""The e+e- pools at the collider anchor energies (plan catalog_coverage step 74): every run of an eea_/dsh_eea_ pool
carries its energy's beam spectrum (lpp by the process string's beam order, pdlabel of EE_ANCHOR), the same values the
step-71 probe ran with; the pools built before keep their runs and identity; a part with a spectrum passes the energy
check. CPU only.

    .venv-cpu/bin/python -m pytest tests/test_ee_anchor.py -q
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recipes"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mg5_pipeline_final as mg  # noqa: E402
import gen_catalog_v4_default as g  # noqa: E402
import lhe_parts  # noqa: E402

R = lambda f: os.path.join(ROOT, "recipes", f)
EEA = yaml.safe_load(open(R("catalog_v4_eea_processes.yaml")))["processes"]
DSHA = yaml.safe_load(open(R("catalog_v4_eea_default_processes.yaml")))["processes"]


def _runs(name):
    return mg.lhe_runs(name, mg.lhe_iseed(name, g.SEED), mg.lhe_n_requested(mg.PROCESSES[name]))


def test_every_anchor_run_has_its_spectrum():
    assert len(EEA) == 170 and len(DSHA) == 115
    for name in list(EEA) + list(DSHA):
        cfg = mg.PROCESSES[name]
        proc = cfg["proc_str"]
        for r in _runs(name):
            E = 2.0 * r[2]["ebeam1"]
            assert r[2]["ebeam2"] == r[2]["ebeam1"]
            assert r[2]["pdlabel"] == g.EE_ANCHOR[E], (name, E)
            # the probe's values (tools/lhe_probe.py run --ee-isr) are the production run's
            assert r[2] == g.ee_beam_patches(E, proc), (name, E)
            # lpp by the beam order of the generate string: 'e+ e- > ...' -> positron beam 1
            b = proc.split(">")[0].split()
            assert r[2]["lpp1"] == (3 if b[0] == "e-" else -3) and r[2]["lpp2"] == (3 if b[1] == "e-" else -3)


def test_beam_order_from_process_string():
    assert mg.ee_beam_lpp("e+ e- > b b~ ve ve~") == {"lpp1": -3, "lpp2": 3}
    assert mg.ee_beam_lpp("e- e+ > mu+ mu-") == {"lpp1": 3, "lpp2": -3}
    for bad in ("u u~ > e+ e-", "e+ > a"):
        try:
            mg.ee_beam_lpp(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_energies_and_names():
    for name, e in EEA.items():
        par = e["parent"]
        Es = mg.PROCESSES[name]["lhe"]["energies"]
        assert set(Es) <= set(g.ANCHOR_ENERGIES) and Es == sorted(Es)
        assert sorted(mg.PROCESSES[name]["lhe"]["beam_spectrum"]) == sorted(f"{E:g}" for E in Es)
        assert not set(Es) & set(g.EEA_DROPPED.get(par, ()))
        assert e["standalone_of"] == "ee4_" + name[len("eea_"):]
    assert 3000.0 not in mg.PROCESSES["eea_epem_cgsxwm"]["lhe"]["energies"]          # D193
    for name, e in DSHA.items():
        assert e["reuse"] == "eea_" + name[len("dsh_eea_"):]
        assert mg.PROCESSES[name]["lhe"]["energies"] == mg.PROCESSES[e["reuse"]]["lhe"]["energies"]
        seeds = lambda x: {r[0] for r in _runs(x)}
        assert not seeds(name) & seeds(e["reuse"]), name


def test_old_pools_unchanged():
    """ee4_/dsh_ runs keep fixed beams and their identity carries no beam_spectrum (their recipe ids unchanged)."""
    old = [n for n in mg.PROCESSES if n.startswith("ee4_")][:20]
    for name in old:
        for r in _runs(name):
            assert set(r[2]) == {"ebeam1", "ebeam2"}, (name, r[2])
        lo, hi = mg.lhe_layout(mg.PROCESSES[name])["test"]
        rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role="test", seed=g.SEED)
        assert "beam_spectrum" not in rec["lhe"]
    name = next(iter(EEA))
    lo, hi = mg.lhe_layout(mg.PROCESSES[name])["test"]
    rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role="test", seed=g.SEED)
    assert rec["lhe"]["beam_spectrum"] == mg.PROCESSES[name]["lhe"]["beam_spectrum"]


def test_gridpacks_one_per_energy_with_spectrum():
    gp = [n for n in list(EEA) + list(DSHA) if mg.PROCESSES[n]["lhe"].get("gridpack")]
    assert gp
    for name in gp[:10]:
        runs = _runs(name)
        grids = mg.lhe_gridpacks(name, mg.lhe_iseed(name, g.SEED), mg.lhe_n_requested(mg.PROCESSES[name]))
        assert len(grids) == len(mg.PROCESSES[name]["lhe"]["energies"])
        assert all("pdlabel" in gr[2] for gr in grids)
        for r in runs:      # every run is drawn from the gridpack of its own energy and spectrum
            j = mg.lhe_gridpack_of_run(r, grids)
            assert grids[j][2] == r[2]


def test_part_energy_ok_with_spectrum():
    run_fixed = (1, 10, {"ebeam1": 120.0, "ebeam2": 120.0}, "run_01")
    run_isr = (1, 10, {"ebeam1": 120.0, "ebeam2": 120.0, "lpp1": -3, "lpp2": 3, "pdlabel": "fcce240ll"}, "run_01")

    def part(e_sum):
        m = np.zeros((len(e_sum), 4, 4))
        m[:, 0, 0] = np.asarray(e_sum) / 2
        m[:, 1, 0] = np.asarray(e_sum) / 2
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
        np.savez(f, mom=m)
        return f
    full = part([240.0] * 5)
    radiated = part([240.0, 239.9, 200.0, 150.0, 239.0])
    above = part([240.0, 241.0])
    low = part([200.0, 210.0])
    assert lhe_parts.part_energy_ok(full, run_fixed)
    assert lhe_parts.part_energy_ok(radiated, run_isr)
    assert not lhe_parts.part_energy_ok(above, run_isr)
    assert not lhe_parts.part_energy_ok(low, run_isr)
    assert not lhe_parts.part_energy_ok(part([120.0] * 3), run_fixed)


def test_dsh_eea_sized_from_stage1():
    """Stage 2 (dsh_eea_) is sized from stage 1's read (lhe_parts_eea_stage1.json): every dsh_eea_ pool measured there
    is in the plan's probe file at its eea_ pool's worst energy's kept x yield, its margin follows the measured kept per energy, and
    a complete-run pool whose eea_ counterpart ran the same layout takes that pool's complete share per energy."""
    import json, math, yaml
    import gen_catalog_v4_default as g
    s1 = json.load(open(g.EEA_STAGE1))
    probe = json.load(open(g.EEA_PROBE))
    dsha = yaml.safe_load(open(os.path.join(os.path.dirname(g.__file__), "catalog_v4_eea_default_processes.yaml")))["processes"]
    eea = yaml.safe_load(open(os.path.join(os.path.dirname(g.__file__), "catalog_v4_eea_processes.yaml")))["processes"]
    assert set(probe) == {k for k, e in dsha.items() if s1.get(e["reuse"], {}).get("kept_by_energy")}
    for k, y in probe.items():
        m = s1[dsha[k]["reuse"]]
        assert y == round(min(m["kept_by_energy"].values()) * m["yield"], 4)     # the worst energy's
    assert probe["dsh_eea_epem_accxh"] < 0.5      # 0.436 at 160 GeV: the plan's YIELD_FACTOR check lists it
    hit = 0
    for k, e in dsha.items():
        rr, l, m = eea[e["reuse"]]["lhe"], e["lhe"], s1.get(e["reuse"], {})
        if not (l.get("complete_runs") and rr.get("complete_runs") and rr.get("gridpack") and m.get("complete_by_energy")):
            continue
        need = sum(int(l[s]) for s in ("n_train", "n_val", "n_test"))
        per = math.ceil(need / len(l["energies"]))
        runs = l.get("energy_runs") or [l["n_runs"]] * len(l["energies"])
        for E, n in zip(l["energies"], runs):
            x = m["complete_by_energy"][f"{float(E):g}"]["complete"]
            assert n == math.ceil(per / (min(g.COMPLETE_SHARE, x / g.RUN_SAFETY) * int(l["run_events"])))
            hit += x < g.COMPLETE_SHARE * g.RUN_SAFETY
    assert hit >= 1      # some energy measured short (eea_epem_bbxwpwm 1500 GeV: 2 of 4 runs complete)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
