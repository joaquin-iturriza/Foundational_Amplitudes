#!/usr/bin/env python
"""Gridpack runs of lhe_pp pools (mg5_pipeline_final lhe_make_gridpack / lhe_run_gridpack, tools/lhe_probe.py
gridpack; catalog_coverage): the gridpack integration gets gridpack = True and the grid's nevents and iseed in its run
card, the packed grids are found, runs drawn from them give their events and the LHE header as the banner, and the
banner's physics key ignores the gridpack switch (analysis/catalog_v4/short_run_bias.py). CPU only: MadEvent's
generate_events and the gridpack's run.sh are replaced by shell fakes.

Run:  python tests/test_lhe_gridpack.py
"""
import gzip
import os
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v4"))
import mg5_pipeline_final as mg  # noqa: E402
import short_run_bias as srb  # noqa: E402

NAME = "zz_gpack_proc"
IDS = [2, 2, 2, 2, 21]

_RUN_SH = r"""#!/bin/bash
# fake gridpack run.sh: N events, seed S; header copies the packed run card
n=$1; s=$2
{ echo '<LesHouchesEvents version="3.0">'; echo '<header>'; echo '<MGRunCard>'; cat run_card.dat; echo '</MGRunCard>';
  echo '</header>'; echo '<init>'; echo '</init>'
  for i in $(seq 1 $n); do
    echo '<event>'; echo " 5 1 1.0 91.0 0.0078 0.118"
    echo " 2 -1 0 0 501 0 0 0 100 100 0 0 9"; echo " 2 -1 0 0 502 0 0 0 -100 100 0 0 9"
    echo " 2 1 1 2 501 0 30 0 10 31.6 0 0 9"; echo " 2 1 1 2 502 0 -20 0 -5 20.6 0 0 9"
    echo " 21 1 1 2 0 0 -10 0 $s 147.8 0 0 9"; echo '</event>'
  done; echo '</LesHouchesEvents>'; } | gzip > events.lhe.gz
"""

_GEN = r"""#!/bin/bash
# fake generate_events: pack the run card and a run.sh into <run_name>_gridpack.tar.gz
cd "$(dirname "$0")/.."
d=$(mktemp -d); cp Cards/run_card.dat "$d/run_card.dat"; cp "$FAKE_RUN_SH" "$d/run.sh"
tar -czf "${1}_gridpack.tar.gz" -C "$d" run_card.dat run.sh
"""


def _register(tmp):
    cards = os.path.join(ROOT, "recipes", "catalog_v4_pp")
    mg.PROCESSES[NAME] = {"pdg_ids": IDS, "m_finals": [0, 0, 0], "nfinal": 3, "mg5_generate": ["generate u u > u u g"],
                          "lhe": {"cards_dir": os.path.relpath(cards, ROOT), "run_card": "run_card.dat",
                                  "param_card": "param_card.dat", "n_val": 10, "n_test": 10, "n_train": 80}}


def _fake_ev(tmp):
    ev = os.path.join(tmp, "ev")
    os.makedirs(os.path.join(ev, "Cards")); os.makedirs(os.path.join(ev, "bin"))
    open(os.path.join(ev, "Cards", "me5_configuration.txt"), "w").write("# run_mode = 2\n")
    rs = os.path.join(tmp, "run.sh")
    open(rs, "w").write(_RUN_SH)
    g = os.path.join(ev, "bin", "generate_events")
    open(g, "w").write(_GEN)
    os.chmod(g, 0o755)
    os.environ["FAKE_RUN_SH"] = rs
    return ev


def test_gridpack_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        _register(tmp)
        ev = _fake_ev(tmp)
        log = os.path.join(tmp, "log")
        tar = mg.lhe_make_gridpack(NAME, ev, 40000, 1234, log, patches={"ebeam1": 500.0})
        assert tar.endswith("grid_gridpack.tar.gz") and os.path.exists(tar), tar
        with tarfile.open(tar) as t:
            card = t.extractfile("run_card.dat").read().decode()
        vals = {l.split("=")[1].split("!")[0].strip(): l.split("=")[0].strip() for l in card.splitlines()
                if "=" in l and not l.lstrip().startswith("#")}
        assert vals["gridpack"] == "True" and vals["nevents"] == "40000" and vals["iseed"] == "1234", vals
        lhe = mg.lhe_run_gridpack(tar, os.path.join(tmp, "r7"), 25, 7, log)
        m, aq, _ = mg.read_lhe_pp(lhe, IDS)
        assert m.shape == (25, 5, 4) and float(m[0, 4, 3]) == 7.0, m.shape
        head = mg.lhe_gridpack_header(lhe)
        assert "<MGRunCard>" in head and "<event" not in head
        # two runs of one gridpack with different seeds are separate draws
        lhe2 = mg.lhe_run_gridpack(tar, os.path.join(tmp, "r8"), 3, 8, log)
        assert mg.read_lhe_pp(lhe2, IDS)[0][0, 4, 3] == 8.0


def test_physics_key_ignores_gridpack():
    proc = "<MGProcCard>\nimport model sm\ngenerate u u > u u g\noutput x\n</MGProcCard>"
    slha = "<slha>\nBLOCK MASS\n 6 1.725e+02\n</slha>"
    card = lambda gp, n, s: (f"<MGRunCard>\n {n} = nevents\n {s} = iseed\n {gp} = gridpack\n 6500.0 = ebeam1\n"
                             "</MGRunCard>")
    k1 = srb.physics_key(proc + card("False", 40282, 11) + slha)
    k2 = srb.physics_key(proc + card("True", 1000, 3) + slha)
    k3 = srb.physics_key(proc + card("True", 1000, 3).replace("6500.0", "7000.0") + slha)
    assert k1 and k1 == k2 and k3 != k1


def test_gpack_file_name():
    m = srb.GPACK_RE.match("dsh_dg_dguux_gpack1234_seed17_ev1000.npz")
    assert m and m["proc"] == "dsh_dg_dguux" and m["grid"] == "1234" and m["seed"] == "17" and m["ev"] == "1000"
    assert not srb.PROBE_RE.match("dsh_dg_dguux_gpack1234_seed17_ev1000.npz")


if __name__ == "__main__":
    for f in (test_gridpack_roundtrip, test_physics_key_ignores_gridpack, test_gpack_file_name):
        f()
        print("ok", f.__name__)
