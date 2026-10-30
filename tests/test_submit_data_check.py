"""submit_guard's data check (E3072): the training recipe an arm picks, and the pass for a job chained on its prebuild.
The site read itself (tools/check_site_data.py over `site run`) is not run here."""
import os, sqlite3, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import cmdparse  # noqa: E402
import submit_guard as g  # noqa: E402

SCRIPT = open(os.path.join(ROOT, "scripts", "job_catalog_v3_short.sh")).read()


def test_arm_recipe():
    assert g.arm_recipe(SCRIPT, "v4_dimensionless_muon_default50") == "recipes/catalog_v4_default50_train.yaml"
    assert g.arm_recipe(SCRIPT, "v4_dimensionless_muon_default25_seed2") == "recipes/catalog_v4_default25_train.yaml"
    assert g.arm_recipe(SCRIPT, "v4_dimensionless_muon_pp_seed2") == "recipes/catalog_v4_pp_mixed_train.yaml"
    assert g.arm_recipe(SCRIPT, "v4_dimensionless_muon") == "recipes/catalog_v4_train.yaml"
    assert g.arm_recipe(SCRIPT, "dimensionless") == "recipes/catalog_v3_train.yaml"        # the default before the case
    assert g.arm_recipe("echo no recipe\n", "x") is None


def _db(rows):
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    c = sqlite3.connect(f)
    c.execute("CREATE TABLE runs (id TEXT, project TEXT, site TEXT, job TEXT, sched_id TEXT, state TEXT, note TEXT)")
    c.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()
    return f


def test_chained_on_prebuild_passes_without_read():
    g.RUNS_DB = _db([("FA-ccin2p3-a", "FA", "ccin2p3", "scripts/job_catalog_v4_regen.sh", "111", "PENDING",
                      "mixture prebuild catalog_v4_default50_train.yaml shard 0 of 8"),
                     ("FA-ccin2p3-b", "FA", "ccin2p3", "scripts/job_catalog_v3_short.sh", "222", "RUNNING", "smoke")])
    calls = []
    g._site_data_ok = lambda site, r: (calls.append(r), (False, "missing"))[1]
    sub = "site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --item x --hdr=--dependency=afterok:%s -- %s"
    assert g.data(cmdparse.commands(sub % ("111", "v4_dimensionless_muon_default50"))) == [] and not calls
    # chained on a job that is not its prebuild (the smoke, as on 2026-10-30): read on the site, refused
    assert g.data(cmdparse.commands(sub % ("222", "v4_dimensionless_muon_default50"))) and calls
    # the prebuild of the other recipe does not cover default25
    assert g.data(cmdparse.commands(sub % ("111", "v4_dimensionless_muon_default25")))
    # not FA, a dry --check: never read
    calls.clear()
    assert g.data(cmdparse.commands("site submit ccin2p3 XX scripts/job_catalog_v3_short.sh -- v4_dimensionless_muon")) == []
    assert g.data(cmdparse.commands("site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --check -- "
                                    "v4_dimensionless_muon")) == [] and not calls


def test_refused_command_keeps_its_pick_unit():
    """a submission refused by another check (here data) does not spend the unit of `site pick` it was allotted."""
    import json, time
    home = tempfile.mkdtemp()
    pl = os.path.join(home, ".local", "share", "ccorch", "placement")
    os.makedirs(pl)
    f = os.path.join(pl, "FA.json")
    json.dump({"ts": int(time.time()), "quota": {"ccin2p3": 1}}, open(f, "w"))
    old_home, old_test = os.environ.get("HOME"), os.environ.get("SUBMIT_TEST")
    os.environ["HOME"] = home
    os.environ.pop("SUBMIT_TEST", None)
    g.RUNS_DB = _db([])
    g.design = lambda *a: []
    g.budget = lambda *a: []
    try:
        cmd = "site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --item x -- v4_dimensionless_muon_default50"
        g._site_data_ok = lambda site, r: (False, "missing")
        assert any(p.startswith("data:") for p in g.problems({"tool_input": {"command": cmd}}))
        assert json.load(open(f))["quota"]["ccin2p3"] == 1
        g._site_data_ok = lambda site, r: (True, "ok")
        assert g.problems({"tool_input": {"command": cmd}}) == []
        assert json.load(open(f))["quota"]["ccin2p3"] == 0
    finally:
        os.environ["HOME"] = old_home or ""
        if old_test is not None:
            os.environ["SUBMIT_TEST"] = old_test


if __name__ == "__main__":
    test_arm_recipe(); test_chained_on_prebuild_passes_without_read(); test_refused_command_keeps_its_pick_unit()
    print("ok")
