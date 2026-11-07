"""cache deletion guard: refuses deleting an amp_mmap_cache key an active FA job names (2026-11-07: 0c897ca2... under
FA-ccin2p3-a77e20b8), allows a key no job uses; data_mmap.prune keeps a key a live job names."""
import os, sqlite3, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import cache_guard as cg  # noqa: E402
import cmdparse  # noqa: E402
import submit_guard as g  # noqa: E402

BASE = "/sps/lpnhe/jiturrizaramirez01/tmp/amp_mmap_cache/"
USED, FREE = "0c897ca2d81c98d3828e2808", "27a1422795e9872faa6b9eaf"
# the two log lines of FA-ccin2p3-a77e20b8 (step 64 scaling grid s42: w32 x16 hp1262) that name its cache
LOG = ("[2026-11-07 00:06:03 I] data.memmap: momentum scale over the train selection at seed 42 in 128 s (" + BASE + USED +
       "/trainall_val200_test200/mom_stats/1a9e66a289ea11ceb5b081f1.json)\n"
       "[2026-11-07 00:07:37 I] data.memmap: 110361200 events of " + BASE + USED + "/trainall_val200_test200 (18.03 GB)\n")


def db(tmp_path, state="RUNNING"):
    p = str(tmp_path / "runs.db")
    c = sqlite3.connect(p)
    c.execute("create table runs (id, project, site, state, note, launch, expect)")
    c.execute("insert into runs values ('FA-ccin2p3-a77e20b8','FA','ccin2p3',?,'step 64 scaling grid s42',null,null)", (state,))
    c.execute("insert into runs values ('FA-ccin2p3-old','FA','ccin2p3','COMPLETED','x',null,null)")
    c.commit()
    return p


def users(tmp_path, keys, state="RUNNING"):
    return cg.users("ccin2p3", keys, fetch=lambda run: LOG if run.endswith("a77e20b8") else "", db=db(tmp_path, state))


def guard(tmp_path, key, state="RUNNING"):
    cmd = "site run ccin2p3 FA --item x -- rm -rf %s%s" % (BASE, key)
    return g.cache_delete(cmdparse.commands(cmd), users=lambda site, keys: users(tmp_path, keys, state))


def test_deleting_the_key_the_running_grid_uses_is_refused(tmp_path):
    out = guard(tmp_path, USED)
    assert len(out) == 1 and "FA-ccin2p3-a77e20b8" in out[0] and USED in out[0], out


def test_key_no_job_uses_is_allowed(tmp_path):
    assert guard(tmp_path, FREE) == []


def test_finished_job_does_not_block(tmp_path):
    assert guard(tmp_path, USED, "COMPLETED") == []


def test_unreadable_log_blocks_and_root_rm_counts_every_job(tmp_path):
    u = cg.users("ccin2p3", [USED], fetch=lambda r: (_ for _ in ()).throw(RuntimeError("down")), db=db(tmp_path))
    assert u and "unreadable" in u[0][3]
    cmd = "site run ccin2p3 FA -- rm -rf $SCRATCH/amp_mmap_cache"
    assert g.cache_delete(cmdparse.commands(cmd), users=lambda s, k: [("r", "RUNNING", k[0], "log")] if k == ["amp_mmap_cache"] else [])


def test_other_commands_not_checked(tmp_path):
    assert g.cache_delete(cmdparse.commands("site run ccin2p3 FA -- ls %s%s" % (BASE, USED)), users=lambda s, k: 1 / 0) == []


def test_prune_keeps_a_key_a_live_job_names(tmp_path, monkeypatch):
    sys.path.insert(0, ROOT)
    pytest = __import__("pytest")
    dm = pytest.importorskip("data_mmap")
    for k in (USED, FREE):
        d = tmp_path / k / "depth"
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        os.utime(d / "last_used", None) if (d / "last_used").exists() else None
        os.utime(d / "meta.json", (1, 1))
    monkeypatch.setattr(cg, "site_users", lambda key: ["62996676"] if key == USED else [])
    removed = dm.prune(str(tmp_path), need_bytes=10 ** 30, keep="/nonexistent", min_idle_s=0)
    assert removed == [str(tmp_path / FREE / "depth")] and (tmp_path / USED / "depth").exists()
