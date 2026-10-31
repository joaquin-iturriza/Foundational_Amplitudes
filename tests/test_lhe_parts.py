#!/usr/bin/env python
"""One-run part jobs of lhe_pp pools (mg5_pipeline_final "One-run part jobs", tools/lhe_parts.py): lhe.run_events in
the runs and the identity, the parts assembled give the serial build's pools byte for byte, and the file queue run
end to end (plan, two concurrent workers, assembly once, rerun is a no-op). CPU only: MadEvent, the LHE reader and
the labels are replaced by deterministic fakes keyed by each run's iseed and event count.

Run:  python tests/test_lhe_parts.py
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
import lhe_parts  # noqa: E402

NAME = "zz_parts_proc"
K = 5   # 2 -> 3


def _register(n_runs=3, run_events=400, parts=True):
    lhe = {"cards_dir": "recipes/catalog_v4_pp", "run_card": "run_card.dat", "param_card": "param_card.dat",
           "label_alphas": "running_sqrt_shat", "n_val": 100, "n_test": 100, "n_train": 800, "n_runs": n_runs}
    if parts:
        lhe["parts"] = True
    if run_events:
        lhe["run_events"] = run_events
    mg.PROCESSES[NAME] = {"kind": "lhe_pp", "pdg_ids": [21] * K, "m_finals": [0.0] * 3, "nfinal": 3,
                          "mg5_generate": ["generate g g > g g g"], "lhe": lhe}


def _fakes(tmp, short=True):
    """Replace MadEvent, the LHE reader, the labels and the physics check by deterministic fakes. short: each run
    returns a few events fewer than asked (the serial build's margin); the queue's workers fail such a part."""
    def prepare(process, work_root, log_path, madevent=True, standalone=True):
        ev = os.path.join(work_root, f"{process}_madevent") if madevent else None
        if ev:
            os.makedirs(ev, exist_ok=True)
        return ev, os.path.join(tmp, "sa")

    def run(process, ev, nevents, iseed, log_path, mg_cores=1, run_name="run_01", patches=None):
        p = os.path.join(ev, f"{run_name}_{iseed}_{nevents}.fake")
        open(p, "w").close()
        return p, None

    def read(path, pdg_ids):
        _, iseed, n = os.path.basename(path)[:-5].rsplit("_", 2)
        rng = np.random.default_rng(int(iseed))
        n = int(n) - (int(iseed) % 7 if short else 0)    # a run returning a few events fewer than asked
        sq = rng.uniform(100, 1000, n)
        P = np.concatenate([mg._beams(sq, np.array([21] * K), {}), mg._rambo_massless_batch(sq, K - 2, rng)], axis=1)
        return P, rng.uniform(0.1, 0.12, n), sq
    mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp = prepare, run, read
    mg.label_lhe_pp = lambda process, m, a, sa=None: 1.0 + np.abs(m[:, 2, 3]) * a
    mg.lhe_parent_physics_check = lambda process: None


def _items(seed=42):
    out = []
    for role, n in (("val", 100), ("test", 100), ("train", 800)):
        rec = mg.variable_energy_recipe(NAME, 13000, 13000, n, role=role, seed=seed)
        out.append((rec, mg.recipe_output_path(rec, datagen.dest_for_role(role))))
    return out


def test_runs_and_identity():
    _register(n_runs=3, run_events=400)
    runs = mg.lhe_runs(NAME, 100, 1020)
    assert [r[1] for r in runs] == [400] * 3 and [r[0] for r in runs] == [(100 + 7919 * j - 1) % 30000 + 1
                                                                          for j in range(3)]
    rec = mg.variable_energy_recipe(NAME, 13000, 13000, 800, role="train", seed=42)
    assert rec["lhe"]["run_events"] == 400 and rec["lhe"]["n_runs"] == 3
    assert rec["lhe"]["build"] == "parts"
    _register(n_runs=3, run_events=None, parts=False)   # without run_events: every run asks n_req, as before
    assert [r[1] for r in mg.lhe_runs(NAME, 100, 1020)] == [1020] * 3
    r0 = mg.variable_energy_recipe(NAME, 13000, 13000, 800, role="train", seed=42)
    assert "run_events" not in r0["lhe"] and "build" not in r0["lhe"]
    print("[runs+identity] ok")


def test_parts_equal_serial():
    _register()
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role)
    try:
        with tempfile.TemporaryDirectory() as d:
            _fakes(d)
            res = {}
            for mode in ("serial", "parts"):
                _register(parts=mode == "parts")
                datagen.dest_for_role = lambda role, m=mode: os.path.join(d, m, role)
                os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
                items = _items()
                if mode == "serial":
                    try:
                        _register(parts=True)
                        mg.build_lhe_pp(NAME, items, work_root=os.path.join(d, "w0"), write_sidecars=False)
                        raise AssertionError("a parts entry built in series")
                    except RuntimeError:
                        _register(parts=False)
                    out = mg.build_lhe_pp(NAME, items, work_root=os.path.join(d, "w1"), write_sidecars=False)
                else:
                    paths = mg.lhe_part_paths(NAME, items)
                    assert len(paths) == 3 and len({p for p, _ in paths}) == 3
                    for k in (2, 0, 1):                           # any order of the part jobs
                        mg.lhe_build_part(NAME, items, k, work_root=os.path.join(d, "w2"))
                    try:
                        mg.lhe_assemble(NAME, [(r, p) for r, p in items], write_sidecars=False)
                    except RuntimeError:
                        raise AssertionError("assembly with every part present failed")
                    out = mg.lhe_assemble(NAME, items, write_sidecars=False, work_root=os.path.join(d, "w2"))
                res[mode] = [(np.load(p), mg.recipe_id(full)) for p, full in out]
            # the same events in code (the fakes are directory-independent); distinct identities, since real MadEvent
            # in a fresh directory per run is not shown to match the serial build's reused one
            for (a, ia), (b, ib) in zip(res["serial"], res["parts"]):
                assert ia != ib and np.array_equal(a, b)
            # a part made under another card is not taken for this one
            p0 = mg.lhe_part_paths(NAME, _items())[0][0]
            mg.PROCESSES[NAME]["lhe"]["run_card"] = "../catalog_v4_ee/run_card.dat"
            assert mg.lhe_part_paths(NAME, _items())[0][0] != p0
            mg.PROCESSES[NAME]["lhe"]["run_card"] = "run_card.dat"
            # a missing part refuses the assembly
            datagen.dest_for_role = lambda role: os.path.join(d, "x", role)
            os.remove(mg.lhe_part_paths(NAME, _items())[1][0])
            try:
                mg.lhe_assemble(NAME, _items(), write_sidecars=False)
                raise AssertionError("assembly with a part missing accepted")
            except RuntimeError:
                pass
    finally:
        (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
         datagen.dest_for_role) = saved
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        del mg.PROCESSES[NAME]
    print("[parts == serial] ok")


def _worker(q, out):
    try:
        lhe_parts.cmd_work(argparse.Namespace(queue=q, workers=2, hours=1.0))
    except SystemExit as e:
        out.put(e.code)


def test_queue_end_to_end():
    _register()
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role, lhe_parts.ASSEMBLE_S)
    try:
        with tempfile.TemporaryDirectory() as d:
            _fakes(d, short=False)
            datagen.dest_for_role = lambda role: os.path.join(d, "pools", role)
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            os.environ["AMP_LHE_WORK_DIR"] = os.path.join(d, "work")
            lhe_parts.ASSEMBLE_S = 0
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            q = os.path.join(d, "queue")
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            plan = lambda: lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q, seed=42, rates=rates,
                                                                 cores=[f"{NAME}=1"]))
            plan()
            assert len(os.listdir(os.path.join(q, "todo"))) == 3
            plan()                                                      # rerun: nothing queued twice
            assert len(os.listdir(os.path.join(q, "todo"))) == 3
            ctx = mp.get_context("fork")
            res = ctx.Queue()
            ps = [ctx.Process(target=_worker, args=(q, res)) for _ in range(2)]
            for p in ps:
                p.start()
            for p in ps:
                p.join(120)
            codes = [res.get(timeout=5) for _ in ps]
            assert codes == [0, 0], codes
            assert len(os.listdir(os.path.join(q, "done"))) == 3 and not os.listdir(os.path.join(q, "todo"))
            assert os.listdir(os.path.join(q, "assembled")) == [NAME] and not os.listdir(os.path.join(q, "locks"))
            for rec, path in _items():
                assert datagen._is_cached(path, mg.recipe_id(rec)), path
            plan()                                                      # built: nothing left to queue
            assert not os.listdir(os.path.join(q, "todo"))
    finally:
        (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
         datagen.dest_for_role, lhe_parts.ASSEMBLE_S) = saved
        for v in ("AMP_LHE_PARTS_DIR", "AMP_LHE_WORK_DIR"):
            os.environ.pop(v, None)
        del mg.PROCESSES[NAME]
    print("[queue end to end] ok")


def test_tidy_and_gc():
    """A worker deletes a saved run's directory (its removals in its gc/ manifest first, nothing deleted when that
    write fails) and reduces a failed or short one to its logs, the short one not done; gc does the same for runs left
    behind (killed, failed before the change), skips runs a live job holds, and one claimed while it runs, writes its
    manifest first, and never deletes outside $SCRATCH/lhe_work/parts."""
    _register(n_runs=4)
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role, lhe_parts.ASSEMBLE_S, lhe_parts._my_jobs)
    env0 = {v: os.environ.get(v) for v in ("SCRATCH", "AMP_LHE_PARTS_DIR", "AMP_LHE_WORK_DIR")}
    try:
        with tempfile.TemporaryDirectory() as d:
            _fakes(d, short=False)
            prep, run0 = mg.lhe_prepare, mg.lhe_run_madevent

            def prepare(process, work_root, log_path, madevent=True, standalone=True):
                ev, sa = prep(process, work_root, log_path, madevent, standalone)
                os.makedirs(os.path.join(ev, "SubProcesses", "P1"), exist_ok=True)
                open(os.path.join(ev, "SubProcesses", "P1", "madevent"), "wb").write(b"x" * 10000)
                open(os.path.join(ev, "ME5_debug"), "w").write("debug")
                os.makedirs(os.path.join(ev, "Events", "run"), exist_ok=True)
                open(os.path.join(ev, "Events", "run", "run_tag_1_debug.log"), "w").write("log")
                return ev, sa

            seeds = [r[0] for r in mg.lhe_runs(NAME, mg._lhe_check_recipes(NAME, _items())[1], 1)]
            fail_seed, short_seed = seeds[1], seeds[3]

            def run(process, ev, nevents, iseed, log_path, mg_cores=1, run_name="run_01", patches=None):
                if iseed == fail_seed:
                    raise RuntimeError("No space left on device (fake)")
                if iseed == short_seed:                                  # MadEvent stopping at half the events
                    return run0(process, ev, nevents // 2, iseed, log_path, mg_cores, run_name, patches)
                return run0(process, ev, nevents, iseed, log_path, mg_cores, run_name, patches)
            mg.lhe_prepare, mg.lhe_run_madevent = prepare, run
            datagen.dest_for_role = lambda role: os.path.join(d, "pools", role)
            os.environ["SCRATCH"] = d
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "lhe_parts")
            os.environ["AMP_LHE_WORK_DIR"] = os.path.join(d, "lhe_work")
            lhe_parts.ASSEMBLE_S = 0
            jobs = {"777"}
            lhe_parts._my_jobs = lambda: set(jobs)
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            q = os.path.join(d, "queue")
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q, seed=42, rates=rates, cores=[]))
            try:
                lhe_parts.cmd_work(argparse.Namespace(queue=q, workers=1, hours=1.0))
            except SystemExit as e:
                assert e.code == 1                                       # the failed and the short run
            root = os.path.join(d, "lhe_work", "parts")
            wds = {k: lhe_parts.run_workdir(NAME, r) for k, (_, r) in enumerate(mg.lhe_part_paths(NAME, _items()))}
            assert not os.path.exists(wds[0]) and not os.path.exists(wds[2]), "a saved run's directory survived"
            for k in (1, 3):                                             # failed, short: their logs only
                left = sorted(os.path.relpath(os.path.join(dp, f), wds[k]) for dp, _, fs in os.walk(wds[k])
                              for f in fs)
                assert left == sorted([f"{NAME}.mg5.log", f"{NAME}_madevent/ME5_debug",
                                       f"{NAME}_madevent/Events/run/run_tag_1_debug.log"]), (k, left)
            # the short part: its file there with half the events, its task failed, not done
            p3, r3 = mg.lhe_part_paths(NAME, _items())[3]
            assert lhe_parts.part_ok(p3) and not lhe_parts.part_ok(p3, r3[1])
            ks = lambda st: sorted(json.load(open(os.path.join(q, st, f)))["k"]
                                   for f in os.listdir(os.path.join(q, st)) if f.endswith(".json"))
            assert ks("done") == [0, 2] and ks("failed") == [1, 3], (ks("done"), ks("failed"))
            assert "fewer events" in open(glob.glob(os.path.join(q, "failed", "*__003.json.err"))[0]).read()
            # the worker's manifest: every file it removed, with its size, and none of the kept logs
            wm = [json.loads(l) for m in glob.glob(os.path.join(q, "gc", "worker_*.jsonl")) for l in open(m)]
            by_run = {}
            for e in wm:
                by_run.setdefault(e["run"], []).append(e)
            assert {e["saved"] for e in by_run[f"{NAME}__000"]} == {True}
            assert any(e["path"].endswith("SubProcesses/P1/madevent") and e["bytes"] == 10000
                       for e in by_run[f"{NAME}__000"])
            assert {e["saved"] for e in by_run[f"{NAME}__003"]} == {False}
            assert not any(e["path"].endswith((".log", "ME5_debug")) for e in by_run[f"{NAME}__003"])
            # a manifest that cannot be written: nothing deleted
            os.makedirs(os.path.join(wds[0], "x"))
            open(os.path.join(wds[0], "x", "f"), "w").write("z")
            q_bad = os.path.join(d, "q_bad")
            os.makedirs(q_bad)
            open(os.path.join(q_bad, "gc"), "w").close()                  # gc/ is a file: the append fails
            assert lhe_parts.tidy_run(NAME, _items(), 0, q_bad) is None
            assert os.path.exists(os.path.join(wds[0], "x", "f"))
            # runs left behind as before the change: run 0 saved with its directory, run 2 killed (no part) and
            # claimed by a dead job, run 1 claimed by a live job
            for k in (0, 1, 2, 3):
                os.makedirs(os.path.join(wds[k], "junk"), exist_ok=True)
                open(os.path.join(wds[k], "junk", "big"), "wb").write(b"y" * 5000)
                open(os.path.join(wds[k], "a.log"), "w").write("l")
            os.remove(mg.lhe_part_paths(NAME, _items())[2][0])
            tasks = {json.load(open(os.path.join(q, "done", f)))["k"]: f for f in os.listdir(os.path.join(q, "done"))}
            os.rename(os.path.join(q, "done", tasks[2]), os.path.join(q, "claimed", tasks[2] + ".666"))
            failed = {json.load(open(os.path.join(q, "failed", f)))["k"]: f
                      for f in os.listdir(os.path.join(q, "failed")) if f.endswith(".json")}
            os.rename(os.path.join(q, "failed", failed[1]), os.path.join(q, "claimed", failed[1] + ".777"))
            outside = os.path.join(d, "keep_me")
            os.makedirs(outside)
            os.symlink(outside, os.path.join(root, f"{NAME}__run_09"))     # never followed (not a run of the spec)
            lhe_parts.cmd_gc(argparse.Namespace(queue=q, dry_run=True))
            assert os.path.exists(os.path.join(wds[0], "junk", "big"))      # a dry run deletes nothing
            # run 3 is claimed by a job started after gc read squeue, between gc's plan and its delete
            plan0 = lhe_parts.tidy_plan

            def plan_then_claim(wd, saved):
                if wd == wds[3]:
                    jobs.add("888")
                    os.rename(os.path.join(q, "failed", failed[3]), os.path.join(q, "claimed", failed[3] + ".888"))
                return plan0(wd, saved)
            lhe_parts.tidy_plan = plan_then_claim
            try:
                lhe_parts.cmd_gc(argparse.Namespace(queue=q, dry_run=False))
            finally:
                lhe_parts.tidy_plan = plan0
            mans = sorted(glob.glob(os.path.join(q, "gc", "manifest_*.json")))
            man = json.load(open([m for m in mans if not m.endswith(("_dry.json", "_skipped.json"))][0]))
            got = {e["path"]: e for e in man["entries"]}
            assert set(got) == {wds[0], wds[2], wds[3]}, got               # run 1 is live: untouched
            assert got[wds[0]]["removed"] == "all" and got[wds[0]]["bytes"] >= 5000
            assert got[wds[2]]["removed"] != "all" and got[wds[2]]["kept"] == [os.path.join(wds[2], "a.log")]
            assert not os.path.exists(wds[0])
            assert os.listdir(wds[2]) == ["a.log"]
            assert os.path.exists(os.path.join(wds[1], "junk", "big"))
            assert os.path.exists(os.path.join(wds[3], "junk", "big")), "a run claimed mid-gc was tidied"
            sk = [m for m in glob.glob(os.path.join(q, "gc", "manifest_*_skipped.json"))]
            assert json.load(open(sk[0]))["skipped_claimed_at_delete"] == [wds[3]]
            assert os.path.isdir(outside)
            # the guard: a work tree outside $SCRATCH/lhe_work/parts is refused, and so is a path out of the tree
            os.environ["AMP_LHE_WORK_DIR"] = os.path.join(d, "elsewhere")
            for f in (lambda: lhe_parts.parts_work_root(),
                      lambda: lhe_parts._under(os.path.realpath(root), outside),
                      lambda: lhe_parts._under(os.path.realpath(root), os.path.join(root, f"{NAME}__run_09"))):
                try:
                    f()
                    raise AssertionError("a delete outside the work tree allowed")
                except RuntimeError:
                    pass
    finally:
        (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
         datagen.dest_for_role, lhe_parts.ASSEMBLE_S, lhe_parts._my_jobs) = saved
        for k, v in env0.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        del mg.PROCESSES[NAME]
    print("[tidy + gc] ok")


def test_stall_layout_short_part_done():
    """The stall layout (run_events None: every run asks the whole pool and returns fewer by design): a short part is
    done, not failed (run_asked None); a pool with run_events still asks the events of its run."""
    try:
        _register(n_runs=2, run_events=None)
        items = _items()
        for part, run in mg.lhe_part_paths(NAME, items):
            assert lhe_parts.run_asked(NAME, run) is None
            os.makedirs(os.path.dirname(part), exist_ok=True)
            with tempfile.TemporaryDirectory() as d:
                f = os.path.join(d, "p.npz")
                np.savez(f, mom=np.zeros((run[1] // 10, 2)))        # a tenth of the events asked
                assert lhe_parts.part_ok(f, lhe_parts.run_asked(NAME, run))
                assert not lhe_parts.part_ok(f, run[1])
        _register(n_runs=2, run_events=400)
        run = mg.lhe_part_paths(NAME, items)[0][1]
        assert lhe_parts.run_asked(NAME, run) == 400
    finally:
        mg.PROCESSES.pop(NAME, None)
    print("[stall layout short part] ok")


if __name__ == "__main__":
    test_runs_and_identity()
    test_parts_equal_serial()
    test_queue_end_to_end()
    test_tidy_and_gc()
    test_stall_layout_short_part_done()
