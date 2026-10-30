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
import time

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


def _register(n_runs=3, run_events=400, parts=True, accept_short=False, complete_runs=False):
    lhe = {"cards_dir": "recipes/catalog_v4_pp", "run_card": "run_card.dat", "param_card": "param_card.dat",
           "label_alphas": "running_sqrt_shat", "n_val": 100, "n_test": 100, "n_train": 800, "n_runs": n_runs}
    if parts:
        lhe["parts"] = True
    if run_events:
        lhe["run_events"] = run_events
    if accept_short:
        lhe["accept_short"] = True
    if complete_runs:
        lhe["complete_runs"] = True
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


def test_queue_end_to_end(run_events=400, short=False):
    """run_events=None, short=True: the stall layout (every run asks the whole pool and comes back short); its short
    parts are done, not failed, and the pools assemble from their sum."""
    _register(run_events=run_events)
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role, lhe_parts.ASSEMBLE_S)
    try:
        with tempfile.TemporaryDirectory() as d:
            _fakes(d, short=short)
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
    print(f"[queue end to end, run_events={run_events}, short={short}] ok")


def test_queue_stall_layout_short_parts():
    test_queue_end_to_end(run_events=None, short=True)


def test_accept_short_and_adopt():
    """A pool whose runs come back short at random (lhe.accept_short): any part counts, and raising its n_runs keeps
    the parts already made (plan links them to the new identity's names, run k < old n_runs is the same run)."""
    saved = (datagen.dest_for_role,)
    try:
        with tempfile.TemporaryDirectory() as d:
            datagen.dest_for_role = lambda role: os.path.join(d, "pools", role)
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            q = os.path.join(d, "queue")
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            plan = lambda: lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q, seed=42, rates=rates,
                                                                 cores=[]))
            _register(n_runs=2, run_events=400, accept_short=True)
            run0 = mg.lhe_runs(NAME, 100, 1020)[0]
            assert lhe_parts.part_need(NAME, run0) is None
            plan()
            old = [p for p, _ in mg.lhe_part_paths(NAME, _items())]
            for p in old:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, mom=np.zeros((37, K, 4)))
            _register(n_runs=3, run_events=400, accept_short=True)
            new = mg.lhe_part_paths(NAME, _items())
            assert [os.path.basename(p) for p in old] != [os.path.basename(p) for p, _ in new[:2]]
            plan()
            assert all(os.path.samefile(a, b) for a, (b, _) in zip(old, new[:2])) and not os.path.exists(new[2][0])
            assert len(os.listdir(os.path.join(q, "todo"))) == 3      # the third run queued, the two first kept
            _register(n_runs=3, run_events=400)                       # without accept_short a short part fails
            assert lhe_parts.part_need(NAME, run0) == 400
    finally:
        (datagen.dest_for_role,) = saved
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        mg.PROCESSES.pop(NAME, None)
    print("[accept_short + adopt] ok")


def test_complete_runs():
    """lhe.complete_runs: in the identity; any part counts as made; the assembly uses only the runs that returned all
    they asked (a short run's events never reach the pool, recorded as runs_dropped_short), refuses before touching
    anything when the complete runs do not hold the blocks, and sets aside a pool of another identity at its path
    (moved, not deleted); parts of the same runs made without the flag are adopted."""
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role)
    try:
        with tempfile.TemporaryDirectory() as d:
            _fakes(d)
            datagen.dest_for_role = lambda role: os.path.join(d, "pools", role)
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            os.environ["AMP_LHE_WORK_DIR"] = os.path.join(d, "work")
            _register(n_runs=4, run_events=400)
            plain = _items()
            _register(n_runs=4, run_events=400, complete_runs=True)
            its = _items()
            assert its[0][0]["lhe"]["runs"] == "complete" and "runs" not in plain[0][0]["lhe"]
            assert mg.recipe_id(its[0][0]) != mg.recipe_id(plain[0][0])
            parts = mg.lhe_part_paths(NAME, its)
            assert lhe_parts.part_need(NAME, parts[0][1]) is None
            rng = np.random.default_rng(0)

            def write(got):
                for (p, run), n in zip(parts, got):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    sq = rng.uniform(100, 1000, n)
                    P = np.concatenate([mg._beams(sq, np.array([21] * K), {}),
                                        mg._rambo_massless_batch(sq, K - 2, rng)], axis=1)
                    if n < run[1]:
                        P[:, 2:, 0] = 7777.0                       # a short run's events, recognisable
                    np.savez(p, mom=P, aqcd=np.full(n, 0.11), seconds=np.array([1.0, 2.0]), mg_cores=np.array(1),
                             second_refine=np.array(np.nan), banner=np.array(""))
            write([400, 400, 100, 120])                            # 800 complete events, the blocks need 1000
            # a pool of the plain identity already at the train path (as assembled from short runs)
            tr_rec, tr_path = its[2]
            os.makedirs(os.path.dirname(tr_path), exist_ok=True)
            np.save(tr_path, np.zeros((3, 3)))
            json.dump({"recipe_id": mg.recipe_id(plain[2][0]), "lhe_run": {}}, open(tr_path + ".recipe.json", "w"))
            try:
                mg.lhe_assemble(NAME, its)
                raise AssertionError("assembled from too few complete runs")
            except RuntimeError as e:
                assert "complete runs hold 800" in str(e), e
            assert os.path.exists(tr_path) and not os.path.exists(os.path.join(os.path.dirname(tr_path), "set_aside"))
            write([400, 400, 100, 400])
            out = mg.lhe_assemble(NAME, its)
            aside = glob.glob(os.path.join(os.path.dirname(tr_path), "set_aside", "*", os.path.basename(tr_path)))
            assert len(aside) == 1 and np.load(aside[0]).shape == (3, 3)
            for (path, full), (rec, _) in zip(out, its):
                assert datagen._is_cached(path, mg.recipe_id(rec)), path
                x = np.load(path)
                assert not (x[:, 4 * 2] == 7777.0).any()               # no event of the short run
                assert full["lhe_run"]["runs_dropped_short"] == [{"run": parts[2][1][3], "iseed": int(parts[2][1][0]),
                                                                   "asked": 400, "got": 100}]
                assert full["lhe_run"]["events_per_run"] == [400, 400, 400]
            # the same runs planned before without the flag: adopted under the complete identity's names
            for p, _ in parts:
                os.remove(p)
            _register(n_runs=4, run_events=400)
            old = [p for p, _ in mg.lhe_part_paths(NAME, plain)]
            for p in old:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, mom=np.zeros((400, K, 4)))
            D = lhe_parts._dirs(os.path.join(d, "q"))
            for v in D.values():
                os.makedirs(v, exist_ok=True)
            json.dump({"seed": 42, "items": [[r, p] for r, p in plain]}, open(os.path.join(D["specs"], NAME + ".json"), "w"))
            _register(n_runs=4, run_events=400, complete_runs=True)
            assert lhe_parts._adopt(D, NAME, its) == 4
            assert all(os.path.samefile(a, b) for a, (b, _) in zip(old, parts))
            # e+e-: the totals fill the blocks but one energy's complete runs miss its equal share: refused, the pool
            # at the path left where it is
            _register(n_runs=3, run_events=300, complete_runs=True)
            mg.PROCESSES[NAME]["lhe"]["energies"] = [500.0, 3000.0]
            its = _items()
            parts = mg.lhe_part_paths(NAME, its)
            write([300, 300, 300, 300, 100, 100])                    # 1200 complete, 300 of them at 3000 GeV
            tr_path = its[2][1]
            np.save(tr_path, np.zeros((3, 3)))
            json.dump({"recipe_id": "other", "lhe_run": {}}, open(tr_path + ".recipe.json", "w"))
            n_aside = len(glob.glob(os.path.join(os.path.dirname(tr_path), "set_aside", "*", "*")))
            try:
                mg.lhe_assemble(NAME, its)
                raise AssertionError("assembled with an energy short of its share")
            except RuntimeError as e:
                assert "raise lhe.n_runs" in str(e), e
            assert np.load(tr_path).shape == (3, 3)
            assert len(glob.glob(os.path.join(os.path.dirname(tr_path), "set_aside", "*", "*"))) == n_aside
    finally:
        (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
         datagen.dest_for_role) = saved
        for v in ("AMP_LHE_PARTS_DIR", "AMP_LHE_WORK_DIR"):
            os.environ.pop(v, None)
        mg.PROCESSES.pop(NAME, None)
    print("[complete_runs] ok")


def test_tidy_and_gc():
    """A worker deletes a saved run's directory (its removals in its gc/ manifest first, nothing deleted when that
    write fails) and reduces a failed or short one to its logs, the short one not done; gc does the same for runs left
    behind (killed, failed before the change), skips runs a live job holds, and one claimed while it runs, writes its
    manifest first, and never deletes outside $SCRATCH/lhe_work/parts."""
    _register(n_runs=4)
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role, lhe_parts.ASSEMBLE_S, lhe_parts._my_jobs)
    env0 = {v: os.environ.get(v) for v in ("SCRATCH", "AMP_LHE_PARTS_DIR", "AMP_LHE_WORK_DIR", "TMPDIR")}
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
            # node-local work tree (TMPDIR): accepted; a failed run's logs copied to Q/logs/<run>, its directory gone
            node = os.path.join(d, "node")
            os.makedirs(node)
            os.environ["TMPDIR"] = node
            os.environ["AMP_LHE_WORK_DIR"] = os.path.join(node, "lhe_work")
            lwd = lhe_parts.run_workdir(NAME, mg.lhe_part_paths(NAME, _items())[1][1])
            os.makedirs(os.path.join(lwd, "junk"))
            open(os.path.join(lwd, "a.log"), "w").write("why it failed")
            open(os.path.join(lwd, "junk", "big"), "wb").write(b"x" * 5000)
            assert lhe_parts.parts_work_root().startswith(os.path.realpath(node) + os.sep)
            assert lhe_parts.tidy_run(NAME, _items(), 1, q) == "logs"
            assert not os.path.exists(lwd), "a node-local run directory left behind"
            assert open(os.path.join(q, "logs", f"{NAME}__001", "a.log")).read() == "why it failed"
            # the guard: a work tree outside $SCRATCH/lhe_work/parts (and outside TMPDIR) is refused, and so is a path
            # out of the tree
            os.environ.pop("TMPDIR", None)
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


def test_ee_energy_runs_and_equal_shares():
    """An e+e- pool (lhe.energies) with lhe.energy_runs: energy j gets its own number of runs, numbered and seeded in
    sequence; an accept_short part counts only with its run's part of its energy's equal share; the assembly cuts the
    energies to equal counts (unchanged when every run returned what it asked) and refuses an energy short of its
    share."""
    _register(n_runs=1, run_events=None, accept_short=True)
    l = mg.PROCESSES[NAME]["lhe"]
    l["energies"] = [91.1876, 500.0, 3000.0]
    r0 = mg.lhe_runs(NAME, 100, 1020)
    assert [r[1] for r in r0] == [340] * 3 and [r[3] for r in r0] == ["run_01", "run_02", "run_03"]
    rec0 = mg.variable_energy_recipe(NAME, 13000, 13000, 800, role="train", seed=42)
    l["energy_runs"] = [1, 1, 4]
    runs = mg.lhe_runs(NAME, 100, 1020)
    assert [r[1] for r in runs] == [340] * 6 and [r[3] for r in runs][-1] == "run_06"
    assert [2 * r[2]["ebeam1"] for r in runs] == [91.1876, 500.0] + [3000.0] * 4
    assert len({r[0] for r in runs}) == 6 and runs[:2] == r0[:2]
    rec = mg.variable_energy_recipe(NAME, 13000, 13000, 800, role="train", seed=42)
    assert rec["lhe"]["energy_runs"] == [1, 1, 4] and mg.recipe_id(rec) != mg.recipe_id(rec0)
    # the share is ceil(1000 / 3) = 334 per energy: 334 for a one-run energy, ceil(334 / 4) = 84 per run at 3 TeV
    assert [lhe_parts.part_need(NAME, r) for r in runs] == [334, 334, 84, 84, 84, 84]
    ev = lambda n: np.arange(n, dtype=float)[:, None, None] * np.ones((1, K, 4))
    full = [ev(340), ev(340), ev(90), ev(90), ev(90), ev(90)]
    m, a = mg.lhe_equal_shares(NAME, full, [x[:, 0, 0] for x in full], runs, 1000)
    assert [len(x) for x in m] == [340, 340, 90, 90, 90, 70]          # each energy cut to 340
    m, _ = mg.lhe_equal_shares(NAME, [ev(340)] * 3, [ev(340)[:, 0, 0]] * 3, r0, 1000)
    assert [len(x) for x in m] == [340] * 3                            # nothing short: unchanged
    try:
        mg.lhe_equal_shares(NAME, [ev(340), ev(340), ev(80)] + [ev(80)] * 3, [ev(1)[:, 0, 0]] * 6, runs, 1000)
        raise AssertionError("an energy short of its share was assembled")
    except RuntimeError as e:
        assert "3000" in str(e) and "334" in str(e), e
    # lhe_parts shares through its command line: one energy short in its parts, the pool's assembled marker moved aside
    saved = (datagen.dest_for_role, sys.argv)
    try:
        with tempfile.TemporaryDirectory() as d:
            datagen.dest_for_role = lambda role: os.path.join(d, "pools", role)
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            q = os.path.join(d, "queue")
            D = lhe_parts._dirs(q)
            for x in D.values():
                os.makedirs(x, exist_ok=True)
            its = _items()
            json.dump({"seed": 42, "items": [[r, p] for r, p in its]}, open(os.path.join(D["specs"], NAME + ".json"), "w"))
            for (p, r), n in zip(mg.lhe_part_paths(NAME, its), [340, 340, 90, 90, 90, 40]):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, mom=np.zeros((n, K, 4)))
            open(os.path.join(D["assembled"], NAME), "w").close()
            sys.argv = ["lhe_parts.py", "shares", "--queue", q, "--requeue"]
            lhe_parts.main()
            assert not os.listdir(D["assembled"])
            assert os.listdir(os.path.join(q, "assembled_superseded"))[0].startswith(NAME + ".")
    finally:
        datagen.dest_for_role, sys.argv = saved
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        mg.PROCESSES.pop(NAME, None)
    print("[e+e- energy runs + equal shares + shares CLI] ok")


def test_stale_worker_returns_tasks():
    """A worker started under an older layout than the queue's plan (the recipes changed and the queue was planned
    again while it ran) builds nothing: each task it claims goes back to todo (never failed), it claims no more and
    assembles nothing; a worker on the current layout then builds them. requeue moves a task a stale worker failed
    (by --match) or a done one whose part is missing back to todo, and an e+e- part at another energy aside."""
    saved = (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
             datagen.dest_for_role, lhe_parts.ASSEMBLE_S, sys.argv)
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
            D = lhe_parts._dirs(q)
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            _register(n_runs=3, run_events=400)
            lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q, seed=42, rates=rates, cores=[]))
            assert len(os.listdir(D["todo"])) == 3
            _register(n_runs=2, run_events=400)               # this worker's recipes: the layout before the replan
            try:
                lhe_parts.cmd_work(argparse.Namespace(queue=q, workers=1, hours=1.0))
            except SystemExit as e:
                assert e.code == 1, e.code
            assert len(os.listdir(D["todo"])) == 3 and not os.listdir(D["failed"]) and not os.listdir(D["done"])
            assert not os.listdir(D["assembled"])
            assert not any(os.path.exists(p) for p, _ in mg.lhe_part_paths(NAME, lhe_parts._spec(D, NAME)))
            # requeue: a task failed by a stale worker before this guard (IndexError), one done with its part missing
            _register(n_runs=3, run_events=400)
            fs = sorted(os.listdir(D["todo"]))
            os.rename(os.path.join(D["todo"], fs[0]), os.path.join(D["failed"], fs[0]))
            open(os.path.join(D["failed"], fs[0] + ".err"), "w").write("IndexError: list index out of range\n")
            os.rename(os.path.join(D["todo"], fs[1]), os.path.join(D["done"], fs[1]))
            sys.argv = ["lhe_parts.py", "requeue", "--queue", q, "--match", "^IndexError"]
            lhe_parts.main()                                  # dry run: nothing moved
            assert len(os.listdir(D["todo"])) == 1
            sys.argv += ["--apply"]
            lhe_parts.main()
            assert len(os.listdir(D["todo"])) == 3 and not os.listdir(D["failed"]) and not os.listdir(D["done"])
            assert os.listdir(os.path.join(q, "failed_requeued"))
            try:
                lhe_parts.cmd_work(argparse.Namespace(queue=q, workers=2, hours=1.0))
            except SystemExit as e:
                assert e.code == 0, e.code
            assert len(os.listdir(D["done"])) == 3 and os.listdir(D["assembled"]) == [NAME]
    finally:
        (mg.lhe_prepare, mg.lhe_run_madevent, mg.read_lhe_pp, mg.label_lhe_pp, mg.lhe_parent_physics_check,
         datagen.dest_for_role, lhe_parts.ASSEMBLE_S, sys.argv) = saved
        for v in ("AMP_LHE_PARTS_DIR", "AMP_LHE_WORK_DIR"):
            os.environ.pop(v, None)
        mg.PROCESSES.pop(NAME, None)
    # an e+e- part at another energy than its run's
    _register(n_runs=1, run_events=None, accept_short=True)
    mg.PROCESSES[NAME]["lhe"]["energies"] = [91.1876, 500.0, 3000.0]
    runs = mg.lhe_runs(NAME, 100, 1020)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npz")
        m = np.zeros((5, K, 4)); m[:, 0, 0] = m[:, 1, 0] = 250.0
        np.savez(p, mom=m)
        assert lhe_parts.part_energy_ok(p, runs[1]) and not lhe_parts.part_energy_ok(p, runs[2])
    mg.PROCESSES.pop(NAME, None)
    print("[stale worker + requeue] ok")


def test_park_and_skip_claimed():
    """park moves a queue's unclaimed tasks aside (nothing deleted) so its workers end with what they hold; a new queue
    planned with --skip-claimed on it plans every run but the ones held there."""
    _register(n_runs=4)
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            q1, q2 = os.path.join(d, "q1"), os.path.join(d, "q2")
            lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q1, seed=42, rates=rates, cores=[]))
            todo = sorted(os.listdir(os.path.join(q1, "todo")))
            assert len(todo) == 4
            os.rename(os.path.join(q1, "todo", todo[0]), os.path.join(q1, "claimed", todo[0] + ".12345"))
            lhe_parts.cmd_park(argparse.Namespace(queue=q1))
            assert not os.listdir(os.path.join(q1, "todo"))
            parked = glob.glob(os.path.join(q1, "parked", "*", "*.json"))
            assert len(parked) == 3
            lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q2, seed=42, rates=rates, cores=[],
                                                  skip_claimed=[q1]))
            assert sorted(os.listdir(os.path.join(q2, "todo"))) == todo[1:]
            # a layout with fewer runs, planned in a third queue taking over q1: the leading runs' parts made under q1's
            # layout are linked to the new names (the previous plan read from q1's specs), not planned again
            for part, _ in mg.lhe_part_paths(NAME, _items())[:2]:
                os.makedirs(os.path.dirname(part), exist_ok=True)
                np.savez(part, mom=np.zeros((5, 5, 4)), aqcd=np.zeros(5))
            mg.PROCESSES[NAME]["lhe"]["n_runs"] = 3
            q3 = os.path.join(d, "q3")
            lhe_parts.cmd_plan(argparse.Namespace(specs=[spec], queue=q3, seed=42, rates=rates, cores=[],
                                                  skip_claimed=[q1]))
            new = mg.lhe_part_paths(NAME, _items())
            assert all(os.path.exists(p) for p, _ in new[:2]) and len(os.listdir(os.path.join(q3, "todo"))) == 1
    finally:
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        del mg.PROCESSES[NAME]
    print("[park + skip-claimed] ok")


def test_generation_rule():
    """agent_redesign C5 on the replan under live workers (2026-10-29: the catalog_v4 default share replanned while
    workers of the old layout held claims; they failed 908 tasks with IndexError): the replan as it happened is refused,
    with park first it passes; `layout` says stale under changed recipes and parked after park; the probe file of that
    night refuses its processes that keep under half the events they ask, a process absent from it is refused, and an
    accepted one (the user's decision named) passes."""
    _register(n_runs=4)
    saved = lhe_parts._my_jobs
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            q = os.path.join(d, "q")
            ns = lambda **k: argparse.Namespace(**dict(dict(specs=[spec], queue=q, seed=42, rates=rates, cores=[]), **k))
            lhe_parts._my_jobs = lambda: {"555"}
            lhe_parts.cmd_plan(ns())
            lay = lambda: json.loads(_out(lambda: lhe_parts.cmd_layout(argparse.Namespace(queue=q))))
            assert lay()["current"] and lay()["todo"] == 4, lay()
            todo = sorted(os.listdir(os.path.join(q, "todo")))
            os.rename(os.path.join(q, "todo", todo[0]), os.path.join(q, "claimed", todo[0] + ".555"))
            mg.PROCESSES[NAME]["lhe"]["n_runs"] = 3                  # the recipes change under the live worker
            L = lay()
            assert not L["current"] and "laid out otherwise" in L["why"], L
            try:
                lhe_parts.cmd_plan(ns())                             # the replan as it happened: refused
                raise AssertionError("replan under a live claim was not refused")
            except SystemExit as e:
                assert "live workers hold 1 claims" in str(e.code), e.code
            lhe_parts._my_jobs = lambda: set()                       # the worker ended: its claim is no longer live
            assert not lhe_parts.live_claims(q)
            lhe_parts._my_jobs = lambda: {"555"}
            lhe_parts.cmd_park(argparse.Namespace(queue=q))
            assert not lay()["current"]
            lhe_parts.cmd_plan(ns())                                 # with park first: passes
            assert lay()["current"], lay()
            # the probe file of that night (committed): its low-yield processes refuse a plan of them
            probe = lhe_parts.PROBE_YIELDS
            y = json.load(open(probe))
            items = {n: [] for n in y}
            why = lhe_parts.plan_refusals(argparse.Namespace(queue=os.path.join(d, "fresh"), probe=probe), items)
            assert len(why) == 1 and "ask more than 2x" in why[0], why
            print("  probe file of 2026-10-30: refused for keeping under half the events they ask:", why[0][:400])
            acc_file = os.path.join(d, "accepted.json")
            fresh = lambda **k: argparse.Namespace(queue=os.path.join(d, "fresh"), probe=probe, accepted=acc_file, **k)
            assert len(lhe_parts.plan_refusals(fresh(), items)) == 1
            # accepting needs a decision in force that names the process (a scratch ledger)
            import sqlite3
            led = os.path.join(d, "ledger.db")
            c = sqlite3.connect(led)
            c.execute("CREATE TABLE decisions (id TEXT, project TEXT, kind TEXT, quote TEXT, covers TEXT, in_force INT)")
            lows = [n for n in y if y[n] <= 0 or 1.0 / y[n] > lhe_parts.YIELD_FACTOR]
            c.execute("INSERT INTO decisions VALUES ('D7','FA','approve','go','%s',1)" % ",".join(lows))
            c.execute("INSERT INTO decisions VALUES ('D8','FA','approve','the redesign','agent_redesign',1)")
            c.execute("INSERT INTO decisions VALUES ('D9','FA','approve','go','%s',0)" % ",".join(lows))
            c.commit(); c.close()
            acc = lambda *names: lhe_parts.cmd_accept_yield(argparse.Namespace(names=list(names), ledger=led,
                                                                               accepted=acc_file))
            for bad in ("D8", "D9", "D99"):                          # names nothing / withdrawn / absent
                try:
                    acc(*["%s=%s" % (n, bad) for n in lows])
                    raise AssertionError("accepted under %s" % bad)
                except SystemExit as e:
                    assert "refused" in str(e.code) and not os.path.exists(acc_file), e.code
            acc(*["%s=D7" % n for n in lows])
            assert sorted(json.load(open(acc_file))) == sorted(lows)
            assert not lhe_parts.plan_refusals(fresh(), items)
            why = lhe_parts.plan_refusals(fresh(), dict(items, zz_absent=[]))
            assert len(why) == 1 and "missing from the probe file" in why[0] and "zz_absent" in why[0], why
    finally:
        lhe_parts._my_jobs = saved
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        mg.PROCESSES.pop(NAME, None)
    print("[generation rule] ok")


def test_layout_dead_claims_park_marker_accept_boundary():
    """repo-reviewer findings on C5: (1) after a last wave dies at its limit, layout counts its claims as claimed_dead
    and the supervisor's queue_plan sends workers for them (todo 0); (2) a parked-then-replanned queue drained to
    todo 0 is not parked again; (3) accept-yield matches the process name on name boundaries, not as a substring;
    and layout reads a queue with no todo/ directory."""
    sys.path.insert(0, os.path.expanduser("~/.local/bin"))
    import site_runends as S
    _register(n_runs=3)
    saved = lhe_parts._my_jobs
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ["AMP_LHE_PARTS_DIR"] = os.path.join(d, "parts_store")
            spec = os.path.join(d, "spec.yaml")
            yaml.safe_dump({"processes": [{"name": NAME, "sqrts": [13000, 13000], "n_train": 800, "n_val": 100,
                                           "n_test": 100}]}, open(spec, "w"))
            rates = os.path.join(d, "rates.json")
            json.dump({"x": {"nfinal": 3, "core_s_per_event": 0.01}}, open(rates, "w"))
            q = os.path.join(d, "q")
            ns = lambda **k: argparse.Namespace(**dict(dict(specs=[spec], queue=q, seed=42, rates=rates, cores=[]), **k))
            lay = lambda qq=q: json.loads(_out(lambda: lhe_parts.cmd_layout(argparse.Namespace(queue=qq))))
            lhe_parts._my_jobs = lambda: {"555"}
            lhe_parts.cmd_plan(ns())
            # (1) the last wave claims all three tasks; one worker (556) dies at TIMEOUT, 555 still runs
            todo = sorted(os.listdir(os.path.join(q, "todo")))
            for f, job in zip(todo, ("555", "556", "556")):
                os.rename(os.path.join(q, "todo", f), os.path.join(q, "claimed", f + "." + job))
            L = lay()
            assert L["current"] and L["todo"] == 0 and L["claimed"] == 3 and L["claimed_dead"] == 2, L
            reg = {"queue": "q", "max_workers": 10, "wall_h": 2.0, "total_hours": 100.0, "submitted_hours": 0}
            n, why, _ = S.queue_plan(reg, 1, L, time.time())
            assert n == 2, (n, why)                                   # before: 0, "nothing to do"
            lhe_parts._my_jobs = lambda: None                         # squeue unreadable: no claim counted dead
            assert lay()["claimed_dead"] == 0
            lhe_parts._my_jobs = lambda: {"555", "556"}
            assert lay()["claimed_dead"] == 0 and S.queue_plan(reg, 2, lay(), time.time())[0] == 0
            # (2) parked, replanned, then drained: current again, and the replan guard holds under live claims
            for f in os.listdir(os.path.join(q, "claimed")):
                os.rename(os.path.join(q, "claimed", f), os.path.join(q, "todo", f.rsplit(".", 1)[0]))
            lhe_parts.cmd_park(argparse.Namespace(queue=q))
            assert lay()["why"] == "parked" and lhe_parts.parked(q)
            lhe_parts.cmd_plan(ns())
            assert not lhe_parts.parked(q) and lay()["current"], lay()
            todo = sorted(os.listdir(os.path.join(q, "todo")))
            for f in todo:                                            # drained: every task claimed by a live worker
                os.rename(os.path.join(q, "todo", f), os.path.join(q, "claimed", f + ".555"))
            assert not os.listdir(os.path.join(q, "todo")) and glob.glob(os.path.join(q, "parked", "*", "*.json"))
            L = lay()
            assert L["current"] and L["why"] == "", L                 # before: "parked"
            try:
                lhe_parts.cmd_plan(ns())
                raise AssertionError("replan under live claims of a once-parked queue was not refused")
            except SystemExit as e:
                assert "live workers hold 3 claims" in str(e.code), e.code
            # a queue with no todo/ directory
            bare = os.path.join(d, "bare")
            os.makedirs(os.path.join(bare, "claimed"))
            L = lay(bare)
            assert L["todo"] == 0 and L["claimed_dead"] == 0 and L["current"], L
            # (3) accept-yield: a decision naming qq_ttbarj does not accept qq_ttbar, nor ttbar; qq_ttbar named is
            import sqlite3
            led, acc_file = os.path.join(d, "ledger.db"), os.path.join(d, "accepted.json")
            c = sqlite3.connect(led)
            c.execute("CREATE TABLE decisions (id TEXT, project TEXT, kind TEXT, quote TEXT, covers TEXT, in_force INT)")
            c.execute("INSERT INTO decisions VALUES ('D1','FA','approve','accept pp_qq_ttbarj','gen',1)")
            c.execute("INSERT INTO decisions VALUES ('D2','FA','approve','go','gen, pp_qq_ttbar',1)")
            c.commit(); c.close()
            acc = lambda *names: lhe_parts.cmd_accept_yield(argparse.Namespace(names=list(names), ledger=led,
                                                                               accepted=acc_file))
            for bad in ("pp_qq_ttbar=D1", "qq_ttbar=D1", "ttbarj=D1", "qq_ttbar=D2"):
                try:
                    acc(bad)
                    raise AssertionError("accepted %s" % bad)
                except SystemExit as e:
                    assert "neither quotes nor covers" in str(e.code) and not os.path.exists(acc_file), e.code
            acc("pp_qq_ttbarj=D1", "pp_qq_ttbar=D2")
            assert sorted(json.load(open(acc_file))) == ["pp_qq_ttbar", "pp_qq_ttbarj"]
    finally:
        lhe_parts._my_jobs = saved
        os.environ.pop("AMP_LHE_PARTS_DIR", None)
        mg.PROCESSES.pop(NAME, None)
    print("[dead claims + park marker + accept boundary] ok")


def _out(f):
    import io, contextlib
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        f()
    return b.getvalue().strip().splitlines()[-1]


if __name__ == "__main__":
    test_runs_and_identity()
    test_parts_equal_serial()
    test_queue_end_to_end()
    test_queue_stall_layout_short_parts()
    test_accept_short_and_adopt()
    test_complete_runs()
    test_tidy_and_gc()
    test_ee_energy_runs_and_equal_shares()
    test_stale_worker_returns_tasks()
    test_park_and_skip_claimed()
    test_generation_rule()
    test_layout_dead_claims_park_marker_accept_boundary()
