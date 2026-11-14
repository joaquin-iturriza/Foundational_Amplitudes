#!/usr/bin/env python3
"""Score a finished single-process run on the KIT hard phase-space benchmark: predictions in its layout, its scores.

    python tools/kit_predict.py <run_dir> [--splits hard_test bulk_test] [--out <dir>]

The run's recipe names one KIT dataset (kit_<file>__<target>[__hard], tools/kit_hardps.py). Every stratum of the
splits asked is written as one test pool, the run is rebuilt on it with its best weights (tools/rebuild_run.rebuild:
its own statistics, dtype and preprocessing), and each event's prediction is put back in the file's order as
log10 |M|^2 of the benchmark's target (the alpha_s stripping undone):
    log10 pred = log10 truth + (pred_z - truth_z) * prepd_std / ln 10,
the residual in the network's standardized ln target, exact for any per-event factor divided out of both.
Writes <out>/predictions.h5 (<split>/<stratum>/log10_<...>, the truth's dataset name) and runs the benchmark's own
src/evaluate.py on it into <out>/kit_result.json (its stdout table into kit_result.txt). <out> defaults to
<run_dir>/kit_benchmark.
--h5 <file>: the strata of another file in the benchmark's layout (new events of the run's process); a group without
the target is predicted from a unit placeholder truth (log10 pred = the residual / ln 10, the same identity), and then
nothing is scored.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402


def run_target(run_dir):
    import yaml
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    rec = yaml.safe_load(open(os.path.expandvars(str(cfg.data.processes_file))))
    (spec,) = rec["processes"]
    return spec["name"]


def predict(run_dir, splits, out, h5=None):
    """`h5`: score the strata of another file in the benchmark's layout and the run's file's metadata (default the
    run's own file), e.g. tools/kit_threshold_scan.py's scan points"""
    import h5py
    import torch
    from tools.rebuild_run import rebuild
    # CPU predictions move with the thread count: tt virtual softpair by 3.8e-4 in log10 at 32 threads (package v2,
    # FA-ccin2p3-2d0c78fe), the asymptotic-factor models by up to 5.5e-3 at 4 threads and not at all at 1 (package v5,
    # FA-ccin2p3-d56c5264 vs FA-ccin2p3-83880457): pinned to 1 (README)
    torch.set_num_threads(int(os.environ.get("KIT_THREADS", 1)))
    name = run_target(run_dir)
    stem, target, _ = kh.parse(name)
    src = h5 or kh.h5_path(stem)
    lname = kh.log10_name(stem, target, src)
    blocks, rows = [], []
    for split in splits:
        for st in kh.strata(stem, split, src):
            r = kh.stratum_rows(stem, target, split, st, src)
            assert (r[:, -1] > 0).all(), f"{split}/{st}: a non-positive target (the signed case is not wired here)"
            blocks.append((split, st, len(r)))
            rows.append(r)
    assert rows, f"none of the splits {splits} in {src}"
    rows = np.concatenate(rows, axis=0)
    os.makedirs(out, exist_ok=True)
    pool = os.path.join(out, "kit_strata_pool.npy")
    np.save(pool, rows)
    # the per-event alpha_s sidecar data.target_physics_orders reads beside every pool (the file's fixed alpha_s)
    with h5py.File(src, "r") as f:    # the scored file's alpha_s, as stratum_rows' strip factor
        a_s = float(json.loads(f.attrs["metadata"])["alpha_s"])
    np.save(pool[:-len(".npy")] + ".alphas.npy", np.full(len(rows), a_s))
    import yaml
    if yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))["data"].get("target_over_born"):
        # a V/B run: the scored events' exact Born beside the pool, from the same file's born target
        meta = kh.metadata(stem)
        kh.write_born(np.concatenate([kh.stratum_rows(stem, "born", sp, st, src)[:, -1] for sp, st, _ in blocks])
                      * kh.strip_factor(meta, "born"), pool)
    exp, loader = rebuild(run_dir, "test", pool, len(rows), event_frame=True)
    trafos = [list(t) for t in (getattr(exp, "_amp_trafos_pp", None) or [exp.cfg.data.amp_trafos])]
    assert all(t == ["log", "standardization"] for t in trafos), (
        f"the residual is ln-exact only under [log, standardization]; the run resolved {trafos}")
    with torch.no_grad():
        pred, truth, _ = exp._collect_predictions(loader)
    res_ln = (np.asarray(pred, np.float64).reshape(-1) - np.asarray(truth, np.float64).reshape(-1)) \
        * float(exp.prepd_std[0])
    perm = np.asarray(exp._role_perm["test"])[:len(res_ln)]
    res = np.full(len(rows), np.nan)
    res[perm] = res_ln
    assert np.isfinite(res).all(), "an event without a prediction"
    path = os.path.join(out, "predictions.h5")
    i = 0
    with h5py.File(src, "r") as t, h5py.File(path, "w") as f:
        for split, st, n in blocks:
            g = t[f"{split}/{st}"]
            tr = g[lname][:] if lname in g else np.zeros(n)    # no truth: log10 of the unit placeholder
            f[f"{split}/{st}/{lname}"] = tr + res[i:i + n] / np.log(10.0)
            i += n
        f.attrs["run_dir"] = os.path.abspath(run_dir)
        f.attrs["dataset"] = name
    os.remove(pool)
    os.remove(pool[:-len(".npy")] + ".alphas.npy")
    return stem, path


def evaluate(stem, pred_path, out, h5=None):
    ev = os.path.join(os.path.dirname(os.path.dirname(kh.kit_dir())), "src", "evaluate.py")
    if not os.path.exists(ev):
        print(f"no benchmark scorer at {ev}: predictions only")
        return None
    res = os.path.join(out, "kit_result.json")
    # the benchmark's own scorer, run isolated (-I) from its directory's files
    p = subprocess.run([sys.executable, "-I", ev, h5 or kh.h5_path(stem), pred_path, res], capture_output=True, text=True)
    assert p.returncode == 0, f"the benchmark's evaluate.py failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}"
    open(os.path.join(out, "kit_result.txt"), "w").write(p.stdout)
    print(p.stdout)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--splits", nargs="+", default=["hard_test", "bulk_test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--h5", default=None, help="score the strata of this file (the benchmark's layout and targets) "
                    "instead of the run's own benchmark file, e.g. new events of the same process")
    ap.add_argument("--check-rebuild", type=int, default=0, metavar="N",
                    help="only rebuild the run on the first N bulk_test events (its stats asserted), then exit: a CPU check")
    a = ap.parse_args()
    if a.check_rebuild:
        import tempfile
        from tools.rebuild_run import rebuild
        stem, target, _ = kh.parse(run_target(a.run_dir))
        r = kh.stratum_rows(stem, target, "bulk_test", kh.strata(stem, "bulk_test")[0])[:a.check_rebuild]
        d = tempfile.mkdtemp(prefix="kitcheck_", dir=os.environ.get("SCRATCH"))
        pool = os.path.join(d, "pool.npy")
        np.save(pool, r)
        np.save(pool[:-len(".npy")] + ".alphas.npy", np.full(len(r), float(kh.metadata(stem)["alpha_s"])))
        exp, _ = rebuild(a.run_dir, "test", pool, len(r))
        print(f"REBUILD_OK {a.run_dir} mom_div {exp.mom_div}")
        return
    out = a.out or os.path.join(a.run_dir, "kit_benchmark")
    stem, path = predict(a.run_dir, a.splits, out, a.h5)
    import h5py
    _, target, _ = kh.parse(run_target(a.run_dir))
    with h5py.File(a.h5 or kh.h5_path(stem), "r") as f:
        truth = all(target in f[f"{sp}/{st}"] for sp in a.splits if sp in f for st in f[sp])
    r = evaluate(stem, path, out, a.h5) if truth else None
    print(json.dumps({"predictions": path, "result": r}))


if __name__ == "__main__":
    main()
