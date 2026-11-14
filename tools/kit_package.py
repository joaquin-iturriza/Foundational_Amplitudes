#!/usr/bin/env python3
"""The checkpoint + inference package for KIT (kit_benchmark step 5): chosen runs with the code that predicts with
them, float64, log10 |M|^2 out in the benchmark's layout.

    python tools/kit_package.py build --out <dir> --runs <run_dir>[=<label>] ...   # on the site holding the runs
    python tools/kit_package.py check --pkg <dir> [--n 200]                        # the package alone reproduces them
    python tools/kit_package.py relocate                                           # at KIT, once, in the unpacked dir

build writes <dir>/: the code (git archive of HEAD: the public core of scripts/publish_main.sh plus EXTRA), each run's
config.yaml, data_stats.json, particle tokenizer, best checkpoint and recipe under runs/kit/, the diagram sidecar of
its (file, target) (tools/kit_hardps.DIAGRAMS_OF) under data/diagrams/, models.json (run, label, process, target,
the commit) and README.txt (how to run), then <dir>.tar.gz. Predicting is tools/kit_predict.py --h5 <file>: the
strata of any file in the benchmark's layout (new events without the target too); the run is rebuilt on its own
statistics from the benchmark's bulk_train (KIT_HARDPS_DIR), so nothing of ours but the package is needed.
check unpacks nothing: it runs the package's own code (cwd = the package, DATA_DIR and SCRATCH fresh temporary dirs)
on the first n events of every hard_test stratum and bulk_test, and compares with the run's predictions from its job
(<run>/kit_benchmark/predictions.h5): max |delta log10| per run, which must be below 1e-9.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import kit_hardps as kh  # noqa: E402

EXTRA = ["keyed_aug.py", "multiplicity_extrap.py", "physics_scales.py",
         "tools/__init__.py", "tools/kit_hardps.py", "tools/kit_predict.py", "tools/rebuild_run.py",
         "tools/kit_package.py", "tools/nlo_conventions.py",
         # imported by the code above but outside publish_main's public paths (tests/test_kit_package_imports.py)
         "data_mmap.py", "dirlock.py", "diagram_graphs.py", "optim/__init__.py", "mix_compose.py", "optim/muon.py", "tools/nlo_virtual_pipeline.py",
         "tools/pool_physics.py", "tools/synthetic_amplitudes.py",
         "analysis/catalog_daniel/data/diagram_couplings.json"]   # read at init (data.target_physics_couplings)
RUN_FILES = ["config.yaml", "data_stats.json", "particle_tokenizer.json", "base_shapes.bsh"]
README = """FA surrogate models for the KIT hard phase-space benchmark (models.json lists them).

Setup (the Python environment of requirements.txt):
    tar xzf {name}.tar.gz && cd {name}
    python tools/kit_package.py relocate          # once: points the run configs at this directory
    export KIT_HARDPS_DIR=<hardps_benchmark>/data/full DATA_DIR=$(mktemp -d) SCRATCH=$(mktemp -d)
Predict (CPU or GPU, float64 throughout):
    python tools/kit_predict.py runs/kit/<run> --h5 <file.h5> --splits <split> ... --out <outdir>
<file.h5> is in the benchmark's layout (<split>/<stratum>/momenta, (N, n, 4), E px py pz in GeV, the CM frame,
order e+, e-, finals) and the root attribute `metadata` of the benchmark's file for that process; groups may lack
the target. Output: <outdir>/predictions.h5, <split>/<stratum>/log10_<target> (log10_abs_ for ee_tt), the benchmark's
name; with the target present, also src/evaluate.py's scores (kit_result.json). Without --h5 the benchmark's own file
is scored. Each run is rebuilt on its own statistics from the benchmark's bulk_train (the first n_train events of its
recipe), which needs KIT_HARDPS_DIR.
Reproducibility: a prediction depends on the event alone (read in its own frame, no augmentation; the frames'
degeneracy regularization uses noise that is a function of the particle, not of a random generator). CPU threads are
pinned to 1 (KIT_THREADS), at which the package on CPU reproduces our predictions (made on GPU) exactly: the
asymptotic-factor models move with the thread count (at 4 threads by up to 5.5e-3 in log10 on ee_tt's Sudakov bins,
4.9e-4 at threshold, 2.6e-5 on ee_uug), a rounding sensitivity well below their errors there.
"""
TOL = 2e-6   # rounding limit of the check (README; at the pinned thread count it reproduces exactly)


def public_paths():
    txt = open(os.path.join(ROOT, "scripts", "publish_main.sh")).read()
    block = re.search(r"PUBLIC_PATHS=\((.*?)\n\)", txt, re.S).group(1)
    return [l.split("#")[0].strip() for l in block.splitlines() if l.split("#")[0].strip()]


def cfg_of(run):
    import yaml
    return yaml.safe_load(open(os.path.join(run, "config.yaml")))


def cmd_build(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), f"{out} exists"
    os.makedirs(out)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = [p for p in public_paths() + EXTRA if subprocess.run(["git", "ls-files", "--error-unmatch", p], cwd=ROOT,
                                                                  capture_output=True).returncode == 0
             or glob.glob(os.path.join(ROOT, p, "*"))]
    tar = subprocess.Popen(["git", "archive", "--format=tar", commit, *paths], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(["tar", "xf", "-"], cwd=out, stdin=tar.stdout, check=True)
    assert tar.wait() == 0, "git archive failed"
    models = []
    for spec in a.runs:
        run, _, label = spec.partition("=")
        run = os.path.abspath(run)
        name = os.path.basename(run.rstrip("/"))
        dst = os.path.join(out, "runs", "kit", name)
        os.makedirs(os.path.join(dst, "models"))
        for f in RUN_FILES:
            if os.path.exists(os.path.join(run, f)):
                shutil.copy2(os.path.join(run, f), dst)
        best = [p for p in (os.path.join(run, "models", "model_run0_best.pt" + x) for x in ("", ".gz")) if os.path.exists(p)]
        assert best, f"{run}: no best checkpoint"
        shutil.copy2(best[0], os.path.join(dst, "models"))
        cfg = cfg_of(run)
        rec = os.path.expandvars(cfg["data"]["processes_file"])
        os.makedirs(os.path.join(out, "runs", "kit", "recipes"), exist_ok=True)
        shutil.copy2(rec, os.path.join(out, "runs", "kit", "recipes"))
        from tools.kit_predict import run_target
        stem, target, _ = kh.parse(run_target(run))
        side = kh.DIAGRAMS_OF.get((stem, target))
        if side and cfg["model"].get("use_diagrams", False):
            src = os.path.join(cfg.get("base_dir", ROOT), cfg["model"].get("diagrams_dir", "data/diagrams"),
                               f"{side}.diagrams.json")
            os.makedirs(os.path.join(out, "data", "diagrams"), exist_ok=True)
            shutil.copy2(src, os.path.join(out, "data", "diagrams"))
        models.append({"run": name, "label": label, "file": stem, "target": target})
    json.dump({"commit": commit, "models": models, "built_from": ROOT}, open(os.path.join(out, "models.json"), "w"), indent=1)
    open(os.path.join(out, "README.txt"), "w").write(README.format(name=os.path.basename(out), tol=TOL))
    subprocess.run(["tar", "czf", out + ".tar.gz", "-C", os.path.dirname(out), os.path.basename(out)], check=True)
    print(f"PACKAGE {out}.tar.gz: {len(models)} runs at {commit[:8]}; {os.path.getsize(out + '.tar.gz') / 1e6:.1f} MB")


def cmd_relocate(a):
    meta = json.load(open(os.path.join(ROOT, "models.json")))
    for m in meta["models"]:
        p = os.path.join(ROOT, "runs", "kit", m["run"], "config.yaml")
        txt = open(p).read()
        old = re.search(r"^base_dir: (.*)$", txt, re.M).group(1).strip()
        open(p, "w").write(txt.replace(old, ROOT))
    print(f"RELOCATED {len(meta['models'])} run configs to {ROOT}")


def cmd_check(a):
    import h5py
    pkg = os.path.abspath(a.pkg)
    meta = json.load(open(os.path.join(pkg, "models.json")))
    # KIT_HARDPS_DIR as the README tells KIT to set it: the runs rebuild their statistics from the benchmark's file
    env = dict(os.environ, DATA_DIR=tempfile.mkdtemp(prefix="kitpkg_data_"), SCRATCH=tempfile.mkdtemp(prefix="kitpkg_"),
               PYTHONPATH=pkg, KIT_HARDPS_DIR=kh.kit_dir())
    subprocess.run([sys.executable, "tools/kit_package.py", "relocate"], cwd=pkg, env=env, check=True)
    worst = {}
    try:
        _check_all(a, meta, env, pkg, worst)
    finally:
        for k in ("DATA_DIR", "SCRATCH"):
            shutil.rmtree(env[k], ignore_errors=True)
    if a.report:    # a diagnostic read: the deltas are the result, a mismatch is not a failure
        print(f"REPORT worst {max(worst.values()):.2e}")
        return
    assert max(worst.values()) < TOL, f"the package does not reproduce the runs' predictions to the rounding limit {TOL}"
    print("PACKAGE_OK")


def _subset(src, small, n_ev):
    """The first n_ev events of every hard_test and bulk_test stratum of `src`, in its layout, at `small`"""
    import h5py
    with h5py.File(src, "r") as f, h5py.File(small, "w") as g:
        g.attrs["metadata"] = f.attrs["metadata"]
        for sp in ("hard_test", "bulk_test"):
            for st in f[sp]:
                grp = f[f"{sp}/{st}"]
                # an azimuthal-scan stratum (its `group` column) is scored in whole groups of its phi values (evaluate.py);
                # a collinear stratum's phi is a continuous per-event angle (its unique count read the whole stratum)
                nphi = len(np.unique(grp["phi"][:])) if "group" in grp else 1
                n = max(n_ev // nphi, 1) * nphi
                for k, v in grp.items():
                    g[f"{sp}/{st}/{k}"] = v[:n]
    return small


def _check_all(a, meta, env, pkg, worst):
    import h5py
    for m in meta["models"]:
        if a.only and m["run"] not in a.only:
            continue
        own = os.path.join(meta["built_from"], "runs", "kit", m["run"], "kit_benchmark", "predictions.h5")
        # --n 0: the benchmark file itself, every event (the pool the job predicted)
        small = _subset(kh.h5_path(m["file"]), os.path.join(env["SCRATCH"], f"{m['run']}_first{a.n}.h5"), a.n) \
            if a.n else kh.h5_path(m["file"])
        out = os.path.join(env["SCRATCH"], m["run"])
        # --repo: the repository's own code on the original run dir (code drift since the predictions vs a package defect)
        src = meta["built_from"] if a.repo else pkg
        subprocess.run([sys.executable, "tools/kit_predict.py", f"runs/kit/{m['run']}", "--h5", small, "--out", out],
                       cwd=src, env=dict(env, PYTHONPATH=src, **({"OMP_NUM_THREADS": str(a.threads), "KIT_THREADS": str(a.threads)}
                                                       if a.threads else {})),
                       check=True, stdout=subprocess.DEVNULL)
        d = []
        with h5py.File(os.path.join(out, "predictions.h5"), "r") as p, h5py.File(own, "r") as o:
            for sp in p:
                for st in p[sp]:
                    for k in p[f"{sp}/{st}"]:
                        x = p[f"{sp}/{st}/{k}"][:]
                        d.append(np.max(np.abs(x - o[f"{sp}/{st}/{k}"][:len(x)])))
                        print(f"  {sp}/{st}: max |delta log10| {d[-1]:.2e}")
        worst[m["run"]] = float(max(d))
        print(f"CHECK {m['run']}: max |delta log10| {worst[m['run']]:.2e} over {len(d)} strata x {a.n} events", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--runs", nargs="+", required=True)
    c = sub.add_parser("check")
    c.add_argument("--pkg", required=True)
    c.add_argument("--n", type=int, default=200, help="first n events per stratum; 0: the whole file")
    c.add_argument("--only", nargs="*", default=None, help="these runs only (a quick end-to-end dry run)")
    c.add_argument("--report", action="store_true", help="print the deltas and exit 0 (a diagnostic, not the gate)")
    c.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS of the prediction (is it reproducible?)")
    c.add_argument("--repo", action="store_true", help="predict with the repository's kit_predict on the original runs")
    sub.add_parser("relocate")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "relocate": cmd_relocate}[a.cmd](a)


if __name__ == "__main__":
    main()
