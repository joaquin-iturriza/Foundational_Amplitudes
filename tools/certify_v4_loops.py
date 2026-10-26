#!/usr/bin/env python3
r"""
certify_v4_loops.py — the pole certification of EVERY catalog_v4 one-loop entry, in one list (CPU).

Runs `tools/nlo_virtual_pipeline.py <entry> --build --certify` for each v4_* entry of recipes/catalog_v4_loops.yaml,
in a work dir of its own ($SCRATCH/zz_v4_certify_all, MG5_WORK_DIR), so that the regeneration jobs reading and writing
the production backends ($WORK/mg5amcnlo/v4_*) at the same time are never touched (the check rewrites the top width in
the card it certifies). A backend already built there is copied (copy_backend: MadGraph's relative links kept, only
links leaving the tree dereferenced; the production tree is only read) instead of rebuilt; a missing one is built in the
work dir. One CERT line per entry, then a summary counting each verdict separately; the whole table goes to
$SCRATCH/zz_v4_certify_all/certify_all.json. Verdicts:
  PASS        the checker passed;
  FAIL        the checker ran and printed its pole numbers, and they fail (a physics verdict);
  ERROR       no verdict: a missing MadLoop resource, a crashed or silent checker, a copy that failed;
  BUILD_FAIL  MG5 generation or the matrix2py build failed.

    python tools/certify_v4_loops.py --workers 16 [--only v4_a v4_b ...]
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# the checker printed numbers: its FAIL is a physics verdict (nlo_pole_check.py, and the loop-induced / EW certifiers
# of nlo_virtual_pipeline.py)
VERDICT_MARKS = ("predicted (const)", "no usable points", "max |pole|/|fin|", "QED double pole")
# no verdict at all: MadLoop could not start, or the checker died
ERROR_MARKS = ("could not automatically find", "(certifier crashed)", "(no checker output)", "Traceback")
RESOURCE = "SubProcesses/P0_*/MadLoop5_resources/MadLoopParams.dat"


def copy_backend(src, dst):
    """Copy a built standalone so that MadLoop finds its resources in the copy. MadGraph's tree is full of relative
    links (P0_*/MadLoop5_resources -> ../MadLoop5_resources -> ../MadLoopParams.dat -> ../Cards/...): they are kept as
    links, so they resolve inside the copy (and the width patch of the check touches the copy's card only). A link
    leaving the tree is replaced by a copy of its target, a dangling one dropped. (copytree(symlinks=False,
    ignore_dangling_symlinks=True) tested each relative link against the CWD, not its own directory, and dropped
    MadLoopParams.dat, ident_card.dat and param_card.dat from MadLoop5_resources: every copied backend failed with
    "could not automatically find the file MadLoopParams.dat", FA-ccin2p3-e425f8aa.)"""
    shutil.copytree(src, dst, symlinks=True)
    top = os.path.realpath(dst)
    for root, dirs, files in os.walk(dst):
        for n in dirs + files:
            p = os.path.join(root, n)
            if not os.path.islink(p):
                continue
            t = os.path.realpath(p)
            if t == top or t.startswith(top + os.sep):
                continue
            os.unlink(p)
            if os.path.isdir(t):
                shutil.copytree(t, p, symlinks=False)
            elif os.path.isfile(t):
                shutil.copy2(t, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    import mg5_pipeline_final as mg          # the production paths, before the override below
    import yaml
    prod, mg5_bin = mg.WORK_DIR, mg.MG5_BIN
    out = os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_v4_certify_all")
    os.makedirs(out, exist_ok=True)
    names = args.only or sorted(yaml.load(open(os.path.join(ROOT, "recipes", "catalog_v4_loops.yaml")),
                                          Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["virt"])
    print(f"CERT {len(names)} one-loop entries, production backends {prod}, work dir {out}", flush=True)
    env = dict(os.environ, MG5_WORK_DIR=out, MG5_BIN=mg5_bin, SETUPTOOLS_USE_DISTUTILS="stdlib",
               OMP_NUM_THREADS="1")

    def one(name):
        t0 = time.time()
        src, dst = f"{prod}/{name}_virt_standalone", f"{out}/{name}_virt_standalone"
        origin = "work dir"
        if os.path.isdir(dst) and not glob.glob(f"{dst}/{RESOURCE}"):
            shutil.rmtree(dst)                              # an incomplete copy (the old copier's): redo it
        if not os.path.isdir(dst) and glob.glob(f"{src}/SubProcesses/P0_*/matrix2py*.so"):
            try:
                copy_backend(src, dst)
            except Exception as e:
                return {"entry": name, "verdict": "ERROR", "rc": None, "origin": "copy failed",
                        "s": round(time.time() - t0), "lines": [f"copy of {src} failed: {e!r}"], "tail": []}
            for m in glob.glob(f"{dst}/.certified"):       # certify afresh
                os.remove(m)
            if not glob.glob(f"{dst}/{RESOURCE}"):
                return {"entry": name, "verdict": "ERROR", "rc": None, "origin": "copied",
                        "s": round(time.time() - t0), "lines": [f"copy lacks {RESOURCE}"], "tail": []}
            origin = "copied"
        elif not os.path.isdir(dst):
            origin = "built"
        r = subprocess.run([sys.executable, f"{HERE}/nlo_virtual_pipeline.py", name, "--build", "--certify"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        txt = r.stdout + r.stderr
        open(f"{out}/{name}.certify.log", "w").write(txt)
        if "[BUILD] MG5 generation failed" in txt or "matrix2py.so build failed" in txt:
            verdict = "BUILD_FAIL"
        elif r.returncode == 0 and not any(k in txt for k in ERROR_MARKS):
            verdict = "PASS"
        elif r.returncode == 2 and any(k in txt for k in VERDICT_MARKS) and not any(k in txt for k in ERROR_MARKS):
            verdict = "FAIL"
        else:
            verdict = "ERROR"
        keep = [l.strip() for l in txt.splitlines()
                if any(k in l for k in ("predicted", "mean/std", "max |", "worst:", "points:", "-> FAIL",
                                        "QED double pole", "ERROR", "Error", "Traceback", "crashed",
                                        "no checker output"))]
        return {"entry": name, "verdict": verdict, "rc": r.returncode, "origin": origin,
                "s": round(time.time() - t0), "lines": keep[:12], "tail": txt.strip().splitlines()[-6:]}

    res = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for f in cf.as_completed([ex.submit(one, n) for n in names]):
            r = f.result()
            res.append(r)
            print(f"CERT {r['entry']}: {r['verdict']} ({r['origin']}, {r['s']} s)", flush=True)
            if r["verdict"] != "PASS":
                for l in (r["lines"] or r["tail"]):
                    print(f"CERT   {r['entry']}: {l}", flush=True)
    res.sort(key=lambda r: r["entry"])
    json.dump(res, open(f"{out}/certify_all.json", "w"), indent=1)
    from collections import Counter
    c = Counter(r["verdict"] for r in res)
    print("CERT SUMMARY " + "  ".join(f"{v} {c.get(v, 0)}" for v in ("PASS", "FAIL", "ERROR", "BUILD_FAIL"))
          + f"  of {len(res)}", flush=True)
    for r in res:
        if r["verdict"] != "PASS":
            print(f"CERT NOT_PASS {r['entry']} {r['verdict']}", flush=True)


if __name__ == "__main__":
    main()
