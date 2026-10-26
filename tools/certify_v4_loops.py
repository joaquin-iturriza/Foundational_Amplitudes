#!/usr/bin/env python3
r"""
certify_v4_loops.py — the pole certification of EVERY catalog_v4 one-loop entry, in one list (CPU).

Runs `tools/nlo_virtual_pipeline.py <entry> --build --certify` for each v4_* entry of recipes/catalog_v4_loops.yaml,
in a work dir of its own ($SCRATCH/zz_v4_certify_all, MG5_WORK_DIR), so that the regeneration jobs reading and writing
the production backends ($WORK/mg5amcnlo/v4_*) at the same time are never touched (the check rewrites the top width in
the card it certifies). A backend already built there is copied (dereferenced, read only) instead of rebuilt; a missing
one is built in the work dir. One CERT line per entry (PASS / FAIL / BUILD_FAIL / CRASH with the checker's worst
lines), then a summary; the whole table goes to $SCRATCH/zz_v4_certify_all/certify_all.json.

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
        if not os.path.isdir(dst) and glob.glob(f"{src}/SubProcesses/P0_*/matrix2py*.so"):
            shutil.copytree(src, dst, symlinks=False, ignore_dangling_symlinks=True)
            for m in glob.glob(f"{dst}/.certified"):       # certify afresh
                os.remove(m)
            origin = "copied"
        elif not os.path.isdir(dst):
            origin = "built"
        r = subprocess.run([sys.executable, f"{HERE}/nlo_virtual_pipeline.py", name, "--build", "--certify"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        txt = r.stdout + r.stderr
        open(f"{out}/{name}.certify.log", "w").write(txt)
        verdict = {0: "PASS", 2: "FAIL"}.get(r.returncode, "CRASH")
        if "[BUILD] MG5 generation failed" in txt or "matrix2py.so build failed" in txt:
            verdict = "BUILD_FAIL"
        keep = [l.strip() for l in txt.splitlines()
                if any(k in l for k in ("predicted", "mean/std", "max |", "worst:", "points:", "exceptional ->",
                                        "Error", "Traceback"))]
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
    print(f"CERT SUMMARY {dict(c)} of {len(res)}", flush=True)
    for r in res:
        if r["verdict"] != "PASS":
            print(f"CERT NOT_PASS {r['entry']} {r['verdict']}", flush=True)


if __name__ == "__main__":
    main()
