"""Give HTCondor sweeps a minimum GPU memory after they were generated without one: the idle jobs are edited in the
queue (condor_qedit RequireGPUs) and the sweep's submit files (subs/*.sub, read when the DAG submits a node) get a
`gpus_minimum_memory` line, so a trial that needs a large GPU stops landing on a 12 GB card and dying of CUDA OOM
(u ubar -> Z+5g at batch 16384 on lxplus, 2026-10-10). Runs on the HTCondor site, in the checkout (site run).
    python sweep/set_gpu_mem.py --mb 24000 SWEEP_PREFIX [...] [--apply]
"""
import argparse, glob, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import siteconf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prefixes", nargs="+")
ap.add_argument("--mb", type=int, required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
roots = {os.path.join(siteconf.SUBMIT_DIR, "sweeps"), siteconf.SWEEP_DIR}
dirs = sorted({d for p in a.prefixes for r in roots for d in glob.glob(os.path.join(r, p + "*"))
               if os.path.isdir(os.path.join(d, "subs"))})
line = f"gpus_minimum_memory     = {a.mb}\n"
for d in dirs:
    subs = glob.glob(os.path.join(d, "subs", "*.sub"))
    todo = [s for s in subs if "gpus_minimum_memory" not in open(s).read()]
    print(f"{os.path.basename(d)}: {len(todo)} of {len(subs)} submit files without a GPU memory floor")
    if a.apply:
        for s in todo:
            txt = open(s).read()
            txt = re.sub(r"^(request_gpus\s*=.*\n)", r"\1" + line, txt, count=1, flags=re.M)
            open(s, "w").write(txt)
    # idle queued jobs of this sweep (their Iwd or Cmd is under the sweep dir)
    q = subprocess.run(["condor_q", "-idle", "-af", "ClusterId", "ProcId", "Cmd"], capture_output=True, text=True).stdout
    ids = [f"{c}.{p}" for c, p, cmd in (l.split(None, 2) for l in q.splitlines() if l.count(" ") >= 2)
           if cmd.startswith(d + "/")]
    print(f"  {len(ids)} idle queued jobs")
    if a.apply and ids:
        subprocess.run(["condor_qedit", *ids, "RequireGPUs", f"GlobalMemoryMb >= {a.mb}"], check=False)
