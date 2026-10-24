"""Diagnostic: the universal double-pole check of catalog_v3 one-loop strings with the top width zeroed (as certified)
and as generated, --n points each; one WIDTH line per string and setting (max relative deviation of c2 from the
prediction, PASS/FAIL at the checker's tolerance).
    python tools/pole_width_diag.py <process> [...]      (registry keys, e.g. v3_bg_bwpwm)
    python tools/pole_width_diag.py --stage2-bt           (the 68 stage-2 strings with a b or t on a leg)"""
import argparse, os, re, sys, subprocess
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import nlo_virtual_pipeline as V, mg5_pipeline_final as mg

ap = argparse.ArgumentParser()
ap.add_argument("procs", nargs="*")
ap.add_argument("--stage2-bt", action="store_true")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
procs = list(a.procs)
if a.stage2_bt:
    import yaml
    a2 = {p["name"] for p in yaml.safe_load(open("recipes/catalog_v3_gen_stage2a.yaml"))["processes"]}
    for p in yaml.safe_load(open("recipes/catalog_v3_gen_stage2.yaml"))["processes"]:
        if p["name"] not in a2:
            procs.append(p["name"][:-4] if p["name"].endswith("_nlo") else p["name"])
for proc in procs:
    if proc not in V.VIRT_PROCESSES:
        print("WIDTH", proc, "not a one-loop registry key (loop-induced or unknown)", flush=True); continue
    cfg = V.VIRT_PROCESSES[proc]
    if cfg.get("loopind"):
        print("WIDTH", proc, "loop-induced: no pole check", flush=True); continue
    sa = V.virt_standalone_dir(proc); p0 = V.find_p0(sa)
    pdg = list(cfg["pdg_ids"]); perm = mg.row_to_slot_perm(pdg, cfg["mg5"])
    masses = [mg.BEAM_MASS.get(abs(int(p)), 0.0) for p in pdg[:2]] + list(cfg["m_finals"])
    cmd = [sys.executable, "tools/nlo_pole_check.py", "--so-dir", p0, "--proc-order", *map(str, pdg), "--m",
           *map(str, masses), "--perm", *map(str, perm), "--n", str(a.n), "--seed", "11"]
    V.massive_beam_params(sa, pdg)
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    for zero in (True, False):
        if zero:
            with V._zero_top_width(os.path.join(sa, "Cards", "param_card.dat")):
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        else:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        m = re.search(r"max rel = ([0-9.e+-]+).*-> (PASS|FAIL)", r.stdout)
        print("WIDTH", proc, "zero" if zero else "generated", m.group(1) if m else "?", m.group(2) if m else "no result",
              flush=True)
