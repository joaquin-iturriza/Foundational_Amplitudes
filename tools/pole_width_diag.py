"""Diagnostic: the universal double-pole check of one one-loop string with the top width zeroed (as certified) and as
generated, 300 points each.  python tools/pole_width_diag.py <process>"""
import os, sys, subprocess
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import nlo_virtual_pipeline as V, mg5_pipeline_final as mg
proc = sys.argv[1]
cfg = V.VIRT_PROCESSES[proc]; sa = V.virt_standalone_dir(proc); p0 = V.find_p0(sa)
pdg = list(cfg["pdg_ids"]); perm = mg.row_to_slot_perm(pdg, cfg["mg5"])
masses = [mg.BEAM_MASS.get(abs(int(p)), 0.0) for p in pdg[:2]] + list(cfg["m_finals"])
cmd = [sys.executable, "tools/nlo_pole_check.py", "--so-dir", p0, "--proc-order", *map(str, pdg), "--m", *map(str, masses),
       "--perm", *map(str, perm), "--n", "300", "--seed", "11"]
V.massive_beam_params(sa, pdg)
for zero in (True, False):
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    if zero:
        with V._zero_top_width(os.path.join(sa, "Cards", "param_card.dat")):
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("TOPWIDTH", "zero" if zero else "as generated")
    print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("DOUBLE", "PASS", "FAIL", "worst", "points", "exceptional"))))
