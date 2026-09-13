#!/usr/bin/env python
"""Probe a built loop-induced standalone: born must be 0, the IR poles must vanish, and
|M_1|^2 must scale with the coupling as its `order` says (alpha_s^a: doubling alpha_s
multiplies it by 2^a). Pins MadLoop's normalisation for the loop-squared target.
Usage: python tools/probe_loop_induced.py uubar_Hg [--sqrts 300]"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import nlo_madloop as ML, mg5_pipeline_final as mg
from nlo_virtual_pipeline import VIRT_PROCESSES, virt_standalone_dir, find_p0

ap = argparse.ArgumentParser(); ap.add_argument("process"); ap.add_argument("--sqrts", type=float, default=300.0)
a = ap.parse_args()
cfg = VIRT_PROCESSES[a.process]; assert cfg.get("loopind"), "not a loop-induced entry"
g = ML.load(find_p0(virt_standalone_dir(a.process)))
perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5"])
ev, _ = mg.sample_nbody_phase_space(3, a.sqrts, a.sqrts, cfg["m_finals"], cfg["pdg_ids"],
                                    rng=np.random.default_rng(1), cuts=mg.FIDUCIAL_CUTS)
a_s_pow = cfg["order"][2]
for mom, _ in ev:
    r1 = ML.evaluate(g, mom[perm], alphas=0.118)
    r2 = ML.evaluate(g, mom[perm], alphas=0.236)
    ratio = r2["fin"] / r1["fin"]
    print(f"born={r1['born']:.1e}  fin={r1['fin']:.4e}  e1/fin={r1['e1']/r1['fin']:.1e}  e2/fin={r1['e2']/r1['fin']:.1e}"
          f"  fin(2as)/fin(as)={ratio:.4f}  expected 2^{a_s_pow}={2**a_s_pow}")
