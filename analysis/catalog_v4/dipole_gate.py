"""The dipole read's gate and main read per cell (plan v3_prep_explore, step 6; design passed 2026-10-30 22:28).

Input: the per-probe rows of analysis/catalog_v4/dipole_residual.py from the CC job FA-ccin2p3-18269717 (copied to
data/dipole_residual/<arm>_<axis>.json). Cells are loop_levels.py --shape-gap's: n particles and initial state, from
data/tree_levels.json. Groups: held out = the H2 reads (the multiplicity pools); training = the TR reads of training
pools in the same cells. Per group and cell, per seed: probes read, refused, no final gluon or photon, and the medians of
  r2_tD    r^2 of the control's target t on log D (the gate: below 0.1 ends the route)
  r2_rawD  r^2 of log|M|^2 = t - logF on log D (how much of D's variance the emission scales removed)
  r2_D, r2_E, r2_F  the main read: residual u on log D and the two placebos
    python3 analysis/catalog_v4/dipole_gate.py -> data/dipole_gate.json; last line the flat JSON"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"v4_dimensionless_muon": 42, "v4_dimensionless_muon_seed2": 43}
KEYS = ("r2_tD", "r2_rawD", "r2_D", "r2_E", "r2_F")


def main():
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    cell = lambda p: f"{info[p]['n']}{info[p]['initial']}" if p in info else None
    out, flat = {}, {}
    for arm, seed in ARMS.items():
        rows = {ax: json.load(open(os.path.join(HERE, "data", "dipole_residual", f"{arm}_{ax}.json"))) for ax in ("H2", "TR")}
        ho_cells = {cell(r["probe"]) for r in rows["H2"]}
        for grp, ax in (("heldout", "H2"), ("train", "TR")):
            sel = [r for r in rows[ax] if cell(r["probe"]) in ho_cells and int(cell(r["probe"])[0]) >= 5]
            by = {"multiplicity cells": sel}
            for c in sorted(ho_cells):    # every held-out multiplicity pool is 2->4 (6 particles); the 5ee, 5bb and
                # 5qg the design named first hold none (data/shape_gap.json: H2 count 0 in every 5-particle cell)
                by[c] = [r for r in sel if cell(r["probe"]) == c]
            for c, rs in by.items():
                ok = [r for r in rs if "r2_D" in r]
                O = {"n_read": len(ok), "n_refused": sum("error" in r for r in rs),
                     "n_no_emitter": sum(bool(r.get("no_emitter")) for r in rs)}
                for k in KEYS:
                    O[f"median_{k}"] = round(float(np.median([r[k] for r in ok])), 3) if ok else None
                out.setdefault(c, {}).setdefault(grp, {})[seed] = O
                for k, v in O.items():
                    flat[f"dip.{c.replace(' ', '_')}.{grp}.s{seed}.{k}"] = v
    gate = {c: all((out[c]["heldout"][s]["median_r2_tD"] or 0) >= 0.1 for s in ARMS.values()) for c in out}
    json.dump({"cells": out, "gate_pass": gate}, open(os.path.join(HERE, "data", "dipole_gate.json"), "w"), indent=1)
    for c, g in out.items():
        for grp, v in g.items():
            print(c, grp, {s: {k: x for k, x in o.items() if k in ("n_read", "n_refused", "n_no_emitter", "median_r2_tD", "median_r2_rawD", "median_r2_D", "median_r2_E", "median_r2_F")} for s, o in v.items()})
    print("gate pass per cell:", gate)
    print(json.dumps(flat))


if __name__ == "__main__":
    main()
