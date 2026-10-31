"""The dipole read's gate and main read per cell (plan v3_prep_explore, step 6; design passed 2026-10-30 22:28).

Input: the per-probe rows of analysis/catalog_v4/dipole_residual.py from the CC job FA-ccin2p3-18269717 (copied to
data/dipole_residual/<arm>_<axis>.json). Cells are loop_levels.py --shape-gap's: n particles and initial state, from
data/tree_levels.json. Groups: held out = the H2 reads (the multiplicity pools); training = the TR reads of training
pools in the same cells. Per group and cell, per seed: probes read, refused, no final gluon or photon, and the medians of
  r2_tD    r^2 of the control's target t on log D (the gate: below 0.1 ends the route)
  r2_rawD  r^2 of log|M|^2 = t - logF on log D (how much of D's variance the emission scales removed)
  r2_D, r2_E, r2_F  the main read: residual u on log D and the two placebos
    python3 analysis/catalog_v4/dipole_gate.py -> data/dipole_gate.json; last line the flat JSON"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"v4_dimensionless_muon": 42, "v4_dimensionless_muon_seed2": 43}
KEYS = ("r2_tD", "r2_rawD", "r2_D", "r2_E", "r2_F")
# the propagator-product read (--prop, FA-ccin2p3-71a96ca2; design passed 2026-10-31): gate r2_tP, ceiling, the mean
# read eta2 of u on log P against the placebos' eta2 (E, logF, log D), the size read r2 of log u^2 on log P
PKEYS = ("r2_tP", "r2_rawP", "ceiling", "eta2_P", "eta2_E", "eta2_F", "eta2_D", "r2size_P", "r2size_E", "r2size_F",
         "r2size_D", "tail_lo_P", "tail_hi_P", "r2_P")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", action="store_true", help="the propagator-product read (data/prop_residual)")
    A = ap.parse_args()
    keys, sub, done, outf = ((PKEYS, "prop_residual", "r2_P", "prop_gate.json") if A.prop
                             else (KEYS, "dipole_residual", "r2_D", "dipole_gate.json"))
    rng = np.random.default_rng(0)
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    cell = lambda p: f"{info[p]['n']}{info[p]['initial']}" if p in info else None
    out, flat = {}, {}
    for arm, seed in ARMS.items():
        rows = {ax: json.load(open(os.path.join(HERE, "data", sub, f"{arm}_{ax}.json"))) for ax in ("H2", "TR")}
        ho_cells = {cell(r["probe"]) for r in rows["H2"]}
        for grp, ax in (("heldout", "H2"), ("train", "TR")):
            sel = [r for r in rows[ax] if cell(r["probe"]) in ho_cells and int(cell(r["probe"])[0]) >= 5]
            by = {"multiplicity cells": sel}
            for c in sorted(ho_cells):    # every held-out multiplicity pool is 2->4 (6 particles); the 5ee, 5bb and
                # 5qg the design named first hold none (data/shape_gap.json: H2 count 0 in every 5-particle cell)
                by[c] = [r for r in sel if cell(r["probe"]) == c]
            for c, rs in by.items():
                ok = [r for r in rs if done in r]
                O = {"n_read": len(ok), "n_refused": sum("error" in r for r in rs),
                     "n_no_emitter": sum(bool(r.get("no_emitter")) for r in rs)}
                for k in keys:
                    v = [r[k] for r in ok if k in r]
                    O[f"median_{k}"] = round(float(np.median(v)), 3) if v else None
                if A.prop and ok:   # the 16-84% bootstrap of the median over pools, 1000 draws
                    for k in ("eta2_P", "r2size_P"):
                        v = np.array([r[k] for r in ok])
                        m = [np.median(rng.choice(v, len(v))) for _ in range(1000)]
                        O[f"boot_{k}"] = [round(float(np.quantile(m, 0.16)), 3), round(float(np.quantile(m, 0.84)), 3)]
                out.setdefault(c, {}).setdefault(grp, {})[seed] = O
                for k, v in O.items():
                    flat[f"dip.{c.replace(' ', '_')}.{grp}.s{seed}.{k}"] = v
    gk = "median_r2_tP" if A.prop else "median_r2_tD"
    gate = {c: all((out[c]["heldout"][s][gk] or 0) >= 0.1 for s in ARMS.values()) for c in out}
    res = {"cells": out, "gate_pass": gate}
    if A.prop:
        res["verdict"] = verdict(out["multiplicity cells"])
        print("verdict over the multiplicity cells:", json.dumps(res["verdict"]))
    json.dump(res, open(os.path.join(HERE, "data", outf), "w"), indent=1)
    for c, g in out.items():
        for grp, v in g.items():
            print(c, grp, {s: {k: x for k, x in o.items() if k in ("n_read", "n_refused", "n_no_emitter")
                               or k.startswith("median_")} for s, o in v.items()})
    print("gate pass per cell:", gate)
    print(json.dumps(flat))


def verdict(g):
    """The design's rule over all multiplicity cells, both seeds (the margins: the seed difference of the held-out
    median and the bootstrap half-range)."""
    H, T = g["heldout"], g["train"]
    S = list(ARMS.values())
    out = {"gate": all(H[s]["median_r2_tP"] >= 0.1 for s in S),
           "ceiling_readable": all(H[s]["median_ceiling"] >= 0.05 for s in S)}
    for read, k, bar in (("mean", "eta2", None), ("size", "r2size", 0.1)):
        per = []
        for s in S:
            h, t = H[s][f"median_{k}_P"], T[s][f"median_{k}_P"]
            b = bar if bar is not None else 0.5 * H[s]["median_ceiling"]
            plac = max(H[s][f"median_{k}_{x}"] or 0 for x in ("E", "F", "D"))
            margin = max(abs(H[S[0]][f"median_{k}_P"] - H[S[1]][f"median_{k}_P"]),
                         (H[s][f"boot_{k}_P"][1] - H[s][f"boot_{k}_P"][0]) / 2)
            per.append({"heldout": h, "train": t, "bar": round(b, 3), "max_placebo": plac, "margin": round(margin, 4),
                        "yes": h >= b and h >= 2 * t and h - plac > margin and h - 2 * t > margin,
                        "below_half_bar": h < 0.5 * b, "not_above": h <= t or h <= plac})
        out[read] = per
    yes = out["gate"] and out["ceiling_readable"] and any(all(p["yes"] for p in out[r]) for r in ("mean", "size"))
    no = all(any(p["below_half_bar"] or p["not_above"] for p in out[r]) for r in ("mean", "size"))
    out["outcome"] = ("gate fails" if not out["gate"] else "unreadable" if not out["ceiling_readable"]
                      else "yes" if yes else "no" if no else "between")
    return out


if __name__ == "__main__":
    main()
