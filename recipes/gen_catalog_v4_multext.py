#!/usr/bin/env python
"""The multiplicity extension's candidates (plan v3_prep_explore, option (b), D216): new 2->4 tree processes in the
held-out multiplicity cells, for a training arm against the control pair.

catalog_v3's census (recipes/gen_catalog_v3.py) is exhaustive over its enumeration: every 2->2 and 2->3, the +g/+a
children of the 2->3, the D100 +z/+h and +pair children, the four-fermion and multi-boson lists. Every 2->4 it
enumerated that MadGraph accepts is a catalog class with a role. New 2->4 processes are therefore the conserved 2->4
over the same BEAMS x FINALS whose symmetry key (charge conjugation, beam order, generation swap) the census never
enumerated (accepted or rejected), with:
  - no top and no tau or tau neutrino (the physical top is H3; the tau the removal twin's never-seen particle),
  - quark flavour conserved per generation (the sm model has a diagonal CKM),
  - an initial state in one of the held-out multiplicity cells (bb, bg, bq, qq, qg, ee, gg: tree_levels.json `initial`).
What MadGraph rejects is dropped by the census check (CC, job_catalog_v3_census.sh with this file), what shares a
class with a held-out probe by the class check (tools/equal_processes.py), both before any pool is built.
    python recipes/gen_catalog_v4_multext.py   -> recipes/catalog_v4_multext_candidates.json
"""
import itertools
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recipes"))
import gen_catalog_v3 as g3  # noqa: E402

OUT = os.path.join(ROOT, "recipes", "catalog_v4_multext_candidates.json")
CELLS = ("bb", "bg", "bq", "qq", "qg", "ee", "gg")
FLAV = {"u": 1, "c": 2, "t": 3, "d": 1, "s": 2, "b": 3}


def initial(ini):
    """the cell's initial-state label as analysis/catalog_v4/data/tree_levels.json writes it"""
    kind = sorted("e" if p in ("e+", "e-") else "g" if p == "g" else "b" if p.rstrip("~") == "b" else "q" for p in ini)
    lab = "".join(kind)
    return {"gq": "qg", "bq": "bq", "bg": "bg"}.get(lab, lab)


def flavour_ok(ini, fin):
    def net(side):
        n = Counter()
        for p in side:
            f = p.rstrip("~")
            if f in FLAV:
                n[f] += -1 if p.endswith("~") else 1
        return n
    a, b = net(ini), net(fin)
    return all(a[f] == b[f] for f in FLAV)


def main():
    old = set()
    for f in ("catalog_v3_candidates.json", "catalog_v3_candidates_ext.json"):
        old |= {c["key"] for c in json.load(open(os.path.join(ROOT, "recipes", f)))["trees"]}
    fins = [p for p in g3.FINALS if p not in ("t", "t~") and not p.startswith(("ta", "vt"))]
    out, seen = [], set()
    for ini in g3.BEAMS:
        cell = initial(ini)
        if cell not in CELLS:
            continue
        for fin in itertools.combinations_with_replacement(fins, 4):
            if not g3.conserved(ini, fin) or not flavour_ok(ini, fin):
                continue
            k = g3.key(ini, fin)
            if k in old or k in seen:
                continue
            seen.add(k)
            out.append({"layer": 4, "beams": list(ini), "finals": list(fin), "why": "multext", "key": k, "cell": "6" + cell,
                        "name": g3.name(ini, fin), "generate": f"generate {' '.join(ini)} > {' '.join(fin)}"})
    json.dump({"trees": out, "bsm": []}, open(OUT, "w"), indent=0)
    print(f"wrote {OUT}: {len(out)} candidates, per cell {dict(sorted(Counter(c['cell'] for c in out).items()))}")


if __name__ == "__main__":
    main()
