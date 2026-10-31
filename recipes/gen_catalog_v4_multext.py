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
    python recipes/gen_catalog_v4_multext.py --write analysis/catalog_v4/data/multext_checked.json
        -> the census check's accepted candidates as catalog_v4 tree entries (recipes/catalog_v4_multext_processes.yaml,
           loaded by mg5_pipeline_final._load_catalog_v4), their generation recipe (recipes/catalog_v4_multext.yaml,
           1e5 / 1e4 / 1e4, the v4 windows and mixture) and the class check's recipe
           (recipes/catalog_v4_multext_classcheck.yaml: the new trees and the held-out 2->4 trees of H1, H2 and H3 in their (multiplicity, final masses) groups,
           no events; tools/equal_processes.py --recipes)
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recipes"))
import gen_catalog_v3 as g3  # noqa: E402
import gen_catalog_v4 as g4  # noqa: E402

OUT = os.path.join(ROOT, "recipes", "catalog_v4_multext_candidates.json")
OUT_PROC = os.path.join(ROOT, "recipes", "catalog_v4_multext_processes.yaml")
OUT_GEN = os.path.join(ROOT, "recipes", "catalog_v4_multext.yaml")
OUT_CLS = os.path.join(ROOT, "recipes", "catalog_v4_multext_classcheck.yaml")
HELD = ("H1", "H2", "H3")
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


def family_children(cands, mg):
    """{key: family} of the candidates that are a child of one of catalog_v4's held-out families
    (recipes/catalog_v4_roles.json held_out_families, class representatives): removing one final g, a, z or h, or one
    fermion-antifermion pair, leaves a tree in that family's class (select_catalog_v4.ADDS, with the generation-swapped
    pairs: the census's child keys are canonical under the generation swap)"""
    roles = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_roles.json")))
    fams = set(roles["held_out_families"])
    cls = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "census_classes.json")))["classes"]
    rep_of = {m: r for r, ms in cls.items() for m in ms}
    by_key = {e["v3_key"]: rep_of.get(n, n) for n, e in mg.PROCESSES.items() if e.get("_v4") and e.get("v3_key")}
    out = {}
    for c in cands:
        ini, fin = c["beams"], list(c["finals"])
        drops = [[i] for i, p in enumerate(fin) if p in ("g", "a", "z", "h")]
        drops += [[i, j] for i, j in itertools.combinations(range(len(fin)), 2) if g3.CONJ.get(fin[i]) == fin[j]
                  and fin[i] not in ("w+", "w-", "g", "a", "z", "h")]
        for d in drops:
            r = by_key.get(g3.key(ini, [p for k, p in enumerate(fin) if k not in d]))
            if r in fams:
                out[c["key"]] = r
                break
    return out


def write(checked):
    """the accepted candidates -> v4 entries, their recipe and the class check's recipe (the module docstring)"""
    import yaml
    sys.path.insert(0, ROOT)
    import mg5_pipeline_final as mg
    ok = [c for c in json.load(open(checked)) if c["exists"]]
    fam = family_children(ok, mg)
    procs = {}
    for c in ok:
        if c["key"] in fam:
            continue
        n4 = g4.v4name(c["name"])
        assert n4 not in mg.PROCESSES or mg.PROCESSES[n4].get("multext"), f"{n4} collides with an existing entry"
        procs[n4] = dict(g4.tree_entry(c["name"], g3.tree_entry(c)), multext=c["cell"])
    hdr = ("# catalog_v4's multiplicity extension (plan v3_prep_explore, D216): generated by "
           "recipes/gen_catalog_v4_multext.py --write; do not edit\n")
    with open(OUT_PROC, "w") as f:
        f.write(hdr); yaml.safe_dump({"processes": procs}, f, sort_keys=True)
    gen = [{"name": n, "sqrts": g4.window(e), "n_train": 100000, "n_val": 10000, "n_test": 10000}
           for n, e in sorted(procs.items())]
    with open(OUT_GEN, "w") as f:
        f.write(hdr + f"# {len(gen)} new 2->4 trees in the held-out multiplicity cells, 1e5 / 1e4 / 1e4, the v4 windows "
                "and mixture\n")
        yaml.safe_dump({"sampling": dict(mg.V4_SAMPLING), "processes": gen}, f, sort_keys=False)
    # equal_processes compares trees only inside one (multiplicity, final masses) group: the held-out trees of the new
    # trees' groups are all it can pair them with
    group = lambda e: (int(e["nfinal"]), tuple(sorted(float(m) for m in e["m_finals"])))
    groups = {group(e) for e in procs.values()}
    held = []
    for h in HELD:
        for e in g4.load(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{h}.yaml"))["processes"]:
            p = mg.PROCESSES.get(e["name"])
            if (p and not p.get("virt") and int(p.get("nfinal", 0)) == 4 and e["name"] not in {x["name"] for x in held}
                    and group(p) in groups):
                held.append({"name": e["name"], "sqrts": list(e["sqrts"]), "n_train": 0, "n_val": 0, "n_test": 0})
    cls = [dict(g, n_train=0, n_val=0, n_test=0) for g in gen] + held
    with open(OUT_CLS, "w") as f:
        f.write(hdr + f"# the class check: the {len(gen)} new trees and the {len(held)} held-out 2->4 trees of "
                f"{', '.join(HELD)}; no events\n")
        yaml.safe_dump({"processes": cls}, f, sort_keys=False)
    print(f"wrote {OUT_PROC} ({len(procs)} of {len(json.load(open(checked)))} candidates accepted), per cell "
          f"{dict(sorted(Counter(p['multext'] for p in procs.values()).items()))}; dropped as children of a held-out "
          f"family {len(fam)}: {fam}; {OUT_GEN}; {OUT_CLS} "
          f"({len(held)} held-out 2->4 trees)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="the census check's output (checked.json)")
    a = ap.parse_args()
    if a.write:
        return write(a.write)
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
