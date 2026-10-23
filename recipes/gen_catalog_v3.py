#!/usr/bin/env python
"""catalog_v3 (D98, the user 2026-10-23: "design a new similarly sized catalog with true diversity, no repetitions or
relabelings, being very intentional on what is left out as heldout"): the census, stage 1, enumeration (no MadGraph).

The unit is an equality class, not a process string. Stage 1 enumerates by rule and collapses what is equal by symmetry
alone; stage 2 (--check, on CC) asks MadGraph which candidates exist and their diagrams and orders; stage 3 (numeric,
tools/equal_processes.py over the survivors) merges what is equal up to a constant and a relabelling but no symmetry
below sees (e+e- > X = u u~ > X for colourless X, ...). The class count after stage 3 is the catalog's size.

Particles: the SM at MadGraph's `sm` (u, d, s, c, e, mu, the neutrinos massless; b 4.7, t 172.5, tau 1.777 GeV).
Beams: e+e-, every quark-antiquark and quark-quark pair of u, d, s, c, b, every quark-gluon pair, gg.
Symmetries collapsed in stage 1 (exact in `sm`, diagonal CKM, massless first two generations): charge conjugation, the
order of the beams, and the swap of the first and second generations (u<->c, d<->s, e<->mu, ve<->vm) applied to the
whole process. Conservation filters: electric charge, baryon number, the three lepton numbers.
Layers:
  2->2, 2->3  exhaustive over the beams and every final multiset of SM particles (MadGraph decides existence)
  2->4        every 2->3 survivor + g or a (rule), the four-fermion finals of e+e- and q q~, the multi-boson finals
              with quartic vertices (as catalog_v2); --max24 caps the layer
  one loop    [virt=QCD] of every coloured 2->2 and 2->3 tree; loop-induced candidates listed for certification
  variants    alpha_s(M_Z) in {0.09, 0.15}, M_Z, M_H at far points on a few families; heavy quarks with the top's quantum
              numbers at m_Q in {50, 300, 600} GeV on the top family (the physical top held out between them, H3)
  BSM         processes of a model whose new particles the particle table can encode (spin, charge, colour, mass, B, L):
              a list here, its existence decided by stage 2 (the model must be on CC)
Output: recipes/catalog_v3_candidates.json and the counts printed.
    python recipes/gen_catalog_v3.py --enumerate
"""
import argparse, itertools, json, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "recipes", "catalog_v3_candidates.json")

# name: (3 x charge, 3 x baryon number, (L_e, L_mu, L_tau), coloured)
P = {}
for q, c, g in (("u", 2, 1), ("c", 2, 2), ("t", 2, 3), ("d", -1, 1), ("s", -1, 2), ("b", -1, 3)):
    P[q] = (c, 1, (0, 0, 0), True); P[q + "~"] = (-c, -1, (0, 0, 0), True)
for i, (l, v) in enumerate((("e", "ve"), ("mu", "vm"), ("ta", "vt"))):
    L = tuple(int(j == i) for j in range(3)); Lb = tuple(-x for x in L)
    P[l + "-"] = (-3, 0, L, False); P[l + "+"] = (3, 0, Lb, False)
    P[v] = (0, 0, L, False); P[v + "~"] = (0, 0, Lb, False)
for b, c in (("g", 0), ("a", 0), ("z", 0), ("h", 0)):
    P[b] = (c, 0, (0, 0, 0), b == "g")
P["w+"] = (3, 0, (0, 0, 0), False); P["w-"] = (-3, 0, (0, 0, 0), False)
FINALS = sorted(P)
CONJ = {}
for k in P:
    if k in ("g", "a", "z", "h"):
        CONJ[k] = k
    elif k.endswith("~"):
        CONJ[k] = k[:-1]
    elif k[-1] in "+-" and k not in ("w+", "w-"):
        CONJ[k] = k[:-1] + ("-" if k[-1] == "+" else "+")
    elif k in ("w+", "w-"):
        CONJ[k] = "w-" if k == "w+" else "w+"
    else:
        CONJ[k] = k + "~"
GEN = {"u": "c", "c": "u", "d": "s", "s": "d", "e": "mu", "mu": "e", "ve": "vm", "vm": "ve"}


def gen_swap(p):
    for base in sorted(GEN, key=len, reverse=True):
        if p == base or p == base + "~" or (p[:-1] == base and p[-1] in "+-"):
            return GEN[base] + p[len(base):]
    return p


QUARKS = ["u", "d", "s", "c", "b"]
BEAMS = [("e+", "e-")]
BEAMS += [(a, b + "~") for a in QUARKS for b in QUARKS]
BEAMS += [(a, b) for a, b in itertools.combinations_with_replacement(QUARKS, 2)]
BEAMS += [(a, "g") for a in QUARKS] + [("g", "g")]


def conserved(ini, fin):
    s = lambda side, i: sum(P[p][i] for p in side)
    if s(ini, 0) != s(fin, 0) or s(ini, 1) != s(fin, 1):
        return False
    return all(sum(P[p][2][j] for p in ini) == sum(P[p][2][j] for p in fin) for j in range(3))


def key(ini, fin):
    """canonical form under charge conjugation, beam order and the generation swap"""
    forms = []
    for f in (lambda p: p, CONJ.get):
        for g in (lambda p: p, gen_swap):
            i2 = sorted(g(f(p)) for p in ini); f2 = sorted(g(f(p)) for p in fin)
            forms.append(" ".join(i2) + " > " + " ".join(f2))
    return min(forms)


def name(ini, fin):
    t = lambda p: p.replace("~", "x").replace("+", "p").replace("-", "m")
    return "v3_" + "".join(t(p) for p in ini) + "_" + "".join(t(p) for p in fin)


def enumerate_candidates(max24):
    seen = {}

    def put(layer, ini, fin, why, extra=None):
        if not conserved(ini, fin):
            return
        k = key(ini, fin) + ("" if not extra else " " + json.dumps(extra, sort_keys=True))
        if k in seen:
            return
        seen[k] = {"layer": layer, "beams": list(ini), "finals": list(fin), "why": why, "key": k,
                   "name": name(ini, fin) + ("" if not extra else "__" + "_".join(f"{a}{b}" for a, b in sorted(extra.items()))),
                   "generate": f"generate {' '.join(ini)} > {' '.join(fin)}", **({"physics": extra} if extra else {})}
    for ini in BEAMS:
        for n in (2, 3):
            for fin in itertools.combinations_with_replacement(FINALS, n):
                put(n, ini, fin, "exhaustive")
    layer3 = [c for c in seen.values() if c["layer"] == 3]
    n24 = 0
    for c in layer3:
        for leg in ("g", "a"):
            put(4, tuple(c["beams"]), tuple(sorted(c["finals"] + [leg])), c["name"] + "+" + leg)
    pairs = [["u", "u~"], ["d", "d~"], ["b", "b~"], ["t", "t~"], ["mu+", "mu-"], ["ta+", "ta-"], ["ve", "ve~"], ["e+", "e-"]]
    for a, b in itertools.combinations_with_replacement(range(len(pairs)), 2):
        for ini in (("e+", "e-"), ("u", "u~"), ("d", "d~")):
            put(4, ini, tuple(sorted(pairs[a] + pairs[b])), "four-fermion")
    for f in (["w+", "w-", "z", "z"], ["w+", "w-", "w+", "w-"], ["w+", "w-", "a", "a"], ["z", "z", "z", "z"],
              ["w+", "w-", "z", "a"], ["z", "z", "z", "a"], ["z", "z", "a", "a"], ["w+", "w-", "z", "h"],
              ["z", "z", "h", "h"], ["w+", "w-", "h", "h"], ["z", "h", "h", "h"], ["h", "h", "h", "h"]):
        for ini in (("e+", "e-"), ("u", "u~"), ("d", "d~"), ("u", "d~"), ("g", "g")):
            put(4, ini, tuple(sorted(f)), "multi-boson")
    trees = [c for c in seen.values()]
    n_tree = Counter(c["layer"] for c in trees)
    # one loop: the QCD virtual of every coloured 2->2, 2->3 tree (MadLoop decides; certification as catalog_v2)
    loops = []
    for c in trees:
        if c["layer"] in (2, 3) and any(P[p][3] for p in c["beams"] + c["finals"]):
            loops.append(dict(c, layer=f"nlo{c['layer']}", name=c["name"] + "_nlo",
                              generate=c["generate"] + " [virt=QCD]", why="virt=QCD of " + c["name"]))
    induced = []
    for ini in (("g", "g"), ("u", "u~"), ("e+", "e-"), ("d", "d~")):
        for fin in (("h",), ("h", "h"), ("z", "z"), ("a", "a"), ("z", "a"), ("h", "g"), ("h", "h", "h"), ("z", "h"),
                    ("a", "a", "g"), ("w+", "w-")):
            if conserved(ini, fin):
                induced.append({"layer": "loop_induced", "beams": list(ini), "finals": list(fin),
                                "name": name(ini, fin) + "_loop", "generate": f"generate {' '.join(ini)} > {' '.join(fin)} [noborn=QCD]",
                                "why": "loop-induced candidate"})
    # variants: far points of the parameters, and the heavy-quark layer (the top's quantum numbers at other masses)
    variants = []
    fam = [c for c in trees if c["layer"] in (2, 3)]
    top = [c for c in fam if any(p.rstrip("~") == "t" for p in c["finals"])]
    for c in top:
        for m in (50, 300, 600):
            variants.append(dict(c, layer="heavy_quark", name=c["name"] + f"__mt{m}", physics={"m_t": m},
                                 why=f"heavy quark at {m} GeV"))
    for c in fam:
        if c["layer"] == 2 and any(P[p][3] for p in c["beams"] + c["finals"]):
            for a in (0.09, 0.15):
                variants.append(dict(c, layer="scan", name=c["name"] + f"__as{int(a * 1000):03d}", physics={"alpha_s": a},
                                     why="alpha_s far point"))
    bsm = [{"layer": "bsm", "model": "MSSM_SLHA2", "beams": list(i), "finals": list(f), "name": "v3bsm_" + n,
            "generate": f"generate {' '.join(i)} > {' '.join(f)}", "why": w}
           for i, f, n, w in ((("u", "u~"), ("ul", "ul~"), "uux_squark_pair", "scalar colour triplet, pair production"),
                              (("g", "g"), ("ul", "ul~"), "gg_squark_pair", "scalar colour triplet, gluon fusion"),
                              (("e+", "e-"), ("mur-", "mur+"), "ee_smuon_pair", "charged scalar, s-channel"),
                              (("u", "g"), ("ul", "n1"), "ug_squark_neutralino", "scalar triplet + Majorana fermion"))]
    return trees, loops, induced, variants, bsm, n_tree


def check(workers, out):
    """stage 2 (CC, CPU job scripts/job_catalog_v3_census.sh): one MadGraph `display processes` per tree and BSM
    candidate (catalog_v2's mg5_check: existence, diagram count, WEIGHTED); one-loop candidates are checked only once the
    loop layer is selected (MadLoop is far slower)"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from gen_catalog_v2 import mg5_check
    cands = json.load(open(OUT))
    todo = [dict(c, layer=c["layer"] if c["layer"] != "bsm" else "bsm") for c in cands["trees"] + cands["bsm"]]
    for c in todo:
        c.setdefault("model", "sm")
    res = mg5_check(todo, workers=workers)
    json.dump(res, open(out, "w"))
    ok = [c for c in res if c["exists"]]
    print(f"CATALOG_V3_CHECK {len(res)} candidates, {len(ok)} exist; by layer "
          f"{dict(sorted(Counter(str(c['layer']) for c in ok).items()))}; errors "
          f"{Counter(c['error'][:40] for c in res if not c['exists']).most_common(5)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_catalog_v3_census", "checked.json"))
    ap.add_argument("--max24", type=int, default=0)
    a = ap.parse_args()
    if a.enumerate:
        trees, loops, induced, variants, bsm, n_tree = enumerate_candidates(a.max24)
        json.dump({"trees": trees, "loops": loops, "loop_induced": induced, "variants": variants, "bsm": bsm},
                  open(OUT, "w"), indent=0)
        print(f"trees after the conservation filters and the symmetry collapse: {dict(sorted(n_tree.items()))}, "
              f"{len(trees)} in all")
        print(f"beams by layer: {dict(Counter((c['layer'], 'ee' if c['beams'] == ['e+', 'e-'] else ('gg' if c['beams'] == ['g', 'g'] else ('qg' if 'g' in c['beams'] else 'qq'))) for c in trees))}")
        print(f"one-loop QCD candidates {len(loops)}, loop-induced {len(induced)}, variants {len(variants)}, BSM {len(bsm)}")
        print("wrote", OUT)
    if a.check:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        check(a.workers, a.out)


if __name__ == "__main__":
    main()
