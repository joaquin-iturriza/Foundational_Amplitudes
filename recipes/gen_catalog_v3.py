#!/usr/bin/env python
"""catalog_v3 (D98, the user 2026-10-23: "design a new similarly sized catalog with true diversity, no repetitions or
relabelings, being very intentional on what is left out as heldout"): the census, stage 1, enumeration (no MadGraph).

The unit is an equality class, not a process string. Stage 1 enumerates by rule and collapses what is equal by symmetry
alone; stage 2 (--check, on CC) asks MadGraph which candidates exist and their diagrams and orders; stage 3 (numeric,
tools/equal_processes.py over the survivors) merges what is equal up to a constant and a relabelling but no symmetry
below sees (e+e- > X = u u~ > X for colourless X, ...). The class count after stage 3 is the catalog's size.

Particles: the SM at MadGraph's `sm` (u, d, s, c, e, mu, the neutrinos massless; b 4.7, t 172.5, tau 1.777 GeV).
Beams: e+e-, every quark-antiquark and quark-quark pair of u, d, s, c, b, every quark-gluon pair, gg.
Symmetries collapsed in stage 1: the order of the beams and the swap of the first and second generations (u<->c, d<->s,
e<->mu, ve<->vm) applied to the whole process, exact in `sm` (massless, Yukawa-free first two generations, diagonal CKM,
flavour-blind default cuts), and charge conjugation, which is exact at 2->2 and from 2->3 on only up to the parity-odd
epsilon(p_i, p_j, p_k, p_l) terms the widths' absorptive parts allow (CP holds, so the conjugate equals the process at
parity-mirrored momenta; the augmentation has no parity flip): a small approximation, stated with the census. Conservation filters: electric charge, baryon number, the three lepton numbers.
Layers:
  2->2, 2->3  exhaustive over the beams and every final multiset of SM particles (MadGraph decides existence)
  2->4        every 2->3 candidate + g or a (rule), the four-fermion finals of e+e- and q q~, the multi-boson finals
              with quartic vertices (as catalog_v2); all of them go to --check, the selection comes after the census
  one loop    [virt=QCD] of every coloured 2->2 and 2->3 tree; loop-induced candidates listed for certification
  variants    alpha_s(M_Z) in {0.09, 0.15} at far points on a few families; the heavy-quark layer (the top's quantum
              numbers at m_Q in {50, 300, 600} GeV, the physical top held out between them, H3) is chosen by
              recipes/select_catalog_v3.py over the census classes
  BSM         the user (D98): "at least one BSM process that can be encoded with our current particle embedding": the
              two non-SM models the table already has rows for, DMsimp_s_spin1 (a Z') and the 2HDM (charged and neutral
              scalars), s-channel, pair and associated production; existence decided by stage 2
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


def enumerate_candidates():
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
    # the heavy-quark layer is chosen over the census classes by recipes/select_catalog_v3.py (top classes at every
    # multiplicity, 2->4 included, D100), not here
    for c in fam:
        if c["layer"] == 2 and any(P[p][3] for p in c["beams"] + c["finals"]):
            for a in (0.09, 0.15):
                variants.append(dict(c, layer="scan", name=c["name"] + f"__as{int(a * 1000):03d}", physics={"alpha_s": a},
                                     why="alpha_s far point"))
    # BSM: the two non-SM models the particle table already encodes (particle_ids.BSM_PARTICLE_PROPERTIES, the
    # generator's point mg5_pipeline_final.NONSM_POINT): DMsimp_s_spin1's vector mediator Y1 (a Z', 500 GeV) and the
    # 2HDM's scalars (H+ 200, h2 250, h3 300 GeV); no new row, no new feature
    B = (("DMsimp_s_spin1", "e+ e- > mu+ mu- QED<=2 DMV<=2", "zp_ee_mumu"), ("DMsimp_s_spin1", "u u~ > mu+ mu- QED<=2 DMV<=2", "zp_uux_mumu"),
         ("DMsimp_s_spin1", "e+ e- > u u~ QED<=2 DMV<=2", "zp_ee_uux"), ("DMsimp_s_spin1", "e+ e- > y1 a QED<=1 DMV<=1", "zp_ee_y1a"),
         ("DMsimp_s_spin1", "u u~ > y1 g DMV<=1", "zp_uux_y1g"), ("DMsimp_s_spin1", "u u~ > y1 z QED<=1 DMV<=1", "zp_uux_y1z"),
         ("2HDM", "e+ e- > h+ h-", "thdm_ee_hphm"), ("2HDM", "e+ e- > h2 h3", "thdm_ee_h2h3"), ("2HDM", "u u~ > h+ h-", "thdm_uux_hphm"),
         ("2HDM", "u d~ > h+ h2", "thdm_udx_hph2"), ("2HDM", "e+ e- > h+ h- a", "thdm_ee_hphma"), ("2HDM", "u u~ > h2 h3 g", "thdm_uux_h2h3g"))
    bsm = [{"layer": "bsm", "model": m, "name": "v3bsm_" + n, "generate": "generate " + g, "why": m,
            "beams": g.split(" > ")[0].split(), "finals": [t for t in g.split(" > ")[1].split() if "<=" not in t]} for m, g, n in B]
    return trees, loops, induced, variants, bsm, n_tree


OUT_EXT = os.path.join(ROOT, "recipes", "catalog_v3_candidates_ext.json")


def extend24(checked):
    """D100 (the user, 2026-10-23: "extend 2 to 4 to get 1500"): more 2->4 candidates from the stage-2 survivors, so the
    catalog reaches ~1500 distinct classes with functions rather than variants: every existing 2->3 class plus a Z or a Higgs (a
    single W never conserves the charge), and every existing 2->2 class plus a quark-antiquark pair (u, d, b) or a lepton pair (e, ve).
    Candidates already enumerated (any layer) are skipped by their symmetry key."""
    old = {c["key"] for c in json.load(open(OUT))["trees"]}
    seen, out = set(old), []
    for c in json.load(open(checked)):
        if not c.get("exists") or c["layer"] not in (2, 3):
            continue
        adds = ([["z"], ["h"]] if c["layer"] == 3 else
                [["u", "u~"], ["d", "d~"], ["b", "b~"], ["e+", "e-"], ["ve", "ve~"]])
        for add in adds:
            ini, fin = tuple(c["beams"]), tuple(sorted(c["finals"] + add))
            if not conserved(ini, fin):
                continue
            k = key(ini, fin)
            if k in seen:
                continue
            seen.add(k)
            out.append({"layer": 4, "beams": list(ini), "finals": list(fin), "why": c["name"] + "+" + "".join(add),
                        "key": k, "name": name(ini, fin), "generate": f"generate {' '.join(ini)} > {' '.join(fin)}"})
    json.dump({"trees": out, "bsm": []}, open(OUT_EXT, "w"), indent=0)
    print(f"wrote {OUT_EXT}: {len(out)} new 2->4 candidates "
          f"({dict(Counter(next(s for s in ('+e+e-', '+veve~', '+uu~', '+dd~', '+bb~', '+z', '+h') if c['why'].endswith(s)) for c in out))})")


def check(workers, out, only_bsm=False, cands_path=None):
    """stage 2 (CC, CPU job scripts/job_catalog_v3_census.sh): one MadGraph `display processes` per tree and BSM
    candidate (catalog_v2's mg5_check: existence, diagram count, WEIGHTED); one-loop candidates are checked only once the
    loop layer is selected (MadLoop is far slower)"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from gen_catalog_v2 import mg5_check
    cands = json.load(open(cands_path or OUT))
    todo = [dict(c) for c in (cands["bsm"] if only_bsm else cands["trees"] + cands["bsm"])]
    for c in todo:
        c.setdefault("model", "sm")
    res = mg5_check(todo, workers=workers)
    json.dump(res, open(out, "w"))
    ok = [c for c in res if c["exists"]]
    print(f"CATALOG_V3_CHECK {len(res)} candidates, {len(ok)} exist; by layer "
          f"{dict(sorted(Counter(str(c['layer']) for c in ok).items()))}; errors "
          f"{Counter(c['error'][:40] for c in res if not c['exists']).most_common(5)}")


MASS = {5: 4.7, 6: 172.5, 15: 1.777, 23: 91.1880, 24: 80.419, 25: 125.0}
OUT_PROC = os.path.join(ROOT, "recipes", "catalog_v3_processes.yaml")
OUT_CENSUS = os.path.join(ROOT, "recipes", "catalog_v3_census.yaml")


def write(*checked):
    """stage 2 -> the tree entries (recipes/catalog_v3_processes.yaml, loaded by mg5_pipeline_final._load_catalog_v3 at
    import) and the census recipe the numeric equality check reads (recipes/catalog_v3_census.yaml: every tree that
    exists, its sqrt(s) window; no events are generated from it)"""
    import sys
    import yaml
    sys.path.insert(0, ROOT)
    import mg5_pipeline_final as mg
    procs, lines = {}, []
    for c in (c for f in checked for c in json.load(open(f))):
        if not c.get("exists") or c["layer"] not in (2, 3, 4, 5):
            continue
        pdg_b, pdg_f = mg.generate_slot_pdgs(c["generate"])
        if pdg_b[1] == -pdg_b[0] and pdg_b[0] < 0:
            pdg_b = [pdg_b[1], pdg_b[0]]
        n = len(pdg_f); W = c["weighted"]; a = 2 * n - W if W is not None else None
        e = {"mg5_generate": [c["generate"]], "nfinal": n, "pdg_ids": pdg_b + pdg_f,
             "m_finals": [MASS.get(abs(p), 0.0) for p in pdg_f], "run_card_patches": {"lpp1": "0", "lpp2": "0"},
             "param_card_patches": (dict(mg.TOP_PATCHES) if 6 in map(abs, pdg_f + pdg_b) else {}),
             "layer": c["layer"], "why": c["why"], "diagrams": c["diagrams"], "v3_key": c["key"]}
        if a:
            e["alphas_power"] = a
        procs[c["name"]] = e
        lo = int(round(max(1.05 * (sum(e["m_finals"]) + 0.0), 25.0)))
        lines.append(f"  - {{name: {c['name']}, sqrts: [{max(lo, 25)}, 1000], n_train: 0, n_val: 0, n_test: 0}}")
    yaml.safe_dump({"processes": procs}, open(OUT_PROC, "w"), sort_keys=True)
    open(OUT_CENSUS, "w").write("# catalog_v3 census (recipes/gen_catalog_v3.py --write): every tree candidate MadGraph "
                                "generates, for the numeric equality check; no events\nprocesses:\n" + "\n".join(lines) + "\n")
    print(f"wrote {OUT_PROC} ({len(procs)} trees) and {OUT_CENSUS}")


OUT_P5 = os.path.join(ROOT, "recipes", "catalog_v3_candidates_p5.json")
COLOURED = {"u", "u~", "d", "d~", "s", "s~", "c", "c~", "b", "b~", "g"}


def stage3(n_p5=40, n_as=7, seed=0):
    """D102, stage 3 (trees only; the one-loop layer and the loop-induced classes are stage 2, MadLoop):
      2->5 probes (H2): --n-p5 training 2->4 classes without a top, drawn per beam, plus a g when anything external is
              coloured, else a photon; their MadGraph check (--check --cands catalog_v3_candidates_p5.json) and entries
              (--write) as stage 2 of the census did
      alpha_s points (H6): --n-as training 2->2 / 2->3 classes with alpha_s in their order, drawn per beam, at
              alpha_s(M_Z) = 0.09 and 0.15 (train) and 0.13 (held out between them)
    Writes recipes/catalog_v3_candidates_p5.json and recipes/catalog_v3_alphas.json (name -> base, alpha_s, role)."""
    import random
    import yaml
    rng = random.Random(seed)
    Pr = yaml.load(open(os.path.join(ROOT, "recipes", "catalog_v3_processes.yaml")), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["processes"]
    roles = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"]
    tr = sorted(n for n, v in roles.items() if v["role"] == "train")
    parse = lambda n: [t.split() for t in Pr[n]["mg5_generate"][0].replace("generate", "").split(">")]
    beam = lambda n: "ee" if parse(n)[0] == ["e+", "e-"] else ("gg" if parse(n)[0] == ["g", "g"] else
                                                                 ("qg" if "g" in parse(n)[0] else "qq"))
    by = {}
    for n in tr:
        if Pr[n]["layer"] == 4 and not any(abs(p) == 6 for p in Pr[n]["pdg_ids"]):
            by.setdefault(beam(n), []).append(n)
    tot = sum(map(len, by.values()))
    pick = [n for b, ns in sorted(by.items()) for n in rng.sample(ns, max(1, round(n_p5 * len(ns) / tot)))]
    out, seen = [], set()
    for n in pick:
        ini, fin = parse(n)
        add = "g" if COLOURED & set(ini + fin) else "a"
        f5 = sorted(fin + [add]); k = key(ini, f5)
        if k in seen:
            continue
        seen.add(k)
        out.append({"layer": 5, "beams": ini, "finals": f5, "why": n + "+" + add, "key": k, "name": name(ini, f5),
                    "generate": f"generate {' '.join(ini)} > {' '.join(f5)}"})
    json.dump({"trees": out, "bsm": []}, open(OUT_P5, "w"), indent=0)
    by = {}
    for n in tr:
        if Pr[n]["layer"] in (2, 3) and Pr[n].get("alphas_power", 0) >= 1 and not any(abs(p) == 6 for p in Pr[n]["pdg_ids"]):
            by.setdefault(beam(n), []).append(n)
    tot = sum(map(len, by.values()))
    fams = [n for b, ns in sorted(by.items()) for n in rng.sample(ns, max(1, round(n_as * len(ns) / tot)))]
    al = {f"{n}__as{int(round(1000 * a)):03d}": {"base": n, "alpha_s": a, "role": "heldout" if a == 0.13 else "train",
                                                 "axis": "H6" if a == 0.13 else None}
          for n in sorted(fams) for a in (0.09, 0.13, 0.15)}
    json.dump(al, open(os.path.join(ROOT, "recipes", "catalog_v3_alphas.json"), "w"), indent=0)
    print(f"stage 3: {len(out)} 2->5 candidates ({dict(Counter(c['why'][-1] for c in out))}) -> {OUT_P5}; "
          f"{len(al)} alpha_s pools on {len(fams)} families -> recipes/catalog_v3_alphas.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs="+", default=None, help="the stage-2 outputs (checked.json, checked_ext.json)")
    ap.add_argument("--stage3", action="store_true", help="choose the 2->5 probes and the alpha_s points (D102)")
    ap.add_argument("--extend24", default=None, help="stage-2 output to extend the 2->4 layer from (D100)")
    ap.add_argument("--cands", default=None, help="--check this candidate file instead of catalog_v3_candidates.json")
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only-bsm", action="store_true", help="--check the BSM candidates only")
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_catalog_v3_census", "checked.json"))
    a = ap.parse_args()
    if a.enumerate:
        trees, loops, induced, variants, bsm, n_tree = enumerate_candidates()
        json.dump({"trees": trees, "loops": loops, "loop_induced": induced, "variants": variants, "bsm": bsm},
                  open(OUT, "w"), indent=0)
        print(f"trees after the conservation filters and the symmetry collapse: {dict(sorted(n_tree.items()))}, "
              f"{len(trees)} in all")
        print(f"beams by layer: {dict(Counter((c['layer'], 'ee' if c['beams'] == ['e+', 'e-'] else ('gg' if c['beams'] == ['g', 'g'] else ('qg' if 'g' in c['beams'] else 'qq'))) for c in trees))}")
        print(f"one-loop QCD candidates {len(loops)}, loop-induced {len(induced)}, variants {len(variants)}, BSM {len(bsm)}")
        print("wrote", OUT)
    if a.check:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        check(a.workers, a.out, a.only_bsm, a.cands)
    if a.extend24:
        extend24(a.extend24)
    if a.stage3:
        stage3()
    if a.write:
        write(*a.write)


if __name__ == "__main__":
    main()
