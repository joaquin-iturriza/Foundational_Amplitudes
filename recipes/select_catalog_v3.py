#!/usr/bin/env python
"""catalog_v3 (D98): the held-out selection over the census classes (analysis/catalog_v3/census_classes.json), one
representative per class, every class in exactly one role. The axes in the user's order (2026-10-23, "good in that
order"), each probe tagged with the structure it needs and its nearest training class:
  H1 new combination   ~5% of the classes of each (layer, beam) cell at random, none of whose particles, multiplicity
                       or order is missing from training (the other axes are removed first)
  H2 multiplicity      every 2->4 class whose parent (the 2->3 or 2->2 class the enumeration added a leg or a pair
                       to, from its `why`; four-fermion and multi-boson finals by final-state kind) is in a held-out
                       family: 15% of the parents that have 2->4 children, so their 2->4 level is inside the trained range but their family's 2->4 is
                       new; plus 2->5 probes (generated later, a training 2->4 plus a g or a)
  H3 particle content  the physical top (every class with an external t, mass 172.5) held out; the heavy-quark layer
                       (the top's quantum numbers at 50, 300, 600 GeV, at 2->2, 2->3 and 2->4) trains. Never-seen
                       particle: every class with an external tau or tau neutrino, held out of the removal twin only
  H4 vertex            the quartic gauge family (four or more external W, Z, a with at least two W), held out of the
                       removal twin only (with H3's Higgs)
  H5 order             a one-loop layer of --n-loops (every coloured 2->2 class, 2->3 ones drawn per beam); 15% of
                       those whose trees train held out, and those of H1's trees; one loop-induced class
  H6 parameters        alpha_s(M_Z) = 0.13 on the scanned families (trained at 0.09 and 0.15), m_Q = 172.5 = the top (H3)
  H7 energy, cuts      the training classes at 100 TeV and at p_T >= 30 GeV (no class leaves training)
Writes recipes/catalog_v3_roles.json ({class representative: role, axis, why}) and prints the counts.
    python recipes/select_catalog_v3.py [--seed 0]
"""
import argparse, json, os, random
from collections import Counter, defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EW = {23, 24, -24, 22}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h1-frac", type=float, default=0.05)
    ap.add_argument("--family-frac", type=float, default=0.15)
    ap.add_argument("--n-loops", type=int, default=150, help="size of the one-loop layer")
    ap.add_argument("--n-heavy", type=int, default=35, help="top classes in the heavy-quark layer (x3 masses)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    P = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_processes.yaml")))["processes"]
    reps = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "census_classes.json")))["representatives"]
    fin = lambda n: P[n]["pdg_ids"][2:]
    beam = lambda n: ("ee" if sorted(P[n]["pdg_ids"][:2]) == [-11, 11] else "gg" if P[n]["pdg_ids"][:2] == [21, 21]
                      else "qg" if 21 in P[n]["pdg_ids"][:2] else "qq")
    key = {n: P[n]["v3_key"] for n in reps}
    role = {}

    def put(n, r, axis, why):
        role.setdefault(n, {"role": r, "axis": axis, "why": why})

    # H3: the physical top out of every training set; H3' and H4 (removal twin): Higgs, quartic gauge family
    for n in reps:
        if any(abs(p) == 6 for p in P[n]["pdg_ids"]):
            put(n, "heldout", "H3", "external top (the heavy-quark layer brackets it)")
    # the twin's never-seen particle: the tau and its neutrino (the user 2026-10-23: another particle than the Higgs, which
    # is a quarter of training; the tau is a tenth: a charged lepton with a mass and a Higgs coupling next to massless e, mu)
    twin_out = [n for n in reps if any(abs(p) in (15, 16) for p in P[n]["pdg_ids"])
                or (sum(abs(p) in (23, 24, 22) for p in fin(n)) >= 4 and sum(abs(p) == 24 for p in fin(n)) >= 2)]
    # H2: families. The parent of a 2->4 class is what the enumeration added a leg (or a pair) to, recorded in its
    # `why` ("<parent>+g", "+a" at enumeration; "+z", "+h", "+uu~", "+dd~", "+bb~", "+e+e-", "+veve~" in the D100
    # extension), mapped to the parent's class; four-fermion and multi-boson finals have none ("other:<beam>")
    cls = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "census_classes.json")))["classes"]
    rep_of = {m: n for ms in cls.values() for n in ms if n in set(reps) for m in ms}
    members = defaultdict(list)
    for m, n in rep_of.items():
        members[n].append(m)
    ADDS = ("+e+e-", "+veve~", "+uu~", "+dd~", "+bb~", "+g", "+a", "+z", "+h")

    def parents(r):
        """the parent classes of class r over ALL its members' `why` (members of one class can come from different
        parents; a held-out family must keep no training child)"""
        out = set()
        for m in members[r]:
            why = P[m].get("why", "")
            for s_ in ADDS:
                if why.endswith(s_) and why[: -len(s_)] in rep_of:
                    out.add(rep_of[why[: -len(s_)]]); break
        return out
    kids, par = defaultdict(list), {}
    for n in reps:
        if P[n]["layer"] == 4:
            par[n] = parents(n)
            for p in sorted(par[n]) or ["other:" + beam(n)]:
                kids[p].append(n)
    fams = sorted(f for f in kids if not f.startswith("other:"))
    out_fams = set(rng.sample(fams, max(1, round(a.family_frac * len(fams)))))
    for f in sorted(out_fams):
        for n in kids[f]:
            put(n, "heldout", "H2", f"2->4 of the held-out family {f}")
    other = sorted(n for f in kids if f.startswith("other:") for n in kids[f])
    for n in rng.sample(other, max(1, round(a.family_frac * len(other)))):
        put(n, "heldout", "H2", "four-fermion / multi-boson 2->4")
    # H1: new combinations, per (layer, beam) cell, among what is left
    cells = defaultdict(list)
    for n in reps:
        if n not in role and n not in out_fams:      # a held-out family's parent trains, so H2 tests multiplicity alone
            cells[(P[n]["layer"], beam(n))].append(n)
    for c, ns in sorted(cells.items()):
        for n in rng.sample(sorted(ns), round(a.h1_frac * len(ns))):
            put(n, "heldout", "H1", f"new combination in {c}")
    for n in reps:
        put(n, "train", None, "")
    # H5 and the one-loop layer: every coloured 2->2 class (top classes aside, H3) and coloured 2->3 classes drawn per
    # beam up to --n-loops in all (a 2->3 MadLoop pool costs 100-200 LO units: the layer is sized, not exhaustive);
    # of those whose tree trains, --family-frac are held out (the LO->NLO map on a seen tree), and the one-loop of every
    # H1 tree in the layer is held out (a new process at a new order)
    col = [n for n in reps if P[n]["layer"] in (2, 3) and role[n]["axis"] != "H3"
           and any(abs(p) in (1, 2, 3, 4, 5, 21) for p in P[n]["pdg_ids"])]
    two = sorted(n for n in col if P[n]["layer"] == 2)
    three = defaultdict(list)
    for n in sorted(n for n in col if P[n]["layer"] == 3):
        three[beam(n)].append(n)
    room = max(0, a.n_loops - len(two)); tot3 = sum(len(v) for v in three.values())
    pick3 = [n for b, ns in sorted(three.items()) for n in rng.sample(ns, min(len(ns), round(room * len(ns) / max(tot3, 1))))]
    layer = two + sorted(pick3)
    tr_col = sorted(n for n in layer if role[n]["role"] == "train")
    loop_out = set(rng.sample(tr_col, round(a.family_frac * len(tr_col))))
    loops = {}
    for n in layer:
        if role[n]["role"] == "train":
            loops[n + "_nlo"] = {"role": "heldout" if n in loop_out else "train", "axis": "H5" if n in loop_out else None,
                                 "why": "one-loop of a training tree" if n in loop_out else "", "tree": n}
        elif role[n]["axis"] == "H1":
            loops[n + "_nlo"] = {"role": "heldout", "axis": "H5", "why": "one-loop of a held-out tree", "tree": n}
    # the heavy-quark layer (H3's brackets; D100: at 2->4 too): top classes drawn per layer, each at the three masses
    tops = defaultdict(list)
    for n in reps:
        if role[n]["axis"] == "H3":
            tops[P[n]["layer"]].append(n)
    # every 2->2 top class (the cheapest, most direct bracket), the rest of --n-heavy drawn over 2->3 and 2->4 by size
    hq_cls = sorted(tops[2])
    tot = len(tops[3]) + len(tops[4]); room = max(0, a.n_heavy - len(hq_cls))
    hq_cls += [n for L in (3, 4) for n in rng.sample(sorted(tops[L]), max(1, round(room * len(tops[L]) / tot)))]
    heavy = {f"{n}__mt{m}": {"role": "train", "tree": n, "m_t": m} for n in sorted(hq_cls) for m in (50, 300, 600)}
    out = {"trees": role, "loops": loops, "heavy_quark": heavy, "twin_out": sorted(twin_out), "seed": a.seed,
           "family_frac": a.family_frac, "h1_frac": a.h1_frac}
    json.dump(out, open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json"), "w"), indent=0)
    c = Counter((v["role"], v["axis"]) for v in role.values())
    print("trees:", dict(sorted(c.items(), key=str)))
    print("trees by layer and role:", dict(sorted(Counter((P[n]["layer"], v["role"]) for n, v in role.items()).items())))
    print(f"held-out 2->4 families {len(out_fams)} of {len(fams)}")
    print("one-loop candidates:", dict(Counter((v["role"], v["axis"]) for v in loops.values())))
    print("heavy-quark layer:", len(heavy), "pools from", len(hq_cls), "top classes by layer",
          dict(Counter(P[n]["layer"] for n in hq_cls)))
    write_stage1(P, role, heavy, a)
    print(f"removal twin also drops {len(twin_out)} classes (tau, quartic gauge), "
          f"{sum(role[n]['role'] == 'train' for n in twin_out)} of them training classes")


BSM = ("zp_ee_mumu", "zp_uux_mumu", "zp_ee_uu", "zp_ee_Y1a", "zp_uux_Y1g", "zp_uux_Y1z", "thdm_ee_HpHm", "thdm_ee_h2h3",
       "thdm_uux_HpHm", "thdm_udx_Hph2", "thdm_ee_HpHma", "thdm_uux_h2h3g")        # mg5_pipeline_final.NONSM, D98


def write_stage1(P, role, heavy, a):
    """the generation recipe of D102's first stage (recipes/catalog_v3_gen_stage1.yaml): every tree class
    representative, train and held-out alike, at the catalog sizes (1e5 train, a tenth each for val and test), the
    heavy-quark layer (the class at m_t = 50, 300, 600 GeV through physics.masses, the top width with it) and the BSM
    layer; sqrt(s) from 1.05 x the final-state masses (at least 25 GeV) to 1 TeV, the catalogs' window (heavy pools: see
    below). Stages 2 (the one-loop layer, after
    its MadLoop certification) and 3 (2->5 probes, alpha_s points, loop-induced) are separate recipes."""
    lo = lambda ms: max(25, int(round(1.05 * sum(ms))))
    sz = "n_train: 100000, n_val: 10000, n_test: 10000"
    L = ["# catalog_v3, generation stage 1 (D102; recipes/select_catalog_v3.py): tree classes, heavy-quark layer, BSM",
         "# roles: recipes/catalog_v3_roles.json (this recipe generates every pool; the training recipes select)",
         "sampling: {mode: mixture}", "processes:"]
    for n in sorted(role):
        assert lo(P[n]["m_finals"]) < 900, n
        L.append(f"  - {{name: {n}, sqrts: [{lo(P[n]['m_finals'])}, 1000], {sz}}}")
    # the heavy-quark pools: the window is the catalog's [threshold, 1 TeV] while that leaves 500 GeV above threshold,
    # else [threshold, threshold + 500] (a t t~ pair at 600 GeV has its threshold at 1260); the top width follows the
    # mass at LO, Gamma(t -> b W) ~ m^3 (1 - x)^2 (1 + 2x), x = M_W^2 / m^2, scaled to the card's 1.4915 at 172.5, and 0
    # below the b W threshold (no on-shell top decays there)
    MW, MB = 80.419, 4.7
    f = lambda m: m ** 3 * (1 - MW ** 2 / m ** 2) ** 2 * (1 + 2 * MW ** 2 / m ** 2) if m > MW + MB else 0.0
    width = lambda m: round(1.4915 * f(m) / f(172.5), 4)
    for h, v in sorted(heavy.items()):
        ms = [v["m_t"] if abs(p) == 6 else m for p, m in zip(P[v["tree"]]["pdg_ids"][2:], P[v["tree"]]["m_finals"])]
        l = lo(ms); hi = max(1000, l + 500)
        assert hi - l >= 500, h
        L.append(f"  - {{name: {h}, base: {v['tree']}, sqrts: [{l}, {hi}], {sz}, "
                 f"physics: {{masses: {{6: {float(v['m_t'])}}}, widths: {{6: {width(v['m_t'])}}}}}}}")
    import sys
    sys.path.insert(0, ROOT)
    import mg5_pipeline_final as mg
    for n in BSM:
        e = mg.PROCESSES[n]
        ms = e.get("m_finals") or [e.get("m_final", 0.0)] * e["nfinal"]
        assert lo(ms) < 900, n
        L.append(f"  - {{name: {n}, sqrts: [{lo(ms)}, 1000], {sz}}}")
    open(os.path.join(ROOT, "recipes", "catalog_v3_gen_stage1.yaml"), "w").write("\n".join(L) + "\n")
    print(f"wrote recipes/catalog_v3_gen_stage1.yaml: {len(role)} trees, {len(heavy)} heavy-quark, {len(BSM)} BSM")


if __name__ == "__main__":
    main()
