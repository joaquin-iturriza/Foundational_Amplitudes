#!/usr/bin/env python
"""catalog_v4 (D130, the user 2026-10-25: "Okay, yes to 1 to 3, yes to electroweak production and the smaller fixes, no
to the decayed final states"): catalog_v3's census and roles regenerated under the approved coverage design. catalog_v3's
entries, pools and recipes are untouched; every v4 name is v4_<the v3 name without v3_> (a BSM entry: v4_<its name>).

What changes against catalog_v3, all of it as fields of the v4 entries that mg5_pipeline_final reads (V4_* there):
  acceptance   hadronic beams pT > 10 GeV, |eta| < 5, Delta R > 0.4, no pair-mass cut; e+e- |cos theta| < 0.99, the
               rest as before; the b and the tau cut like the other legs (fiducial_cuts, cut_pdgs)
  sqrt(s)      [V4_SQRTS_LO, 6 TeV], log-uniform (sqrts_measure: log); the lower edge is the threshold the masses and
               the pT cut give (mg5_pipeline_final.v4_sqrts_lo), no 1.05 strip, no 25 GeV floor
  mixture      a forward component (f_fwd 0.15 of each pool; fwd_cand a third of the flat part's candidates turned
               forward), the splitter and f_flat 0.35 kept, every role the same mixture (V4_SAMPLING)
  orders       full coupling orders: each tree string carries "QED<=b" with b = n - n_g (n final legs, n_g external
               gluons) when a quark is external, 0 for an all-gluon process, n for a colourless one. Every tree diagram
               has QCD + QED = n (three-point vertices count one, four-point ones two, and a tree with n + 2 legs has
               n such powers) and QCD >= n_g when a quark line exists (each external gluon costs a g_s: it attaches to
               a quark line or to a gluon tree that does), so QED <= b holds for every diagram and the string admits
               them all: VBF, VBS and the EW-QCD interference. QCD stays unbounded (MadGraph applies no default order
               once one is given; catalog_v2's four-quark entries, "QED<=2", are the precedent). The order vector reads
               the bound (order_vector: [0, 0, a_max, b]); a_max is catalog_v3's (minimal QED = maximal QCD)
  top          MT = ymt = 172.5 in every entry, internal tops and the one-loop layer too, WT = V4_WT = 1.4915, the width
               the pools built before the DECAY-column fix have (the user 2026-10-26, "do 1.4915 for all"); the
               heavy-quark layer's width the LO formula at its mass (mg5_pipeline_final.top_width_lo); the one-loop
               b b~ > h w+ w- at zero top width (WIDTH_OVERRIDE, the user 2026-10-26, D154)
  b            massless in beams and finals: model sm-no_b_mass (loop_sm-no_b_mass for MadLoop), beam_masses {}
Classes and roles: catalog_v4's own (recipes/select_catalog_v4.py on the v4 census's equality, after --write; it
writes the training and held-out recipes and gen_stage5, the pools of the classes the electroweak orders split off).
Not changed: the pool sizes, the BSM strings (their orders are explicit already), the alpha_s points.
The energy-extrapolation axis H7E (catalog_v3: sqrt(s_hat) in [1, 10] TeV, inside the v4 training window up to 6 TeV)
moves above it (the user 2026-10-26, "Yes, okay, that's fine, go ahead.", to the proposal of 6 to 20 TeV, log-uniform
like the window): each catalog_v3 probe <name>__E10 becomes v4_<name>__E20 on E_HOLDOUT, in gen_stage4 and its
held-out recipe heldout_H7E. Dropped too (D136, the user 2026-10-25: "Drop the b to higgs events."): the 34 trees
with no diagram once the b is massless (a Higgs on a b line with nothing else to couple to it: b b~ > h + photons,
gluons or Higgses, b g > b h ..., g g > b b~ h ...), MadGraph's verdict in analysis/catalog_v4/data/checked.json
(exists false), with their one-loop entries and every recipe entry (any role) named after or based on them.
    python3 recipes/gen_catalog_v4.py --write            entries + generation recipes (no MadGraph, laptop or site)
    python recipes/gen_catalog_v4.py --check [--workers N] [--only NAME ...]   (CC) MadGraph on the v4 strings
    python3 recipes/gen_catalog_v4.py --summary <checked.json>   census change: existence, diagrams
    python3 recipes/gen_catalog_v4.py --write-loops-signext   the one-loop sign extension (write_loops_signext)
"""
import argparse, json, os, re, sys
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT_PROC = os.path.join(HERE, "catalog_v4_processes.yaml")
OUT_LOOPS = os.path.join(HERE, "catalog_v4_loops.yaml")
OUT_CENSUS = os.path.join(HERE, "catalog_v4_census.yaml")
OUT_LOOPS_SIGNEXT = os.path.join(HERE, "catalog_v4_loops_signext.yaml")
OUT_GEN_SIGNEXT = os.path.join(HERE, "catalog_v4_gen_signext.yaml")
OUT_SIGNEXT_AUDIT = os.path.join(ROOT, "analysis", "catalog_v4", "data", "signext_classes.json")
COLOURED = {1, 2, 3, 4, 5, 21}
QUARKS = {1, 2, 3, 4, 5, 6}
CHECKED = os.path.join(ROOT, "analysis", "catalog_v4", "data", "checked.json")
MIXED_AUDIT = os.path.join(ROOT, "analysis", "catalog_v4", "data", "mixed_order_audit.json")   # tools/mixed_order_audit.py
LOOP_MODEL = {"loop_sm": "loop_sm-no_b_mass", "loop_qcd_qed_sm": "loop_qcd_qed_sm-no_b_mass"}
# the v3 stage and training recipes mapped to v4
# (the generation stages only: the training and held-out recipes come from catalog_v4's own classes and roles,
# recipes/select_catalog_v4.py, run after this)
RECIPES = ["gen_stage1", "gen_stage2", "gen_stage3", "gen_stage4"]
E_HOLDOUT = [6000, 20000]      # GeV, the energy hold-out's window (the user 2026-10-26, see the docstring)
# One-loop entries whose full-order born mixes QCD and electroweak powers (QED<=n with QCD diagrams too): MadLoop's
# [virt=QCD] on that born fails the pole check (double pole identically 0 for the two 2->3, off by up to 25% at a few
# points for the three 2->2; plan catalog_coverage). They keep catalog_v3's pure-QCD born string (no QED order), which
# order_vector encodes as a lower electroweak maximum ([1, 0, 2, 1] or [1, 0, 2, 0] against the tree's [0, 0, 2, 3] or
# [0, 0, 2, 2]), so the difference is in the data (the user 2026-10-26, D142: "As long as there is a way to encode this
# difference in the data, then yes, qcd only is fine, if not drop them."). Their trees keep the full orders. The names
# carry _qcdborn (virt base v4_<x>_qcdborn, entry v4_<x>_qcdborn_nlo), so no pool, standalone or quarantined chunk of
# the mixed-order one-loop definition is ever reused for them.
QCD_BORN = {"v3_uux_ccxz", "v3_ddx_bbxz", "v3_ddx_ccx", "v3_ddx_bbx", "v3_bbx_ccx"}
# One-loop entries with an internal top that can go on shell: the physical top width moves MadLoop's double pole off the
# analytic one by ~width/mass (4.2e-3 to 4.6e-3 at WT, below 5e-5 with it zeroed; plan catalog_coverage), above the 1e-3
# per-event guard, so they carry catalog_v3's v3_bg_bwpwm tolerance (c2_rel_tol 0.01, pole_width_diag.txt).
C2_WIDTH_TOL = {"v3_bg_bwpwm": 0.01, "v3_bbx_tapvtwm": 0.01}
# One-loop entries generated at another top width than V4_WT (width_override: written into the virt entry's
# param_card_patches, so the card patcher puts it on the column-1 DECAY line, and into the process entry's, so the
# pool's recipe_id and sidecar carry it). b b~ > h w+ w-: the top is t-channel only, never resonant, so the exact
# amplitude has no width; the fixed width breaks gauge cancellations and moves MadLoop's double pole off the analytic
# one by up to 2e-2 at 4 to 6 TeV (19% of points refused by the pole guard over [286, 6000] GeV; 1.6e-5 at zero width;
# plan catalog_coverage). The user 2026-10-26 (D154): "isn't there I single correct way to calculate the amplitude?
# ... if this one is blowing up, then it's clearly wrong". Exact at zero width, it takes the default pole tolerance.
WIDTH_OVERRIDE = {"v3_bbx_hwpwm": {"WT": 0.0}}


def qcdborn_name(n, virt_key=False):
    """v4_<x>_nlo -> v4_<x>_qcdborn_nlo for the QCD_BORN entries, and with virt_key their one-loop table key
    v4_<x> -> v4_<x>_qcdborn (v4_<x> as a process name is the tree, which stays); any other name unchanged"""
    for b in QCD_BORN:
        vb = v4name(b)
        if virt_key and n == vb:
            return vb + "_qcdborn"
        if n == vb + "_nlo":
            return vb + "_qcdborn_nlo"
    return n


def load(path):
    return yaml.load(open(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def v4name(n):
    return "v4_" + (n[3:] if n.startswith("v3_") else n)


def qed_max(pdg_ids):
    """the largest amplitude-level QED power of a tree with these external legs (the docstring's rule)"""
    n = len(pdg_ids) - 2
    n_g = sum(int(p) == 21 for p in pdg_ids)
    if any(abs(int(p)) in QUARKS for p in pdg_ids):
        return n - n_g
    return 0 if n_g else n


def v4_string(gen, pdg_ids):
    assert not re.search(r"\b(QED|QCD)\s*[<>=]", gen), gen
    return f"{gen} QED<={qed_max(pdg_ids)}"


def b_massless(pdg_ids, m_finals):
    return [0.0 if abs(int(p)) == 5 else float(m) for p, m in zip(pdg_ids[2:], m_finals)]


def common(pdg_ids):
    import mg5_pipeline_final as mg
    return {"fiducial_cuts": dict(mg.V4_CUTS[mg.v4_cut_kind(pdg_ids)]), "sqrts_measure": "log", "beam_masses": {}}


def tree_entry(n, e):
    import mg5_pipeline_final as mg
    pdg = list(e["pdg_ids"]); n_fin = int(e["nfinal"])
    k = int(e.get("alphas_power", 0)); b = qed_max(pdg)
    assert b >= n_fin - k, (n, b, n_fin, k)          # the bound never cuts below catalog_v3's minimal-QED diagrams
    out = {"mg5_generate": [v4_string(e["mg5_generate"][0], pdg)], "model": mg.V4_MODEL, "nfinal": n_fin,
           "pdg_ids": pdg, "m_finals": b_massless(pdg, e["m_finals"]), "run_card_patches": dict(e["run_card_patches"]),
           "param_card_patches": dict(mg.V4_TOP_PATCHES), "layer": e["layer"],
           "why": re.sub(r"\bv3_", "v4_", e.get("why", "")), "v3_name": n, "v3_key": e.get("v3_key"),
           "diagrams_v3": e.get("diagrams"), "qed_v3": n_fin - k, **common(pdg)}
    if k:
        out["alphas_power"] = k
    return out


def bsm_entry(n):
    """a catalog_v3 BSM entry (a hand-written PROCESSES entry, its own model and orders) under v4's acceptance,
    measure and top width; its b-mass and strings as they are (no external b in any of the twelve)"""
    import mg5_pipeline_final as mg
    e = dict(mg.PROCESSES[n]); pdg = list(e["pdg_ids"])
    assert not any(abs(int(p)) == 5 for p in pdg), n
    e["param_card_patches"] = {**e.get("param_card_patches", {}), "MT": mg.V4_MT, "ymt": mg.V4_MT, "WT": mg.V4_WT}
    e.update(common(pdg)); e["v3_name"] = n
    if "m_finals" not in e:
        mf = e.get("m_final", 0.0)
        e["m_finals"] = [float(x) for x in mf] if isinstance(mf, (list, tuple)) else [float(mf)] * int(e["nfinal"])
    return {k: v for k, v in e.items() if not k.startswith("_")}


def loop_entries(trees):
    """catalog_v3's one-loop layer (QCD virtuals, loop-induced) on v4's strings, models, cuts, measure and width"""
    import mg5_pipeline_final as mg
    L = load(os.path.join(HERE, "catalog_v3_loops.yaml"))
    virt, procs = {}, {}
    for k, v in L["virt"].items():
        nk = v4name(k); pdg = list(v["pdg_ids"])
        w = {"pdg_ids": pdg, "m_finals": b_massless(pdg, v["m_finals"]), "certified": False, "v3_name": k,
             "param_card_patches": {"WT": mg.V4_WT}, **common(pdg)}
        if v.get("loopind"):
            w.update(mg5=v["mg5"], model=LOOP_MODEL[v["model"]], loopind=True, order=list(v["order"]))
        else:
            gen, tail = v["mg5"].split(" [", 1)
            assert gen == trees[nk]["v3_gen"], (k, gen)
            born = gen if k in QCD_BORN else trees[nk]["mg5_generate"][0]   # QCD_BORN: catalog_v3's string as is
            w.update(mg5=born + " [" + tail, model=LOOP_MODEL[v.get("model", "loop_sm")])
            if k in QCD_BORN:
                w["qcd_born"] = True
        if k in C2_WIDTH_TOL:
            w["c2_rel_tol"] = C2_WIDTH_TOL[k]
        if k in WIDTH_OVERRIDE:
            w["width_override"] = dict(WIDTH_OVERRIDE[k])
            w["param_card_patches"] = {**w["param_card_patches"], **WIDTH_OVERRIDE[k]}
        virt[qcdborn_name(nk, virt_key=True)] = w
    for n, p in L["processes"].items():
        e = {k: v for k, v in p.items() if k not in ("param_card_patches",)}
        vb = qcdborn_name(v4name(p["virt_base"]), virt_key=True); pdg = list(p["pdg_ids"])
        e.update(virt_base=vb, m_finals=b_massless(pdg, p["m_finals"]),
                 param_card_patches=dict(WIDTH_OVERRIDE.get(p["virt_base"], {})), certified=False,
                 v3_name=n, why=re.sub(r"\bv3_", "v4_", p.get("why", "")), **common(pdg))
        procs[qcdborn_name(v4name(n))] = e
    return virt, procs


def window(e, physics=None):
    """[lo, 6 TeV] of an entry, its physics block applied (a heavy-quark mass, a harder pT cut)"""
    import mg5_pipeline_final as mg
    physics = physics or {}
    pdg = e["pdg_ids"]
    mf = [float((physics.get("masses") or {}).get(abs(int(p)), m)) for p, m in zip(pdg[2:], e["m_finals"])]
    cuts = {**e["fiducial_cuts"], **(physics.get("cuts") or {})}
    return [int(mg.v4_sqrts_lo(pdg, mf, cuts)), int(mg.V4_SQRTS_MAX)]


def vanished():
    """the v4 tree strings MadGraph found no diagram for (--check's output, committed)"""
    return {c["name"] for c in json.load(open(CHECKED)) if not c["exists"]}


def write():
    import mg5_pipeline_final as mg
    P3 = load(os.path.join(HERE, "catalog_v3_processes.yaml"))["processes"]
    gone = vanished()
    assert len(gone) == 34 and all("h" in n.split("_", 2)[2] and "b" in n for n in gone), sorted(gone)
    trees = {}
    for n, e in sorted(P3.items()):
        trees[v4name(n)] = dict(tree_entry(n, e), v3_gen=e["mg5_generate"][0])
    assert gone <= set(trees), sorted(gone - set(trees))
    procs = {n: {k: v for k, v in t.items() if k != "v3_gen"} for n, t in trees.items()}
    bsm = sorted({e["name"] for e in load(os.path.join(HERE, "catalog_v3_gen_stage1.yaml"))["processes"]
                  if not e["name"].startswith("v3_")})
    for n in bsm:
        procs[v4name(n)] = bsm_entry(n)
    virt, lprocs = loop_entries(trees)
    for n in gone:
        trees.pop(n); procs.pop(n)
    virt = {k: v for k, v in virt.items() if k not in gone}
    lgone = {n for n, e in lprocs.items() if e["virt_base"] in gone or n in gone}
    lprocs = {n: e for n, e in lprocs.items() if n not in lgone}
    drop_names = gone | lgone
    hdr = "# catalog_v4 (D130): generated by recipes/gen_catalog_v4.py --write; do not edit\n"
    with open(OUT_PROC, "w") as f:
        f.write(hdr); yaml.safe_dump({"processes": procs}, f, sort_keys=True)
    with open(OUT_LOOPS, "w") as f:
        f.write(hdr); yaml.safe_dump({"virt": virt, "processes": lprocs}, f, sort_keys=True)
    allp = {**procs, **lprocs}
    # the census recipe (every tree that exists in catalog_v3, for the v4 equality step) and the v4 recipes
    lines = [f"  - {{name: {n}, sqrts: {window(e)}, n_train: 0, n_val: 0, n_test: 0}}"
             for n, e in sorted(procs.items()) if n in trees]
    open(OUT_CENSUS, "w").write("# catalog_v4 census (recipes/gen_catalog_v4.py --write): every catalog_v3 tree on its v4 "
                                "string, for the numeric equality check; no events\nprocesses:\n" + "\n".join(lines) + "\n")
    energies, bdrop, counts = Counter(), Counter(), {}
    for r in RECIPES:
        src = load(os.path.join(HERE, f"catalog_v3_{r}.yaml"))
        out = []
        for e in src["processes"]:
            if v4name(e["name"]) in drop_names or v4name(e.get("base", e["name"])) in drop_names:
                bdrop[r] += 1; continue
            energy = e["name"].endswith("__E10")
            ne = dict(e, name=qcdborn_name(v4name(e["name"][:-len("__E10")] + "__E20" if energy else e["name"])))
            ph = dict(e.get("physics") or {})
            if "base" in e:
                ne["base"] = qcdborn_name(v4name(e["base"]))
            if (ph.get("masses") or {}).get(6) is not None:           # the heavy-quark layer: width at its mass
                ph["widths"] = {6: round(mg.top_width_lo(float(ph["masses"][6])), 6)}
            if ph:
                ne["physics"] = ph
            ne["sqrts"] = list(E_HOLDOUT) if energy else window(allp[ne.get("base", ne["name"])], ph)
            assert not energy or window(allp[ne["base"]], ph)[0] < E_HOLDOUT[0], ne["name"]
            energies[r] += energy
            out.append(ne)
        with open(os.path.join(HERE, f"catalog_v4_{r}.yaml"), "w") as f:
            f.write(hdr + f"# from catalog_v3_{r}.yaml: names v3_ -> v4_, windows [threshold, 6 TeV] (the energy hold-out [6, 20] TeV), the v4 mixture; the 34 b-Higgs trees with no "
                    "diagram (massless b) dropped; "
                    "generation recipe (roles: recipes/select_catalog_v4.py)\n")
            yaml.safe_dump({"sampling": dict(mg.V4_SAMPLING), "processes": out}, f, sort_keys=False)
        counts[r] = len(out)
    floor = sum(1 for n in trees if window(procs[n])[0] == int(mg.V4_SQRTS_FLOOR) and
                not any(m > 0 for m in procs[n]["m_finals"]))
    lo = Counter(min(window(procs[n])[0], 100) // 10 * 10 for n in trees)
    print(f"wrote {OUT_PROC} ({len(procs)}: {len(trees)} trees, {len(bsm)} BSM), {OUT_LOOPS} ({len(virt)} MadLoop "
          f"strings, {len(lprocs)} entries), {OUT_CENSUS}")
    print(f"recipes: {counts}; energy hold-out entries on {E_HOLDOUT} GeV: {dict(energies)}")
    print(f"b-Higgs classes with no diagram dropped: {len(gone)} trees, {len(lgone)} one-loop entries; recipe entries "
          f"dropped: {dict(bdrop)} ({sum(bdrop.values())})")
    print(f"QED bound above catalog_v3's minimal order: {sum(t['qed_v3'] < qed_max(t['pdg_ids']) for t in trees.values())}"
          f" of {len(trees)} trees; lower sqrt(s) edges (GeV, binned by 10 up to 100): {dict(sorted(lo.items()))}; "
          f"{floor} trees at the {mg.V4_SQRTS_FLOOR:.0f} GeV floor")


def signext_trees(P=None, roles=None):
    """The classes of catalog_v4's one-loop sign extension (plan v3_prep_explore, the design of 2026-10-29): every
    training tree class (recipes/catalog_v4_roles.json, keyed by class representative) that is 2->3, coloured (a quark
    or gluon leg) and holds no catalog_v4 one-loop pool, train or held out (each one-loop entry's `class`); catalog_v3's
    rule (gen_catalog_v3.write_loops_signext) applied at class level. Returns (sorted representatives, audit dict)."""
    P = P if P is not None else load(OUT_PROC)["processes"]
    roles = roles if roles is not None else json.load(open(os.path.join(HERE, "catalog_v4_roles.json")))
    loop_cls = {v["class"] for v in roles["loops"].values()}
    col = lambda e: any(abs(int(q)) in COLOURED for q in e["pdg_ids"])
    trees = sorted(n for n, v in roles["trees"].items() if v["role"] == "train" and n in P and int(P[n]["nfinal"]) == 3
                   and col(P[n]) and n not in loop_cls)
    # the class-level check, independent of the filter above: no member of a chosen class (the census classes) is the
    # tree of any one-loop pool (train or held out, duplicates included) or a pool of any held-out recipe
    members = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "census_classes.json")))["classes"]
    loop_trees = {v["tree"] for v in roles["loops"].values()}
    # (the removal twin's axes H3p and H4 are held out of the twin's recipe only; their classes train in
    # catalog_v4_train.yaml, which is the arm's base, and are listed under twin_out below)
    held = set()
    for f in sorted(os.listdir(HERE)):
        if f.startswith("catalog_v4_heldout_") and f.endswith(".yaml") and f not in ("catalog_v4_heldout_H3p.yaml",
                                                                                     "catalog_v4_heldout_H4.yaml"):
            held |= {e["name"] for e in load(os.path.join(HERE, f))["processes"]}
    bad = sorted(n for n in trees for m in members.get(n, [n]) if m in loop_trees or m in held or m + "_nlo" in held)
    assert not bad, bad
    MIXED = json.load(open(MIXED_AUDIT))["entries"]
    assert all(n in MIXED for n in trees), sorted(n for n in trees if n not in MIXED)
    # held-out probes that are physics variants of a training class (alpha_s points, energy probes, heavy quark): the
    # tree of those classes trains already; listed, not excluded (the one-loop pool is no held-out pool's process)
    var = {"alphas_heldout": sorted({v["base"] for v in roles["alphas"].values() if v["role"] != "train"} & set(trees)),
           "h7_classes": sorted(set(roles["h7_classes"]) & set(trees)),
           "heavy_quark_heldout": sorted({v["tree"] for v in roles["heavy_quark"].values() if v["role"] != "train"}
                                         & set(trees)),
           # the removal twin's held-out classes: the arm builds on catalog_v4_train.yaml only, never the twin's recipe
           "twin_out": sorted(set(roles["twin_out"]) & set(trees)),
           "held_out_families_parents": sorted(set(roles["held_out_families"]) & set(trees))}
    audit = {"n": len(trees), "by_beams": dict(Counter(n.split("_")[1] for n in trees)),
             # read from the trees' diagrams (tools/mixed_order_audit.py: more than one amplitude-level QCD power), not
             # from the QED bound: with massless b the three b-Higgs trees have no Yukawa diagram and are pure
             # electroweak at full order, though QED<=3 exceeds 3 - alphas_power (listed under qed_bound_only)
             "mixed_order_born": sorted(n for n in trees if MIXED[n]["mixed"]),
             "qed_bound_only": sorted(n for n in trees if not MIXED[n]["mixed"]
                                      and qed_max(P[n]["pdg_ids"]) > 3 - int(P[n].get("alphas_power", 0))),
             "one_loop_classes": len(loop_cls), "training_classes": sum(v["role"] == "train" for v in roles["trees"].values()),
             "checked": "role train, not the class of any one-loop pool (the filter); then no member of the class is the tree of "
                        "any one-loop pool or a pool of any catalog_v4_heldout_*.yaml (the assertion)",
             "variants_of_training_classes": var, "trees": trees}
    return trees, audit


def signext_names(trees, audit):
    """tree -> its sign-extension entry's name: v4_<x>_nlo, v4_<x>_qcdborn_nlo for a mixed-order born"""
    mixed = set(audit["mixed_order_born"])
    return {n: n + ("_qcdborn_nlo" if n in mixed else "_nlo") for n in trees}


def write_loops_signext():
    """The one-loop sign extension: [virt=QCD] of every signext_trees class on its catalog_v4 tree string (full
    coupling orders), model, cuts, measure and top width, as loop_entries' (role signext); a class whose tree mixes QCD
    and electroweak powers (the audit's mixed_order_born) takes the pure-QCD born instead, as the QCD_BORN entries: the
    tree string without its QED bound (MadGraph's minimal QED order, the largest QCD power alone), which order_vector
    encodes as the lower electroweak maximum, under the names v4_<x>_qcdborn / v4_<x>_qcdborn_nlo (D142, see QCD_BORN);
    to
    recipes/catalog_v4_loops_signext.yaml, and their generation recipe at 1e4 / 1e3 / 1e3 in the v4 mixture and the
    tree's window, recipes/catalog_v4_gen_signext.yaml; the class audit to analysis/catalog_v4/data/signext_classes.json.
    No string is fixed by hand: an entry whose MadLoop build or pole certification fails is dropped by the generation
    (prebuild_recipes.py --skip-cert-failed) and listed. Which pools change sign is read from the generated pools
    (analysis/catalog_v3/pool_audit.py --stages v4_signext); only those enter the arm's training recipe
    (recipes/make_catalog_v4_train_signext.py)."""
    import mg5_pipeline_final as mg
    P = load(OUT_PROC)["processes"]
    trees, audit = signext_trees(P)
    virt, procs = {}, {}
    mixed = set(audit["mixed_order_born"])
    names = signext_names(trees, audit)
    for n in trees:
        e = P[n]; pdg = list(e["pdg_ids"]); vb = names[n][:-len("_nlo")]
        assert n not in mg.PROCESSES or mg.PROCESSES[n].get("_v4"), n
        born = e["mg5_generate"][0]
        if n in mixed:   # the pure-QCD born: the tree string without its QED bound (v4_string's inverse)
            born = re.sub(r"\s+QED<=\d+$", "", born)
            assert born != e["mg5_generate"][0] and "QED" not in born, (n, born)
        virt[vb] = {"mg5": born + " [virt=QCD]", "model": LOOP_MODEL["loop_sm"], "pdg_ids": pdg,
                    "m_finals": list(e["m_finals"]), "certified": False, "param_card_patches": {"WT": mg.V4_WT},
                    **common(pdg)}
        if n in mixed:
            virt[vb]["qcd_born"] = True
        procs[names[n]] = {"kind": "virt", "virt": True, "virt_base": vb, "nfinal": int(e["nfinal"]), "n_loops": 1,
                             "alphas_power": int(e.get("alphas_power", 0)) + 1, "pdg_ids": pdg,
                             "m_finals": list(e["m_finals"]), "param_card_patches": {}, "layer": "nlo",
                             "why": f"one-loop QCD of {n}" + (" at its pure-QCD born" if n in mixed else "")
                                    + " (sign extension)", "role": "signext", "certified": False,
                             **common(pdg)}
    hdr = "# catalog_v4's one-loop sign extension: generated by recipes/gen_catalog_v4.py --write-loops-signext; do not edit\n"
    with open(OUT_LOOPS_SIGNEXT, "w") as f:
        f.write(hdr); yaml.safe_dump({"virt": virt, "processes": procs}, f, sort_keys=True)
    out = [{"name": names[n], "sqrts": window(P[n]), "n_train": 10000, "n_val": 1000, "n_test": 1000} for n in trees]
    with open(OUT_GEN_SIGNEXT, "w") as f:
        f.write(hdr + "# [virt=QCD] of every coloured 2->3 training tree class outside catalog_v4's one-loop layer, "
                "1e4 / 1e3 / 1e3, the trees' windows, the v4 mixture\n")
        yaml.safe_dump({"sampling": dict(mg.V4_SAMPLING), "processes": out}, f, sort_keys=False)
    os.makedirs(os.path.dirname(OUT_SIGNEXT_AUDIT), exist_ok=True)
    json.dump(audit, open(OUT_SIGNEXT_AUDIT, "w"), indent=1)
    print(f"wrote {OUT_LOOPS_SIGNEXT}, {OUT_GEN_SIGNEXT} ({len(trees)} one-loop entries, by beams "
          f"{audit['by_beams']}; {len(audit['mixed_order_born'])} with a mixed-order born, on the pure-QCD born) and "
          f"{OUT_SIGNEXT_AUDIT}")


def check(workers, out, only=None):
    """MadGraph `display processes` on every v4 tree string (catalog_v2's mg5_check): existence under sm-no_b_mass and
    the diagram count; diagrams_v3 is catalog_v3's count of the same process (sm, minimal QED)"""
    import mg5_pipeline_final as mg
    sys.path.insert(0, HERE)
    from gen_catalog_v2 import mg5_check
    todo = []
    for n, e in sorted(mg.PROCESSES.items()):
        if not (e.get("_v4") and e.get("v3_key") is not None) or (only and n not in only):
            continue
        todo.append({"name": n, "layer": e["layer"], "generate": e["mg5_generate"][0], "model": e["model"],
                     "diagrams_v3": e.get("diagrams_v3"), "qed_v3": e.get("qed_v3"), "v3_name": e["v3_name"]})
    res = mg5_check(todo, workers=workers)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"CATALOG_V4_CHECK {len(res)} strings, {sum(c['exists'] for c in res)} exist -> {out}")
    summary(out)


def summary(path):
    res = json.load(open(path))
    by = {}
    for c in res:
        L = by.setdefault(str(c["layer"]), {"n": 0, "exists": 0, "diagrams_v3": 0, "diagrams_v4": 0, "grew": 0})
        L["n"] += 1; L["diagrams_v3"] += int(c.get("diagrams_v3") or 0)
        if c["exists"]:
            L["exists"] += 1; L["diagrams_v4"] += int(c["diagrams"]); L["grew"] += int(c["diagrams"] > (c.get("diagrams_v3") or 0))
    gone = sorted(c["name"] for c in res if not c["exists"])
    print("CATALOG_V4_SUMMARY", json.dumps({"by_layer": by, "vanished": gone[:60], "n_vanished": len(gone),
                                            "errors": Counter(c["error"][:50] for c in res if not c["exists"]).most_common(5)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-loops-signext", action="store_true", help="the one-loop sign extension")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_catalog_v4", "checked.json"))
    a = ap.parse_args()
    if a.write:
        write()
    if a.write_loops_signext:
        write_loops_signext()
    if a.check:
        check(a.workers, a.out, a.only)
    if a.summary:
        summary(a.summary)


if __name__ == "__main__":
    main()
