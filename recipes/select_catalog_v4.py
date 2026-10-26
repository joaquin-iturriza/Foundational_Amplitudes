#!/usr/bin/env python
"""catalog_v4's classes and roles (D130), from the v4 census's numeric equality (tools/equal_processes.py over the 2370
v4 trees, analysis/catalog_v4/equality/equal_processes.json) and catalog_v3's held-out design (recipes/catalog_v3_roles.json,
recipes/select_catalog_v3.py), kept wherever the v4 classes allow.

Classes: catalog_v3's rule (analysis/catalog_v3/census_classes.json is reproduced exactly by it): the union of the exactly
equal classes (spread < 1e-9) and the near pairs (spread < 1e-2), over the v4 trees of catalog_v3's census (the 40
2->5 probes are H2p5 probes, not classes; an equality between two of them is reported and the pair kept once). The full
electroweak orders and the massless b change the classes against catalog_v3:
  split   a catalog_v3 class whose members are no longer equal (a t-channel W now connects d to u and b to t, so
          d d~ > u u~ X differs from d d~ > c c~ X, b b~ > t t~ X from d d~ > t t~ X, ...): every part keeps the v3
          class's role; a part with no generated pool (catalog_v4 generated catalog_v3's representatives only) gets one
          in recipes/catalog_v4_gen_stage5.yaml, at the catalog's sizes, window and mixture
  merge   v4 classes holding members of several catalog_v3 classes (a massless b equals a d or s where no W, top or
          Higgs tells them apart): one representative per class; a held-out process is held out with its whole class,
          so a merge of a held-out and a training v3 class is held out (axis by catalog_v3's selection order: H3, H2,
          H1) and the training pool of the other leaves training; a second generated pool in the class leaves every recipe
Representative: among the members with a generated pool, one whose catalog_v3 role is the class's (so a held-out read is
the process catalog_v3 held out), then the first by name; a class with no pool: the first by name (gen_stage5).
Roles then rechecked, not redrawn:
  H3      every class with an external top (rule, as catalog_v3)
  H2      catalog_v3's held-out 2->4 families (parents recorded in `why`) mapped to v4 classes: every 2->4 class with a
          parent in a held-out family is held out (no held-out family keeps a training child); a family's parent stays in
          training (reported if a merge made it held out)
  H5      a one-loop pool follows its tree: a held-out tree class holds its one-loop out (H5, "one-loop of a held-out
          tree"); a training tree's one-loop keeps catalog_v3's draw; two or three one-loop pools of one tree class
          (equal trees) take one role, held out if either is, and one of them is kept, as one tree pool per class; the
          loop-induced draw as catalog_v3's
  H6, H7  the alpha_s points and the H7 bases must be training classes; the H7 energy probes sit on [6, 20] TeV
          (the user 2026-10-25, D137, to the proposal of 6 to 20 TeV once the window reached 6 TeV); an entry whose base
          left training is dropped and reported
  twin    the removal twin's tau / tau-neutrino (H3p) and quartic gauge (H4) classes by catalog_v3's rule on the v4
          representatives (the pool a class trains on)
Writes analysis/catalog_v4/census_classes.json ({classes: {representative: members}, representatives}),
recipes/catalog_v4_roles.json, recipes/catalog_v4_gen_stage5.yaml, catalog_v4_train.yaml, catalog_v4_twin_train.yaml
and catalog_v4_heldout_<axis>.yaml, and prints every split and merge with what it changed
(analysis/catalog_v4/data/class_changes.json).
    python3 recipes/select_catalog_v4.py
"""
import json, os, sys
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
EQ = os.path.join(ROOT, "analysis", "catalog_v4", "equality", "equal_processes.json")
OUT_CLS = os.path.join(ROOT, "analysis", "catalog_v4", "census_classes.json")
OUT_CHG = os.path.join(ROOT, "analysis", "catalog_v4", "data", "class_changes.json")
OUT_ROLES = os.path.join(HERE, "catalog_v4_roles.json")
NEAR = 1e-2                                          # catalog_v3's near-pair merge (par. catalog_v3, the census)
AXIS_ORDER = ("H3", "H2", "H1")                      # catalog_v3's selection order
TWIN_PDG = {15, 16}
ADDS = ("+e+e-", "+veve~", "+uu~", "+dd~", "+bb~", "+g", "+a", "+z", "+h")


def load(p):
    return yaml.load(open(p), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def v4n(n):
    return "v4_" + (n[3:] if n.startswith("v3_") else n)


class UF(dict):
    def find(self, x):
        self.setdefault(x, x)
        while self[x] != x:
            self[x] = self[self[x]]; x = self[x]
        return x

    def union(self, a, b):
        self[self.find(a)] = self.find(b)


def main():
    from gen_catalog_v4 import qcdborn_name
    P4 = load(os.path.join(HERE, "catalog_v4_processes.yaml"))["processes"]
    census = [e["name"] for e in load(os.path.join(HERE, "catalog_v4_census.yaml"))["processes"]]
    c3 = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "census_classes.json")))["classes"]
    R3 = json.load(open(os.path.join(HERE, "catalog_v3_roles.json")))
    al3 = json.load(open(os.path.join(HERE, "catalog_v3_alphas.json")))
    eq = json.load(open(EQ))
    v3cls = {m: min(ms) for ms in c3.values() for m in ms}          # member -> catalog_v3 representative (= min)
    T3 = R3["trees"]
    stage = {k: load(os.path.join(HERE, f"catalog_v4_gen_{k}.yaml")) for k in ("stage1", "stage2", "stage3", "stage4")}
    top = {k: v for k, v in stage["stage1"].items() if k != "processes"}
    s1 = {e["name"]: e for e in stage["stage1"]["processes"]}

    # ---- classes
    uf = UF()
    for n in census:
        uf.find(n)
    for c in eq["classes"]:
        for m in c[1:]:
            uf.union(c[0], m)
    for p in eq["pairs"]:
        if p["spread"] < NEAR:
            uf.union(p["a"], p["b"])
    groups = defaultdict(set)
    for n in census:
        groups[uf.find(n)].add(n)
    p5 = {n for n in census if P4[n]["v3_name"] not in v3cls}
    p5_equal = [sorted(g) for g in groups.values() if g & p5 and len(g) > 1]
    assert all(g <= p5 for g in groups.values() if g & p5), "a 2->5 probe equals a census tree"
    G = [g for g in groups.values() if not g & p5]
    old = lambda n: v3cls[P4[n]["v3_name"]]                         # a v4 tree -> its catalog_v3 class representative
    role3 = lambda r: (T3[r]["role"], T3[r]["axis"])

    # ---- roles inherited from catalog_v3, then the representative
    has_top = lambda n: any(abs(int(p)) == 6 for p in P4[n]["pdg_ids"])
    cls, info = {}, {}
    for g in G:
        olds = sorted({old(n) for n in g})
        axes = {T3[r]["axis"] for r in olds if T3[r]["role"] == "heldout"}
        if any(has_top(n) for n in g):
            r, ax = "heldout", "H3"
        elif axes:
            r, ax = "heldout", next(a for a in AXIS_ORDER if a in axes)
        else:
            r, ax = "train", None
        pooled = sorted(n for n in g if n in s1)
        pref = [n for n in pooled if role3(old(n)) == (r, ax)]
        rep = (pref or pooled or sorted(g))[0]
        cls[rep] = sorted(g)
        info[rep] = {"role": r, "axis": ax, "v3_classes": olds, "pooled": pooled}
    reps = sorted(cls)
    rep_of = {m: r for r, ms in cls.items() for m in ms}

    # ---- H2: catalog_v3's held-out families on v4 classes
    out_fams3 = {T3[r]["why"].rsplit(" ", 1)[1] for r in T3 if T3[r]["axis"] == "H2" and "family" in T3[r]["why"]}
    out_fams = {rep_of[v4n(f)] for f in out_fams3 if v4n(f) in rep_of}
    lost_fams = sorted(f for f in out_fams3 if v4n(f) not in rep_of)

    def parents(r):
        out = set()
        for m in cls[r]:
            why = P4[m].get("why", "")
            for s_ in ADDS:
                if why.endswith(s_) and why[: -len(s_)] in rep_of:
                    out.add(rep_of[why[: -len(s_)]]); break
        return out
    h2_added, fam_parent_out = [], []
    for r in reps:
        if P4[r]["layer"] == 4 and parents(r) & out_fams and info[r]["axis"] != "H3":
            if info[r]["role"] != "heldout" or info[r]["axis"] != "H2":
                h2_added.append({"class": r, "was": [info[r]["role"], info[r]["axis"]]})
                info[r].update(role="heldout", axis="H2")
    for f in sorted(out_fams):
        if info[f]["role"] != "train":
            fam_parent_out.append({"family": f, "role": [info[f]["role"], info[f]["axis"]]})
    for r in reps:
        info[r]["why"] = {"H3": "external top (the heavy-quark layer brackets it)", "H2": "2->4 of a held-out family or "
                          "four-fermion / multi-boson 2->4 (catalog_v3's draw)", "H1": "new combination (catalog_v3's "
                          "draw)", None: ""}[info[r]["axis"]]

    # ---- split / merge report
    v3_to_v4 = defaultdict(set)
    for r in reps:
        for o in info[r]["v3_classes"]:
            v3_to_v4[o].add(r)
    splits = [{"v3_class": o, "v3_role": list(role3(o)), "v4_classes": {r: {"members": cls[r], "role": [info[r]["role"],
               info[r]["axis"]], "pool": "generated" if info[r]["pooled"] else "gen_stage5"} for r in sorted(rs)}}
              for o, rs in sorted(v3_to_v4.items()) if len(rs) > 1]
    merges = []
    for r in reps:
        if len(info[r]["v3_classes"]) > 1:
            merges.append({"v4_class": r, "members": cls[r], "role": [info[r]["role"], info[r]["axis"]],
                           "v3_classes": {o: list(role3(o)) for o in info[r]["v3_classes"]},
                           "pools_dropped": [n for n in info[r]["pooled"] if n != r],
                           "left_training": [o for o in info[r]["v3_classes"] if T3[o]["role"] == "train"
                                             and info[r]["role"] != "train"]})
    vanished = sorted(o for o in T3 if o not in v3_to_v4)
    changed = [{"class": r, "v3": [list(role3(o)) for o in info[r]["v3_classes"]], "v4": [info[r]["role"], info[r]["axis"]]}
               for r in reps if any(role3(o) != (info[r]["role"], info[r]["axis"]) for o in info[r]["v3_classes"])]

    # ---- one-loop layer, loop-induced, alpha_s, heavy-quark, BSM, H7
    tree_of = lambda n: (n[: -len("_nlo")] if n.endswith("_nlo") else n).replace("_qcdborn", "")
    loops, induced = {}, {}
    for e in stage["stage2"]["processes"]:
        n = e["name"]
        if n.endswith("_loop"):
            v3 = R3["loop_induced"]["v3_" + n[3:]]
            induced[n] = {"role": v3["role"], "axis": v3["axis"]}
            continue
        t = tree_of(n); r = rep_of[t]
        v3l = R3["loops"]["v3_" + t[3:] + "_nlo"]
        if info[r]["role"] == "heldout":
            loops[n] = {"role": "heldout", "axis": "H5", "why": "one-loop of a held-out tree", "tree": t, "class": r}
        else:
            loops[n] = {"role": v3l["role"], "axis": v3l["axis"], "why": v3l["why"], "tree": t, "class": r}
    by_cls = defaultdict(list)
    for n, v in loops.items():
        by_cls[v["class"]].append(n)
    loop_shared = []
    for r, ns in sorted(by_cls.items()):
        if len(ns) > 1:
            out = any(loops[n]["role"] == "heldout" for n in ns)
            loop_shared.append({"class": r, "loops": sorted(ns), "roles": [[loops[n]["role"], loops[n]["axis"]] for n in sorted(ns)]})
            v3r = {n: R3["loops"]["v3_" + loops[n]["tree"][3:] + "_nlo"]["role"] for n in ns}
            if out:
                for n in ns:
                    loops[n].update(role="heldout", axis="H5", why=loops[n]["why"] or "one-loop of a class whose other one-loop is held out")
            # one one-loop pool per tree class, as one tree pool: the one catalog_v3 gave this role, then the
            # representative's, then the first by name; the others leave every recipe
            keep = sorted(ns, key=lambda n: (v3r[n] != loops[n]["role"], loops[n]["tree"] != r, n))[0]
            loop_shared[-1]["kept"] = keep
            for n in ns:
                if n != keep:
                    loops[n]["dup_of"] = keep
    loop_changed = [{"loop": n, "v3": [R3["loops"]["v3_" + v["tree"][3:] + "_nlo"]["role"],
                                       R3["loops"]["v3_" + v["tree"][3:] + "_nlo"]["axis"]], "v4": [v["role"], v["axis"]]}
                    for n, v in sorted(loops.items())
                    if (R3["loops"]["v3_" + v["tree"][3:] + "_nlo"]["role"], R3["loops"]["v3_" + v["tree"][3:] + "_nlo"]["axis"]) != (v["role"], v["axis"])]
    alphas, al_dropped = {}, []
    for e in stage["stage3"]["processes"]:
        n = e["name"]
        if "base" not in e:
            continue
        v3 = al3["v3_" + n[3:]]
        if info[rep_of[e["base"]]]["role"] != "train":
            al_dropped.append({"entry": n, "base_class": rep_of[e["base"]], "role": info[rep_of[e["base"]]]["axis"]}); continue
        alphas[n] = {"base": e["base"], "alpha_s": v3["alpha_s"], "role": v3["role"], "axis": v3["axis"]}
    heavy = {}
    for n, e in s1.items():
        if "__mt" in n:
            assert info[rep_of[e["base"]]]["axis"] == "H3", n
            heavy[n] = {"role": "train", "tree": e["base"], "m_t": float(e["physics"]["masses"][6])}
    bsm = sorted(n for n in s1 if n.startswith(("v4_zp_", "v4_thdm_")))
    assert all(n in rep_of or n in heavy or n in bsm for n in s1), sorted(n for n in s1 if not (n in rep_of or n in heavy or n in bsm))
    twin_out = sorted(r for r in reps if any(abs(int(p)) in TWIN_PDG for p in P4[r]["pdg_ids"])     # by its pool
                      or (sum(abs(int(p)) in (23, 24, 22) for p in P4[r]["pdg_ids"][2:]) >= 4
                          and sum(abs(int(p)) == 24 for p in P4[r]["pdg_ids"][2:]) >= 2))
    h7_ok, h7_dropped = [], []
    for v3b in R3["h7_classes"]:
        b = v4n(v3b)
        if b not in rep_of:
            h7_dropped.append({"base": b, "why": "no diagram with a massless b (D136)"}); continue
        r = rep_of[b]
        if info[r]["role"] != "train" or r in twin_out:
            h7_dropped.append({"base": b, "why": f"its class {r} is {info[r]['axis'] or 'in the twin removal'}"}); continue
        h7_ok.append(b)

    # ---- recipes
    hdr = ("# generated by recipes/select_catalog_v4.py from catalog_v4's classes (analysis/catalog_v4/census_classes.json) "
           "and roles (recipes/catalog_v4_roles.json); do not edit\n")
    import gen_catalog_v4 as g4

    def tree_entry(r):
        if r in s1:
            return s1[r]
        sz = {k: v for k, v in next(iter(s1.values())).items() if k in ("n_train", "n_val", "n_test")}
        return {"name": r, "sqrts": g4.window(P4[r]), **sz}

    def dump(name, es):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(hdr)
            yaml.safe_dump({**top, "processes": es}, f, sort_keys=False)
        return len(es)
    stage5 = [tree_entry(r) for r in reps if r not in s1]
    train, held = [], {a: [] for a in ("H1", "H2", "H2p5", "H3", "H5", "H6")}
    for r in reps:
        (train if info[r]["role"] == "train" else held[info[r]["axis"]]).append(tree_entry(r))
    for n in sorted(heavy):
        train.append(s1[n])
    for n in bsm:
        train.append(s1[n])
    for e in stage["stage2"]["processes"]:
        v = loops.get(e["name"]) or induced[e["name"]]
        if "dup_of" in v:
            continue
        (train if v["role"] == "train" else held["H5"]).append(e)
    p5_drop = {n for g in p5_equal for n in g[1:]}
    for e in stage["stage3"]["processes"]:
        n = e["name"]
        if n in alphas:
            (train if alphas[n]["role"] == "train" else held[alphas[n]["axis"]]).append(e)
        elif "base" not in e and n not in p5_drop:
            held["H2p5"].append(e)
    e4 = {e["name"]: e for e in stage["stage4"]["processes"]}
    h7e = [e4[b + "__E20"] for b in h7_ok]
    h7c = [e4[b + "__c30"] for b in h7_ok]
    pdgs = lambda n: {abs(int(x)) for x in P4[n]["pdg_ids"]}
    base = {**{n: v["tree"] for n, v in heavy.items()}, **{n: v["base"] for n, v in alphas.items()},
            **{n: v["tree"] for n, v in loops.items()}}
    seen = [e for e in train if e["name"] not in induced and not e["name"].startswith(("v4_zp_", "v4_thdm_"))
            and pdgs(base.get(e["name"], e["name"])) & TWIN_PDG]
    quart = [e for e in train if e not in seen and rep_of.get(base.get(e["name"], e["name"])) in set(twin_out)]
    out = {e["name"] for e in seen + quart}
    counts = {"catalog_v4_gen_stage5.yaml": dump("catalog_v4_gen_stage5.yaml", stage5),
              "catalog_v4_train.yaml": dump("catalog_v4_train.yaml", train),
              "catalog_v4_twin_train.yaml": dump("catalog_v4_twin_train.yaml", [e for e in train if e["name"] not in out])}
    for a, es in held.items():
        counts[f"catalog_v4_heldout_{a}.yaml"] = dump(f"catalog_v4_heldout_{a}.yaml", es)
    counts["catalog_v4_heldout_H3p.yaml"] = dump("catalog_v4_heldout_H3p.yaml", seen)
    counts["catalog_v4_heldout_H4.yaml"] = dump("catalog_v4_heldout_H4.yaml", quart)
    counts["catalog_v4_heldout_H7E.yaml"] = dump("catalog_v4_heldout_H7E.yaml", h7e)
    counts["catalog_v4_heldout_H7c.yaml"] = dump("catalog_v4_heldout_H7c.yaml", h7c)
    names = [e["name"] for e in train]
    assert len(names) == len(set(names))
    for a, es in held.items():
        assert not {e["name"] for e in es} & set(names), a

    # ---- outputs
    json.dump({"classes": cls, "representatives": reps}, open(OUT_CLS, "w"), indent=0)
    roles = {"trees": {r: {k: info[r][k] for k in ("role", "axis", "why")} for r in reps}, "loops": loops,
             "loop_induced": induced, "alphas": alphas, "heavy_quark": heavy, "bsm": bsm, "twin_out": twin_out,
             "h7_classes": h7_ok, "held_out_families": sorted(out_fams), "gen_stage5": [e["name"] for e in stage5],
             "source": "catalog_v3_roles.json carried to catalog_v4's classes (recipes/select_catalog_v4.py)"}
    json.dump(roles, open(OUT_ROLES, "w"), indent=1)
    lay = lambda r: P4[r]["layer"]
    summary = {
        "trees_in_scope": sum(len(g) for g in G), "v4_classes": len(reps), "v3_classes": len(T3),
        "v3_classes_vanished": len(vanished), "p5_equal": p5_equal,
        "classes_by_layer_role": {f"{k[0]} {k[1]}": v for k, v in sorted(Counter((lay(r), info[r]["axis"] or "train") for r in reps).items(), key=str)},
        "v3_by_role": dict(Counter(T3[o]["axis"] or "train" for o in T3)),
        "v4_by_role": dict(Counter(info[r]["axis"] or "train" for r in reps)),
        "n_split_v3_classes": len(splits), "n_new_classes_without_pool": len(stage5),
        "new_without_pool_by_layer_role": {f"{k[0]} {k[1]}": v for k, v in sorted(Counter((lay(e["name"]), info[e["name"]]["axis"] or "train") for e in stage5).items(), key=str)},
        "n_merges": len(merges), "pools_dropped_by_merges": sum(len(m["pools_dropped"]) for m in merges),
        "v3_training_classes_held_out_by_merge": sorted(o for m in merges for o in m["left_training"]),
        "role_changed_classes": len(changed), "h2_added": h2_added, "family_parent_not_train": fam_parent_out,
        "lost_families": lost_fams, "loop_role_changed": loop_changed, "loop_shared_class": loop_shared,
        "alphas_dropped": al_dropped, "h7_dropped": h7_dropped, "twin_out": len(twin_out),
        "recipes": counts}
    os.makedirs(os.path.dirname(OUT_CHG), exist_ok=True)
    json.dump({"summary": summary, "splits": splits, "merges": merges, "role_changed": changed, "vanished": vanished},
              open(OUT_CHG, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
