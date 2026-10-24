"""catalog_v3's pretraining recipes (D115) from its generated stages and its roles (recipes/catalog_v3_roles.json,
recipes/catalog_v3_alphas.json):
  catalog_v3_train.yaml        every training pool: the train tree classes, the heavy-quark layer (m_Q = 50, 300, 600
                               GeV), the 12 BSM processes, the train one-loop pools, the loop-induced pools but one, the
                               alpha_s points at 0.09 and 0.15
  catalog_v3_twin_train.yaml   the same without every pool, in any layer, with an external tau or nu_tau (H3p, the
                               never-seen particle) or built on a quartic-gauge tree class of twin_out (H4)
  catalog_v3_heldout_<A>.yaml  one per held-out axis: H1 new combinations, H2 multiplicity (2->4 families), H2p5 the 2->5
                               probes, H3 the physical top, H5 order (the held-out one-loop pools and one loop-induced
                               class), H6 alpha_s = 0.13; H3p and H4, the twin's, read by the twin only
Every recipe keeps the stage files' top-level keys (sampling: mixture), so a run resolves the _smix pools generated.
The held-out loop-induced class is drawn with the selection's seed among the 17 and recorded in catalog_v3_roles.json
("loop_induced"), as the design asks (H5: "one loop-induced class").
H7, the regime axis (D98: "100 TeV and pT >= 30"), on catalog_v3's sampler (sqrt(s_hat) windows, not pp beams): 40
training tree classes drawn with the selection's seed (h7_classes in catalog_v3_roles.json) among those outside the
twin's removals with at least one massless visible final leg (the legs the cuts act on), each as
  <name>__E10   sqrt(s_hat) in [1, 10] TeV, above every training window (the energy extrapolation)
  <name>__c30   its own window with pT >= 30 GeV in place of the catalog's 10 GeV (physics.cuts)
written as the generation recipe catalog_v3_gen_stage4.yaml (the catalog's sizes, 1e5 / 1e4 / 1e4) and the held-out files
H7E and H7c.
    python3 recipes/make_catalog_v3_recipes.py
"""
import json, os, random, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TWIN_PDG = {15, 16}                                   # the twin's never-seen particle: the tau and its neutrino


def stage(name):
    return yaml.safe_load(open(os.path.join(HERE, f"catalog_v3_gen_{name}.yaml")))


def entries(name):
    return stage(name)["processes"]


def main():
    import mg5_pipeline_final as mg
    rpath = os.path.join(HERE, "catalog_v3_roles.json")
    ro = json.load(open(rpath))
    al = json.load(open(os.path.join(HERE, "catalog_v3_alphas.json")))
    T, L, Hq, twin = ro["trees"], ro["loops"], ro["heavy_quark"], set(ro["twin_out"])
    induced = sorted(e["name"] for e in entries("stage2") if e["name"] not in L)
    if "loop_induced" not in ro:                     # drawn once with the selection's seed, then fixed in the roles
        h = random.Random(ro["seed"]).choice(induced)
        ro["loop_induced"] = {n: {"role": "heldout" if n == h else "train", "axis": "H5" if n == h else None}
                              for n in induced}
        json.dump(ro, open(rpath, "w"), indent=1)
    LI = ro["loop_induced"]
    if "h7_classes" not in ro:
        # eligible: a training class outside the twin's removals (so the twin reads H7 as H7, not H3p/H4) with at least
        # one massless visible final leg, the legs the fiducial cuts act on (else pT >= 30 changes nothing)
        def cut_legs(n):
            cfg = mg.PROCESSES[n]
            return sum(m == 0.0 and abs(int(q)) not in (12, 14, 16) for m, q in zip(cfg["m_finals"], cfg["pdg_ids"][2:]))
        ok = sorted(n for n, v in T.items() if v["role"] == "train" and n not in twin and cut_legs(n) >= 1)
        ro["h7_classes"] = sorted(random.Random(ro["seed"]).sample(ok, 40))
        json.dump(ro, open(rpath, "w"), indent=1)
    base = {**{n: v["tree"] for n, v in Hq.items()}, **{n: v["base"] for n, v in al.items()}}

    def pdg(n):
        return {abs(int(x)) for x in mg.PROCESSES[base.get(n, n)]["pdg_ids"]}

    def tree_of(n):                                  # the tree class a pool is built on
        b = base.get(n, n)
        return b[:-len("_nlo")] if b.endswith("_nlo") else b

    train, held = [], {a: [] for a in ("H1", "H2", "H2p5", "H3", "H5", "H6")}
    for e in entries("stage1"):
        n = e["name"]
        if n in T:
            (train if T[n]["role"] == "train" else held[T[n]["axis"]]).append(e)
        else:                                            # the heavy-quark layer (all train) and the 12 BSM processes
            assert n in Hq or n.startswith(("zp_", "thdm_")), n
            train.append(e)
    for e in entries("stage2"):
        n = e["name"]
        if n in L:
            (train if L[n]["role"] == "train" else held["H5"]).append(e)
        else:                                            # the 17 loop-induced
            assert n in LI, n
            (train if LI[n]["role"] == "train" else held["H5"]).append(e)
    for e in entries("stage3"):
        n = e["name"]
        if n in al:
            (train if al[n]["role"] == "train" else held[al[n]["axis"]]).append(e)
        else:                                            # the 40 2->5 probes
            held["H2p5"].append(e)
    top = {k: v for k, v in stage("stage1").items() if k != "processes"}
    for st in ("stage2", "stage3"):
        assert {k: v for k, v in stage(st).items() if k != "processes"} == top, st
    hdr = "# generated by recipes/make_catalog_v3_recipes.py (D115); do not edit\n"

    def dump(name, es):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(hdr)
            yaml.safe_dump({**top, "processes": es}, f, sort_keys=False)
        print(f"{name}: {len(es)}")
    seen = [e for e in train if pdg(e["name"]) & TWIN_PDG]
    quart = [e for e in train if not pdg(e["name"]) & TWIN_PDG and tree_of(e["name"]) in twin]
    out = {e["name"] for e in seen + quart}
    dump("catalog_v3_train.yaml", train)
    dump("catalog_v3_twin_train.yaml", [e for e in train if e["name"] not in out])
    for a, es in held.items():
        dump(f"catalog_v3_heldout_{a}.yaml", es)
    dump("catalog_v3_heldout_H3p.yaml", seen)
    dump("catalog_v3_heldout_H4.yaml", quart)
    s1 = {e["name"]: e for e in entries("stage1")}
    sz = {k: s1[n][k] for n in ro["h7_classes"][:1] for k in ("n_train", "n_val", "n_test")}   # the catalog's sizes
    h7e = [{"name": n + "__E10", "base": n, "sqrts": [1000, 10000], **sz} for n in ro["h7_classes"]]
    h7c = [{"name": n + "__c30", "base": n, "sqrts": list(s1[n]["sqrts"]), **sz, "physics": {"cuts": {"pt_min": 30.0}}}
           for n in ro["h7_classes"]]
    dump("catalog_v3_gen_stage4.yaml", h7e + h7c)
    dump("catalog_v3_heldout_H7E.yaml", h7e)
    dump("catalog_v3_heldout_H7c.yaml", h7c)


if __name__ == "__main__":
    main()
