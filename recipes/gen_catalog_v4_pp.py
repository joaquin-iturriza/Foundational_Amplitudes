"""The 13 TeV pp pools with catalog_v4's physics (plan catalog_coverage; the user 2026-10-26, "Yes. delete catalog 3 and
do the regeneration"). recipes/gen_catalog_v3_pp.py built the pp pools (pp13_*) as the pp versions of catalog_v3's
classes: model sm (b at 4.7 GeV), the minimal QED order, MT = ymt = 173 unless a top is external. catalog_v4's classes
are another |M|^2 function wherever a b is external (massless b, sm-no_b_mass: a five-flavour proton) or the full
coupling orders (QED<=b) add electroweak diagrams (analysis/catalog_v4/pp_physics_check.py: 566 and 253 pp classes).
This script writes the pp versions of catalog_v4's entries (pp13v4_<x>, parent v4_<x>), the same way as the catalog_v3
script: the parent's generate string, its model (sm-no_b_mass) and catalog_v4's parameter card
(recipes/catalog_v4_pp/param_card.dat, copied verbatim from a catalog_v4 tree standalone on CC, sha256 bcd8f754...:
MT = ymt = 172.5, WT 1.4915, WZ 2.441404, WW 2.0476 on column-1 DECAY lines; every v4 tree has the same card), Daniel's
13 TeV run card as is (recipes/catalog_v4_pp/run_card.dat = recipes/catalog_v3_pp/run_card.dat: nn23lo1,
maxjetflavor 5, pT > 10 GeV, |eta| < 5), labels at alpha_s(sqrt(s_hat)) run at one loop from 0.118, 1e4 / 2e3 / 2e3
events (the pp13 pools' sizes). The pipeline's physics check (mg5_pipeline_final.lhe_parent_physics_check) compares
each pp standalone's physics block with its parent's before MadEvent runs, and every pool sidecar carries the block.
Which entries are written (pp_physics_check.json's classes, the pp13 classes against catalog_v4):
  regenerated  every pp13 class whose catalog_v4 counterpart has a b or more diagrams (cls b or ew: 819), and every
               hadronic catalog_v4 class (recipes/catalog_v4_roles.json) with no pp13 pool (173)
  reused       the pp13 classes with neither (cls none: 544): the same |M|^2 function (catalog_v4's standalone gives
               their labels back, mean squared log ratio at most 4e-25 on 200 events each), so their pp13 pools stay
  dropped      the 33 b-Higgs pp13 classes with no catalog_v4 diagram
Writes
  catalog_v4_pp_processes.yaml   the regenerated entries (kind lhe_pp), loaded by mg5_pipeline_final._load_catalog_2to4
  catalog_v4_pp_test.yaml        every hadronic catalog_v4 class (training and the held-out H1, H2, H3): pp13v4_<x> when
                                 regenerated, its pp13_<x> pool when reused
  catalog_v4_pp_train.yaml       every hadronic catalog_v4 training class, the same names
  catalog_v4_pp_mixed_train.yaml catalog_v4_train.yaml with every hadronic tree class swapped for its pp version
  catalog_v4_pp_part<k>.yaml     the regenerated entries in 8 parts, one CPU generation job each
A process whose catalog_v3 pp run came back short or stalled starts at that run's margin or run count
(catalog_v3_pp_margins.json, catalog_v3_pp_runs.json, read by the pp13 name); a pp13v4 process whose own first run came
back short takes its margin from catalog_v4_pp_margins.json (read by the pp13v4 name, it wins): 1.25 x the events asked
x the events needed / the events returned (the 1.25 is open, unsourced). A pp13v4 process whose MadEvent run stalls
(refine "fail to reach target": a fixed few thousand events whatever is asked) runs lhe.n_runs times instead
(catalog_v4_pp_runs.json, read by the pp13v4 name, its margin kept): ceil(1.25 x the events needed / the events one
run returned).
Every entry carries its parent's beam_masses (catalog_v4: {}, massless b beams), so the b's particle-table row and the
pipeline's physics check see the massless b the sm-no_b_mass standalone generates; without it a b beam defaulted to
the pipeline's 4.7 GeV shell and the check refused all 277 b-beam entries. beam_masses is not in an lhe_pp pool's
identity (lhe_pp_recipe), so the built pools stay cache hits.
    python3 recipes/gen_catalog_v4_pp.py
"""
import json, os, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from gen_catalog_v3_pp import PARTONS, check_parent_physics

CARDS = "recipes/catalog_v4_pp"
PARTS = 8


def main():
    import mg5_pipeline_final as mg
    T = json.load(open(os.path.join(HERE, "catalog_v4_roles.json")))["trees"]
    C = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "data", "pp_physics_check.json")))["classes"]
    hadronic = sorted(n for n in T if n in mg.PROCESSES
                      and all(abs(int(p)) in PARTONS for p in mg.PROCESSES[n]["pdg_ids"][:2]))
    pp3 = {c["v4"]: n for n, c in C.items() if c["v4"]}            # v4 name -> its pp13 pool
    changed = sorted(c["v4"] for c in C.values() if c["cls"] in ("b", "ew"))
    reused = {c["v4"] for c in C.values() if c["cls"] == "none"}
    new = sorted(n for n in hadronic if n not in pp3)
    regen = sorted(set(changed) | set(new))
    assert not set(regen) & reused
    assert all(n in reused or n in set(regen) for n in hadronic), "a hadronic v4 class with no pp pool"
    margins = json.load(open(os.path.join(HERE, "catalog_v3_pp_margins.json")))
    runs = json.load(open(os.path.join(HERE, "catalog_v3_pp_runs.json")))
    own_margins = json.load(open(os.path.join(HERE, "catalog_v4_pp_margins.json")))
    own_runs = json.load(open(os.path.join(HERE, "catalog_v4_pp_runs.json")))
    pp = lambda n: "pp13v4_" + n[len("v4_"):]
    # built as parallel one-run part jobs (lhe.parts, tools/lhe_parts.py; catalog_coverage step 7): the main session's
    # call of 2026-10-29 for pp13v4_ud_dggu, whose 32 stalling runs took 13 to 24 h in series in one job
    parts = {"pp13v4_ud_dggu"}
    procs = {}
    for n in regen:
        c = mg.PROCESSES[n]
        own = {}
        old = pp3.get(n)
        if pp(n) in own_runs:
            own = {"lhe": {"n_runs": own_runs[pp(n)], **({"margin": own_margins[pp(n)]} if pp(n) in own_margins else {})}}
        elif pp(n) in own_margins:
            own = {"lhe": {"margin": own_margins[pp(n)]}}
        elif old in runs:
            own = {"lhe": {"n_runs": runs[old]}}
        elif old in margins:
            own = {"lhe": {"margin": margins[old]}}
        if pp(n) in parts:
            own = {**own, "lhe": {**own.get("lhe", {}), "parts": True}}
        split = T[n]["role"] if n in T else mg.PROCESSES[old].get("split")
        procs[pp(n)] = {**own,
            "kind": "lhe_pp", "folder": pp(n), "mg5_generate": list(c["mg5_generate"]),
            "proc_str": c["mg5_generate"][0].replace("generate ", ""), "nfinal": int(c["nfinal"]),
            "pdg_ids": [int(p) for p in c["pdg_ids"]], "m_finals": [float(m) for m in c["m_finals"]],
            "order": [int(x) for x in mg.order_vector(c)], "alphas_power": int(c.get("alphas_power", 0)),
            **({"beam_masses": dict(c["beam_masses"])} if c.get("beam_masses") is not None else {}),
            "split": split, "test_group": f"pp 13 TeV v4: {(T.get(n) or {}).get('axis') or 'train'}", "parent": n,
            **({} if n in T else {"note": "not a catalog_v4 class (a member merged into another class); written "
                                          "because its pp13 pool carries catalog_v3's physics"})}
    lhe = {"cards_dir": CARDS, "run_card": "run_card.dat", "param_card": "param_card.dat",
           "model": mg.V4_MODEL, "n_val": 2000, "n_test": 2000, "n_train": 10000, "label_alphas": "running_sqrt_shat",
           "source": "catalog_v4's parameter card (a v4 tree standalone's), Daniel's 13 TeV run card"}
    bad = check_parent_physics(procs, lhe)
    from particle_ids import b_mass_of_process
    bad += [f"{n}: b at {b_mass_of_process(e)} GeV, parent {e['parent']} {b_mass_of_process(mg.PROCESSES[e['parent']])}"
            for n, e in procs.items() if b_mass_of_process(e) != b_mass_of_process(mg.PROCESSES[e["parent"]])]
    if bad:
        sys.exit(f"{len(bad)} pp entries would not carry their parent's physics: " + "; ".join(bad[:5]))
    hdr = "# generated by recipes/gen_catalog_v4_pp.py (the 13 TeV pp pools with catalog_v4's physics); do not edit\n"
    with open(os.path.join(HERE, "catalog_v4_pp_processes.yaml"), "w") as f:
        f.write(hdr)
        yaml.safe_dump({"lhe": lhe, "processes": procs}, f, sort_keys=False)
    sz = {"n_train": 10000, "n_val": 2000, "n_test": 2000}
    name = lambda n: pp(n) if pp(n) in procs else pp3[n]
    test = [{"name": name(n), "sqrts": [13000, 13000], **sz} for n in hadronic]
    train = [n for n in hadronic if T[n]["role"] == "train"]
    tr = [{"name": name(n), "sqrts": [13000, 13000], **sz} for n in train]
    for fn, es in (("catalog_v4_pp_test.yaml", test), ("catalog_v4_pp_train.yaml", tr)):
        with open(os.path.join(HERE, fn), "w") as f:
            f.write(hdr)
            yaml.safe_dump({"processes": es}, f, sort_keys=False)
    base = yaml.safe_load(open(os.path.join(HERE, "catalog_v4_train.yaml")))
    swap = {n: {"name": name(n), "sqrts": [13000, 13000], **sz} for n in train}
    mixed = [swap.get(e["name"], e) for e in base["processes"]]
    assert sum(e["name"] in swap for e in base["processes"]) == len(train), "a hadronic training class not in the recipe"
    with open(os.path.join(HERE, "catalog_v4_pp_mixed_train.yaml"), "w") as f:
        f.write(hdr)
        yaml.safe_dump({**{k: v for k, v in base.items() if k != "processes"}, "processes": mixed}, f, sort_keys=False)
    # the generation in PARTS CPU jobs (CLAUDE.md 0c): every regenerated process once, dealt round-robin by multiplicity
    # so each part carries its share of the slow 2->4 MadEvent runs
    gen = sorted(({"name": p, "sqrts": [13000, 13000], **sz} for p in procs),
                 key=lambda e: (-procs[e["name"]]["nfinal"], e["name"]))
    for k in range(PARTS):
        with open(os.path.join(HERE, f"catalog_v4_pp_part{k}.yaml"), "w") as f:
            f.write(hdr)
            yaml.safe_dump({"processes": gen[k::PARTS]}, f, sort_keys=False)
    print(json.dumps({"hadronic_v4_classes": len(hadronic), "regenerated": len(procs), "changed": len(changed),
                      "changed_not_v4_class": sum(n not in T for n in changed), "new": len(new),
                      "reused_in_test": sum(e["name"].startswith("pp13_") for e in test), "test": len(test),
                      "train": len(tr), "mixed_pp": sum(e["name"].startswith("pp13") for e in mixed),
                      "mixed": len(mixed), "with_margin": sum("margin" in (e.get("lhe") or {}) for e in procs.values()),
                      "with_runs": sum("n_runs" in (e.get("lhe") or {}) for e in procs.values())}))


if __name__ == "__main__":
    main()
