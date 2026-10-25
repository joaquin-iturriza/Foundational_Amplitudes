"""The pp study (the user, 2026-10-25: most users will apply the model to pp-sampled data; show whether training on
catalog_v3's shaped sqrt(s_hat) windows serves pp-sampled use, maybe more than training on pp): catalog_v3's classes with
quark or gluon beams as 13 TeV proton-proton processes. Each keeps its partonic channel (the same generate string, so the
same |M|^2 function) and catalog_v3's parameter card (recipes/catalog_v3_pp/param_card.dat, copied from a catalog_v3
standalone); its events are MadEvent's unweighted pp events (the pipeline's lhe_pp path, as Daniel's catalog): PDFs
(nn23lo1), the cross-section weighting, lab-frame cuts (Daniel's run card, recipes/catalog_v3_pp/run_card.dat: pT > 10
GeV, |eta| < 5 for jets, b, photons and charged leptons). The |M|^2 labels are the window pools' function: the parent's
param-card patches (its own card file) and alpha_s(sqrt(s_hat)) run at one loop from 0.118 (lhe.label_alphas; MadEvent's
AQCDUP kept beside each pool as <pool>.aqcdup.npy). The e+e-
classes have no pp analogue and are not written.
Writes
  catalog_v3_pp_processes.yaml   every hadronic catalog_v3 tree class as pp13_<name> (kind lhe_pp), loaded by
                                 mg5_pipeline_final._load_catalog_2to4's lhe_pp loader
  catalog_v3_pp_test.yaml        the test set: every hadronic held-out class of H1, H2, H3 and every hadronic training
                                 tree class (the in-distribution reference)
  catalog_v3_pp_train.yaml       every hadronic training tree class, for the pp-trained comparison (the heavy-quark
                                 layer, the alpha_s points, the BSM layer, the one-loop pools and the e+e- classes
                                 stay on their windows there: their pp versions need patched masses, another alpha_s
                                 rule or MadLoop labels of pp events, not built yet)
  catalog_v3_pp_mixed_train.yaml catalog_v3_train.yaml with every hadronic tree class swapped for its pp version
  catalog_v3_pp_part<k>.yaml     the same processes in 8 parts, one CPU generation job each
Every pool: 1e4 train (the short pretraining's size), 2e3 val and test (the 2000 events a held-out read scores).
    python3 recipes/gen_catalog_v3_pp.py
"""
import json, os, shutil, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PARTONS = {1, 2, 3, 4, 5, 21}


def main():
    import mg5_pipeline_final as mg
    ro = json.load(open(os.path.join(HERE, "catalog_v3_roles.json")))
    T = ro["trees"]
    hadronic = sorted(n for n in T if n.startswith("v3_") and n in mg.PROCESSES
                      and all(abs(int(p)) in PARTONS for p in mg.PROCESSES[n]["pdg_ids"][:2]))
    # the processes whose MadEvent run came back short at the default margin: their own margin, from the shortfall
    # measured on the first pass (recipes/catalog_v3_pp_margins.json: name -> margin)
    mp = os.path.join(HERE, "catalog_v3_pp_margins.json")
    margins = json.load(open(mp)) if os.path.exists(mp) else {}
    procs, cards = {}, {}
    base_card = os.path.join(HERE, "catalog_v3_pp", "param_card.dat")
    for n in hadronic:
        c = mg.PROCESSES[n]
        # the parent's own param-card patches (the classes with an external top run at MT = ymt = 172.5): one patched
        # card per patch set, so MadEvent, the standalone and the mass-width read all see the parent's parameters
        pat = {k: v for k, v in sorted((c.get("param_card_patches") or {}).items())}
        own = {}
        if pat:
            tag = "_".join(f"{k}{v}" for k, v in pat.items()).replace(".", "p")
            fn = f"param_card_{tag}.dat"
            if fn not in cards:
                shutil.copy2(base_card, os.path.join(HERE, "catalog_v3_pp", fn))
                left = mg.patch_param_card_slha(os.path.join(HERE, "catalog_v3_pp", fn), pat)
                assert not left, (n, left)
                cards[fn] = pat
            own = {"lhe": {"param_card": fn}}
        if margins.get("pp13_" + n[len("v3_"):]):
            own.setdefault("lhe", {})["margin"] = margins["pp13_" + n[len("v3_"):]]
        procs["pp13_" + n[len("v3_"):]] = {**own, 
            "kind": "lhe_pp", "folder": "pp13_" + n[len("v3_"):], "mg5_generate": list(c["mg5_generate"]),
            "proc_str": c["mg5_generate"][0].replace("generate ", ""), "nfinal": int(c["nfinal"]),
            "pdg_ids": [int(p) for p in c["pdg_ids"]], "m_finals": [float(m) for m in c["m_finals"]],
            "order": [int(x) for x in mg.order_vector(c)], "alphas_power": int(c.get("alphas_power", 0)),
            "split": T[n]["role"], "test_group": f"pp 13 TeV: {T[n]['axis'] or 'train'}", "parent": n}
    lhe = {"cards_dir": "recipes/catalog_v3_pp", "run_card": "run_card.dat", "param_card": "param_card.dat",
           "model": "sm", "n_val": 2000, "n_test": 2000, "n_train": 10000, "label_alphas": "running_sqrt_shat",
           "source": "catalog_v3's parameter card, Daniel's 13 TeV run card"}
    hdr = "# generated by recipes/gen_catalog_v3_pp.py (the pp study); do not edit\n"
    with open(os.path.join(HERE, "catalog_v3_pp_processes.yaml"), "w") as f:
        f.write(hdr)
        yaml.safe_dump({"lhe": lhe, "processes": procs}, f, sort_keys=False)
    held = [n for n in hadronic if T[n]["role"] == "heldout" and T[n]["axis"] in ("H1", "H2", "H3")]
    train = [n for n in hadronic if T[n]["role"] == "train"]
    ref = train                                   # every hadronic training class: its val pool exists for the pp training
    pp = lambda n: "pp13_" + n[len("v3_"):]
    sz = {"n_train": 10000, "n_val": 2000, "n_test": 2000}     # one size per process, whichever recipe names it
    test = [{"name": pp(n), "sqrts": [13000, 13000], **sz} for n in held + ref]
    tr = [{"name": pp(n), "sqrts": [13000, 13000], **sz} for n in train]
    for name, es in (("catalog_v3_pp_test.yaml", test), ("catalog_v3_pp_train.yaml", tr)):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(hdr)
            yaml.safe_dump({"processes": es}, f, sort_keys=False)
    # the pp-trained comparison's training set: catalog_v3's training recipe with every hadronic tree class swapped for
    # its pp version; every other pool (e+e- classes, heavy-quark layer, alpha_s points, BSM, one-loop) stays as it is
    base = yaml.safe_load(open(os.path.join(HERE, "catalog_v3_train.yaml")))
    swap = {n: {"name": pp(n), "sqrts": [13000, 13000], **sz} for n in train}
    mixed = [swap.get(e["name"], e) for e in base["processes"]]
    with open(os.path.join(HERE, "catalog_v3_pp_mixed_train.yaml"), "w") as f:
        f.write(hdr)
        yaml.safe_dump({**{k: v for k, v in base.items() if k != "processes"}, "processes": mixed}, f, sort_keys=False)
    print(f"mixed training set: {len(mixed)} pools, {sum(e['name'].startswith('pp13_') for e in mixed)} of them pp")
    # the generation in 8 CPU jobs (CLAUDE.md 0c): every process once, dealt round-robin by multiplicity so each part
    # carries its share of the slow 2->4 MadEvent runs
    order = sorted(test, key=lambda e: (-mg.PROCESSES[e["name"]]["nfinal"], e["name"]))
    for k in range(8):
        with open(os.path.join(HERE, f"catalog_v3_pp_part{k}.yaml"), "w") as f:
            f.write(hdr)
            yaml.safe_dump({"processes": order[k::8]}, f, sort_keys=False)
    print(f"patched cards: {cards}")
    print(f"{len(procs)} hadronic classes; test {len(test)} ({len(held)} held-out, {len(ref)} training); "
          f"train {len(tr)}")


if __name__ == "__main__":
    main()
