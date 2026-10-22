#!/usr/bin/env python
"""catalog_2to4: 2->4 processes beyond the catalog of arXiv:2606.23791, for the multiplicity test of the
dimensionless path (D90, the user 2026-10-22: "we are of course not constrained to this catalog"). Every catalog
2->3 training process has its parent in a model trained on the catalog's 1352 training processes; one more
final-state particle makes a 2->4 process that no such model has seen at that multiplicity, and its level against
its parent's is the extra particle's cost:
  * 20 parents + one gluon  (the parent's QCD order + 1): the cost of an extra alpha_s and a soft/collinear gluon;
  * 20 parents + one photon (the parent's QED order + 1): the same with alpha, the contrast in coupling.
Parents: 40 catalog 2->3 TRAINING processes drawn at random (seed 0), one per relabelling class, 20 for each kind; generated with the
catalog's own MadEvent cards (pp at 13 TeV, pT >= 10 / |eta| <= 5 on j, b, a, l, nu: the extra gluon or photon
is cut like every other jet or photon), 100k train / 10k val / 10k test as the catalog (D54), names d4_<parent
folder>_<g|a>.

Writes recipes/catalog_2to4_processes.yaml (loaded by mg5_pipeline_final._load_catalog_2to4) and
recipes/catalog_2to4_heldout.yaml (the probes). The coupling-order vector is [0, 0, QCD, QED] with the parent's
maximum orders plus the new vertex (classes_default.json "orders", as gen_catalog_daniel.py).
    python recipes/gen_catalog_2to4.py
"""
import argparse, json, os, random, shutil, sys

import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_catalog_daniel import MG5_PDG, SRC, CLASSES, card_masses  # noqa: E402

N_EACH, SEED = 20, 0
ROOT_CARDS = os.path.join(HERE, "catalog_daniel", "src")
EXTRA = {"g": ("g", 21, "QCD"), "a": ("a", 22, "QED")}


def build():
    cat = json.load(open(os.path.join(SRC, "process_catalog_sm.json")))
    split = json.load(open(os.path.join(SRC, "test_split.json")))
    held = {f for gs in split.values() for g in gs for f in g["folders"]}
    by_folder = {r["folder"]: r for r in json.load(open(CLASSES))["processes"]}
    masses = card_masses(os.path.join(SRC, "param_card_sm.dat"))
    parents = sorted((x for x in cat if len(x["outgoing"]) == 3 and x["folder"] not in held), key=lambda x: x["folder"])
    # one parent per relabelling class (classes_default.json exact_key: massless flavour swaps such as u<->c, d<->s
    # give the same |M|^2), so no two probes are the same process; adding the same g or a keeps them distinct
    random.Random(SEED).shuffle(parents)
    pick, seen = [], set()
    for x in parents:
        k = by_folder[x["folder"]]["exact_key"]
        if k not in seen:
            seen.add(k); pick.append(x)
        if len(pick) == 2 * N_EACH:
            break
    entries = {}
    for i, x in enumerate(pick):
        kind = "g" if i < N_EACH else "a"
        mg_name, pdg_new, order_key = EXTRA[kind]
        rec = by_folder[x["folder"]]
        qcd = max(int(o.get("QCD", 0)) for o in rec["orders"]) + (order_key == "QCD")
        qed = max(int(o.get("QED", 0)) for o in rec["orders"]) + (order_key == "QED")
        pdg = [MG5_PDG[p] for p in x["incoming"]] + [MG5_PDG[p] for p in x["outgoing"]] + [pdg_new]
        assert qcd + qed == 4, (x["folder"], rec["orders"], kind)
        entries[f"d4_{x['folder']}_{kind}"] = {
            "kind": "lhe_pp",
            "folder": f"{x['folder']}_{kind}",
            "mg5_generate": [f"generate {x['madgraph_process']} {mg_name}"],
            "proc_str": f"{x['madgraph_process']} {mg_name}",
            "nfinal": 4,
            "pdg_ids": pdg,
            "m_finals": [float(masses.get(abs(p), 0.0)) for p in pdg[2:]],
            "order": [0, 0, qcd, qed],
            "alphas_power": qcd,
            "split": "heldout",
            "test_group": f"2->4: parent + {'gluon' if kind == 'g' else 'photon'}",
            "parent": f"dc_{x['folder']}",
        }
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=None,
                    help="the same 40 processes with ptj = ptb = pta = CUT GeV (the catalog's run card otherwise), named "
                         "d4c<CUT>_..., val and test only (1000 train events): the cut test of the dimensionless path "
                         "(D90; the user: \"what if different cuts are chosen for different processes?\")")
    ap.add_argument("--parents-at", type=float, default=None, metavar="TEV",
                    help="the 40 parents themselves (catalog 2->3 TRAINING processes) at a pp energy of TEV, named "
                         "dE<TEV>_<folder>, val and test only: the collider-energy test (D93, the held-out sets along "
                         "every axis a user can change), the energy the only difference from what the model trained on")
    a = ap.parse_args()
    entries = build()
    lhe = yaml.safe_load(open(os.path.join(HERE, "catalog_daniel_processes.yaml")))["lhe"]
    tag, n_train = "", 100000
    if a.cut is not None:
        tag, n_train = f"c{int(a.cut)}", 1000
        cards = os.path.join(HERE, f"catalog_2to4{tag}")
        os.makedirs(cards, exist_ok=True)
        src = os.path.join(ROOT_CARDS, lhe["run_card"])
        card = open(src).read()
        for key in ("ptj", "ptb", "pta"):
            card, n = re.subn(rf"^\s*[0-9.eE+-]+(\s*=\s*{key}\b)", f"  {a.cut}\\1", card, flags=re.M)
            assert n == 1, (key, n)
        open(os.path.join(cards, lhe["run_card"]), "w").write(card)
        shutil.copy(os.path.join(ROOT_CARDS, lhe["param_card"]), cards)
        lhe = dict(lhe, cards_dir=os.path.relpath(cards, os.path.dirname(HERE)), n_train=n_train,
                   source=f"catalog run card with ptj = ptb = pta = {a.cut} GeV")
        entries = {n.replace("d4_", f"d4{tag}_", 1): dict(e, folder=e["folder"] + f"_{tag}") for n, e in entries.items()}
    rs = 13000
    if a.parents_at is not None:
        tag, n_train, rs = f"E{int(a.parents_at)}", 1000, int(round(a.parents_at * 1000))
        cards = os.path.join(HERE, f"catalog_2to4{tag}")
        os.makedirs(cards, exist_ok=True)
        card = open(os.path.join(ROOT_CARDS, lhe["run_card"])).read()
        for key in ("ebeam1", "ebeam2"):
            card, n = re.subn(rf"^\s*[0-9.eE+-]+(\s*=\s*{key}\b)", f"  {rs / 2:.1f}\\1", card, flags=re.M)
            assert n == 1, (key, n)
        open(os.path.join(cards, lhe["run_card"]), "w").write(card)
        shutil.copy(os.path.join(ROOT_CARDS, lhe["param_card"]), cards)
        lhe = dict(lhe, cards_dir=os.path.relpath(cards, os.path.dirname(HERE)), n_train=n_train,
                   source=f"catalog run card with ebeam1 = ebeam2 = {rs / 2:.1f} GeV")
        par = {}
        for n, e in entries.items():
            x = dict(e)
            x["mg5_generate"] = [e["mg5_generate"][0].rsplit(" ", 1)[0]]
            x["proc_str"] = e["proc_str"].rsplit(" ", 1)[0]
            x["pdg_ids"], x["m_finals"], x["nfinal"] = e["pdg_ids"][:-1], e["m_finals"][:-1], 3
            q, ed = e["order"][2], e["order"][3]
            x["order"] = [0, 0, q - (n.endswith("_g")), ed - (n.endswith("_a"))]
            x["alphas_power"] = x["order"][2]
            x["test_group"] = f"2->3 training process at {a.parents_at:g} TeV"
            fold = e["folder"].rsplit("_", 1)[0]
            x["folder"], x["parent"] = f"{fold}_{tag}", f"dc_{fold}"
            par[f"d{tag}_{fold}"] = x
        entries = par
    with open(os.path.join(HERE, f"catalog_2to4{tag}_processes.yaml"), "w") as f:
        f.write("# catalog_2to4: catalog 2->3 training processes plus one gluon or photon (D90). Written by\n"
                "# recipes/gen_catalog_2to4.py; do not edit by hand. Loaded by mg5_pipeline_final._load_catalog_2to4.\n")
        yaml.safe_dump({"lhe": lhe, "processes": entries}, f, sort_keys=False, width=120)
    with open(os.path.join(HERE, f"catalog_2to4{tag}_heldout.yaml"), "w") as f:
        f.write("# catalog_2to4 probes (recipes/gen_catalog_2to4.py): 20 catalog 2->3 training processes + g, 20 + a\n")
        yaml.safe_dump({"processes": [{"name": n, "sqrts": [rs, rs], "n_train": n_train, "n_val": 10000,
                                       "n_test": 10000} for n in entries]}, f, sort_keys=False, width=120)
    for n, e in entries.items():
        print(f"{n:28s} {e['proc_str']:40s} order {e['order']}")


if __name__ == "__main__":
    main()
