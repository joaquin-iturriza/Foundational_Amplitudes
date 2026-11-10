#!/usr/bin/env python3
"""The two catalog_v4 pretraining recipes of kit_benchmark, from the parent pretraining's recipe
(recipes/catalog_v4_default25_train.yaml, trial 1262's), with the KIT targets' whole families held out.

    python recipes/gen_kit_pretrain.py [--check]

A family is every pool whose amplitude is the KIT target's up to a flavour relabelling, a coupling or mass scan
variant (`__as*`, `__mt*`) or a crossing (Bakar: "if uug is the target, then ddg, bbg and the scan variants should go
too"): the pools with e+ e- and exactly the target's quarks and gluons, in any initial/final assignment, at tree level
and at one loop. In the parent recipe these are (listed by recipes' names, checked by --check against a scan of every
entry):
  ee->tt        v4_epem_ttx__mt50, __mt300, __mt600 (the physical-mass v4_epem_ttx is held out of the parent already)
  ee->qqbar g   v4_epem_bbxg (+ __as090, __as150), v4_bbx_epemg, v4_bg_bepem, v4_uux_epemg, v4_ug_epemu, and the
                one-loop v4_uux_epemg_nlo, v4_ug_epemu_nlo
  ee->qqbar gg  v4_epem_ccxgg, v4_ug_epemgu
Writes
  recipes/kit_pretrain_heldout.yaml   the parent's pools minus the three families: protocol A's pretrained parent for
                                      every KIT target (tt Born, tt virtual, uug, uugg)
  recipes/kit_pretrain_fair_nlo.yaml  Bakar's fair NLO pretraining for the ee->tt virtual target: every tree pool of
                                      the parent, the target's own tree v4_epem_ttx added (recipes/
                                      catalog_v4_datasaving_epem_ttx.yaml's entry), and 10 one-loop pools drawn at
                                      random (numpy seed 0) from the parent's one-loop pools (n_loops > 0 in the catalog
                                      entry, the loop-induced `*_loop` pools included) outside the families
"""
import argparse
import os
import re

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.join(ROOT, "recipes", "catalog_v4_default25_train.yaml")
TTX = os.path.join(ROOT, "recipes", "catalog_v4_datasaving_epem_ttx.yaml")
FAMILIES = {
    "ee_tt": ["v4_epem_ttx__mt50", "v4_epem_ttx__mt300", "v4_epem_ttx__mt600"],
    "ee_qqg": ["v4_epem_bbxg", "v4_epem_bbxg__as090", "v4_epem_bbxg__as150", "v4_bbx_epemg", "v4_bg_bepem",
               "v4_uux_epemg", "v4_ug_epemu", "v4_uux_epemg_nlo", "v4_ug_epemu_nlo"],
    "ee_qqgg": ["v4_epem_ccxgg", "v4_ug_epemgu"],
}
N_LOOP = 10
_TOK = re.compile(r"(ux|dx|sx|cx|bx|tx|u|d|s|c|b|t|g)")


def family_of(name):
    """The KIT family of a pool name (ee_tt, ee_qqg, ee_qqgg) by its particle content, None when it is in none"""
    m = re.match(r"v4_([a-z]+)_([a-z]+?)(_nlo)?$", name.split("__")[0])
    if not m or "epem" not in m.group(1) + "|" + m.group(2):
        return None
    rest = (m.group(1) + m.group(2)).replace("epem", "", 1)
    if _TOK.sub("", rest):
        return None
    q = _TOK.findall(rest)
    ng, quarks = q.count("g"), [x for x in q if x != "g"]
    if len(quarks) != 2:
        return None
    if ng == 0:
        return "ee_tt" if set(quarks) <= {"t", "tx"} else None
    if any(x in ("t", "tx") for x in quarks):
        return None
    return {1: "ee_qqg", 2: "ee_qqgg"}.get(ng)


def build():
    par = yaml.safe_load(open(PARENT))
    ps = par["processes"]
    names = [p["name"] for p in ps]
    listed = {n for v in FAMILIES.values() for n in v}
    scanned = {n for n in names if family_of(n)}
    assert scanned == listed, f"family scan and list differ: {sorted(scanned ^ listed)}"
    head = {k: v for k, v in par.items() if k != "processes"}
    held = dict(head, processes=[p for p in ps if p["name"] not in listed])
    # tree or loop by the catalog entry's loop count (mg5_pipeline_final.PROCESSES), not the name: the loop-induced
    # `*_loop` pools are one-loop too
    import sys
    sys.path.insert(0, ROOT)
    import mg5_pipeline_final as mg
    mg.register_recipe_processes(ps)
    n_loops = {p["name"]: int((mg.PROCESSES.get(p["name"]) or mg.PROCESSES[p.get("base", p["name"])]).get("n_loops", 0) or 0)
               for p in ps}
    trees = [p for p in ps if n_loops[p["name"]] == 0]
    loops = sorted(p["name"] for p in ps if n_loops[p["name"]] > 0 and p["name"] not in listed)
    pick = sorted(np.random.default_rng(0).choice(loops, N_LOOP, replace=False).tolist())
    (ttx,) = yaml.safe_load(open(TTX))["processes"]
    assert ttx["name"] == "v4_epem_ttx" and ttx["name"] not in names
    fair = dict(head, processes=trees + [ttx] + [p for p in ps if p["name"] in pick])
    return held, fair, pick, len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="build and report, write nothing")
    a = ap.parse_args()
    held, fair, pick, n = build()
    print(f"parent {n} pools; heldout {len(held['processes'])}; fair_nlo {len(fair['processes'])} "
          f"(one loop: {', '.join(pick)})")
    if a.check:
        return
    for name, rec, what in (("kit_pretrain_heldout", held, "the parent's pools minus the KIT families"),
                            ("kit_pretrain_fair_nlo", fair, "trees + v4_epem_ttx + 10 one-loop pools (seed 0)")):
        with open(os.path.join(ROOT, "recipes", f"{name}.yaml"), "w") as f:
            f.write(f"# kit_benchmark: {what}; written by recipes/gen_kit_pretrain.py from "
                    f"recipes/catalog_v4_default25_train.yaml\n")
            yaml.safe_dump(rec, f, sort_keys=False)


if __name__ == "__main__":
    main()
