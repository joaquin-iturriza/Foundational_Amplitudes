#!/usr/bin/env python3
"""Recipes of the regeneration of catalog_v4's 1301 hadronic tree classes with the channel-grid flat sampler
(plan catalog_coverage step 65; sampler settled by step 75: the flat part filled flat in log|M|^2 over each class's pp
range, recipes/catalog_v4_flat_range.json, by the quota sampler with the multichannel proposal (resonance, initial-state
collinear, both multiperipheral ladders, joint-resonance and its forward variant), each channel's uniform numbers
through an adaptive grid, at most 12 matrix elements per flat event: policy grid_both_res2f_budget12 of
analysis/catalog_v4/multichannel_check.py).

Only the flat part's sampler changes: every other field of an entry (window, sizes, base, physics, the mixture's
f_flat, forward share, default share) stays as in the recipe it comes from. The new pools carry `_smixq` in their file
names (mg.recipe_output_path), so the pools every earlier read used stay where they are.

    python3 tools/regen_recipes.py pull     on CC-IN2P3's login node through site run (read-only): each class's
                                            resonance channels from its diagram sidecar (data/diagrams), whether its
                                            compiled standalone exists, free disk -> analysis/catalog_v4/data/regen_g12_inputs.json
    python3 tools/regen_recipes.py write    -> recipes/catalog_v4_g12_regen.yaml (the 1301 classes once, from the
                                            generation stages 1 and 5) and a _g12 copy of every recipe holding them
                                            (training mixtures and the catalog-sampled held-out recipes)
    python3 tools/regen_recipes.py check    every _g12 reader resolves each hadronic pool to the regeneration's (recipe
                                            id and file name), every regenerated pool is new (_smixq, no old path)
"""
import argparse, json, os, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "recipes")
FLAT_RANGE = os.path.join(REC, "catalog_v4_flat_range.json")
INPUTS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "regen_g12_inputs.json")
STAGES = ["catalog_v4_gen_stage1.yaml", "catalog_v4_gen_stage5.yaml"]     # where the 1301 classes are generated
# recipes that read the hadronic tree pools: each gets a _g12 copy reading the regenerated ones
READERS = ["catalog_v4_base_train.yaml", "catalog_v4_default25_train.yaml", "catalog_v4_default50_train.yaml",
           "catalog_v4_default25_pponly_train.yaml", "catalog_v4_heldout_H1.yaml", "catalog_v4_heldout_H2.yaml",
           "catalog_v4_heldout_H3.yaml", "catalog_v4_heldout_H3p.yaml", "catalog_v4_heldout_H4.yaml"]
OUT_REGEN = "catalog_v4_g12_regen.yaml"
EE_PREFIX = "v4_epem"
# grid_both_res2f_budget12 (multichannel_check.py, step 75): the sampler's switches; the constants behind them enter
# the policy through mg.sampling_policy
G12 = {"multichannel": True, "mc_isc": True, "mc_mperi": True, "mc_mperi_both": True, "mc_res2": True,
       "mc_res2_fwd": True, "flat_grid": True, "flat_budget": 12}


def hadronic():
    return sorted(c for c in json.load(open(FLAT_RANGE)) if not c.startswith(EE_PREFIX))


def stage_entries():
    """{name: (entry, recipe-level sampling)} of the 1301 classes in the generation stages"""
    had = set(hadronic()); out = {}
    for f in STAGES:
        spec = yaml.safe_load(open(os.path.join(REC, f)))
        for p in spec["processes"]:
            if p["name"] in had:
                assert p["name"] not in out, f"{p['name']} in two generation stages"
                out[p["name"]] = (p, spec.get("sampling"))
    missing = had - set(out)
    if missing:
        sys.exit(f"regen_recipes: {len(missing)} hadronic classes in no generation stage: {sorted(missing)[:5]}")
    return out


PULL = r'''
import json, os, sys, shutil
sys.path.insert(0, os.getcwd())
import yaml
import mg5_pipeline_final as mg
names = NAMES
specs = {}
for f in %(stages)r:
    s = yaml.safe_load(open(os.path.join("recipes", f)))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in s["processes"] if p["name"] in names],
                                 default_sampling=s.get("sampling"))
    for p in s["processes"]:
        specs[p["name"]] = p
res, err, sa = {}, {}, {}
for n in names:
    p = specs[n]
    side = os.path.join("data", "diagrams", p.get("base", n) + ".diagrams.json")
    try:
        res[n] = [[list(S), q] for S, q in mg.resonance_channels(json.load(open(side)), mg.PROCESSES[n]["pdg_ids"])]
    except Exception as e:
        err[n] = "%%s: %%s" %% (type(e).__name__, e)
    d = "%%s/%%s_standalone" %% (mg.WORK_DIR, mg.standalone_name(n))
    b, sub, drv, _ = mg.detect_compiled_backend(d)
    sa[n] = bool(sub) and not (b == "cpp" and drv is None)
free = {k: shutil.disk_usage(os.environ[k]).free for k in ("DATA_DIR", "SCRATCH")}
print("REGEN_G12 " + json.dumps({"resonances": res, "errors": err, "standalone_built": sa, "free_bytes": free}))
'''


def cmd_pull(a):
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    names = hadronic()
    # the class list goes over inside the script (site_python sends only the script)
    script = "import json\nNAMES = json.loads(%r)\n" % json.dumps(names) + PULL % {"stages": STAGES}
    d = site_python(script, [], "REGEN_G12", timeout=1800)
    d["classes"] = len(names)
    json.dump(d, open(INPUTS, "w"), indent=1, sort_keys=True)
    print(f"{len(d['resonances'])} classes with channels read, {len(d['errors'])} errors, "
          f"{sum(d['standalone_built'].values())} standalones built; free {d['free_bytes']} -> {INPUTS}")


def g12_sampling(name, base_sampling, physics, inp, fr):
    import mg5_pipeline_final as mg
    s = dict(base_sampling or {})
    s.update(G12)
    s["flat_range"] = [float(x) for x in fr[name]]
    s["mc_resonances"] = inp["resonances"][name]
    s["mc_res_params"] = {str(k): [float(x) for x in v] for k, v in sorted(mg.mc_res_params(physics).items())}
    return s


def cmd_write(a):
    sys.path.insert(0, ROOT)
    inp = json.load(open(INPUTS))
    if inp["errors"]:
        sys.exit(f"regen_recipes: {len(inp['errors'])} classes without resonance channels: "
                 f"{list(inp['errors'].items())[:3]}")
    fr = json.load(open(FLAT_RANGE)); had = set(hadronic())
    head = ("# generated by tools/regen_recipes.py write (plan catalog_coverage step 65: the flat part by the channel-grid "
            "sampler at 12 labels per flat event); do not edit\n")
    ents = stage_entries(); procs = []
    for n in sorted(ents):
        p, top = ents[n]
        q = dict(p); q["sampling"] = g12_sampling(n, p.get("sampling") or top, p.get("physics"), inp, fr)
        procs.append(q)
    with open(os.path.join(REC, OUT_REGEN), "w") as f:
        f.write(head); yaml.safe_dump({"processes": procs}, f, sort_keys=False, default_flow_style=None, width=200)
    print(f"{OUT_REGEN}: {len(procs)} classes")
    for r in READERS:
        spec = yaml.safe_load(open(os.path.join(REC, r))); k = 0
        for p in spec["processes"]:
            if p["name"] in had:
                p["sampling"] = g12_sampling(p["name"], p.get("sampling") or spec.get("sampling"), p.get("physics"),
                                             inp, fr)
                k += 1
        out = r.replace("catalog_v4_", "catalog_v4_g12_", 1)
        with open(os.path.join(REC, out), "w") as f:
            f.write(head.replace("; do not edit", f" from {r}; do not edit"))
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None, width=200)
        print(f"{out}: {k} of {len(spec['processes'])} entries on the new sampler")


def _ids(path, strip_default=False):
    """{(name, role): (recipe id, pool file name)} of a recipe's hadronic entries, as a run resolves them (a default-share
    entry: its control pool, the one the loader composes from)"""
    import mg5_pipeline_final as mg
    import datagen
    spec = yaml.safe_load(open(path)); had = set(hadronic()); out = {}
    procs = [p for p in spec["processes"] if p["name"] in had]
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in procs], default_sampling=spec.get("sampling"))
    for p in procs:
        if strip_default:
            s = dict(mg.PROCESSES[p["name"]]["sampling"]); s.pop("f_default", None); s.pop("default_source", None)
            mg.PROCESSES[p["name"]]["sampling"] = s
        for role in ("train", "val", "test"):
            r = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_" + role]),
                                          role=role, seed=42)
            out[(p["name"], role)] = (mg.recipe_id(r), os.path.basename(mg.recipe_output_path(r, datagen.dest_for_role(role))))
    return out


def cmd_check(a):
    """Every _g12 reader resolves each hadronic pool to the regeneration recipe's pool (same recipe id and file), and
    the regeneration's files are new (_smixq) and differ from the old recipe's"""
    sys.path.insert(0, ROOT)
    reg = _ids(os.path.join(REC, OUT_REGEN)); bad = 0
    assert all("_smixq_" in f for _, f in reg.values()), "a regenerated pool without the _smixq tag"
    old = {}
    for f in STAGES:
        old.update(_ids(os.path.join(REC, f)))
    same = [k for k in reg if old.get(k, (None, None))[1] == reg[k][1]]
    assert not same, f"{len(same)} regenerated pools at an old pool's path: {same[:3]}"
    for r in READERS:
        out = r.replace("catalog_v4_", "catalog_v4_g12_", 1)
        ids = _ids(os.path.join(REC, out), strip_default=True)
        miss = [k for k, v in ids.items() if reg.get(k) != v]
        bad += len(miss)
        print(f"{out}: {len(ids)} hadronic pools, {len(ids) - len(miss)} resolve to the regeneration's"
              + (f"; differing e.g. {miss[:2]}: {[ids[k] for k in miss[:2]]} vs {[reg.get(k) for k in miss[:2]]}" if miss else ""))
    print(f"REGEN_G12_CHECK {'OK' if not bad else 'FAIL'}: {len(reg)} pools of {len(reg) // 3} classes, all _smixq, "
          f"none at an old path; {bad} reader pools differ")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull"); sub.add_parser("write"); sub.add_parser("check")
    a = ap.parse_args()
    {"pull": cmd_pull, "write": cmd_write, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
