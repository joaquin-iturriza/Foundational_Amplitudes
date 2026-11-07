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
                                            generation stages 1 and 5)
    python3 tools/regen_recipes.py write --r1  -> recipes/catalog_v4_r1_regen.yaml (step 75's revision, R1 below,
                                            ranges from recipes/catalog_v4_flat_range_dsh.json; pools _smixr)
    python3 tools/regen_recipes.py readers  a _g12 copy of every recipe holding them (found by scanning recipes/,
                                            NOT_READERS excepted with reasons); refused while a reader holds an e+e-
                                            tree class no regeneration covers (step 68)
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
# recipes holding hadronic tree classes that get no _g12 copy, each with its reason (every other recipes/catalog_v4_*.yaml
# holding one is a reader: readers_of() scans for them, so a new recipe is never left on the old pools unnoticed)
NOT_READERS = {"catalog_v4_census.yaml": "the numeric equality census: 0 events per class, no pools",
               "catalog_v4_gen_stage1.yaml": "a generation stage: the regeneration recipe replaces it",
               "catalog_v4_gen_stage5.yaml": "a generation stage: the regeneration recipe replaces it"}
OUT_REGEN = "catalog_v4_g12_regen.yaml"
EE_PREFIX = "v4_epem"
# grid_both_res2f_budget12 (multichannel_check.py, step 75): the sampler's switches; the constants behind them enter
# the policy through mg.sampling_policy
G12 = {"multichannel": True, "mc_isc": True, "mc_mperi": True, "mc_mperi_both": True, "mc_res2": True,
       "mc_res2_fwd": True, "flat_grid": True, "flat_budget": 12}


# plan step 75's SHORT/COST revision (one design-reviewed revision of step 65's sampler; regen_profile.py, regen_where.py):
# the stall stop (a round past the first that fills < 0.2 of what was missing ends the flat part: on step 75's five
# classes no grid round ever filled less than 0.5, so it only stops chunks whose short bins the proposal does not reach),
# the ladder solves at 1e-12, both ladders on every hadronic class with >= 3 finals, one resonance-ladder channel per
# resonance, the range joined with the dsh_ train pool (tools/flat_range.py --dsh); pools tagged _smixr
R1 = dict(G12, flat_stall=0.2, mc_solve_tol=1e-12, mc_mperi_all=True, mc_resladder=True)
# plan step 75's r2 (one revision after r1's NO; data/r1_check.json, r1_where.json, r1_rounds.json): r1 plus
# - flat_keep_ext: the highest and lowest in-range candidate always kept (r1's one held-out event outside, on
#   v4_bbx_htaptam, lay above the kept max but below r1's highest candidate at both seeds),
# - flat_stall_patience 2: the stall stop after two consecutive stalled rounds (v4_uux_bgtxwp: the chunks that stopped
#   after one left a margin bin at 0-8 of its 26 events, those that ran one more round filled it),
# - flat_budget 18: two more rounds of oversample/2 x n_flat for chunks still converging at r1's budget (uu_bbxuu,
#   us_ddxsu, bb_bbvevex, ug_auuux: missing at the stop and the last round's fill in r1_rounds.json); pools _smixs
R2 = dict(R1, flat_keep_ext=True, flat_stall_patience=2, flat_budget=18)
# plan step 78's r3 (after r2's NO on v4_uux_bgtxwp's reachable top bin [-1.206, -0.96]; data/r2_check.json,
# r2_where.json): r2 plus flat_empty_near: an empty bin's deficit steers the proposal through the deficit-many candidates
# nearest to it in log|M|^2, not through every candidate of the nearest occupied bin; pools _smixt
R3 = dict(R2, flat_empty_near=True)
# plan step 79's r4 (after r3's NO: v4_uux_bgtxwp's reachable bin [-1.206, -0.96] up from 0.73/0.79 to 0.925/0.917, but
# v4_uu_bbxuu, at 1.0 under r2, down to 0.90/0.80 in its top bin, its chunks converging slower from the second round
# with 3-9 top bins empty; data/r3_check.json, r3_rounds.json): r3 plus flat_empty_unique: each empty bin steers
# through the deficit-many nearest candidates no other empty bin took, so adjacent empty bins never stack their units on
# the same few candidates; pools _smixu
R4 = dict(R3, flat_empty_unique=True)
# plan step 81's r5 (after r4's NO; data/r4_check.json, r4_acted.json, r3_rounds.json): r3 with its nearest-candidate
# rule only for an empty bin whose nearest occupied bin(s) hold more candidates than its deficit, r2's rule (every
# candidate of the nearest occupied bin) otherwise: the fill moved with how few candidates carry an empty bin's deficit
# on both v4_uu_bbxuu (r2 1.0, r3 0.90/0.80, r4 0.70/0.72; its nearest occupied bin itself short, so r3 reached below
# it) and v4_uux_bgtxwp (r2 0.73/0.79, r3 0.925/0.917, r4 0.835/0.809; its nearest occupied bin ~900 candidates);
# pools _smixv
R5 = dict(R3, flat_empty_near_min=True)
# plan step 82's r6 (after r5's NO on v4_uux_bgtxwp; data/r5_check.json, r5_acted.json): r5 without the stall stop, the
# rounds ending at quota or the label budget 18: every chunk short in bgtxwp's reachable top bin stopped on the stall
# rule under r3 and r5, every budget-stopped chunk filled it; pools _smixw
R6 = dict(R5, flat_stall_off=True)
FLAT_RANGE_R1 = os.path.join(REC, "catalog_v4_flat_range_dsh.json")
OUT_R1 = "catalog_v4_r1_regen.yaml"


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


def g12_sampling(name, base_sampling, physics, inp, fr, switches=G12):
    import mg5_pipeline_final as mg
    s = dict(base_sampling or {})
    s.update(switches)
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
    r1 = getattr(a, "r1", False)
    fr = json.load(open(FLAT_RANGE_R1 if r1 else FLAT_RANGE)); had = set(hadronic())
    head = ("# generated by tools/regen_recipes.py write%s (plan catalog_coverage %s); do not edit\n"
            % ((" --r1", "step 75's revision of the step-65 sampler: stall stop, ladders everywhere, resonance ladder, "
                "range joined with the dsh_ train pool") if r1 else
               ("", "step 65: the flat part by the channel-grid sampler at 12 labels per flat event")))
    ents = stage_entries(); procs = []
    for n in sorted(ents):
        p, top = ents[n]
        q = dict(p); q["sampling"] = g12_sampling(n, p.get("sampling") or top, p.get("physics"), inp, fr,
                                                  R1 if r1 else G12)
        procs.append(q)
    out = OUT_R1 if r1 else OUT_REGEN
    with open(os.path.join(REC, out), "w") as f:
        f.write(head); yaml.safe_dump({"processes": procs}, f, sort_keys=False, default_flow_style=None, width=200)
    print(f"{out}: {len(procs)} classes")


def readers_of():
    """recipes/catalog_v4_*.yaml (not _g12) with at least one hadronic tree class at events > 0, NOT_READERS left out"""
    had = set(hadronic()); out = []
    for f in sorted(os.listdir(REC)):
        if not (f.startswith("catalog_v4_") and f.endswith(".yaml")) or f.startswith("catalog_v4_g12_") or f in NOT_READERS:
            continue
        spec = yaml.safe_load(open(os.path.join(REC, f)))
        ps = spec.get("processes") if isinstance(spec, dict) else None
        if ps and any(isinstance(p, dict) and p.get("name") in had and sum(int(p.get(k, 0) or 0) for k in
                      ("n_train", "n_val", "n_test")) > 0 for p in ps):
            out.append(f)
    return out


def cmd_readers(a):
    """A _g12 copy of every reader recipe, the hadronic entries on the new sampler. Refused while a reader holds a tree
    class of the range table (an e+e- class) that no regeneration covers: Joaquin 2026-11-06 ("If I asked for pp sampled
    data, why would you produce e+e- datasets that are different in any way?"), so a training or held-out recipe never
    mixes the two samplers; the e+e- classes join with step 68."""
    sys.path.insert(0, ROOT)
    inp = json.load(open(INPUTS)); fr = json.load(open(FLAT_RANGE)); had = set(hadronic())
    covered = had      # the classes a regeneration recipe covers (step 68 adds the e+e- ones)
    readers = readers_of(); refuse = {}
    for r in readers:
        spec = yaml.safe_load(open(os.path.join(REC, r)))
        left = sorted(p["name"] for p in spec["processes"] if p["name"] in fr and p["name"] not in covered)
        if left:
            refuse[r] = left
    if refuse:
        sys.exit("regen_recipes readers: refused, these readers hold e+e- tree classes not yet regenerated (step 68): "
                 + "; ".join(f"{r}: {len(v)} (e.g. {v[0]})" for r, v in refuse.items()))
    head = ("# generated by tools/regen_recipes.py readers (plan catalog_coverage step 65); do not edit\n")
    for r in readers:
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
    readers = readers_of()
    print(f"{len(readers)} reader recipes hold hadronic tree classes: {', '.join(readers)}")
    for r in readers:
        out = r.replace("catalog_v4_", "catalog_v4_g12_", 1)
        if not os.path.exists(os.path.join(REC, out)):
            print(f"{out}: not written (readers is refused until the e+e- classes are regenerated)"); continue
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
    sub.add_parser("pull"); w = sub.add_parser("write"); sub.add_parser("readers"); sub.add_parser("check")
    w.add_argument("--r1", action="store_true", help="step 75's revision -> recipes/catalog_v4_r1_regen.yaml")
    a = ap.parse_args()
    {"pull": cmd_pull, "write": cmd_write, "readers": cmd_readers, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
