#!/usr/bin/env python
"""Whether a finished run read the same pools a new recipe would, entry by entry (plan catalog_coverage, the pure-default
bound: catalog_v4's Muon short runs as the control of the pp-trained arm on recipes/catalog_v4_pp_mixed_train.yaml).

For every entry of RECIPE not swapped by the arm (name not starting with one of --skip prefixes) and every role, the pool
the current code resolves (mg.variable_energy_recipe, recipe_output_path, recipe_id) is compared with what the run read:
  same      the sidecar's recipe_id is the current one and neither the .npy nor its sidecar changed after --before
            (the run's submission): the run read this file under this id
  migrated  the .npy is older than --before, its sidecar was rewritten after it by tools/width_regen.py migrate (the
            backup <sidecar>.wt1480327 holds the id the run looked up): same bytes, the top width label of the recipe
            identity relabelled 1.480327 -> 1.4915
  changed   the .npy is newer than --before (rebuilt or swapped in after the run started, or listed as swapped in by a
            width_regen swap manifest): the run read other bytes
  missing   no pool at the current path
Run on the site the run ran on (its DATA_DIR and caches), read-only:
    python analysis/catalog_v4/control_pool_check.py RECIPE --base BASE --before UNIX [--skip pp13] [--manifest TSV ...]
--base is the run's training recipe: every checked entry must be in it with the same fields. Prints one CONTROL_POOL line
per pool that is not `same` and a CONTROL_POOL_SUMMARY json line.
"""
import argparse, collections, json, os, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ROLES = (("train", "n_train"), ("val", "n_val"), ("test", "n_test"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--base", required=True, help="the run's training recipe")
    ap.add_argument("--before", type=float, required=True, help="unix time of the run's submission")
    ap.add_argument("--skip", nargs="*", default=["pp13"], help="name prefixes of the arm's swapped entries")
    ap.add_argument("--manifest", nargs="*", default=[], help="width_regen swap manifests (tsv)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import mg5_pipeline_final as mg
    import datagen
    r = yaml.safe_load(open(a.recipe))
    base = {p["name"]: p for p in yaml.safe_load(open(a.base))["processes"]}
    procs = [p for p in r["processes"] if not p["name"].startswith(tuple(a.skip))]
    not_in_base = [p["name"] for p in procs if base.get(p["name"]) != p]
    if not_in_base:
        sys.exit(f"CONTROL_POOL_FAIL {len(not_in_base)} entries not in {a.base} as they are: {not_in_base[:10]}")
    mg.register_recipe_processes(procs, default_sampling=r.get("sampling"))   # as experiment does
    swapped_in = set()
    for m in a.manifest:
        for line in open(m):
            f = line.rstrip("\n").split("\t")
            # a pool swapped in after the run was submitted (the manifest's times are the site's local time)
            if len(f) >= 4 and f[1] == "in" and time.mktime(time.strptime(f[0], "%Y-%m-%dT%H:%M:%S")) >= a.before:
                swapped_in.add(os.path.join(f[3], os.path.basename(f[2])))
    seen = collections.Counter()
    for p in procs:
        smin, smax = float(p["sqrts"][0]), float(p["sqrts"][1])
        for role, key in ROLES:
            rec = mg.variable_energy_recipe(p["name"], smin, smax, int(p[key]), role=role, seed=a.seed)
            out, rid = mg.recipe_output_path(rec, datagen.dest_for_role(role)), mg.recipe_id(rec)
            side = out + ".recipe.json"
            if not (os.path.exists(out) and os.path.exists(side)):
                kind, why = "missing", out
            else:
                sid = json.load(open(side)).get("recipe_id")
                old_npy = os.path.getmtime(out) < a.before and out not in swapped_in
                bak = side + ".wt1480327"
                if sid != rid:
                    kind, why = "changed", f"sidecar id {sid} != current {rid}"
                elif not old_npy:
                    kind, why = "changed", f"npy written {os.path.getmtime(out):.0f} after {a.before:.0f}"
                elif os.path.getmtime(side) < a.before:
                    kind, why = "same", ""
                elif os.path.exists(bak) and os.path.getmtime(bak) < a.before:
                    kind, why = "migrated", f"id {json.load(open(bak)).get('recipe_id')} -> {rid}"
                else:
                    kind, why = "changed", f"sidecar rewritten after {a.before:.0f}, no older backup"
            seen[kind] += 1
            seen[f"{kind}:{role}"] += 1
            if kind != "same":
                print(f"CONTROL_POOL {kind} {p['name']} {role}: {why}")
    print("CONTROL_POOL_SUMMARY " + json.dumps({"entries": len(procs), "pools": sum(seen[k] for k in
          ("same", "migrated", "changed", "missing")), **dict(sorted(seen.items()))}))


if __name__ == "__main__":
    main()
