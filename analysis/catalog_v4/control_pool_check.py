#!/usr/bin/env python
"""Whether a finished run read the same pools a new recipe would, entry by entry (plan catalog_coverage, the pure-default
bound: catalog_v4's Muon short runs as the control of the pp-trained arm on recipes/catalog_v4_pp_mixed_train.yaml).

For every entry of RECIPE not swapped by the arm (name not starting with one of --skip prefixes) and every role, the pool
the current code resolves (mg.variable_energy_recipe, recipe_output_path, recipe_id) is compared with the one the run
resolved at its deployed commit (--ids-at, written by --write-ids from that commit's checkout; no data needed) and with
the file on the site:
  same        the same file name and recipe id at both commits, the sidecar's id the current one, the .npy older than
              the run's submission (--before) and not swapped in after it (width_regen swap manifests): the run read
              these bytes under this id. A sidecar may be newer (tools/backfill_pool_physics.py adds its physics
              block, recipe id unchanged), so its mtime is not read
  relabelled  the same file and bytes, the recipe id changed between the commits
  changed     the .npy written after the submission, another sidecar id, or another file name: the run read other bytes
  missing     no pool at the current path
    python analysis/catalog_v4/control_pool_check.py RECIPE --base BASE --write-ids IDS.json   (the run's commit)
    python analysis/catalog_v4/control_pool_check.py RECIPE --base BASE --ids-at IDS.json --before UNIX [--manifest TSV]
the second on the site the run ran on, read-only. --base is the run's training recipe: every checked entry must be in
it with the same fields. Prints one CONTROL_POOL line per pool that is not `same` and a CONTROL_POOL_SUMMARY json line.
"""
import argparse, collections, json, os, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ROLES = (("train", "n_train"), ("val", "n_val"), ("test", "n_test"))


def pools(recipe_path, base_path, skip, seed):
    """[(name, role, file name, recipe_id)] of the recipe's entries not swapped by the arm, as the code in use resolves
    them (no data read: runs anywhere, at any commit's checkout)"""
    import mg5_pipeline_final as mg
    r = yaml.safe_load(open(recipe_path))
    base = {p["name"]: p for p in yaml.safe_load(open(base_path))["processes"]}
    procs = [p for p in r["processes"] if not p["name"].startswith(tuple(skip))]
    not_in_base = [p["name"] for p in procs if base.get(p["name"]) != p]
    if not_in_base:
        sys.exit(f"CONTROL_POOL_FAIL {len(not_in_base)} entries not in {base_path} as they are: {not_in_base[:10]}")
    mg.register_recipe_processes(procs, default_sampling=r.get("sampling"))   # as experiment does
    out = []
    for p in procs:
        smin, smax = float(p["sqrts"][0]), float(p["sqrts"][1])
        for role, key in ROLES:
            rec = mg.variable_energy_recipe(p["name"], smin, smax, int(p[key]), role=role, seed=seed)
            out.append((p["name"], role, os.path.basename(mg.recipe_output_path(rec, "x")), mg.recipe_id(rec)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--base", required=True, help="the run's training recipe")
    ap.add_argument("--skip", nargs="*", default=["pp13"], help="name prefixes of the arm's swapped entries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-ids", help="only write the pools' file names and recipe ids (json) at this checkout")
    ap.add_argument("--ids-at", help="the json --write-ids wrote at the run's deployed commit")
    ap.add_argument("--before", type=float, help="unix time of the run's submission")
    ap.add_argument("--manifest", nargs="*", default=[], help="width_regen swap manifests (tsv)")
    a = ap.parse_args()
    cur = pools(a.recipe, a.base, a.skip, a.seed)
    if a.write_ids:
        json.dump({f"{n} {role}": [f, rid] for n, role, f, rid in cur}, open(a.write_ids, "w"), indent=0)
        print(f"CONTROL_POOL_IDS {len(cur)} pools -> {a.write_ids}")
        return
    assert a.ids_at and a.before, "--ids-at and --before (or --write-ids)"
    import datagen
    then = json.load(open(a.ids_at))
    swapped_in = set()
    for m in a.manifest:
        for line in open(m):
            f = line.rstrip("\n").split("\t")
            # a pool swapped in after the run was submitted (the manifest's times are the site's local time)
            if len(f) >= 4 and f[1] == "in" and time.mktime(time.strptime(f[0], "%Y-%m-%dT%H:%M:%S")) >= a.before:
                swapped_in.add(os.path.join(f[3], os.path.basename(f[2])))
    seen = collections.Counter()
    for name, role, fname, rid in cur:
        out = os.path.join(datagen.dest_for_role(role), fname)
        side = out + ".recipe.json"
        old = then.get(f"{name} {role}")
        if not (os.path.exists(out) and os.path.exists(side)):
            kind, why = "missing", out
        elif os.path.getmtime(out) >= a.before or out in swapped_in:
            kind, why = "changed", f"npy written {os.path.getmtime(out):.0f}, after the run's submission"
        elif json.load(open(side)).get("recipe_id") != rid:
            kind, why = "changed", f"sidecar id {json.load(open(side)).get('recipe_id')} != current {rid}"
        elif old is None or old[0] != fname:
            kind, why = "changed", f"the run resolved {old}, the current code {fname}"
        elif old[1] != rid:
            kind, why = "relabelled", f"same file and bytes, recipe id {old[1]} at the run's commit -> {rid}"
        else:
            kind, why = "same", ""
        seen[kind] += 1
        seen[f"{kind}:{role}"] += 1
        if kind != "same":
            print(f"CONTROL_POOL {kind} {name} {role}: {why}")
    print("CONTROL_POOL_SUMMARY " + json.dumps({"entries": len(cur) // len(ROLES), "pools": len(cur),
                                                 **dict(sorted(seen.items()))}))


if __name__ == "__main__":
    main()
