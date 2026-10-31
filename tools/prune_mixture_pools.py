#!/usr/bin/env python3
"""Delete the stored default-share mixture pools (_smixd<NN>_) of the given recipes, which the loader now composes at
load (mix_compose.ensure_composed; the user, 2026-10-31 [D207]: "just build the mixture from the data that already
exists when you load the data"). Per entry with sampling.f_default > 0 and per role, the stored pool is the path
datagen.ensure_dataset would read (with its .recipe.json and .alphas.npy); it is listed for deletion only when its file
name carries _smixd and the inputs of its composition are built: the entry's catalog_v4 control pool (recipe id
checked) and every pool of its default_source (mg.default_share_pools, recipe ids checked). Any entry missing an input
keeps its stored pool and is listed as KEEP with why.
Without --apply it prints the plan; with it, the manifest (every path and its size) is written first, through a
temporary file renamed into place, to <the train cache dir>/.prune_manifests/mixture_pools_<time>.json, and only then are the
files deleted; a failure to write it deletes nothing. The last line is one JSON summary.
    site run --item catalog_coverage ccin2p3 FA -- python tools/prune_mixture_pools.py \
        recipes/catalog_v4_default50_train.yaml recipes/catalog_v4_default25_train.yaml [--apply]"""
import argparse, json, os, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SEED = 42   # data.seed of every catalog run (tools/check_site_data.py)


def plan(recipes, seed=SEED):
    """([(path, bytes)] to delete, [(entry/role, why)] kept), over the recipes' default-share entries."""
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose
    delete, keep, seen = [], [], set()
    for spec in recipes:
        doc = yaml.safe_load(open(os.path.join(ROOT, spec)))
        procs = doc["processes"]
        mg.register_recipe_processes(procs, default_sampling=doc.get("sampling"))
        for p in procs:
            if float((p.get("sampling") or {}).get("f_default", 0) or 0) <= 0:
                continue
            lo, hi = float(p["sqrts"][0]), float(p["sqrts"][1])
            for role in ("train", "val", "test"):
                n = int(p["n_" + role])
                rec = mg.variable_energy_recipe(p["name"], lo, hi, n, role=role, seed=seed)
                path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
                if "_smixd" not in os.path.basename(path):
                    keep.append((f"{p['name']}/{role}", f"not a _smixd pool: {path}")); continue
                if not os.path.exists(path):
                    continue
                with mix_compose._Swap(p["name"], mix_compose.control_config(p["name"])):
                    rc = mg.variable_energy_recipe(p["name"], lo, hi, n, role=role, seed=seed)
                    ctrl = mg.recipe_output_path(rc, datagen.dest_for_role(role))
                if not datagen._is_cached(ctrl, mg.recipe_id(rc)):
                    keep.append((f"{p['name']}/{role}", f"control pool not built: {ctrl}")); continue
                src = mg.sampling_policy(mg.PROCESSES[p["name"]], role)["default_source"]
                miss = [nm for nm, r2, p2 in mg.default_share_pools(src, role, seed)
                        if not datagen._is_cached(p2, mg.recipe_id(r2))]
                if miss:
                    keep.append((f"{p['name']}/{role}", f"default source pools not built: {miss}")); continue
                for f in (path, path + ".recipe.json", path[:-len(".npy")] + ".alphas.npy"):
                    if os.path.exists(f) and f not in seen:
                        seen.add(f); delete.append((f, os.path.getsize(f)))
    return delete, keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipes", nargs="+"); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    delete, keep = plan(a.recipes)
    for k, why in keep:
        print(f"KEEP {k}: {why}")
    gb = round(sum(s for _, s in delete) / 1e9, 2)
    summary = {"recipes": a.recipes, "apply": a.apply, "files": len(delete),
               "pools": sum(1 for f, _ in delete if f.endswith("_amplitudes.npy")), "gb": gb, "kept": len(keep)}
    if a.apply and delete:
        import datagen
        mdir = os.path.join(datagen.dest_for_role("train"), ".prune_manifests")
        os.makedirs(mdir, exist_ok=True)
        man = os.path.join(mdir, time.strftime("mixture_pools_%Y%m%d_%H%M%S.json"))
        tmp = man + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"recipes": a.recipes, "decision": "D207", "files": delete}, fh)
        os.replace(tmp, man)
        freed = 0
        for f, s in delete:
            os.remove(f); freed += s
        summary.update(manifest=man, freed_gb=round(freed / 1e9, 2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
