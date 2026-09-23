"""Is this site's data ready for a recipe run? Read-only, a few seconds.

For every process of each recipe: the train/val/test pools are prebuilt (the check a
`data.require_cache=true` run makes before it touches a GPU), and the diagram sidecar
exists (data/diagrams/<process>.diagrams.json, resolved through the recipe's `base` for a
scan variant, as the off-shellness masks and the target propagators resolve it; without it
those features are silently off for that process). Also that the env has the packages
a run imports. Prints one line per recipe and exits non-zero if anything is missing.
    python tools/check_site_data.py [recipes/<spec>.yaml ...]     (default: the catalog recipe)
Used by sites/setup.sh --verify, so `site pick` only picks a site that can run the catalog."""
import importlib.util, os, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT = ["recipes/catalog_v2_train_scan.yaml"]
SEED = 42                      # data.seed of every catalog run

ok = True
env_missing = [m for m in ("torch", "lloca", "xformers") if importlib.util.find_spec(m) is None]
if env_missing:
    print(f"env: missing {' '.join(env_missing)}"); ok = False
import datagen
import mg5_pipeline_final as mg
for spec in sys.argv[1:] or DEFAULT:
    doc = yaml.safe_load(open(os.path.join(ROOT, spec)))
    procs = doc["processes"] if isinstance(doc, dict) else doc
    # as training does: scan variants into the registry, the recipe's sampling onto every
    # entry (both are part of a pool's recipe id)
    mg.register_recipe_processes(procs, default_sampling=doc.get("sampling") if isinstance(doc, dict) else None)
    n_key = {"train": "n_train", "val": "n_val", "test": "n_test"}
    uncached, nosidecar = [], []
    for p in procs:
        for role in ("train", "val", "test"):
            rec = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]),
                                            int(p[n_key[role]]), role=role, seed=SEED)
            if not datagen._is_cached(mg.recipe_output_path(rec, datagen.dest_for_role(role)), mg.recipe_id(rec)):
                uncached.append(f"{p['name']}/{role}")
        if not any(os.path.exists(os.path.join(ROOT, "data", "diagrams", f"{c}.diagrams.json"))
                   for c in (p["name"], p.get("base"))  if c):
            nosidecar.append(p["name"])
    line = f"{os.path.basename(spec)}: {len(procs)} processes"
    if uncached:
        line += f", {len(uncached)} pools not prebuilt (e.g. {', '.join(uncached[:3])})"; ok = False
    if nosidecar:
        line += f", {len(nosidecar)} without a diagram sidecar (e.g. {', '.join(nosidecar[:3])})"; ok = False
    print(line + ("" if (uncached or nosidecar) else ", pools prebuilt, sidecars present"))
sys.exit(0 if ok else 1)
