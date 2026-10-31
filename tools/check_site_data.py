"""Is this site's data ready for a recipe run? Read-only, a few seconds.

For every process of each recipe: the train/val/test pools are prebuilt (the check a
`data.require_cache=true` run makes before it touches a GPU), and the diagram sidecar
exists (data/diagrams/<process>.diagrams.json, resolved through the recipe's `base` for a
scan variant, as the off-shellness masks and the target propagators resolve it; without it
those features are silently off for that process). Also that the env has the packages
a run and a sweep trial import. Prints one line per recipe and exits non-zero if anything is missing.
    python tools/check_site_data.py [--list] [recipes/<spec>.yaml ...]     (default: the catalog recipe)
--list prints every pool not prebuilt, one `MISSING <process>/<role>` line each, after the recipe's line.
--alphas also requires each pool whose recipe says per_event_alphas true to carry <pool>.alphas.npy (what a run with
data.target_physics_orders reads; submit_guard passes it), and lists the ones without (`NOALPHAS <process>/<role>`).
--paths prints every pool's files instead (the .npy and its .recipe.json, `POOL <path>`, prebuilt or not), and
every sidecar found (`SIDECAR <path>`), deduplicated over the recipes, for sizing and `site copy`.
Used by sites/setup.sh --verify, so `site pick` only picks a site that can run the catalog."""
import importlib.util, os, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT = ["recipes/catalog_v4_train.yaml"]   # the catalog in use (catalog_v4, D130); catalog_v3 deleted by the user's decision, catalog_v2 (D110), Daniel's (D131)
SEED = 42                      # data.seed of every catalog run

ok = True
LIST = "--list" in sys.argv[1:]
ALPHAS = "--alphas" in sys.argv[1:]
PATHS = "--paths" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if a not in ("--list", "--paths", "--alphas")]
seen = set()
# what a training run imports, and what every DyHPO sweep's run_trial imports on top; lloca.mup exists only
# in the mup-parametrization build (requirements.txt), not in the PyPI release
env_missing = [m for m in ("torch", "lloca", "lloca.mup", "xformers", "gpytorch", "sklearn", "scipy")
               if importlib.util.find_spec(m) is None]
if env_missing:
    print(f"env: missing {' '.join(env_missing)}"); ok = False
import datagen
import mg5_pipeline_final as mg
for spec in ARGS or DEFAULT:
    doc = yaml.safe_load(open(os.path.join(ROOT, spec)))
    procs = doc["processes"] if isinstance(doc, dict) else doc
    # as training does: scan variants into the registry, the recipe's sampling onto every
    # entry (both are part of a pool's recipe id)
    mg.register_recipe_processes(procs, default_sampling=doc.get("sampling") if isinstance(doc, dict) else None)
    n_key = {"train": "n_train", "val": "n_val", "test": "n_test"}
    uncached, nosidecar, noalphas = [], [], []
    for p in procs:
        for role in ("train", "val", "test"):
            rec = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]),
                                            int(p[n_key[role]]), role=role, seed=SEED)
            path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
            if float((rec.get("sampling") or {}).get("f_default", 0)) > 0 and not datagen._is_cached(path, mg.recipe_id(rec)):
                # a default-share mixture is composed at load (mix_compose.py): its catalog_v4 pool and the default
                # source's lhe pools must be here instead
                import mix_compose
                with mix_compose._Swap(p["name"], mix_compose.control_config(p["name"])):
                    rec = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]),
                                                    int(p[n_key[role]]), role=role, seed=SEED)
                    path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
                src = mg.sampling_policy(mg.PROCESSES[p["name"]], role)["default_source"]
                for nm, r2, p2 in mg.default_share_pools(src, role, SEED):
                    if not datagen._is_cached(p2, mg.recipe_id(r2)):
                        uncached.append(f"{nm}/{role} (default share of {p['name']})")
            if PATHS:
                for f in (path, path + ".recipe.json"):
                    if f not in seen:
                        seen.add(f); print(f"POOL {f}")
            if not datagen._is_cached(path, mg.recipe_id(rec)):
                uncached.append(f"{p['name']}/{role}")
            elif ALPHAS and rec.get("per_event_alphas") is True and not os.path.exists(path[:-len(".npy")] + ".alphas.npy"):
                noalphas.append(f"{p['name']}/{role}")
        cars = [f for f in (os.path.join(ROOT, "data", "diagrams", f"{c}.diagrams.json")
                            for c in (p["name"], p.get("base")) if c) if os.path.exists(f)]
        if PATHS:
            for f in cars:
                if f not in seen:
                    seen.add(f); print(f"SIDECAR {f}")
        if not cars:
            nosidecar.append(p["name"])
    line = f"{os.path.basename(spec)}: {len(procs)} processes"
    if uncached:
        line += f", {len(uncached)} pools not prebuilt (e.g. {', '.join(uncached[:3])})"; ok = False
    if noalphas:
        line += f", {len(noalphas)} pools without per-event alpha_s (e.g. {', '.join(noalphas[:3])})"; ok = False
    if nosidecar:
        line += f", {len(nosidecar)} without a diagram sidecar (e.g. {', '.join(nosidecar[:3])})"; ok = False
    print(line + ("" if (uncached or nosidecar or noalphas) else ", pools prebuilt, sidecars present"))
    if LIST:
        for u in uncached:
            print(f"MISSING {u}")
        for u in noalphas:
            print(f"NOALPHAS {u}")
if PATHS:   # what is on this site of the listed files: count and bytes
    have = [f for f in seen if os.path.exists(f)]
    print(f"PRESENT {len(have)} of {len(seen)} files, {sum(os.path.getsize(f) for f in have) / 1e9:.2f} GB")
sys.exit(0 if ok else 1)
