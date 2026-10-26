#!/usr/bin/env python
"""Fill the `physics` block (tools/pool_physics.py) into the sidecars of the pools made before generation wrote it
(plan catalog_coverage, the guard after the pp pools were found labelled with catalog_v3's physics). Runs on the site
that holds the pools and their backends; writes only the new key of each sidecar (<pool>.npy.recipe.json, read, the key
added, written back through a temporary file), and checks that the pool's recipe_id is unchanged by it.

Per sidecar: its process (the recipe scans of every recipes/**/*.yaml registered first, so a scan name such as
<base>__as0.13 resolves), its backend's Cards/param_card.dat and proc_card_mg5.dat, the block. Two cross-checks go into
the block's `backfill` record, never into the values:
  card_vs_sidecar   the sidecar's param_card_patches against the card's values for the fields both have (a backend
                    re-patched after the pool was made would differ)
  lhe_card_sha256   an lhe_pp pool: the standalone's card against the card the sidecar records MadEvent ran with
Not filled, and listed with the reason: a sidecar that already has a block, one in the catalog_v3 deletion manifest
(analysis/catalog_v4/data/catalog_v3_delete_manifest.json, group pools: its re-listing refuses a file whose size or
mtime changed), an unknown process, a recipe name two recipes define differently, a block that cannot be built
(no backend: the block is still written with what the entry gives and its `missing` fields, and listed).
    site run ccin2p3 FA --item catalog_coverage -- python tools/backfill_pool_physics.py [--dry-run] [--roots D ...]
The last stdout line is the JSON report: counts per outcome, the block values' families, the sidecars not filled.
"""
import argparse, collections, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
MANIFEST = os.path.join(ROOT, "analysis", "catalog_v4", "data", "catalog_v3_delete_manifest.json")


def register_all_recipes(mg):
    """every recipe entry with a physics scan or a base, registered; names two recipes define differently are
    returned (their pools are not filled: which definition made them is not known)"""
    import yaml
    seen, clash = {}, set()
    for f in sorted(glob.glob(os.path.join(ROOT, "recipes", "**", "*.yaml"), recursive=True)):
        try:
            d = yaml.load(open(f), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        except Exception:
            continue
        procs = d.get("processes") if isinstance(d, dict) else d
        if not isinstance(procs, list):
            continue
        for p in procs:
            if not isinstance(p, dict) or "name" not in p:
                continue
            if not (p.get("physics") or p.get("base", p["name"]) != p["name"]):
                continue
            key = json.dumps({"base": p.get("base", p["name"]), "physics": p.get("physics")}, sort_keys=True, default=str)
            if p["name"] in seen and seen[p["name"]][0] != key:
                clash.add(p["name"])
            seen.setdefault(p["name"], (key, p))
    specs = [{"name": n, "base": v[1].get("base", n), "physics": v[1].get("physics")} for n, v in seen.items()
             if n not in clash]
    ok = []
    for s in specs:
        try:
            mg.register_recipe_processes([s])
            ok.append(s["name"])
        except Exception:
            clash.add(s["name"])
    return {s["name"]: s["physics"] for s in specs}, clash


def manifest_paths():
    if not os.path.exists(MANIFEST):
        return set()
    d = json.load(open(MANIFEST))
    out = set()
    for root, items in ((d.get("groups") or {}).get("pools") or {}).items():
        out |= {os.path.join(root, it[0]) for it in items}
    return out


def roots_default():
    import datagen
    return sorted({datagen.frozen_dir(), datagen.train_cache_dir()})


def cross_checks(blk, rec):
    out = {}
    pat = rec.get("param_card_patches") or {}
    diff = {k: [v, blk.get(k)] for k, v in pat.items() if k in blk and blk.get(k) is not None
            and abs(float(v) - float(blk[k])) > 1e-6 * max(1.0, abs(float(v)))}
    if diff:
        out["card_vs_sidecar"] = diff
    lhe = rec.get("lhe") or {}
    if lhe.get("param_card_sha256") and blk.get("card_sha256") and lhe["param_card_sha256"] != blk["card_sha256"]:
        out["lhe_card_sha256"] = "the standalone's card differs from the card MadEvent ran with"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", help="pool directories (default: the frozen val/test dir and the train cache)")
    ap.add_argument("--dry-run", action="store_true", help="build every block, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="at most this many sidecars (a timing test)")
    a = ap.parse_args()
    import mg5_pipeline_final as mg
    import pool_physics as pp
    scans, clash = register_all_recipes(mg)
    skip = manifest_paths()
    roots = a.roots or roots_default()
    counts, notfilled, families, checks = collections.Counter(), collections.defaultdict(list), collections.Counter(), []
    n = 0
    for root in roots:
        for side in sorted(glob.glob(os.path.join(root, "*.npy.recipe.json"))):
            if a.limit and n >= a.limit:
                break
            n += 1
            pool = side[:-len(".recipe.json")]
            if pool in skip or side in skip:
                counts["skipped: in the catalog_v3 deletion manifest"] += 1
                continue
            try:
                rec = json.load(open(side))
            except ValueError as e:
                notfilled["unreadable sidecar"].append(side); counts["unreadable sidecar"] += 1
                continue
            if rec.get("physics") and "error" not in rec["physics"]:
                counts["already has a block"] += 1
                continue
            proc = rec.get("process")
            if proc in clash:
                notfilled["name defined differently by two recipes"].append(side); counts["ambiguous scan name"] += 1
                continue
            if proc not in mg.PROCESSES and not mg.is_synthetic(proc or ""):
                notfilled["unknown process"].append(side); counts["unknown process"] += 1
                continue
            try:
                blk = pp.physics_block(proc, recipe=rec, spec_physics=scans.get(proc))
            except Exception as e:
                notfilled["block not built: " + repr(e)[:80]].append(side); counts["block not built"] += 1
                continue
            cc = cross_checks(blk, rec)
            blk["backfill"] = {"by": "tools/backfill_pool_physics.py", **cc}
            if cc:
                checks.append({"sidecar": side, **cc})
            if blk.get("card") is None:
                notfilled["no backend card (entry fields only; missing: card values)"].append(side)
                counts["filled without a card"] += 1
            else:
                counts["filled"] += 1
            families[json.dumps([blk.get("model"), blk.get("MB"), blk.get("MT"), blk.get("WT"),
                                 pp.clause_kind(blk.get("order_clause"), (mg.PROCESSES.get(proc) or {}).get("pdg_ids")),
                                 (blk.get("alpha_s") or {}).get("rule")])] += 1
            rid0 = mg.recipe_id(rec)
            new = dict(rec, physics=blk)
            assert mg.recipe_id(new) == rid0, f"{side}: the block moved the recipe_id"
            if a.dry_run:
                continue
            tmp = side + ".physics_tmp"
            with open(tmp, "w") as f:
                json.dump(new, f, indent=2, sort_keys=True)
            os.replace(tmp, side)
    rep = {"roots": roots, "sidecars_seen": n, "counts": dict(counts), "dry_run": a.dry_run,
           "families [model, MB, MT, WT, clause, alpha_s rule]": dict(families.most_common()),
           "not_filled": {k: {"n": len(v), "first": v[:20]} for k, v in notfilled.items()},
           "cross_check_flags": {"n": len(checks), "first": checks[:30]}}
    for k, v in sorted(counts.items()):
        print(f"[backfill] {k}: {v}", file=sys.stderr)
    print(json.dumps(rep, default=str))


if __name__ == "__main__":
    main()
