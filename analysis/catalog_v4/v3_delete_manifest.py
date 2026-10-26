#!/usr/bin/env python3
"""The manifest of catalog_v3's files on CC for their deletion (catalog_coverage; the user's call of 2026-10-26, "I'd
wait to see what happens with the first runs for catalog 4, but I don't see a reason to keep catalog 3 after" [D163],
after the switch "Okay, switching makes sense then" [D166]). Local, CPU, no cluster access: it joins

  LISTING  the read-only listing of the site (analysis/catalog_v4/v3_disk_listing.py, run there per root and shard:
           site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/v3_disk_listing.py --only R --shard I/6)
  RUNS     the site's run configs and the recipe each trained on:
           site run --item catalog_coverage ccin2p3 FA -- 'find runs -maxdepth 3 -name config.yaml -mtime -5 -print0 |
               xargs -0 grep -H "processes_file:"'
  QUEUE    a note of what was running or queued when the listing was taken (site status ccin2p3, site runs)

with every recipe under recipes/ (which recipe names each pool: process name, sqrt(s) window, role, sampling tag), and
writes analysis/catalog_v4/data/catalog_v3_delete_manifest.json. Nothing is deleted here.

Groups (each with its items: path, allocated bytes, file count, mtime):
  pools          catalog_v3's pools (v3_* and its BSM zp_* / thdm_* shaped pools), val/test in $DATA_DIR, train in the
                 train cache, with their sidecars (.recipe.json, .alphas.npy, ...): what D163 names
  quarantine     catalog_v3's pre-fix pools and standalones the width rebuild moved aside ($SCRATCH/zz_width_regen/quarantine)
  rebuild_mg     the width rebuild's staging backends of catalog_v3 names ($SCRATCH/zz_width_regen/mg)
  diagrams       catalog_v3's diagram sidecars (data/diagrams/v3_*, zp_*, thdm_*)
  backends       catalog_v3's MadGraph standalones and MadLoop standalones ($WORK/mg5amcnlo/v3_*)
  keep_pp        the 13 TeV pp pools, sidecars and standalones (pp13_*): read by catalog_v4's models (the pp axis)
Only `pools` is named by D163; the others are listed for the user's call.

    python analysis/catalog_v4/v3_delete_manifest.py LISTING.jsonl RUNS.txt --queue "<note>"
"""
import argparse, collections, glob, json, os, re, time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "analysis", "catalog_v4", "data", "catalog_v3_delete_manifest.json")
POOL = re.compile(r"^(?P<name>.+?)_(?P<lo>\d+)-(?P<hi>\d+)GeV_(?P<role>train|val|test)(?P<tag>_smix|_lhe|_ssteer)?"
                  r"_amplitudes(?P<suffix>.*)$")
TAG = {"mixture": "_smix", "steered": "_ssteer", "flat": ""}


# what of catalog_v3 stays, what reads the pools besides training, and what the deletion waits for (read 2026-10-26)
JUDGEMENT = {
    "keep": {
        "keep_pp": "the 13 TeV pp pools, their sidecars and standalones (pp13_*): catalog_v4's models are read on them "
                   "(the pp axis, recipes/catalog_v3_pp_test.yaml, scripts/job_catalog_v3_short.sh CROSS=PP), the density "
                   "check reads them, and the pp blend proposal waiting on the user would draw its pp events from them",
        "run_directories_and_reads": "not pools and not in this manifest: runs/catalog_v3_short/*, runs/muon_test_search_*, "
                   "runs/sign_head_search/* (configs, result.json, curves) and the zero-shot reads in "
                   "$SCRATCH/zz_v3_short_<arm> and zz_v3_pp_<arm> (saved truth and prediction per event) stay; the notes' "
                   "catalog_v3 numbers rest on the committed reductions of them (analysis/catalog_v3/data/short_reads*, "
                   "sign_head_search.json, prep_signed.json, pp_rows.json, compare_v3.json); model files fall under the "
                   "separate call on finished runs' models (D149)",
        "no_pool_shared_by_name": "catalog_v4 names every pool v4_* (its recipes name no v3_*, zp_* or thdm_* pool; "
                   "v3_name in recipes/catalog_v4_processes.yaml is metadata), so no catalog_v3 file is a catalog_v4 pool",
        "optional_shared_probe_val_pools": "the validation pools of the 555 probes held out in both catalogs "
                   "(recipes/catalog_v3_shared_heldout.yaml, about 2 GB with sidecars): catalog_v4's models were read on "
                   "them (CROSS=XV3) for the like-for-like comparison, which is stored (compare_v3.json); kept only if a new "
                   "catalog_v4 arm must also be scored on catalog_v3's events; the pp blend test's design compares against "
                   "catalog_v3's arms through stored reads only, so not needed by default",
    },
    "other_readers": {
        "held-out axes (catalog_v3_heldout_*.yaml)": "zero-shot reads (val, 2000 events) by every catalog_v3 arm of "
                   "scripts/job_catalog_v3_short.sh at the end of its job; all those jobs have ended",
        "catalog_v3_shared_heldout.yaml": "read (val) by runs/catalog_v4_short/v4_dimensionless_muon and _seed2 (XV3); ended",
        "catalog_signed_ab_*.yaml": "the signed pools' A/B (signed_ab), held by the user's call of 2026-10-24 (D114); its "
                   "adaptation to catalog_v4 is the switch's other step",
        "nonsm_*.yaml": "name the same 15 BSM pools (zp_ee_*, thdm_ee_*, 0.13 GB); the non-SM study is held (D49) and "
                   "would rebuild them on demand",
        "muon_hp_ft/v3_*.yaml": "the Muon fine-tune search's four probes (done entry muon_hp)",
        "scripts that recompute notes' inputs from these pools on CC": "analysis/catalog_v3/coverage_check.py (run), "
                   "analysis/catalog_v4/density_check.py (catalog_v3 control), pool_audit.py, pool_precision.py, "
                   "prep_signed_pools.py, prep_sampling_pools.py, prep_level.py, prep_shape.py: their outputs are "
                   "committed and verify_claims reruns only the local reductions; after the deletion they cannot be "
                   "recomputed from the pools",
    },
    "waits_for": [
        "the pair sign head test's collection and the sign study's write-up (v3_prep_explore): its seed-43 runs ended "
        "COMPLETED at 22:21 and are not collected; a missing or broken read would need an eval re-read on catalog_v3's "
        "held-out pools",
        "the manifest shown to the user (the deletion call requires it first, D163)",
        "the user's call on the groups D163 does not name: backends, quarantine, rebuild_mg, diagrams",
    ],
}


def recipe_refs():
    """(name, lo, hi, role, tag) -> [recipe paths]; recipes that are a list of processes with windows only"""
    refs = collections.defaultdict(set)
    for f in sorted(glob.glob(os.path.join(ROOT, "recipes", "**", "*.yaml"), recursive=True)):
        try:
            spec = yaml.safe_load(open(f))
        except Exception:
            continue
        procs = spec.get("processes") if isinstance(spec, dict) else spec
        if not isinstance(procs, list):
            continue
        top = (spec.get("sampling") or {}).get("mode", "flat") if isinstance(spec, dict) else "flat"
        rel = os.path.relpath(f, ROOT)
        for p in procs:
            if not isinstance(p, dict) or "sqrts" not in p or "name" not in p:
                continue
            mode = (p.get("sampling") or {}).get("mode", top) if isinstance(p.get("sampling"), dict) else top
            lo, hi = int(round(float(p["sqrts"][0]))), int(round(float(p["sqrts"][1])))
            for role in ("train", "val", "test"):
                if int(p.get(f"n_{role}", 0) or 0) > 0:
                    tags = {TAG.get(mode, "_smix")}
                    if p["name"].startswith("pp13_"):
                        tags.add("_lhe")
                    if p["name"].endswith(("_nlo", "_loop")):   # one-loop and loop-induced pools are never shaped: no tag
                        tags = {""}
                    for tag in tags:
                        refs[(p["name"], lo, hi, role, tag)].add(rel)
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("listing")
    ap.add_argument("runs")
    ap.add_argument("--queue", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.listing) if l.startswith('{"kind": "item"')]
    refs = recipe_refs()

    # runs: recipe basename -> run dirs
    runs = collections.defaultdict(list)
    for l in open(a.runs):
        if "processes_file:" not in l:
            continue
        cfg, _, val = l.strip().partition(":  processes_file:")
        if not val:
            cfg, val = l.split(":", 1)[0], l.split("processes_file:", 1)[1]
        rec = val.strip().split("/recipes/")[-1] if "/recipes/" in val else val.strip()
        runs["recipes/" + rec if not rec.startswith("/") else rec].append(os.path.dirname(cfg.strip()))

    groups = collections.defaultdict(list)
    unref = collections.Counter()
    by_recipe = collections.defaultdict(lambda: {"pools": set(), "bytes": 0})
    for r in rows:
        base = os.path.basename(r["path"])
        item = {"path": r["path"], "bytes": r["alloc"], "files": r["files"], "mtime": r["mtime"]}
        pp = base.startswith("pp13_")
        if r["root"] in ("datasets", "amp_data_cache"):
            m = POOL.match(base)
            if m:
                key = (m["name"], int(m["lo"]), int(m["hi"]), m["role"], m["tag"] or "")
                item["pool"] = f'{m["name"]}_{m["lo"]}-{m["hi"]}GeV_{m["role"]}{m["tag"] or ""}'
                item["recipes"] = sorted(refs.get(key, ()))
                if not pp:
                    if not item["recipes"]:
                        unref[r["root"]] += 1
                    for rc in item["recipes"]:
                        by_recipe[rc]["pools"].add(item["pool"]); by_recipe[rc]["bytes"] += r["alloc"]
            g = "keep_pp" if pp else "pools"
        elif r["root"] == "width_regen":
            g = "keep_pp" if pp else ("quarantine" if "/quarantine/" in r["path"] else "rebuild_mg")
        elif r["root"] == "diagrams":
            g = "keep_pp" if pp else "diagrams"
        else:
            g = "keep_pp" if pp else "backends"
        groups[g].append(item)

    def tot(items):
        return {"items": len(items), "files": sum(i["files"] for i in items), "bytes": sum(i["bytes"] for i in items),
                "GB": round(sum(i["bytes"] for i in items) / 1e9, 2)}

    pools = groups["pools"]
    RL = sorted({rc for v in groups.values() for i in v for rc in i.get("recipes", [])})
    RI = {rc: k for k, rc in enumerate(RL)}

    def compact(items):
        d = collections.defaultdict(list)
        for i in sorted(items, key=lambda i: i["path"]):
            d[os.path.dirname(i["path"])].append([os.path.basename(i["path"]), i["bytes"], i["files"], i["mtime"]]
                                                 + [RI[rc] for rc in i.get("recipes", [])])
        return dict(d)
    other = [rc for rc in by_recipe if not os.path.basename(rc).startswith("catalog_v3")]
    out = {
        "what": "catalog_v3's files on CC (/sps/lpnhe), read-only listing; nothing deleted. Only the group 'pools' is "
                "named by the user's deletion call (D163, after the switch D166); the others are listed for the user's call.",
        "listed": "2026-10-26 22:25 (site run listings), newest file at " + time.strftime("%Y-%m-%dT%H:%M", time.localtime(max(r["mtime"] for r in rows) if rows else 0)),
        "running_or_queued": a.queue,
        **JUDGEMENT,
        "totals": {g: tot(v) for g, v in sorted(groups.items())},
        "pools_by_root": {k: tot([i for i in pools if ("/datasets/" in i["path"]) == (k == "datasets")])
                          for k in ("datasets", "amp_data_cache")},
        "pools_not_named_by_any_recipe": dict(unref),
        "recipes_naming_the_pools": {rc: {"pools": len(v["pools"]), "GB": round(v["bytes"] / 1e9, 2),
                                          "runs_trained_on_it": sorted(runs.get(rc, []))}
                                     for rc, v in sorted(by_recipe.items())},
        "non_catalog_v3_recipes_naming_the_pools": sorted(other),
        "item_format": "groups[g][directory] = [[name, allocated bytes, files, mtime, index into recipe_list per "
                       "recipe naming the pool]]",
        "recipe_list": RL,
        "groups": {g: compact(v) for g, v in sorted(groups.items())},
    }
    with open(OUT, "w") as f:
        f.write("{\n")
        keys = list(out)
        for k in keys:
            if k != "groups":
                f.write(f"{json.dumps(k)}: {json.dumps(out[k], indent=1)},\n")
        f.write('"groups": {\n' + ",\n".join(
            f"{json.dumps(g)}: {{\n" + ",\n".join(f"{json.dumps(d)}: [\n" + ",\n".join(json.dumps(x, separators=(',', ':'))
                                                 for x in xs) + "]" for d, xs in gd.items()) + "}"
            for g, gd in out["groups"].items()) + "}}\n")
    print(json.dumps({k: out[k] for k in ("totals", "pools_by_root", "pools_not_named_by_any_recipe",
                                          "non_catalog_v3_recipes_naming_the_pools")}, indent=1))
    for rc, v in out["recipes_naming_the_pools"].items():
        print(rc, v["pools"], v["GB"], "GB", len(v["runs_trained_on_it"]), "runs")


if __name__ == "__main__":
    main()
