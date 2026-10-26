#!/usr/bin/env python3
"""Split one or more generation recipes into K shards for parallel prebuild jobs.

Every entry that shares a compiled backend with another (a coupling-only scan and its base, an energy probe
`<name>__E20` and its base, an NLO alpha_s child and its stripped parent) lands in the same shard, so no two jobs
ever compile the same MadGraph/MadLoop directory at once. Groups are balanced over the shards by their estimated
generation cost (datagen.chunk_tasks' est_cost, 2->2-equivalent events over the three roles), longest first; a
group with a one-loop entry also counts a fixed compile cost. Deterministic: the same recipes and K give the same
shards on any machine (a CPU dry-run on the laptop reproduces what a job builds).

The recipes must share one recipe-level `sampling` block (it is written into every shard).

    python tools/shard_recipes.py recipes/catalog_v4_gen_stage{1,2,3,4}.yaml --n 8 --plan
    python tools/shard_recipes.py recipes/catalog_v4_gen_stage{1,2,3,4}.yaml --n 8 --k 3 --out shard3.yaml

`--skip a,b` drops the named entries from shard k's output AFTER sharding, so every shard keeps the entries it
had without the skip (a part rerun with a skip builds the same pools as the others expect, minus the named ones).
A name that is not in the shard is an error; each skipped name is printed as `SHARD_SKIP <name>`.
`--only a,b` keeps just the named entries (with their backend-group mates) before sharding, for a job that builds a
few entries (use --n 1 --k 0); a name in no recipe is an error.
"""
import argparse, os, sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import datagen
import mg5_pipeline_final as mg

ROLES = (("train", "n_train"), ("val", "n_val"), ("test", "n_test"))
VIRT_COMPILE_COST = 2.0e6   # a MadLoop compile, in 2->2-equivalent events (~6 min at 0.18 ms): scheduling only


def load(paths):
    procs, sampling = [], None
    for p in paths:
        spec = yaml.safe_load(open(p))
        s = spec.get("sampling")
        if sampling is not None and s != sampling:
            sys.exit(f"{p}: sampling block differs from the first recipe's")
        sampling = s
        procs += spec["processes"]
    names = [e["name"] for e in procs]
    if len(set(names)) != len(names):
        sys.exit("duplicate entry names across the recipes")
    mg.register_recipe_processes(procs, default_sampling=sampling)
    return procs, sampling


def groups_of(procs):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for e in procs:
        n = e["name"]
        keys = ["n:" + n, "k:" + datagen._backend_key(n)]
        cfg = mg.PROCESSES.get(n, {})
        for b in (e.get("base"), cfg.get("scan_base")):
            if b and b in mg.PROCESSES:
                keys += ["n:" + b, "k:" + datagen._backend_key(b)]
        for k in keys[1:]:
            union(keys[0], k)
    out = {}
    for e in procs:
        out.setdefault(find("n:" + e["name"]), []).append(e)
    return list(out.values())


def cost(e):
    n = e["name"]
    w = datagen.process_gen_weight(n)
    c = sum(int(e[k]) * w for _, k in ROLES)
    if datagen.is_virt(n) and not datagen.is_reweightable_virt_scan(n):
        c += VIRT_COMPILE_COST
    return c


def shard(procs, n):
    gs = sorted(groups_of(procs), key=lambda g: (-sum(map(cost, g)), g[0]["name"]))
    bins = [[0.0, []] for _ in range(n)]
    for g in gs:
        b = min(range(n), key=lambda i: (bins[i][0], i))
        bins[b][0] += sum(map(cost, g)); bins[b][1] += g
    order = {e["name"]: i for i, e in enumerate(procs)}
    return [(c, sorted(es, key=lambda e: order[e["name"]])) for c, es in bins]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="+")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--out")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated entry names left out of shard k (after sharding)")
    ap.add_argument("--only", default="", help="comma-separated entry names kept, with their backend groups (before sharding)")
    a = ap.parse_args()
    procs, sampling = load(a.recipes)
    only = [s for s in a.only.split(",") if s]
    if only:
        missing = [s for s in only if s not in {e["name"] for e in procs}]
        if missing:
            sys.exit(f"--only: in no recipe: {', '.join(missing)}")
        procs = [e for g in groups_of(procs) if any(e["name"] in only for e in g) for e in g]
        for e in procs:
            print(f"SHARD_ONLY {e['name']}", file=sys.stderr)
    shards = shard(procs, a.n)
    if a.plan or a.k is None:
        tot = sum(c for c, _ in shards)
        print(f"{len(procs)} entries, {len(groups_of(procs))} backend groups, {a.n} shards, total cost {tot/1e6:.1f}M")
        for i, (c, es) in enumerate(shards):
            nv = sum(datagen.is_virt(e["name"]) for e in es)
            ev = sum(int(e[k]) for e in es for _, k in ROLES)
            gb = sum(int(e[k]) * (5 * (mg.PROCESSES[e["name"]].get("nfinal", 2) + 2) + 1) * 8
                     for e in es for _, k in ROLES) / 1e9
            print(f"  shard {i:2d}: {len(es):4d} entries ({nv:3d} one-loop), {ev/1e6:6.1f}M events, "
                  f"{gb:5.1f} GB, cost {c/1e6:7.1f}M ({100*c/tot:4.1f}%)")
    if a.k is not None:
        es = shards[a.k][1]
        skip = [s for s in a.skip.split(",") if s]
        missing = [s for s in skip if s not in {e["name"] for e in es}]
        if missing:
            sys.exit(f"--skip: not in shard {a.k}: {', '.join(missing)}")
        for s in skip:
            print(f"SHARD_SKIP {s}", file=sys.stderr)
        es = [e for e in es if e["name"] not in skip]
        spec = {"processes": es}
        if sampling is not None:
            spec = {"sampling": sampling, **spec}
        txt = (f"# shard {a.k} of {a.n} of {' '.join(map(os.path.basename, a.recipes))} (tools/shard_recipes.py)\n"
               + (f"# skipped: {', '.join(skip)}\n" if skip else "")
               + yaml.safe_dump(spec, sort_keys=False))
        if a.out:
            open(a.out, "w").write(txt)
        else:
            sys.stdout.write(txt)


if __name__ == "__main__":
    main()
