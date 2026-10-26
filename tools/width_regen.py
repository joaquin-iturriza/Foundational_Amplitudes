#!/usr/bin/env python
"""The pool families whose patched widths MadGraph ignored, rebuilt at their intended widths (plan catalog_coverage).

MadGraph reads a width only from a DECAY line that starts in column 1; our card patchers indented it until 7b00eb21, so
the compiled default width was used (analysis/catalog_v4/card_width_audit.py, data/card_width_audit.json). The user
(2026-10-26): "do 1.4915 for all, let's not waste time regenerating everything now. For the others, if it's not too
much time for regeneration, regenerate." So:
  - catalog_v4's physical top is 1.4915 (V4_WT), what its pools have; a built tree pool's sidecar still records the
    1.480327 its card asked for, so `migrate` rewrites that sidecar (and renames its chunk dir) to the new recipe_id
    after checking the standalone's card really gave 1.4915; a pool whose card did give 1.480327 is quarantined;
  - FAMILIES are rebuilt: the heavy-quark layer (catalog_v3 and catalog_v4), the Z' and 2HDM points (both
    catalogs), catalog_v4's six loop-induced HH/HHH entries (loop_qcd_qed_sm-no_b_mass: top 1.4915, sm Z and W widths).
    Their recipe_ids do not see the width, so the old pools would be cache hits: the rebuild runs in a staging area
    (MG5_WORK_DIR, AMP_FROZEN_DIR, AMP_TRAIN_CACHE_DIR set by the job), `verify-cards` reads the new standalones' cards
    back, and `swap` moves the old pools, chunk dirs and standalones to the quarantine dir (never deleted, one TSV line
    per move) and the staged pools into place.
    python tools/width_regen.py spec OUT.yaml
    python tools/width_regen.py verify-cards SPEC.yaml            (MG5_WORK_DIR = the staging standalones)
    python tools/width_regen.py migrate --quarantine Q [--apply]
    python tools/width_regen.py swap SPEC.yaml --stage-frozen D --stage-train D --quarantine Q [--apply]
Without --apply, migrate and swap only print what they would do.
"""
import argparse, glob, json, os, re, shutil, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STAGES = sorted(glob.glob(os.path.join(ROOT, "recipes", "catalog_v3_gen_stage*.yaml"))) + \
    sorted(glob.glob(os.path.join(ROOT, "recipes", "catalog_v4_gen_stage*.yaml")))
LOOPIND_V4 = re.compile(r"v4_(ddx|uux|epem)_hhh?_loop$")
BSM = re.compile(r"(v4_)?(zp|thdm)_")
OLD_V4_WT = 1.480327          # what catalog_v4's tree sidecars record (top_width_lo(172.5)); the cards gave 1.4915
ROLES = ("val", "test", "train")
COUNT = {"train": "n_train", "val": "n_val", "test": "n_test"}


def family(e):
    """the family of a stage-recipe entry, None if it is not rebuilt"""
    n, b = e["name"], e.get("base", e["name"])
    if ((e.get("physics") or {}).get("masses") or {}).get(6) is not None:
        return "heavy_quark"
    if BSM.match(b) or BSM.match(n):
        return "bsm"
    if LOOPIND_V4.match(b) or LOOPIND_V4.match(n):
        return "loop_induced_v4"
    return None


def stage_entries():
    """every (entry with its recipe's sampling) of the catalog_v3 and catalog_v4 generation stages, deduplicated"""
    out, seen = [], {}
    for f in STAGES:
        d = yaml.safe_load(open(f))
        for e in d["processes"]:
            e = dict(e)
            if d.get("sampling") and "sampling" not in e:
                e["sampling"] = dict(d["sampling"])
            key = json.dumps(e, sort_keys=True)
            if e["name"] in seen:
                assert seen[e["name"]] == key, f"{e['name']}: two different stage entries"
                continue
            seen[e["name"]] = key
            out.append(e)
    return out


def load(spec_path):
    import mg5_pipeline_final as mg
    procs = yaml.safe_load(open(spec_path))["processes"]
    mg.register_recipe_processes(procs)
    return procs


def pools(procs, seed=42):
    """[(entry, role, recipe, file name, recipe_id)], α_s-scan children's stripped parents included"""
    import mg5_pipeline_final as mg
    import datagen
    out = []
    for p in procs:
        smin, smax = float(p["sqrts"][0]), float(p["sqrts"][1])
        for role in ROLES:
            n = int(p[COUNT[role]])
            names = [p["name"]] + ([mg.PROCESSES[p["name"]]["scan_base"]]
                                   if datagen.is_reweightable_virt_scan(p["name"]) else [])
            for name in names:
                r = mg.variable_energy_recipe(name, smin, smax, n, role=role, seed=seed)
                out.append((name, role, r, os.path.basename(mg.recipe_output_path(r, "x")), mg.recipe_id(r)))
    return out


def backend_dirs(name):
    """the standalone dirs a process reads (production or staging, by MG5_WORK_DIR)"""
    import mg5_pipeline_final as mg
    import datagen
    if datagen.is_virt(name):
        return [datagen._virt_standalone_dir(name)]
    sa = f"{mg.WORK_DIR}/{mg.standalone_name(name)}_standalone"
    return [sa, mg.fortran_dir_for(sa), f"{mg.WORK_DIR}/{mg.standalone_name(name)}_events"]


def card_patches(name):
    import mg5_pipeline_final as mg
    import datagen
    if datagen.is_virt(name):
        from tools import nlo_conventions as C
        from tools import nlo_virtual_pipeline as V
        cfg = V.VIRT_PROCESSES[mg.PROCESSES[name]["virt_base"]]
        return {**C.PARAM_CARD_PATCHES, **cfg.get("param_card_patches", {})}
    return dict(mg.PROCESSES[mg.standalone_name(name)].get("param_card_patches", {}))


def read_widths(card):
    """{width name (lower case): (value, column-1?)} from the DECAY lines of a param card"""
    out = {}
    for line in open(card):
        m = re.match(r"(\s*)DECAY\s+-?\d+\s+(\S+)\s*#\s*(\w+)", line, re.I)
        if m:
            out[m.group(3).lower()] = (float(m.group(2).replace("d", "e").replace("D", "e")), m.group(1) == "")
    return out


def cmd_spec(a):
    sel = [e for e in stage_entries() if family(e)]
    fams = {}
    for e in sel:
        fams[family(e)] = fams.get(family(e), 0) + 1
    with open(a.out, "w") as f:
        f.write("# tools/width_regen.py spec: the stage-recipe entries of the families whose widths MadGraph ignored "
                "(plan catalog_coverage)\n")
        yaml.safe_dump({"processes": sel}, f, sort_keys=False)
    print(f"WIDTH_SPEC {len(sel)} entries {fams} -> {a.out}")


def cmd_verify(a):
    """every staged standalone card of the spec: no indented DECAY line, every patched width at its value"""
    procs = load(a.spec)
    bad, n = [], 0
    for p in procs:
        want = {k.lower(): float(v) for k, v in card_patches(p["name"]).items() if k.upper().startswith("W")}
        for d in backend_dirs(p["name"]):
            card = os.path.join(d, "Cards", "param_card.dat")
            if d.endswith("_events") or not os.path.exists(card):
                continue
            n += 1
            got = read_widths(card)
            why = [f"{k} indented" for k, (v, col1) in got.items() if not col1]
            why += [f"{k} {got.get(k, (None,))[0]} != {v}" for k, v in want.items()
                    if k not in got or abs(got[k][0] - v) > 1e-6 * max(abs(v), 1e-12)]
            print(f"CARD {'OK ' if not why else 'BAD'} {card} {' '.join(f'{k}={v[0]:g}' for k, v in sorted(got.items()) if k in want)}"
                  + (f"  [{'; '.join(why)}]" if why else ""))
            if why:
                bad.append(card)
    print(f"VERIFY_CARDS {n} cards, {len(bad)} bad")
    sys.exit(1 if bad or not n else 0)


def move(src, qdir, log, apply):
    dst = os.path.join(qdir, os.path.basename(src.rstrip("/")))
    if os.path.exists(dst):
        dst += time.strftime(".%Y%m%d-%H%M%S")
    print(f"QUARANTINE {src} -> {dst}")
    if apply:
        os.makedirs(qdir, exist_ok=True)
        shutil.move(src, dst)
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{src}\t{dst}\n"); log.flush()


def cmd_migrate(a):
    """catalog_v4 tree pools (not a rebuilt family) whose sidecar records WT 1.480327: if every card of its backend gave
    1.4915 (an indented line read as the compiled default, or a column-1 1.4915), the sidecar's WT and recipe_id are
    rewritten (the old sidecar kept as <sidecar>.wt1480327) and its chunk dir renamed; otherwise the pool is
    quarantined."""
    import mg5_pipeline_final as mg
    import datagen
    sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v4"))
    from card_width_audit import defaults
    assert mg.V4_WT == 1.4915, mg.V4_WT
    procs = [e for e in stage_entries() if e["name"].startswith("v4_") and not family(e)]
    mg.register_recipe_processes(procs)
    log = open(os.path.join(a.quarantine, "migrate_manifest.tsv"), "a") if a.apply else None
    if a.apply:
        os.makedirs(a.quarantine, exist_ok=True)
    eff_cache, counts = {}, {"rewritten": 0, "chunks_renamed": 0, "quarantined": 0, "current": 0, "absent": 0}
    for name, role, r, fname, rid in pools(procs):
        if (r.get("param_card_patches") or {}).get("WT") != mg.V4_WT:
            continue                                    # no top width in its identity (one-loop pools: card only)
        old = json.loads(json.dumps(r)); old["param_card_patches"]["WT"] = OLD_V4_WT
        oid = mg.recipe_id(old)
        dest = datagen.dest_for_role(role)
        out, side = os.path.join(dest, fname), os.path.join(dest, fname) + ".recipe.json"
        och, nch = os.path.join(dest, f".chunks_{oid}"), os.path.join(dest, f".chunks_{rid}")
        sid = json.load(open(side)).get("recipe_id") if os.path.exists(side) else None
        if sid == rid:
            counts["current"] += 1
            continue
        if sid != oid and not os.path.isdir(och):
            counts["absent"] += 1
            continue
        if name not in eff_cache:
            eff = []
            for d in backend_dirs(name)[:2]:
                card = os.path.join(d, "Cards", "param_card.dat")
                if not os.path.exists(card):
                    continue
                w = read_widths(card).get("wt")
                eff.append(w[0] if w and w[1] else defaults(d)[0].get(6))
            eff_cache[name] = eff
        eff = eff_cache[name]
        if not eff:
            print(f"MIGRATE_UNKNOWN {name} {role}: no standalone card to read, left as it is")
            counts["unknown"] = counts.get("unknown", 0) + 1
            continue
        if not all(e is not None and abs(e - mg.V4_WT) < 1e-6 for e in eff):
            print(f"MIGRATE_REBUILD {name} {role}: its cards gave WT {eff}")
            for x in (out, side, och):
                if os.path.exists(x):
                    move(x, a.quarantine, log, a.apply)
                    counts["quarantined"] += 1
            continue
        if sid == oid:
            rec = json.load(open(side))
            rec["param_card_patches"]["WT"] = mg.V4_WT
            rec["recipe_id"] = rid
            assert mg.recipe_id(rec) == rid, (name, role)
            print(f"MIGRATE {side}: WT {OLD_V4_WT} -> {mg.V4_WT}, recipe_id {oid} -> {rid}")
            if a.apply:
                shutil.copy2(side, side + ".wt1480327")
                tmp = side + ".tmp"
                json.dump(rec, open(tmp, "w"), indent=2, sort_keys=True)
                os.replace(tmp, side)
                log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tsidecar\t{side}\t{oid}->{rid}\n"); log.flush()
            counts["rewritten"] += 1
        if os.path.isdir(och) and not os.path.exists(nch):
            print(f"MIGRATE_CHUNKS {och} -> {nch}")
            if a.apply:
                os.rename(och, nch)
                log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tchunks\t{och}\t{nch}\n"); log.flush()
            counts["chunks_renamed"] += 1
    print(f"MIGRATE_SUMMARY {counts} apply={a.apply}")


def cmd_swap(a):
    """the spec's production pools (every file of the pool's stem, its chunk dir) and standalones to the quarantine dir,
    the staged pools (sidecar recipe_id checked) into place"""
    import mg5_pipeline_final as mg
    import datagen
    procs = load(a.spec)
    stage = {"val": a.stage_frozen, "test": a.stage_frozen, "train": a.stage_train}
    plan, missing = [], []
    for name, role, r, fname, rid in pools(procs):
        side = os.path.join(stage[role], fname + ".recipe.json")
        if not (os.path.exists(side) and json.load(open(side)).get("recipe_id") == rid):
            missing.append(f"{name} {role}")
        plan.append((name, role, fname, rid))
    if missing:
        print(f"SWAP_REFUSED: {len(missing)} staged pools missing or with another recipe_id: {missing[:20]}")
        sys.exit(1)
    log = None
    if a.apply:
        os.makedirs(a.quarantine, exist_ok=True)
        log = open(os.path.join(a.quarantine, "swap_manifest.tsv"), "a")
    nq = ns = 0
    for name, role, fname, rid in plan:
        dest = datagen.dest_for_role(role)
        stem = fname[:-len(".npy")]
        old = [os.path.join(dest, f) for f in os.listdir(dest) if f.startswith(stem)] if os.path.isdir(dest) else []
        old += [x for x in [os.path.join(dest, f".chunks_{rid}")] if os.path.isdir(x)]
        for x in old:
            move(x, os.path.join(a.quarantine, "pools", role), log, a.apply); nq += 1
        for f in sorted(os.listdir(stage[role])):
            if f.startswith(stem):
                print(f"SWAP_IN {os.path.join(stage[role], f)} -> {os.path.join(dest, f)}")
                if a.apply:
                    shutil.move(os.path.join(stage[role], f), os.path.join(dest, f))
                    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tin\t{os.path.join(stage[role], f)}\t{dest}\n")
                ns += 1
    for d in sorted({d for p in procs for d in backend_dirs(p["name"]) if os.path.isdir(d)}):
        move(d, os.path.join(a.quarantine, "standalones"), log, a.apply); nq += 1
    print(f"SWAP_SUMMARY quarantined {nq} files/dirs, swapped in {ns} files, apply={a.apply}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec"); s.add_argument("out")
    s = sub.add_parser("verify-cards"); s.add_argument("spec")
    s = sub.add_parser("migrate"); s.add_argument("--quarantine", required=True); s.add_argument("--apply", action="store_true")
    s = sub.add_parser("swap"); s.add_argument("spec"); s.add_argument("--stage-frozen", required=True)
    s.add_argument("--stage-train", required=True); s.add_argument("--quarantine", required=True)
    s.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    {"spec": cmd_spec, "verify-cards": cmd_verify, "migrate": cmd_migrate, "swap": cmd_swap}[a.cmd](a)


if __name__ == "__main__":
    main()
