#!/usr/bin/env python3
"""Rebuild a run's config.yaml that a later job overwrote, from its seed twin (the same arm at another seed).

The case it was written for (plan warm_start, 2026-11-06): the cache prebuild FA-ccin2p3-5e02275b, run with
--config-path runs/catalog_v4_long/v4_het_default25_w16_grow_k533, warm-started that run's directory and rewrote its
config.yaml with its own exp_name, run_name, run_idx, training.num_workers and data.memmap_prebuild. The run's
twin (v4_het_default25_seed2_w16_grow_k533, same job, seed 43) differs from the original only in the seed and in the
paths that carry the run's own name. So the original is: the twin's config, with every key where the damaged file and
the twin differ taken from the damaged file (seed, run_dir, result_path, pretrained_path: the run's own), except the
keys the overwriting job set (--from-twin: taken from the twin; --drop: removed when the twin lacks them), and
run_name the basename of run_dir.

    python3 tools/restore_run_config.py <run_dir> <twin_dir> [--from-twin exp_name,run_idx,training.num_workers]
                                        [--drop data.memmap_prebuild] [--dry-run]

Writes <run_dir>/config.yaml.overwritten_<n> (the damaged file, kept) and the rebuilt <run_dir>/config.yaml; touches
nothing else (config_<i>.yaml and out_<i>.log of the overwriting job stay). Prints one RESTORE_CONFIG JSON line: the
keys taken from each source, and the rebuilt file's differences from the twin and from the damaged file.
"""
import argparse, json, os, shutil, sys

from omegaconf import OmegaConf


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k: [fa.get(k, "<absent>"), fb.get(k, "<absent>")] for k in sorted(set(fa) | set(fb)) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def setkey(d, key, val):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = val


def delkey(d, key):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.get(p, {})
    d.pop(parts[-1], None)


def rebuild(damaged, twin, from_twin, drop, run_dir):
    out = json.loads(json.dumps(twin))
    fd, ft = flat(damaged), flat(twin)
    own = []
    for k, (vt, vd) in diff(twin, damaged).items():
        if k in from_twin or k in drop or k == "run_name":
            continue
        if vd == "<absent>":
            delkey(out, k)
        else:
            setkey(out, k, fd[k])
        own.append(k)
    for k in drop:
        if k not in ft:
            delkey(out, k)
    setkey(out, "run_name", os.path.basename(os.path.normpath(run_dir)))
    return out, own


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("twin_dir")
    ap.add_argument("--from-twin", default="exp_name,run_idx,training.num_workers")
    ap.add_argument("--drop", default="data.memmap_prebuild")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from_twin = [k for k in a.from_twin.split(",") if k]
    drop = [k for k in a.drop.split(",") if k]
    path = os.path.join(a.run_dir, "config.yaml")
    damaged = OmegaConf.to_container(OmegaConf.load(path))
    twin = OmegaConf.to_container(OmegaConf.load(os.path.join(a.twin_dir, "config.yaml")))
    out, own = rebuild(damaged, twin, from_twin, drop, a.run_dir)
    rec = {"run_dir": a.run_dir, "twin": a.twin_dir, "own_keys_from_damaged": own, "from_twin": from_twin,
           "dropped": drop, "diff_vs_twin": diff(out, twin), "diff_vs_damaged": diff(out, damaged),
           "dry_run": a.dry_run}
    if not a.dry_run:
        n = 0
        while os.path.exists(f"{path}.overwritten_{n}"):
            n += 1
        backup = f"{path}.overwritten_{n}"
        shutil.copy2(path, backup)
        OmegaConf.save(OmegaConf.create(out), path)
        rec["backup"] = backup
    print("RESTORE_CONFIG " + json.dumps(rec, default=str))


if __name__ == "__main__":
    sys.exit(main())
