"""The pp study's pools, checked before the reads and the pp-trained run are released: for every process of
recipes/catalog_v3_pp_test.yaml, the val and test pools in $DATA_DIR and the train pool in the $SCRATCH cache exist,
load, hold the recipe's row counts, carry their alpha_s sidecar (<pool>.alphas.npy, one value per row), and the
process has its diagram sidecar (data/diagrams/<name>.diagrams.json, parses). Prints one line per problem and a
summary; exit 1 if any.
    python tools/check_pp_pools.py [--spec recipes/<spec>.yaml] [--labels]
--labels: also each train pool's labels (2 pairs per process required, else BAD) against its val and test pools (datagen.mixed_generation, any generator
version): BADZ above |z| = 5; prints the largest |z|.
"""
import json, os, sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CACHE = os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml"))
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    procs = yaml.safe_load(open(a.spec))["processes"]
    bad, paths = [], {"train": {}, "val": {}, "test": {}}
    for e in procs:
        n = e["name"]
        for role, d, want in (("val", DATA, e["n_val"]), ("test", DATA, e["n_test"]), ("train", CACHE, e["n_train"])):
            p = os.path.join(d, f"{n}_13000-13000GeV_{role}_lhe_amplitudes.npy")
            paths[role][n] = p
            try:
                pool = np.load(p, mmap_mode="r")
                if pool.shape[0] < want:
                    bad.append(f"{n} {role}: {pool.shape[0]} rows < {want}")
                s = np.load(p[:-4] + ".alphas.npy", mmap_mode="r")
                if s.shape[0] != pool.shape[0]:
                    bad.append(f"{n} {role}: alphas sidecar {s.shape[0]} rows != {pool.shape[0]}")
            except Exception as ex:                     # noqa: BLE001
                bad.append(f"{n} {role}: {type(ex).__name__}: {ex}")
        dg = os.path.join(ROOT, "data", "diagrams", f"{n}.diagrams.json")
        try:
            json.load(open(dg))
        except Exception as ex:                         # noqa: BLE001
            bad.append(f"{n} diagrams: {type(ex).__name__}")
    if a.labels:
        sys.path.insert(0, ROOT)
        import datagen
        zs = datagen.mixed_generation(paths, z_max=0, any_version=True)
        allz = [(abs(v["z"]), n, r) for n, d in zs.items() for r, v in d.items()]
        bad += [f"{n} {r}: train labels against {r} z={z:.1f}" for z, n, r in allz if z > 5]
        if len(allz) != 2 * len(procs):      # a pool without a generator sha is skipped by mixed_generation: never a pass
            bad.append(f"labels: {len(allz)} process-role pairs compared, {2 * len(procs)} expected")
        print("LABELS %d process-role pairs compared, largest |z| %s" % (len(allz), max(allz)[:3] if allz else None))
    for b in bad:
        print("BAD", b)
    print(f"checked {len(procs)} processes, {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
