"""The pp study's pools, checked before the reads and the pp-trained run are released: for every process of
recipes/catalog_v3_pp_test.yaml, the val and test pools in $DATA_DIR and the train pool in the $SCRATCH cache exist,
load, hold the recipe's row counts, carry their alpha_s sidecar (<pool>.alphas.npy, one value per row), and the
process has its diagram sidecar (data/diagrams/<name>.diagrams.json, parses). Prints one line per problem and a
summary; exit 1 if any.
    python tools/check_pp_pools.py
"""
import json, os, sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CACHE = os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")


def main():
    procs = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]
    bad = []
    for e in procs:
        n = e["name"]
        for role, d, want in (("val", DATA, e["n_val"]), ("test", DATA, e["n_test"]), ("train", CACHE, e["n_train"])):
            p = os.path.join(d, f"{n}_13000-13000GeV_{role}_lhe_amplitudes.npy")
            try:
                a = np.load(p, mmap_mode="r")
                if a.shape[0] < want:
                    bad.append(f"{n} {role}: {a.shape[0]} rows < {want}")
                s = np.load(p[:-4] + ".alphas.npy", mmap_mode="r")
                if s.shape[0] != a.shape[0]:
                    bad.append(f"{n} {role}: alphas sidecar {s.shape[0]} rows != {a.shape[0]}")
            except Exception as ex:                     # noqa: BLE001
                bad.append(f"{n} {role}: {type(ex).__name__}: {ex}")
        dg = os.path.join(ROOT, "data", "diagrams", f"{n}.diagrams.json")
        try:
            json.load(open(dg))
        except Exception as ex:                         # noqa: BLE001
            bad.append(f"{n} diagrams: {type(ex).__name__}")
    for b in bad:
        print("BAD", b)
    print(f"checked {len(procs)} processes, {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
