#!/usr/bin/env python3
"""A one-loop pool's Born sidecar identity record (<pool>.born.npy.json, asserted by experiment._born_of_pool) rewritten
for a sidecar written at generation beside its pool (datagen before 2026-11-12 wrote the sidecar but not the record, so
a regenerated pool kept the record of the pool it replaced). The check before writing: the sidecar has the pool's rows,
every Born is positive, and the spread std ln|V/B| over the pool's rows is printed (a sidecar of another pool pairs
unrelated events: its spread is far above the process's, e.g. b b~ > Z Z 0.27, vb_spread.py).

    python tools/born_record.py <pool.npy> [<pool.npy> ...] [--write]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyed_aug import pool_identity  # noqa: E402

for pool in [a for a in sys.argv[1:] if not a.startswith("--")]:
    bp = pool[:-len(".npy")] + ".born.npy"
    v = np.load(pool, mmap_mode="r")[:, -1].astype(float)
    b = np.load(bp).astype(float).ravel()
    bad = f"{len(b)} rows, the pool {len(v)}" if len(b) != len(v) else "non-positive Born" if not np.all(b > 0) else None
    if bad:  # reported, never written; the other pools are still checked
        print("BORN_RECORD " + json.dumps({"pool": pool, "bad": bad}))
        continue
    old = json.load(open(bp + ".json")) if os.path.exists(bp + ".json") else None
    new = {"pool_identity": pool_identity(pool), "n_rows": int(len(v))}
    print("BORN_RECORD " + json.dumps({"pool": pool, "rows": len(v), "std_ln_vb": float(np.std(np.log(np.abs(v / b)))),
                                       "record_matches": bool(old and list(old["pool_identity"]) == list(new["pool_identity"])),
                                       "record_exists": old is not None}))
    if "--write" in sys.argv:
        json.dump(new, open(bp + ".json", "w"))
        print("wrote", bp + ".json")
