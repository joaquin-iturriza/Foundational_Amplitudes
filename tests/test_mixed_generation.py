"""datagen.mixed_generation: flags a process only when its pools differ in generator version AND in labels."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datagen


def _pool(d, name, sha, n, shift, seed):
    rng = np.random.default_rng(seed)
    e = rng.uniform(400, 1000, n) / 2                     # beam energy: two incoming massless legs along z
    rows = np.zeros((n, 2 * 5 + 1))
    rows[:, 0], rows[:, 3], rows[:, 4], rows[:, 7] = e, e, e, -e
    rows[:, -1] = np.exp(-2 * np.log(2 * e) + rng.normal(0, 0.05, n) + shift)
    pth = os.path.join(d, name + ".npy"); np.save(pth, rows)
    json.dump({"provenance": {"pipeline_sha256": sha}}, open(pth + ".recipe.json", "w"))
    return pth


def test_mixed_generation():
    with tempfile.TemporaryDirectory() as d:
        tr = _pool(d, "tr", "a" * 64, 20000, 0.0, 1)
        same = _pool(d, "same", "a" * 64, 5000, 0.01, 2)      # same version: never compared, even when shifted
        clean = _pool(d, "clean", "b" * 64, 5000, 0.0, 3)     # other version, same labels
        bad = _pool(d, "bad", "b" * 64, 5000, 0.01, 4)        # other version, 1% label shift
        assert datagen.mixed_generation({"train": {"p": tr}, "val": {"p": same}}) == {}
        assert datagen.mixed_generation({"train": {"p": tr}, "val": {"p": clean}}) == {}
        r = datagen.mixed_generation({"train": {"p": tr}, "val": {"p": bad}})
        assert set(r) == {"p"} and r["p"]["val"]["z"] < -5, r
        assert datagen.mixed_generation({"train": {"kit_ee_tt__virt_finite__hard": tr},
                                         "val": {"kit_ee_tt__virt_finite__hard": bad}}) == {}   # KIT: no generator
        # any_version (a train pool rebuilt on another site's MadGraph): the same-version shift is caught, the clean
        # pool passes, and z_max=0 returns its z too
        r = datagen.mixed_generation({"train": {"p": tr}, "val": {"p": same}}, any_version=True)
        assert set(r) == {"p"} and r["p"]["val"]["z"] < -5, r
        r = datagen.mixed_generation({"train": {"p": tr}, "val": {"p": clean}}, z_max=0, any_version=True)
        assert abs(r["p"]["val"]["z"]) < 5, r


if __name__ == "__main__":
    test_mixed_generation(); print("ok")
