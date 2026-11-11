"""The regeneration with the channel-grid flat sampler (plan catalog_coverage step 65): its pools get their own file
(_smixq), never an old pool's path, and every chunk's flat fill reaches the assembled pool (<pool>.flatfill.json)."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import json, os, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg
import datagen


def _recipe(sampling):
    return {"process": "v4_x", "sqrts_min": 30.0, "sqrts_max": 6000.0, "role": "train", "sampling": sampling}


def test_path_tag():
    base = {"mode": "mixture", "f_fwd": 0.15}
    old = mg.recipe_output_path(_recipe(base), "/D")
    new = mg.recipe_output_path(_recipe(dict(base, flat_range=[-10.0, 2.0])), "/D")
    newd = mg.recipe_output_path(_recipe(dict(base, flat_range=[-10.0, 2.0], f_default=0.25)), "/D")
    assert old.endswith("_train_smix_amplitudes.npy"), old
    assert new.endswith("_train_smixq_amplitudes.npy"), new
    assert newd.endswith("_train_smixqd25_amplitudes.npy"), newd
    # a flat (unshaped) recipe never takes the tag
    assert mg.recipe_output_path(_recipe({"mode": "flat", "flat_range": [0, 1]}), "/D").endswith("_train_amplitudes.npy")


def test_flat_fill_written_and_skipped():
    with tempfile.TemporaryDirectory(prefix="zz_probe_flatfill_") as td:
        out = os.path.join(td, "chunk.npy")
        mg.FLAT_LAST.clear()
        datagen._write_flat_fill(out, 1.0)
        assert not os.path.exists(os.path.join(td, "chunk.flatfill.json"))     # no quota sampler, no file
        mg.FLAT_LAST.update(quota=np.array([3, 3]), kept_per_bin=np.array([3, 2]), cand_per_bin=np.array([9, 4]),
                            budget_hit=np.bool_(True), stop="budget", n_candidates=13, n_labels=13, t_propose=0.1,
                            t_label=0.2, kept_per_channel_bin=np.array([[3, 2]]), channels=[["flat"]],
                            edges=np.log(10.0) * np.array([0.0, 0.25, 0.5]))
        datagen._write_flat_fill(out, 2.5)
        d = json.load(open(os.path.join(td, "chunk.flatfill.json")))
        assert d["quota"] == [3, 3] and d["kept_per_bin"] == [3, 2] and d["budget_hit"] is True
        assert np.allclose(d["edges_log10"], [0, 0.25, 0.5]) and d["chunk_seconds"] == 2.5
        mg.FLAT_LAST.clear()


if __name__ == "__main__":
    test_path_tag(); test_flat_fill_written_and_skipped(); print("ok")
