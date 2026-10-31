"""datagen.composed_record: a run records which of its pools were composed at load (mix_compose.ensure_composed
writes the stored pool's recipe id, so the sidecar's derived_from is the only trace, and it sits in the job's $TMPDIR).
CPU only.
    python3 tests/test_composed_record.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datagen  # noqa: E402


def test_composed_record():
    with tempfile.TemporaryDirectory() as d:
        stored, composed, bare = (os.path.join(d, f) for f in ("a_train.npy", "b_train.npy", "c_val.npy"))
        der = {"composed_at_load": True, "control_pool": "/x/b_smix.npy", "control_recipe_id": "r1",
               "shares": {"bulk": 1, "fwd": 2, "flat": 3, "default": 4}}
        json.dump({"recipe_id": "s"}, open(stored + ".recipe.json", "w"))
        json.dump({"recipe_id": "c", "derived_from": der}, open(composed + ".recipe.json", "w"))
        out = datagen.composed_record({"train": {"a": stored, "b": composed}, "val": {"c": bare}})
        assert out == {"train": {"b": der}}, out
        assert datagen.composed_record({"train": {"a": stored}}) == {}
    print("ok composed record")


if __name__ == "__main__":
    test_composed_record()
