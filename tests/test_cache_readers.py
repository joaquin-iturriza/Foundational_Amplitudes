"""analysis/storage/cache_readers.py on a fake SCRATCH and a fake plan (cc_disk, D556): a recipe a not-done entry names
(whole word, with or without .yaml, or through a family glob) is in use, one only a done entry names is not; a memmap
key is linked to an in-use recipe through its train pools' recipe_id."""
import os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "analysis", "storage", "cache_readers.py")
PLAN = """entries:
- id: a
  status: held
  notes: "pools of catalog_v3_sep_probes and the catalog_v4_datasaving_* family; vob_ee_uu_nlo.yaml"
- id: b
  status: done
  notes: "ref_solo_ee_uu"
"""


def _run(t, rid):
    os.makedirs(os.path.join(t, "amp_mmap_cache", "k1", "d1"))
    open(os.path.join(t, "amp_mmap_cache", "k1", "key_doc.json"), "w").write(
        '{"names": ["x"], "pools": {"train": [["%s", 1, "h"]]}}' % rid)
    open(os.path.join(t, "amp_mmap_cache", "k1", "d1", "last_used"), "w").close()
    open(os.path.join(t, "plan.yaml"), "w").write(PLAN)
    env = dict(os.environ, SCRATCH=t, SWEEP_DIR="", PATH="/nonexistent:" + os.environ["PATH"])
    return subprocess.run([sys.executable, TOOL, "--days", "1", "--plan", os.path.join(t, "plan.yaml")], env=env,
                          capture_output=True, text=True, check=True, cwd=ROOT).stdout


def test_plan_recipes_and_key_link():
    sys.path.insert(0, os.path.join(ROOT, "analysis", "storage"))
    from cache_census import pools_of
    rid = pools_of((os.path.join(ROOT, "recipes", "catalog_v3_sep_probes.yaml"), 42))[1][0][1]
    with tempfile.TemporaryDirectory() as t:
        out = _run(t, rid)
        inuse = set(next(l for l in out.splitlines() if l.startswith("INUSE")).split("\t")[1].split())
        assert {"catalog_v3_sep_probes.yaml", "vob_ee_uu_nlo.yaml", "catalog_v4_datasaving_trees.yaml"} <= inuse, inuse
        assert "ref_solo_ee_uu.yaml" not in inuse
        mm = next(l for l in out.splitlines() if l.startswith("MMAP"))
        assert "1 train pools, 1 of in-use recipes catalog_v3_sep_probes.yaml" in mm, mm
    with tempfile.TemporaryDirectory() as t:
        assert "1 train pools, 0 of in-use recipes -" in next(l for l in _run(t, "madeuprid").splitlines()
                                                             if l.startswith("MMAP"))


if __name__ == "__main__":
    test_plan_recipes_and_key_link(); print("ok")
