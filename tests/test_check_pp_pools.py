"""tools/check_pp_pools.py --labels: runs end to end on a one-process spec, passes on same-distribution pools and
fails on a 1% label shift and on a pool without a generator sha (nothing compared is never a pass)."""
import json, os, subprocess, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_mixed_generation import _pool


def _run(d, shift, sha=True):
    data, cache = os.path.join(d, "data"), os.path.join(d, "tmp", "amp_data_cache")
    os.makedirs(data); os.makedirs(cache)
    base = "p_13000-13000GeV_%s_lhe_amplitudes"
    for role, dd, n, sh in (("train", cache, 20000, shift), ("val", data, 5000, 0.0), ("test", data, 5000, 0.0)):
        pth = _pool(dd, base % role, "a" * 64, n, sh, hash(role) % 100)
        np.save(pth[:-4] + ".alphas.npy", np.full(n, 0.118))
        if not sha:
            json.dump({}, open(pth + ".recipe.json", "w"))
    spec = os.path.join(d, "spec.yaml")
    open(spec, "w").write("processes:\n  - {name: p, n_train: 20000, n_val: 5000, n_test: 5000}\n")
    env = dict(os.environ, DATA_DIR=data, SCRATCH=os.path.join(d, "tmp"))
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_pp_pools.py"), "--spec", spec, "--labels"],
                          capture_output=True, text=True, env=env)


def test_labels():
    for shift, sha, ok in ((0.0, True, True), (0.01, True, False), (0.0, False, False)):
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, shift, sha)
            assert "LABELS" in r.stdout, r.stdout + r.stderr
            bad = [l for l in r.stdout.splitlines() if l.startswith("BAD") and "diagrams" not in l]   # no sidecar here
            assert (not bad) == ok, (shift, sha, r.stdout)


if __name__ == "__main__":
    test_labels(); print("ok")
