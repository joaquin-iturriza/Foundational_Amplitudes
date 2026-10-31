"""dipole_residual.pool_rows: a read's idx (positions in the role-contiguous train|val|test array, each role shuffled by
one rng seeded 42 as experiment._init_data does) map back to the pool rows the events came from."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.catalog_v4.dipole_residual import pool_rows  # noqa: E402


class Z(dict):
    @property
    def files(self):
        return list(self)


def _layout(counts):
    rng = np.random.default_rng(seed=42)
    return [rng.permutation(c) for c in counts]


def test_val_and_test_rows():
    nt, nv, ns = 10000, 2000, 2000
    pv, ps = _layout([nt, nv, ns])[1:]
    for split, perm, start in (("val", pv, nt), ("test", ps, nt + nv)):
        idx = np.arange(start, start + len(perm))           # scored in loader order
        rows = pool_rows(Z(idx=idx, split=np.array(split)))
        assert np.array_equal(rows, perm)                   # position i of the role holds pool row perm[i]


def test_saved_pool_row_wins():
    z = Z(idx=np.arange(10, 13), split=np.array("val"), pool_row=np.array([2, 0, 1]))
    assert list(pool_rows(z)) == [2, 0, 1]


if __name__ == "__main__":
    test_val_and_test_rows()
    test_saved_pool_row_wins()
    print("ok")


def test_eta2_and_tail_share():
    """eta2 sees a mean that depends on x with no linear tilt; tail_share sees an error concentrated at one end."""
    from analysis.catalog_v4.dipole_residual import eta2, tail_share
    r = np.random.default_rng(0)
    x = r.normal(size=4000)
    u = np.sin(3 * x) + 0.3 * r.normal(size=4000)
    assert eta2(u, x) > 0.6 and np.corrcoef(u, x)[0, 1] ** 2 < 0.05
    v = np.where(x > 1.3, 3.0, 0.0) * r.normal(size=4000) + 0.1 * r.normal(size=4000)
    lo, hi = tail_share(v, x)
    assert hi > 0.9 and lo < 0.05
    w = r.normal(size=4000)
    assert eta2(w, x) < 0.02 and all(0.07 < s < 0.13 for s in tail_share(w, x))


def test_joint_r2_finds_signal_and_not_noise():
    """The joint read's out-of-fold fit recovers a nonlinear signal in one feature and reads no signal in noise."""
    from analysis.catalog_v4.dipole_residual import joint_features, joint_r2
    r = np.random.default_rng(1)
    m = r.normal(size=(2000, 6, 4))
    m[..., 0] = np.abs(m[..., 0]) + 3
    t = r.normal(size=2000)
    X = joint_features(m, t, t * 0, t * 0, None)
    u = np.sin(2 * t) + 0.5 * r.normal(size=2000)
    rj, rn = joint_r2(X, u)
    assert rj > 0.5 and rn < 0.05
    assert joint_r2(X, r.normal(size=2000))[0] < 0.05


def test_lof_features_ignore_leg_order():
    """The leave-class-out read's features are the same when the final legs are listed in another order."""
    from analysis.catalog_v4.dipole_residual import lof_features
    r = np.random.default_rng(1)
    n, N = 6, 50
    m = np.zeros((N, n, 4))
    m[:, 2:, 1:] = r.normal(size=(N, n - 2, 3))
    m[:, 2:, 0] = np.linalg.norm(m[:, 2:, 1:], axis=-1)
    tot = m[:, 2:].sum(1)
    m[:, 2:, 1:] -= tot[:, None, 1:] / (n - 2)       # roughly balanced; exact balance is not needed here
    m[:, 2:, 0] = np.linalg.norm(m[:, 2:, 1:], axis=-1)
    E = m[:, 2:, 0].sum(1) / 2
    m[:, 0] = np.stack([E, 0 * E, 0 * E, E], 1)
    m[:, 1] = np.stack([E, 0 * E, 0 * E, -E], 1)
    pdg = np.tile([11, -11, 21, 1, -1, 22], (N, 1))
    t = r.normal(size=N)
    X1, names = lof_features(m, pdg, t, t, t, None)
    perm = [0, 1, 5, 3, 2, 4]
    X2, _ = lof_features(m[:, perm], pdg[:, perm], t, t, t, None)
    assert np.allclose(X1, X2) and X1.shape[1] == len(names) == 38
    assert X1[0, names.index("n_g")] == 1 and X1[0, names.index("n_q")] == 2 and X1[0, names.index("n_a")] == 1


def test_lco_scored_never_sees_the_scored_group():
    """A pool's prediction comes from a map fitted without its group: a per-group signal no other group shares
    is not recovered, one every group shares is."""
    from analysis.catalog_v4.lco_map import scored
    r = np.random.default_rng(0)
    pools, groups = {}, {}
    for k in range(8):
        X = r.normal(size=(300, 3))
        shared = np.sin(2 * X[:, 0])
        own = (k - 3.5) * X[:, 1]          # a slope only this pool has
        pools[f"p{k}"] = (X, shared + own + 0.1 * r.normal(size=300), np.nan)
        groups[f"p{k}"] = k
    vr = scored(pools, groups, [0, 1, 2])
    for p, (X, u, _) in pools.items():
        own_var = np.var(u - np.sin(2 * X[:, 0]))
        assert vr[p] > 0.5 * own_var       # its own slope is left in the residual


def test_lco_map_end_to_end_on_a_synthetic_dump(tmp_path):
    """Stage 2 runs on a small synthetic dump: a residual shared by every held-out pool gives f_lco near f_ceil."""
    import json
    import subprocess
    import sys
    from analysis.catalog_v4 import lco_map
    tl = json.load(open(os.path.join(os.path.dirname(lco_map.__file__), "data", "tree_levels.json")))
    ho = [p for ax in tl["heldout"].values() for p, v in ax.items() if f"{v['n']}{v['initial']}" == "6qq"][:12]
    tr = [p for p, v in tl["train"].items() if f"{v['n']}{v['initial']}" == "6qq"][:6]
    names = [f"sij{k}" for k in range(15)] + [f"lE{k}" for k in range(4)] + [f"cth{k}" for k in range(4)] + ["lshat"] \
        + [f"n_{k}" for k in ("g", "a", "q", "b", "l", "nu", "W", "Z", "H", "t")] + ["t", "logF", "logP", "logD"]
    r = np.random.default_rng(0)
    for arm in ("A",):
        for ax, ps in (("H2", ho), ("TR", tr)):
            d = tmp_path / f"{arm}_{ax}"
            d.mkdir()
            for p in ps:
                X = r.normal(size=(500, 38)).astype(np.float32)
                X[:, 37] = 0                  # no final gluon or photon: the drop-group branch names the arm
                u = (np.sin(2 * X[:, 0]) if ax == "H2" else 0) + 0.5 * r.normal(size=500)
                np.savez(d / f"{p}.npz", X=X, u=u, names=np.array(names), var_u=np.var(u), n=2000, ceiling_own=0.6)
    out = tmp_path / "o.json"
    run = subprocess.run([sys.executable, lco_map.__file__, "--root", str(tmp_path), "--arms", "A", "--out", str(out)],
                   check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    R = json.load(open(out))["A"]
    assert R["f_lco"] > 0.5 * R["f_ceil"] > 0.2
    assert R["f_lco_factors"] < 0.2
    V = json.load(open(out))["verdict"]
    assert V["per_seed"]["A"]["branch"] == "yes" and V["per_seed"]["A"]["next_arm"] == "the pairs group's term"
