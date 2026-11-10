"""The KIT hard phase-space benchmark loader (tools/kit_hardps.py) and the float64 model path (training.dtype=float64).

CPU only, on a small file in the benchmark's layout written here (no KIT data needed): e+e- -> u u~ g with a deep
collinear stratum at y = s_ug / s ~ 1e-7, and e+e- -> t t~ with a Born and an alpha_s-carrying one-loop target.
    python tests/test_kit_hardps.py
Checks
  1. pools: roles, sizes, the hard variant, val disjoint from train, the beams reordered to (e-, e+), the one-loop
     target stripped by alpha_s / 2 pi and its order vector [1, 0, 0, 2];
  2. a recipe run with training.dtype=float64: the network's momenta are float64, the deep stratum's s_ug / s read back
     from them matches the file's to 1e-6 relative (float32 momenta miss it by percents at y = 3e-7: the reason for the path), every
     module's floating output in the forward pass is float64, and the attention isolates events (a second event in the
     batch does not change the first's prediction).
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

P_EPLUS = {"pdg": -11, "Q": 1, "spin": 0.5, "mass": 0.0, "T3": 0.5, "B": 0, "L": -1, "color": 1, "C2": 0}
P_EMINUS = {"pdg": 11, "Q": -1, "spin": 0.5, "mass": 0.0, "T3": -0.5, "B": 0, "L": 1, "color": 1, "C2": 0}


def _q(pdg, mass):
    return {"pdg": pdg, "Q": 2 / 3 * np.sign(pdg), "spin": 0.5, "mass": mass, "T3": 0.5 * np.sign(pdg),
            "B": np.sign(pdg) / 3, "L": 0, "color": 3, "C2": 4 / 3}


G = {"pdg": 21, "Q": 0, "spin": 1.0, "mass": 0.0, "T3": 0, "B": 0, "L": 0, "color": 8, "C2": 3}


def _beams(n, sqrts):
    e = np.full(n, sqrts / 2)
    z = np.zeros(n)
    return np.stack([e, z, z, e], -1), np.stack([e, z, z, -e], -1)       # e+ along +z, e-


def _rot(v, rng):
    """v (n, 3) rotated by a random rotation per event"""
    q = rng.normal(size=(len(v), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    return np.einsum("nij,nj->ni", R, v)


def uug_events(n, rng, y=None, sqrts=1000.0):
    """massless u u~ g at fixed sqrt(s): the u along a random axis with energy fraction x1, the g at angle theta to it
    (y = s_ug / s set exactly when given, via 1 - cos theta), the u~ balancing. Float64."""
    ep, em = _beams(n, sqrts)
    E = sqrts / 2
    if y is None:
        x1 = rng.uniform(0.3, 0.7, n); x3 = rng.uniform(0.3, 0.7, n)
        x2 = 2 - x1 - x3
        ok = (x2 < 1) & (x2 > 0)
        x1, x3 = np.where(ok, x1, 0.6), np.where(ok, x3, 0.6)
        x2 = 2 - x1 - x3
        cos13 = 1 - 2 * (1 - x2) / (x1 * x3)
    else:
        x1 = np.full(n, 0.5); x3 = np.full(n, 0.4)
        # s_13 = 2 E1 E3 (1 - cos) = y s  ->  1 - cos = 2 y / (x1 x3)
        cos13 = 1 - 2 * y / (x1 * x3)
    th = np.arccos(np.clip(cos13, -1, 1))
    E1, E3 = x1 * E, x3 * E
    p1 = np.stack([np.zeros(n), np.zeros(n), E1], -1)
    p3 = np.stack([E3 * np.sin(th), np.zeros(n), E3 * np.cos(th)], -1)
    p2 = -(p1 + p3)
    # one common rotation per event: rotate the three with the same draw
    seed = int(rng.integers(1 << 30))
    v = [_rot(p, np.random.default_rng(seed)) for p in (p1, p2, p3)]
    E2 = np.linalg.norm(v[1], axis=1)
    fin = [np.concatenate([E1[:, None], v[0]], 1), np.concatenate([E2[:, None], v[1]], 1),
           np.concatenate([E3[:, None], v[2]], 1)]
    return np.stack([ep, em] + fin, 1)


def _mdot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def write_uug(path, rng):
    import h5py
    meta = {"process": "e+ e- -> u u~ g", "particles": [P_EPLUS, P_EMINUS, _q(2, 0.0), _q(-2, 0.0), G],
            "targets": {"me2": {"n_loops": 0, "alpha_s_power": 1, "alpha_power": 2}},
            "alpha_s": 0.118, "alpha_inv": 132.507, "sqrts": 1000.0}
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(meta)
        for split, st, n, y in [("bulk_train", "bulk", 600, None), ("bulk_test", "bulk", 200, None),
                                ("hard_train", "coll_y_1e-07_1e-06", 50, 3e-7),
                                ("hard_test", "coll_y_1e-07_1e-06", 100, 3e-7)]:
            mom = uug_events(n, rng, y)
            s = 2 * _mdot(mom[:, 0], mom[:, 1])
            s13, s23 = 2 * _mdot(mom[:, 2], mom[:, 4]), 2 * _mdot(mom[:, 3], mom[:, 4])
            me2 = 1e-3 * s / s13 * s / s23 * 1e-2
            g = f.create_group(f"{split}/{st}")
            g["momenta"] = mom
            g["me2"] = me2
            g["log10_me2"] = np.log10(me2)
            g["singular_var"] = s13 / s if y is not None else np.minimum(s13, s23) / s


def write_tt(path, rng):
    import h5py
    top = _q(6, 173.0)
    meta = {"process": "e+ e- -> t t~", "particles": [P_EPLUS, P_EMINUS, top, dict(_q(-6, 173.0))],
            "targets": {"born": {"n_loops": 0, "alpha_s_power": 0, "alpha_power": 2},
                        "virt_finite": {"n_loops": 1, "alpha_s_power": 1, "alpha_power": 2}},
            "alpha_s": 0.118, "alpha_inv": 132.507}
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(meta)
        for split, n in [("bulk_train", 300), ("bulk_test", 100)]:
            rs = rng.uniform(400, 1500, n)
            ep, em = _beams(n, 1.0)
            ep, em = ep * rs[:, None], em * rs[:, None]
            k = np.sqrt((rs / 2) ** 2 - 173.0 ** 2)
            d = _rot(np.stack([np.zeros(n), np.zeros(n), k], -1), rng)
            t = np.concatenate([rs[:, None] / 2, d], 1)
            tb = np.concatenate([rs[:, None] / 2, -d], 1)
            mom = np.stack([ep, em, t, tb], 1)
            born = 1e-2 * (1 + (d[:, 2] / k) ** 2)
            virt = 0.118 / (2 * np.pi) * born * 3.0
            g = f.create_group(f"{split}/bulk")
            g["momenta"] = mom
            g["born"] = born; g["virt_finite"] = virt
            g["log10_abs_born"] = np.log10(born); g["log10_abs_virt_finite"] = np.log10(virt)


def section1_pools(d):
    import datagen
    import mg5_pipeline_final as mg
    from tools import kit_hardps as kh
    assert kh.parse("kit_ee_tt__virt_finite") == ("ee_tt", "virt_finite", None)
    assert kh.parse("kit_ee_uug__me2__hard") == ("ee_uug", "me2", "hard")
    cfg = mg.register_kit("kit_ee_tt__virt_finite")
    assert mg.order_vector(cfg) == [1, 0, 0, 2], mg.order_vector(cfg)
    assert cfg["pdg_ids"] == [11, -11, 6, -6] and cfg["m_finals"] == [173.0, 173.0]
    assert mg.order_vector(mg.register_kit("kit_ee_uug__me2")) == [0, 0, 1, 2]

    p = datagen.ensure_dataset("kit_ee_tt__virt_finite", 353, 1739, 200, role="train", seed=42, require_cache=True)
    rows = np.load(p)
    assert rows.dtype == np.float64 and rows.shape == (200, 21), rows.shape
    import h5py
    with h5py.File(os.path.join(d, "ee_tt.h5")) as f:
        raw = f["bulk_train/bulk/virt_finite"][:200]
        mom = f["bulk_train/bulk/momenta"][:200]
    assert np.allclose(rows[:, -1], raw / (0.118 / (2 * np.pi)), rtol=1e-14), "the one-loop target is not stripped"
    assert np.array_equal(rows[:, :4], mom[:, 1]) and np.array_equal(rows[:, 4:8], mom[:, 0]), "beams not (e-, e+)"
    assert list(rows[0, 16:20]) == [11, -11, 6, -6]
    # the second call is a cache hit (the recipe sidecar's id)
    assert datagen.ensure_dataset("kit_ee_tt__virt_finite", 353, 1739, 200, role="train", seed=42,
                                  require_cache=True) == p

    tr = np.load(datagen.ensure_dataset("kit_ee_uug__me2", 1000, 1000, 500, role="train", seed=42))
    va = np.load(datagen.ensure_dataset("kit_ee_uug__me2", 1000, 1000, 100, role="val", seed=42))
    te = np.load(datagen.ensure_dataset("kit_ee_uug__me2", 1000, 1000, 200, role="test", seed=42))
    hb = np.load(datagen.ensure_dataset("kit_ee_uug__me2__hard", 1000, 1000, 500, role="train", seed=42))
    assert tr.shape[0] == 500 and va.shape[0] == 100 and te.shape[0] == 200 and hb.shape[0] == 550
    assert np.array_equal(hb[:500], tr), "the hard variant's bulk part differs"
    assert "_n500_" in os.path.basename(datagen.ensure_dataset("kit_ee_uug__me2", 1000, 1000, 500, role="train",
                                                                seed=42)), "the pool name lacks its event count"
    try:
        mg.register_recipe_processes([{"name": "kit_ee_uug__me2", "n_train": 550, "n_val": 100}])
        raise AssertionError("an overlapping train/val layout was accepted")
    except AssertionError as e:
        assert "overlap" in str(e), e
    trs, vas = {r.tobytes() for r in tr}, {r.tobytes() for r in va}
    assert not trs & vas, "val shares events with train"
    print("section 1: pools (roles, hard variant, val disjoint, beams, alpha_s stripping, order vectors) OK")


def _cfg(out, recipe, dtype):
    from hydra import compose, initialize_config_dir
    ov = ["data.source=recipes", f"data.processes_file={recipe}", "model=lloca", f"run_dir={out}",
          "data.require_cache=true", "data.use_PIDs=false", "data.offshell_per_event=false",
          "data.internal_mass_scalars=false", "data.coupling_scalars=false", "model.use_diagrams=false",
          "training.sign_head=false", "data.preprocess_per_dataset=true", "data.eval_subsample=null",
          "training.batchsize=64", "evaluation.batchsize=64", "training.num_workers=0", f"training.dtype={dtype}",
          "model.net.num_blocks=2", "model.net.num_heads=2", "seed=7"]
    with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
        return compose("amplitudes", overrides=ov)


def _build(cfg):
    from experiment import AmplitudeExperiment
    torch.set_default_dtype(torch.float64 if cfg.training.dtype == "float64" else torch.float32)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    return exp


def _deep_y(exp, n_test):
    """s_ug / s of the test role's events, from the momenta the network reads (times the momentum scale)"""
    off = np.asarray(exp.offsets)
    idx = np.asarray(exp._split_indices["test"])
    out = []
    for i in idx[:n_test]:
        a, b = off[i]
        p = np.asarray(exp.particles_flat[a:b], dtype=np.float64) * float(exp.mom_div)
        s = _mdot(p[0] + p[1], p[0] + p[1])
        out.append(2 * _mdot(p[2], p[4]) / s)
    return np.asarray(out)


def section2_float64(d, tmp):
    import yaml
    from tools import kit_hardps as kh
    import h5py
    # the deep stratum as the test role: a recipe whose test pool is that stratum (what tools/kit_predict.py does)
    recipe = os.path.join(tmp, "kit_uug.yaml")
    yaml.safe_dump({"processes": [{"name": "kit_ee_uug__me2", "sqrts": [1000, 1000],
                                   "n_train": 400, "n_val": 100, "n_test": 100}]}, open(recipe, "w"))
    deep = kh.stratum_rows("ee_uug", "me2", "hard_test", "coll_y_1e-07_1e-06")
    with h5py.File(os.path.join(d, "ee_uug.h5")) as f:
        y_true = f["hard_test/coll_y_1e-07_1e-06/singular_var"][:]
    import datagen
    real = datagen.ensure_split_set
    deep_path = os.path.join(tmp, "deep_test.npy")
    np.save(deep_path, deep)

    def split(specs, role, seed, dest_dir=None, require_cache=False):
        return {"kit_ee_uug__me2": deep_path} if role == "test" else real(specs, role=role, seed=seed,
                                                                          dest_dir=dest_dir,
                                                                          require_cache=require_cache)
    datagen.ensure_split_set = split
    try:
        res = {}
        for dtype in ("float32", "float64"):
            exp = _build(_cfg(os.path.join(tmp, f"run_{dtype}"), recipe, dtype))
            perm = np.asarray(exp._role_perm["test"])
            y = _deep_y(exp, 100)
            res[dtype] = np.abs(y / y_true[perm[:len(y)]] - 1)
            if dtype == "float64":
                assert exp.particles_flat.dtype == np.float64
                _check_forward(exp)
    finally:
        datagen.ensure_split_set = real
    print(f"section 2: deep collinear y = 3e-7, |y read back / y - 1|: float64 max {res['float64'].max():.2e}, "
          f"float32 median {np.median(res['float32']):.2e}")
    assert res["float64"].max() < 1e-6, res["float64"].max()
    assert np.median(res["float32"]) > 1e-2, "float32 resolves y = 3e-7: the float64 path would not be needed"


def _check_forward(exp):
    model = exp.model.to(exp.device, dtype=exp.dtype).eval()
    bad = []

    def hook(mod, inp, out):
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(t) and t.is_floating_point() and t.dtype != torch.float64:
                bad.append((type(mod).__name__, t.dtype))
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    batch = next(iter(exp.test_loader))
    with torch.no_grad():
        y1 = exp._collect_predictions([batch])[0]
    for h in hs:
        h.remove()
    assert not bad, f"modules with non-float64 outputs: {sorted(set(bad))[:10]}"
    # event isolation: the first 8 events alone predict what they predict inside the full batch
    from torch.utils.data import Subset, DataLoader
    from dataset import collate_variable_length
    sub = next(iter(DataLoader(Subset(exp.test_loader.dataset, list(range(8))), batch_size=8,
                               collate_fn=collate_variable_length)))
    with torch.no_grad():
        y2 = exp._collect_predictions([sub])[0]
    first8 = next(iter(DataLoader(Subset(exp.test_loader.dataset, list(range(64))), batch_size=64,
                                  collate_fn=collate_variable_length)))
    with torch.no_grad():
        y3 = exp._collect_predictions([first8])[0]
    assert np.allclose(y2[:8], y3[:8], rtol=0, atol=1e-10), np.abs(y2[:8] - y3[:8]).max()
    print("section 2: float64 forward: every module float64, events isolated in attention")


EVAL_STUB = """import json, sys, h5py
t, p = h5py.File(sys.argv[1]), h5py.File(sys.argv[2])
out = {}
for sp in p:
    for st in p[sp]:
        for k in p[sp][st]:
            d = p[sp][st][k][:] - t[sp][st][k][:]
            out[sp + "/" + st + "/" + k] = {"n": len(d), "logmse": float((d ** 2).mean())}
json.dump({"strata": out}, open(sys.argv[3], "w"))
print("stub", len(out))
"""


def section3_predict(d, tmp):
    """a 20-step float64 CPU run on kit_ee_uug__me2, scored by tools/kit_predict.py: every stratum predicted, in the
    file's order (the bulk test pool's prediction equals the run's own test-role prediction event by event)"""
    import subprocess
    import yaml
    import h5py
    src = os.path.join(os.path.dirname(os.path.dirname(d)), "src")
    os.makedirs(src, exist_ok=True)
    open(os.path.join(src, "evaluate.py"), "w").write(EVAL_STUB)
    recipe = os.path.join(tmp, "kit_uug_run.yaml")
    yaml.safe_dump({"processes": [{"name": "kit_ee_uug__me2", "sqrts": [1000, 1000],
                                   "n_train": 400, "n_val": 100, "n_test": 200}]}, open(recipe, "w"))
    run = os.path.join(tmp, "run_predict")
    ov = ["data.source=recipes", f"data.processes_file={recipe}", "model=lloca", f"run_dir={run}",
          "data.require_cache=true", "data.offshell_per_event=false", "data.internal_mass_scalars=false",
          "data.coupling_scalars=false", "model.use_diagrams=false", "training.sign_head=false",
          "training.batchsize=64", "evaluation.batchsize=64", "training.num_workers=0", "training.dtype=float64",
          "model.net.num_blocks=2", "model.net.num_heads=2", "training.iterations=20", "plot=false",
          "use_mlflow=false", "count_flops=false", "seed=7"]
    p = subprocess.run([sys.executable, os.path.join(ROOT, "run.py")] + ov, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "kit_predict.py"), run], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with h5py.File(os.path.join(run, "kit_benchmark", "predictions.h5")) as f:
        keys = sorted(f"{a}/{b}" for a in f for b in f[a])
        assert keys == ["bulk_test/bulk", "hard_test/coll_y_1e-07_1e-06"], keys
        pb = f["bulk_test/bulk/log10_me2"][:]
    res = json.load(open(os.path.join(run, "kit_benchmark", "kit_result.json")))
    assert len(res["strata"]) == 2
    # the run's own test role is bulk_test's first 200 events: its test predictions, back in pool order, are these
    from tools.rebuild_run import rebuild
    from tools import kit_hardps as kh
    pool = os.path.join(tmp, "own_test.npy")
    np.save(pool, kh.stratum_rows("ee_uug", "me2", "bulk_test", "bulk")[:200])
    exp, loader = rebuild(run, "test", pool, 200)
    with torch.no_grad():
        pr, tr, _ = exp._collect_predictions(loader)
    z = (np.asarray(pr, np.float64).reshape(-1) - np.asarray(tr, np.float64).reshape(-1)) * float(exp.prepd_std[0])
    own = np.full(200, np.nan)
    own[np.asarray(exp._role_perm["test"])[:200]] = z
    with h5py.File(os.path.join(d, "ee_uug.h5")) as f:
        truth = f["bulk_test/bulk/log10_me2"][:200]
    assert np.allclose(pb[:200], truth + own / np.log(10), rtol=0, atol=1e-9), np.abs(pb[:200] - truth - own / np.log(10)).max()
    print(f"section 3: kit_predict on a 20-step float64 run: {len(keys)} strata scored, bulk predictions in file order")


def main():
    tmp = tempfile.mkdtemp(prefix="kit_test_")
    d = os.path.join(tmp, "kit")
    os.makedirs(d)
    rng = np.random.default_rng(3)
    write_uug(os.path.join(d, "ee_uug.h5"), rng)
    write_tt(os.path.join(d, "ee_tt.h5"), rng)
    os.environ["KIT_HARDPS_DIR"] = d
    for k in ("AMP_FROZEN_DIR", "AMP_TRAIN_CACHE_DIR", "DATA_DIR", "SCRATCH"):
        os.environ[k] = os.path.join(tmp, k.lower())
        os.makedirs(os.environ[k], exist_ok=True)
    os.environ.setdefault("MG5_OUTPUT_DIR", os.path.join(tmp, "mg5_out"))
    os.makedirs(os.environ["MG5_OUTPUT_DIR"], exist_ok=True)
    section1_pools(d)
    section2_float64(d, tmp)
    section3_predict(d, tmp)


if __name__ == "__main__":
    main()
