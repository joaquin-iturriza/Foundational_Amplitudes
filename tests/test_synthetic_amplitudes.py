#!/usr/bin/env python
"""CPU validation of the synthetic-amplitude generator (tools/synthetic_amplitudes.py).

  (a) Lorentz invariance: |M|^2 unchanged under random rotations x boosts;
  (b) mass dimension: |M|^2(lam p; lam M, lam Gamma) = lam^(8-2N) |M|^2(p);
  (c) reproducibility: the structure from the name alone (fresh interpreter, other hash
      seed), the pool bytes from the recipe seed;
  (d) log|M|^2 statistics of the pilot pools, next to real pools when some are found
      (AMP_REAL_POOLS, default ~/datasets), with a figure (png + pdf);
  (e) the recipe path: the pools prebuild_recipes.py wrote for recipes/synthetic_pilot.yaml
      (AMP_FROZEN_DIR / AMP_TRAIN_CACHE_DIR) in the exact format of the real pools, their
      recipe ids, the diagram sidecars read by diagram_graphs as experiment._setup_offshell_masks
      reads them, and, when experiment.py imports, init_physics + init_data of the real
      AmplitudeExperiment on the pilot recipe with the ladder's data settings.

Run (after `python prebuild_recipes.py recipes/synthetic_pilot.yaml` with the same env):
    AMP_FROZEN_DIR=... AMP_TRAIN_CACHE_DIR=... python tests/test_synthetic_amplitudes.py [--fig-dir D]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg          # noqa: E402
import datagen                           # noqa: E402
from tools import synthetic_amplitudes as syn   # noqa: E402

RECIPE = os.path.join(ROOT, "recipes", "synthetic_pilot.yaml")
LADDER_DATA = {   # data settings of sweep/sweep_config_tp3_ladder_r9.yaml
    "data.source": "recipes", "data.eval_subsample": 10000, "data.phase_space_on_pool": True,
    "data.preprocess_per_dataset": True, "data.signedlog_quantile": 0.01, "data.seed": 42,
    "data.use_PIDs": False, "data.spin_onehot": True, "data.color_onehot": True,
    "data.prop_is_massless": True, "data.standardize_props": True, "data.generation_onehot": True,
    "data.generation_feature": True, "data.mass_from_momenta": False, "data.coupling_scalars": True,
    "data.internal_mass_scalars": True, "data.offshell_per_event": True,
    "data.target_propagators": False, "data.internal_mass_pdgs": [23, 6, 25],
    "training.sign_head": True, "model.use_diagrams": False,
}


def _recipe():
    import yaml
    with open(RECIPE) as f:
        return yaml.safe_load(f)


def _events(name, n, rng, democratic=0.5):
    """n events of `name` under the fiducial cuts, half from the IR-democratic splitter
    (the corners the mixture pools reach)."""
    cfg = mg.register_synthetic(name)
    lo, hi = cfg["sqrts"]
    ev, _ = mg._candidates(n, lambda k: rng.uniform(lo, hi, k), cfg["m_finals"], cfg["pdg_ids"], rng,
                           mg.FIDUCIAL_CUTS, democratic, 1e-6)
    return np.stack([e[0] for e in ev])


def _lorentz(rng, B, max_rapidity=2.0):
    """B random proper orthochronous transforms: rotation x boost (rapidity up to max)."""
    L = np.zeros((B, 4, 4))
    for b in range(B):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        y = rng.uniform(0, max_rapidity); g, bg = np.cosh(y), np.sinh(y)
        K = np.eye(4); K[0, 0] = g; K[0, 1:] = K[1:, 0] = bg * n
        K[1:, 1:] += (g - 1) * np.outer(n, n)
        Rot = np.eye(4); Rot[1:, 1:] = Q
        L[b] = K @ Rot
    return L


def check_lorentz(names, rng, n=400):
    worst = 0.0
    for nm in names:
        P = _events(nm, n, rng)
        P2 = np.einsum("bij,bkj->bki", _lorentz(rng, n), P)
        st = syn.structure(nm)
        a, b = syn.evaluate(st, P), syn.evaluate(st, P2)
        worst = max(worst, float(np.max(np.abs(b / a - 1))))
    print(f"(a) Lorentz invariance, {len(names)} processes x {n} events, rapidity <= 2: "
          f"max |dM2/M2| = {worst:.2e}")
    assert worst < 1e-9, worst
    return worst


def check_dimension(names, rng, n=200):
    worst = 0.0
    for nm in names:
        P = _events(nm, n, rng)
        st = syn.structure(nm)
        N = P.shape[1]
        a = syn.evaluate(st, P)
        for lam in (0.1, 0.37, 2.9, 10.0):
            b = syn.evaluate(st, lam * P, scale=lam)
            worst = max(worst, float(np.max(np.abs(b / (lam ** (8 - 2 * N) * a) - 1))))
    print(f"(b) mass dimension 8-2N, lambda in (0.1, 0.37, 2.9, 10): max deviation = {worst:.2e}")
    assert worst < 1e-10, worst
    return worst


def check_reproducible(names):
    code = ("import sys, json; sys.path.insert(0, %r); from tools import synthetic_amplitudes as s; "
            "print(json.dumps([s.structure(n)['sha'] for n in %r]))" % (ROOT, list(names)))
    shas = []
    for seed in ("0", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env=dict(os.environ, PYTHONHASHSEED=seed)).stdout
        shas.append(json.loads(out.strip().splitlines()[-1]))
    here = [syn.structure(n)["sha"] for n in names]
    assert shas[0] == shas[1] == here, "structure depends on the interpreter"
    assert len(set(here)) == len(here), "two names drew the same structure"
    # the bytes: one chunk built twice from the same seed, through the pipeline's builder
    cfg = dict(mg.register_synthetic(names[0])); cfg["sampling"] = {"mode": "mixture"}
    arrs = []
    with tempfile.TemporaryDirectory() as d:
        for k in range(2):
            out = os.path.join(d, f"p{k}.npy")
            mg.build_dataset_variable_energy(3000, *cfg["sqrts"], None, "synthetic", ["synthetic"], None,
                                             cfg, out, rng=np.random.default_rng(7), role="train")
            arrs.append(np.load(out))
    assert np.array_equal(arrs[0], arrs[1]), "pool not reproducible from the seed"
    print(f"(c) reproducibility: {len(names)} structure hashes identical across interpreters "
          f"(PYTHONHASHSEED 0 / 12345) and distinct; a 3000-event mixture chunk is bit-identical")


COUNT = {"train": "n_train", "val": "n_val", "test": "n_test"}


def check_pools(doc):
    """Every pool as the experiment reads it: shape, pdg block, kinematics, target, recipe id."""
    mg.register_recipe_processes(doc["processes"], default_sampling=doc.get("sampling"))
    stats = {}
    for role in ("train", "val", "test"):
        specs = [{"process": p["name"], "sqrts_min": p["sqrts"][0], "sqrts_max": p["sqrts"][1],
                  "n_events": p[COUNT[role]]} for p in doc["processes"]]
        paths = datagen.ensure_split_set(specs, role=role, seed=42, require_cache=True)
        for p in doc["processes"]:
            nm = p["name"]
            cfg = mg.PROCESSES[nm]
            d = np.load(paths[nm])
            N = len(cfg["pdg_ids"])
            assert d.shape == (p[COUNT[role]], 5 * N + 1), d.shape
            mom = d[:, :4 * N].reshape(-1, N, 4)
            assert np.array_equal(d[:, 4 * N:-1], np.tile(np.asarray(cfg["pdg_ids"], float), (len(d), 1)))
            sq = mom[:, 0, 0] + mom[:, 1, 0]
            assert np.allclose(mom[:, :2, 1:3], 0)
            assert np.allclose(mom[:, 0, 3], sq / 2) and np.allclose(mom[:, 1, 3], -sq / 2)
            cons = np.abs(mom[:, :2].sum(1) - mom[:, 2:].sum(1)).max() / sq.max()
            m2 = mom[:, 2:, 0] ** 2 - (mom[:, 2:, 1:] ** 2).sum(-1)
            onshell = np.abs(np.sqrt(np.clip(m2, 0, None)) - np.asarray(cfg["m_finals"])).max()
            assert cons < 1e-9 and onshell < 1e-3, (nm, cons, onshell)
            assert sq.min() >= cfg["sqrts"][0] - 1e-9 and sq.max() <= cfg["sqrts"][1] + 1e-9
            y = d[:, -1]
            assert np.all(np.isfinite(y)) and np.all(y > 0), nm
            rec = mg.variable_energy_recipe(nm, *p["sqrts"], len(d), role=role, seed=42)
            with open(paths[nm] + ".recipe.json") as f:
                side = json.load(f)
            assert side["recipe_id"] == mg.recipe_id(rec) and side["synthetic"]["sha"] == cfg["synthetic"]["sha"]
            assert paths[nm].endswith("_smix_amplitudes.npy")
            if role == "train":
                stats[nm] = {"logm": np.log(y), "mom": mom}
    print(f"(e) pools: {len(doc['processes'])} processes x 3 roles in the real pools' format "
          f"(rows = 4N momenta + N pdg ids + |M|^2 > 0), on shell, momentum conserved, beams on z, "
          f"recipe ids match (cache hits)")
    return stats


def check_sidecars(doc, stats):
    """The sidecars through diagram_graphs, the way experiment._setup_offshell_masks reads them:
    every process maps (N = P), and every structure propagator is a sidecar mask row with the
    same invariant and pole mass."""
    from diagram_graphs import build_process_diagrams, build_process_virtuality
    from particle_ids import build_property_matrix
    prop_matrix, _ = build_property_matrix(spin_onehot=True, color_onehot=True, is_massless=True,
                                           standardize=True, generation_onehot=True, generation_feature=True)
    pdgs = LADDER_DATA["data.internal_mass_pdgs"]
    n_ok, n_sel, min_massless, worst_s = 0, 0, np.inf, 0.0
    for p in doc["processes"]:
        nm = p["name"]
        pd = build_process_diagrams(syn.write_sidecar(nm), prop_matrix, k_pe=8)
        n_initial = sum(1 for leg in pd.external if leg["state"] == "in")
        mo = {g: mg.internal_mass(nm, g) for g in pdgs}
        vt = build_process_virtuality(pd, [int(x) for x in mg.PROCESSES[nm]["pdg_ids"]], n_initial,
                                      mass_override=mo, offshell=True)
        assert vt is not None, nm
        n_ok += 1
        mask, m2, ppdg = vt["mask"].numpy().astype(np.float64), vt["prop_mass2"].numpy(), vt["prop_pdgs"]
        n_sel += sum(abs(x) in pdgs for x in ppdg)
        mom = stats[nm]["mom"][:2000]
        q = np.einsum("ks,nsc->nkc", mask, mom)
        s_side = q[..., 0] ** 2 - (q[..., 1:] ** 2).sum(-1)              # (n, K)
        sign = np.array([-1.0, -1.0] + [1.0] * (mom.shape[1] - 2))
        for d in syn.structure(nm)["diagrams"]:
            for pr in d["props"]:
                v = (sign[None, pr["legs"], None] * mom[:, pr["legs"]]).sum(1)
                s_me = v[:, 0] ** 2 - (v[:, 1:] ** 2).sum(-1)
                err = np.abs(s_side - s_me[:, None]).max(0) / np.abs(s_me).max()
                ok = [k for k in range(len(err)) if err[k] < 1e-9 and abs(ppdg[k]) == abs(pr["pdg"])]
                assert ok, (nm, pr)
                worst_s = max(worst_s, float(err[ok[0]]))
                if abs(pr["pdg"]) in pdgs:
                    assert any(abs(m2[k] - pr["mass"] ** 2) <= 1e-6 * pr["mass"] ** 2 for k in ok), (nm, pr)
                if pr["mass"] == 0.0:
                    min_massless = min(min_massless, float(np.abs(s_me).min()))
    P = len(doc["processes"])
    print(f"(e) sidecars: offshell_per_event masks build for {n_ok}/{P} processes "
          f"({n_sel} sidecar propagators carry pdgs {pdgs}); every structure propagator is in its "
          f"sidecar (max rel. invariant mismatch {worst_s:.1e}; pole masses incl. the random scalar's); "
          f"smallest |s_S| of a massless propagator over the train pools: {min_massless:.1f} GeV^2")
    assert n_ok == P
    return min_massless


def _flat_logm(name, n, rng):
    """log|M|^2 of n events sampled as the real flat pools are (uniform sqrt(s), RAMBO, cuts)."""
    cfg = mg.register_synthetic(name)
    sampler = mg.sample_2to2_phase_space if cfg["nfinal"] == 2 else mg.sample_nbody_phase_space
    ev, _ = sampler(n, *cfg["sqrts"], cfg["m_finals"], cfg["pdg_ids"], rng=rng, cuts=mg.FIDUCIAL_CUTS)
    return np.log(syn.label_events(name, ev))


def check_stats(stats, fig_dir, rng):
    """log|M|^2 spread of the synthetic processes, flat-sampled like the real pools found under
    AMP_REAL_POOLS (default ~/datasets: legacy flat pools), and of their mixture pools."""
    real_dir = os.environ.get("AMP_REAL_POOLS", os.path.expanduser("~/datasets"))
    real, nf_real = {}, {}
    for f in sorted(glob.glob(os.path.join(real_dir, "*-*GeV_amplitudes.npy"))):
        a = np.load(f, mmap_mode="r")
        y = np.asarray(a[:100000, -1])
        y = y[y > 0]
        if len(y) > 1000:
            k = os.path.basename(f).replace("_amplitudes.npy", "")
            real[k], nf_real[k] = np.log(y), (a.shape[1] - 1) // 5 - 2
    flat = {k: _flat_logm(k, 20000, rng) for k in stats}
    sd = lambda l: float(l.std())
    r99 = lambda l: float(np.quantile(l, 0.99) - np.quantile(l, 0.01))
    print("(d) log|M|^2 of the synthetic processes: flat sampling (as the real pools) | mixture train pool")
    for k in stats:
        m = stats[k]["logm"]
        print(f"      {k}  2->{syn.structure(k)['nfinal']}  flat: std {sd(flat[k]):5.2f} 1-99% {r99(flat[k]):5.1f}"
              f" median {np.median(flat[k]):7.2f} | mixture: std {sd(m):5.2f} 1-99% {r99(m):5.1f}")
    for lab, d in (("synthetic, flat", flat), ("synthetic, mixture", {k: v["logm"] for k, v in stats.items()}),
                   (f"real ({real_dir}), flat", real)):
        if d:
            a = np.array([sd(v) for v in d.values()])
            print(f"    {lab}: {len(a)} pools, std median {np.median(a):.2f} [{a.min():.2f}, {a.max():.2f}]")
    for k, l in real.items():
        print(f"      {k:28s} 2->{nf_real[k]}  std {sd(l):5.2f}  1-99% {r99(l):5.1f}  median {np.median(l):7.2f}")
    if fig_dir:
        import plot_style as ps
        fig, (axL, axR) = ps.figure(ncols=2)
        for i, l in enumerate(flat.values()):
            axL.hist(l - np.median(l), bins=60, range=(-12, 12), histtype="step", density=True,
                     color=ps.C.blue, alpha=0.5, label="synthetic" if i == 0 else None)
        for i, l in enumerate(real.values()):
            axL.hist(l - np.median(l), bins=60, range=(-12, 12), histtype="step", density=True,
                     color=ps.C.vermillion, label="real" if i == 0 else None)
        axL.set_xlabel(r"$\log|\mathcal{M}|^2 - $ median"); axL.set_ylabel("density"); axL.set_yscale("log")
        ps.legend(axL, "upper left")
        axR.scatter([syn.structure(k)["nfinal"] - 0.06 for k in flat], [sd(v) for v in flat.values()],
                    color=ps.C.blue, label="synthetic")
        if real:
            axR.scatter([nf_real[k] + 0.06 for k in real], [sd(v) for v in real.values()],
                        color=ps.C.vermillion, marker="s", label="real")
        axR.set_xlabel("final-state multiplicity"); axR.set_ylabel(r"std of $\log|\mathcal{M}|^2$")
        axR.set_xticks([2, 3, 4])
        os.makedirs(fig_dir, exist_ok=True)
        base = os.path.join(fig_dir, "synthetic_logm2")
        ps.save(fig, base)
        print(f"    figure: {base}.png / .pdf (both flat-sampled)")


def check_experiment():
    """init_physics + init_data of the real AmplitudeExperiment on the pilot recipe (CPU)."""
    try:
        import logging
        from hydra import compose, initialize_config_dir
        from experiment import AmplitudeExperiment
        from logger import LOGGER
    except Exception as e:      # the laptop has no training env
        print(f"(e) experiment path NOT exercised: experiment.py does not import here ({e!r})")
        return None
    lines = []

    class _Grab(logging.Handler):
        def emit(self, r):
            lines.append(r.getMessage())
    LOGGER.addHandler(_Grab()); LOGGER.setLevel(logging.INFO)
    run_dir = tempfile.mkdtemp(prefix="syn_exp_")
    fmt = lambda v: json.dumps(v) if isinstance(v, list) else (str(v).lower() if isinstance(v, bool) else v)
    ov = [f"{k}={fmt(v)}" for k, v in LADDER_DATA.items()]
    ov += [f"data.processes_file={RECIPE}", "data.require_cache=true", "model=lloca", f"run_dir={run_dir}"]
    with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
        cfg = compose("amplitudes", overrides=ov)
    if int(np.__version__.split(".")[0]) >= 2:
        # the sites pin numpy 1.26, where float() of the (1,)-shaped standardization stats works;
        # numpy 2 refuses it, so hand the data path scalars here
        import experiment as _ex
        _pp = _ex.preprocess_amplitude
        _sc = lambda v: float(np.asarray(v).reshape(-1)[0]) if np.size(v) == 1 else v
        _ex.preprocess_amplitude = lambda *a, **k: (lambda r: (r[0], _sc(r[1]), _sc(r[2])))(_pp(*a, **k))
    exp = AmplitudeExperiment(cfg)
    exp.warm_start = False
    exp.init_physics()
    exp.init_data()
    ok = [l for l in lines if l.startswith("offshell_per_event: built propagator masks")]
    P = len(cfg.data.dataset)
    print(f"(e) experiment: init_physics + init_data ran on {P} synthetic processes; "
          f"amp_orders {[list(o) for o in cfg.data.amp_orders][:3]}...; log: {ok[0] if ok else 'NO offshell line'}")
    for l in lines:
        if any(t in l for t in ("Preprocessing amplitudes", "Recipe data source", "offshell_per_event ON")):
            print(f"      {l[:240]}")
    assert ok and f"for {P}/{P} processes" in ok[0], ok
    return ok[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--n-check", type=int, default=30, help="synthetic processes in (a)-(c)")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    names = [f"syn_{k:05d}" for k in range(a.n_check)] + [f"syn{n}_{k:05d}" for n in (2, 3, 4) for k in range(5)]
    check_lorentz(names, rng)
    check_dimension(names, rng)
    check_reproducible(names)
    doc = _recipe()
    stats = check_pools(doc)
    check_stats(stats, a.fig_dir, rng)
    check_sidecars(doc, stats)
    check_experiment()
    print("all synthetic-amplitude checks passed")
