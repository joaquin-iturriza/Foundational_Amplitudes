#!/usr/bin/env python3
"""Does the singular variable survive the data path into the network? Per KIT stratum, float32 against float64.

    python tools/kit_precision_check.py [--files ee_tt ee_uug ee_uugg] [--n 2000] [--out <json>]

For each file's first target and each hard_test stratum (and bulk_test), the recipe path is built (CPU, a few
events per role) with that stratum as the test role, for training.dtype float32 and float64; the momenta the network
reads (augmented, scaled; times the momentum scale) give the stratum's singular variable back, compared with the
file's `singular_var`: |read back / file - 1|, its median and maximum. The variable from the momenta:
  soft_x_*           x = min over gluons of 2 p_g.Q / s          (Q = p1 + p2)
  coll*_y_*, azimuth y = min over pairs with a gluon of s_ij / s
  threshold_beta_*   beta = sqrt(1 - 4 m^2 / s), m^2 = (p3^2 + p4^2) / 2
  sudakov_sqrts_*, bulk without gluons (ee_tt)   sqrt(s); bulk with gluons: y as for coll
The local frames are float64 when the network is (tests/test_kit_hardps.py checks every module); this reads the
input side, where float32 rounding happens first.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402


def _mdot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def singular_from_momenta(stratum, p, pdgs):
    """the stratum's singular variable from momenta p (N, n, 4) in the pool's leg order (e-, e+, finals)"""
    Q = p[:, 0] + p[:, 1]
    s = _mdot(Q, Q)
    g = [i for i, x in enumerate(pdgs) if int(x) == 21]
    if stratum.startswith("soft_x"):
        return np.min([2 * _mdot(p[:, i], Q) / s for i in g], axis=0)
    if stratum.startswith(("coll", "azimuth")) or (stratum == "bulk" and g):
        fin = range(2, p.shape[1])
        return np.min([2 * _mdot(p[:, i], p[:, j]) / s for i in fin for j in fin if i < j and (i in g or j in g)],
                      axis=0)
    if stratum.startswith("threshold_beta"):
        m2 = 0.5 * (_mdot(p[:, 2], p[:, 2]) + _mdot(p[:, 3], p[:, 3]))
        return np.sqrt(np.clip(1 - 4 * m2 / s, 0, None))
    return np.sqrt(s)


def build(name, test_rows, dtype, tmp, n):
    import torch
    import yaml
    import datagen
    from hydra import compose, initialize_config_dir
    from experiment import AmplitudeExperiment
    recipe = os.path.join(tmp, f"{name}.yaml")
    yaml.safe_dump({"processes": [{"name": name, "sqrts": [1000, 1000], "n_train": n, "n_val": n,
                                   "n_test": len(test_rows)}]}, open(recipe, "w"))
    pool = os.path.join(tmp, "stratum.npy")
    np.save(pool, test_rows)
    real = datagen.ensure_split_set
    datagen.ensure_split_set = lambda specs, role, seed, dest_dir=None, require_cache=False: (
        {name: pool} if role == "test" else real(specs, role=role, seed=seed, dest_dir=dest_dir,
                                                 require_cache=require_cache))
    ov = ["data.source=recipes", f"data.processes_file={recipe}", "model=lloca", f"run_dir={tmp}/run",
          "data.require_cache=true", "data.offshell_per_event=false", "data.internal_mass_scalars=false",
          "data.coupling_scalars=false", "model.use_diagrams=false", "training.sign_head=false",
          "data.eval_subsample=null", "training.num_workers=0", f"training.dtype={dtype}", "seed=42"]
    try:
        torch.set_default_dtype(torch.float64 if dtype == "float64" else torch.float32)
        with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
            cfg = compose("amplitudes", overrides=ov)
        exp = AmplitudeExperiment(cfg)
        exp._init(); exp.init_physics(); exp.init_geometric_algebra(); exp.init_data(); exp._init_dataloader()
    finally:
        datagen.ensure_split_set = real
    off = np.asarray(exp.offsets)
    idx = np.asarray(exp._split_indices["test"])
    P = test_rows.shape[1] // 5
    g = (off[idx, 0][:, None] + np.arange(P)[None, :]).reshape(-1)
    p = np.asarray(exp.particles_flat[g], dtype=np.float64).reshape(len(idx), P, 4) * float(exp.mom_div)
    perm = np.asarray(exp._role_perm["test"])[:len(idx)]
    return p, perm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--files", nargs="+", default=["ee_tt", "ee_uug", "ee_uugg"])
    ap.add_argument("--n", type=int, default=2000, help="events read per stratum (and per train/val role)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import h5py
    out = {}
    for stem in a.files:
        meta = kh.metadata(stem)
        target = sorted(meta["targets"])[0]
        name = f"kit_{stem}__{target}"
        for split in ("hard_test", "bulk_test"):
            for st in kh.strata(stem, split):
                rows = kh.stratum_rows(stem, target, split, st)[:a.n]
                with h5py.File(kh.h5_path(stem), "r") as f:
                    sv = f[f"{split}/{st}/singular_var"][:a.n]
                pdgs = rows[0, 4 * (rows.shape[1] // 5):-1]
                rec = {}
                # the file's own momenta against its singular_var: the definition check
                P = rows.shape[1] // 5
                fm = singular_from_momenta(st, rows[:, :4 * P].reshape(-1, P, 4), pdgs)
                rec["definition"] = float(np.max(np.abs(fm / sv - 1)))
                for dtype in ("float32", "float64"):
                    tmp = tempfile.mkdtemp(prefix="kitprec_", dir=os.environ.get("SCRATCH"))
                    p, perm = build(name, rows, dtype, tmp, min(a.n, 2000))
                    r = np.abs(singular_from_momenta(st, p, pdgs) / sv[perm] - 1)
                    rec[dtype] = {"median": float(np.median(r)), "max": float(np.max(r))}
                out[f"{stem}/{split}/{st}"] = rec
                print(f"{stem}/{split}/{st}: definition {rec['definition']:.1e}  float32 median "
                      f"{rec['float32']['median']:.1e} max {rec['float32']['max']:.1e}  float64 median "
                      f"{rec['float64']['median']:.1e} max {rec['float64']['max']:.1e}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
