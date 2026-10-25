"""The CPU check of data.target_physics_hybrid (experiment._physics_logF, _physics_hybrid_cp; D129): on a few catalog_v3
pools of every particle count, through the experiment's own functions,
  cp_val    the constant on the validation pool's first 2000 events, against prep_level's c_p on the same events
            (analysis/catalog_v3/data/prep_level.json): the same number up to rounding
  cp_train  the constant the run uses (the first 10000 train events, a held-out probe's too), its distance to cp_val
            beside the sampling error c_p_sd / sqrt(2000)
  shape     the standard deviation over events of (hybrid target - s_hat target): zero, the shape is the s_hat target's
  level     the hybrid target's class mean minus the emission-scale target's (data.target_physics_dim_ref=qperp), both
            on the validation events with cp_val: zero; with cp_train it is cp_train - cp_val
The targets as the dimensionless arm builds them (dim_only, s0 = 1), the coupling orders off: they enter every target
alike and need the per-event alpha_s.
    site run ccin2p3 FA -- python analysis/catalog_v3/hybrid_check.py   (CPU, login node; imports experiment)
"""
import glob, json, os, sys
from types import SimpleNamespace

import numpy as np
import yaml
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from experiment import AmplitudeExperiment as E  # noqa: E402

PROBES = {"train": ["v3_bbx_epem", "v3_bbx_epemg", "v3_bbx_ccxh", "v3_bbx_epemgg", "v3_bbx_epemgz"],
          "H2": ["v3_bbx_aagg", "v3_bbx_accxz"], "H2p5": ["v3_epem_ccxgvevex", "v3_dg_cxghsu"],
          "H7E": ["v3_bg_bccxg__E10"]}


def sqrts(setname, name):
    f = "catalog_v3_train.yaml" if setname == "train" else f"catalog_v3_heldout_{setname}.yaml"
    return next(p.get("sqrts") for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]
                if p["name"] == name)


def load(name, sq, role, n):
    import datagen
    d = datagen.dest_for_role(role)                         # val and test frozen in $WORK/datasets, train cached on $SCRATCH
    g = sorted(glob.glob(os.path.join(d, f"{name}_{sq[0]}-{sq[1]}GeV_{role}_*amplitudes.npy"))) if sq else []
    g = g or sorted(glob.glob(os.path.join(d, f"{name}_*_{role}_*amplitudes.npy")))
    x = np.asarray(np.load(g[0], mmap_mode="r")[:n], dtype=np.float64)
    N = (x.shape[1] - 1) // 5
    return x[:, : 4 * N].reshape(len(x), N, 4), x[:, -1]


def stub(ref, hybrid, cp=None):
    cfg = OmegaConf.create({"data": {"target_physics_factor": True, "target_physics_dim_only": True,
                                     "target_physics_s0": 1.0, "target_physics_dim_ref": ref,
                                     "target_physics_hybrid": hybrid}})
    s = SimpleNamespace(cfg=cfg, _phys_cp=cp or {})
    s._eikonal_logS = lambda *a: (0, 0.0)
    return s


def main():
    pl = json.load(open(os.path.join(HERE, "data", "prep_level.json")))["classes"]
    out = {}
    for setname, names in PROBES.items():
        for name in names:
            sq = sqrts(setname, name)
            Pv, av = load(name, sq, "val", 2000)
            Pt, _ = load(name, sq, "train", 10000)
            cpv, cpt = E._physics_hybrid_cp(Pv), E._physics_hybrid_cp(Pt)
            y = np.log(av)
            tgt = lambda s: y + E._physics_logF(s, name, Pv)
            t_shat, t_q = tgt(stub("shat", False)), tgt(stub("qperp", False))
            t_hv, t_ht = tgt(stub("shat", True, {name: cpv})), tgt(stub("shat", True, {name: cpt}))
            ref = pl[setname][name]
            r = {"N": int(Pv.shape[1]), "cp_val": cpv, "cp_prep_level": ref["c_p"], "cp_val_minus_prep": cpv - ref["c_p"],
                 "cp_train": cpt, "cp_train_minus_val": cpt - cpv, "sampling_err": ref["c_p_sd"] / np.sqrt(2000),
                 "shape_sd": float(np.std(t_hv - t_shat)), "offset": float(np.mean(t_hv - t_shat)),
                 "level_val": float(t_hv.mean() - t_q.mean()), "level_train_cp": float(t_ht.mean() - t_q.mean())}
            out[name] = r
            print("HYBRID_CHECK", setname, name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                                              for k, v in r.items()}), flush=True)
    worst = {k: max(abs(r[k]) for r in out.values()) for k in ("cp_val_minus_prep", "shape_sd", "level_val")}
    print("HYBRID_CHECK_WORST", json.dumps(worst))
    print("HYBRID_CHECK_TRAIN_VS_VAL", json.dumps({n: [round(r["cp_train_minus_val"], 4), round(r["sampling_err"], 4)]
                                                   for n, r in out.items()}))


if __name__ == "__main__":
    main()
