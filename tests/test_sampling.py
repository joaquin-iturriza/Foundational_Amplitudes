#!/usr/bin/env python
"""Sampling-policy guard (cluster: needs the compiled ee_uu C++ driver).

Builds two small ee_uu pools through the real pipeline path, mixture vs legacy flat, and
checks what the divergence study asked of the sampler: the Z pole is covered by generation
(fraction of events within [88, 95] GeV far above flat RAMBO's ~0.4%), the kept events are
much flatter in log|M|^2, the bulk is still there, and a seed reproduces the pool exactly.
Run: python tests/test_sampling.py
"""
import os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg


def build(policy, seed, n=6000, lo=25.0, hi=1000.0):
    cfg = dict(mg.PROCESSES["ee_uu"]); cfg["sampling"] = policy
    sa = f"{mg.WORK_DIR}/ee_uu_standalone"
    backend, dirs, driver, eff = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and driver, "compiled ee_uu driver_v2 needed"
    out = os.path.join(tempfile.mkdtemp(), "pool.npy")
    mg.build_dataset_variable_energy(n, lo, hi, eff, backend, dirs, driver, cfg, out, rng=np.random.default_rng(seed))
    d = np.load(out); mom = d[:, :16].reshape(-1, 4, 4)
    pin = mom[:, 0] + mom[:, 1]; sq = np.sqrt(pin[:, 0] ** 2 - (pin[:, 1:] ** 2).sum(1))
    return d, sq, np.log(d[:, -1])


def flatness(logm, q=(0.02, 0.98), nb=25):
    lo, hi = np.quantile(logm, q); h = np.histogram(logm, bins=nb, range=(lo, hi))[0]
    return h.max() / max(h.min(), 1)


if __name__ == "__main__":
    flat, sq_f, lm_f = build({"mode": "flat"}, 1)
    mix, sq_m, lm_m = build({"mode": "mixture"}, 1)
    mix2, _, _ = build({"mode": "mixture"}, 1)
    pole = lambda sq: np.mean((sq > 88) & (sq < 95))
    bulk = lambda sq: np.mean(sq > 300)
    print(f"[sampling] Z-pole fraction [88,95]: flat {pole(sq_f):.3%}  mixture {pole(sq_m):.3%}")
    print(f"[sampling] log|M|^2 flatness (max/min bin over 2-98%): flat {flatness(lm_f):.1f}  mixture {flatness(lm_m):.1f}")
    print(f"[sampling] bulk share (sqrt s > 300): flat {bulk(sq_f):.2f}  mixture {bulk(sq_m):.2f}")
    assert pole(sq_m) > 8 * pole(sq_f), "the pole is not being covered"
    assert flatness(lm_m) < 0.5 * flatness(lm_f), "log|M|^2 not flatter"
    assert bulk(sq_m) > 0.4, "bulk starved"
    assert np.array_equal(mix, mix2), "not reproducible from the seed"
    assert np.all(np.isfinite(lm_m)) and mix.shape == flat.shape
    # every role samples the same distribution: a shaped entry shapes val/test like train,
    # with their own identity and path (a shaped pool never shares a path with a flat one)
    cfg = dict(mg.PROCESSES["ee_uu"]); cfg["sampling"] = {"mode": "mixture"}
    assert mg.sampling_policy(cfg, "val")["mode"] == "mixture" and mg.sampling_policy(cfg, "train")["mode"] == "mixture"
    assert mg.sampling_policy(dict(cfg, kind="virt"), "train")["mode"] == "flat"   # one-loop never shaped
    mg.PROCESSES["ee_uu"]["sampling"] = {"mode": "mixture"}
    rv, rt = mg.variable_energy_recipe("ee_uu", 25, 1000, 100, role="val", seed=1), mg.variable_energy_recipe("ee_uu", 25, 1000, 100, role="train", seed=1)
    del mg.PROCESSES["ee_uu"]["sampling"]
    assert "sampling" in rv and "sampling" in rt and "_smix" in mg.recipe_output_path(rt, ".") and "_smix" in mg.recipe_output_path(rv, ".")
    print("[sampling] ok")
