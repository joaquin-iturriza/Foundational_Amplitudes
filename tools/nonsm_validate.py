"""Validate the non-SM pools (D49; mg5_pipeline_final.NONSM, docs/results.tex sec:nonsm). CPU, on the site that holds
the pools (CC-IN2P3), after prebuild_recipes.py has built recipes/nonsm_all.yaml:

  1. every pool: finite, positive, its |M|^2 range;
  2. the SM limit: each model with its non-SM couplings at zero (zp0_*, smeft0_*), compiled here, against the SM
     standalone of the same process on the same momenta (the non-SM pools' validation momenta): the models' SM
     inputs were aligned with the sm card, so the ratio must be 1 to rounding;
  3. analytic |M|^2: e+e- -> mu+mu- through gamma, Z and the Z'-like Y1 (and the SM pool through gamma, Z alone,
     which pins the conventions), e+e- -> H+H- through gamma and Z, against the stored labels.

Prints one NONSM_VALIDATE JSON line (the write-up reads it from the job log).
    python tools/nonsm_validate.py [--n 2000]
"""
import argparse, json, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402

# recipes/nonsm_all.yaml's windows (the validation pools are read, not regenerated)
WINDOWS = {"zp_ee_mumu": (25, 1000), "zp_ee_uu": (25, 1000), "zp_ee_Y1a": (525, 1000),
           "thdm_ee_HpHm": (420, 1000), "thdm_ee_h2h3": (578, 1000),
           "smeft_ee_mumu": (25, 1000), "smeft_ee_ttbar": (363, 1000), "ee_mumu": (25, 1000)}
SM_LIMIT = [("zp0_ee_mumu", "ee_mumu", "zp_ee_mumu"), ("smeft0_ee_mumu", "ee_mumu", "smeft_ee_mumu"),
            ("smeft0_ee_ttbar", "ee_ttbar", "smeft_ee_ttbar")]


def pool(name, n_val=10000):
    import yaml
    spec = yaml.safe_load(open(os.path.join(ROOT, "recipes", "nonsm_all.yaml")))
    mg.PROCESSES[name]["sampling"] = dict(spec["sampling"])        # as the training recipe path attaches it
    lo, hi = WINDOWS[name]
    rec = mg.variable_energy_recipe(name, lo, hi, n_val, role="val", seed=42)
    path = mg.recipe_output_path(rec, datagen.frozen_dir())
    a = np.load(path)
    n_part = len(mg.PROCESSES[name]["pdg_ids"])
    assert a.shape[1] == 5 * n_part + 1, (path, a.shape)          # momenta, PDG ids, |M|^2
    return a[:, :4 * n_part].reshape(-1, n_part, 4), a[:, -1], path


def backend(name):
    sa = f"{mg.WORK_DIR}/{name}_standalone"
    be, dirs, drv, eff = mg.detect_compiled_backend(sa)
    if not dirs or drv is None:
        mg.generate_mg5_process(name, mg.PROCESSES[name])
        be, dirs, drv, eff = mg.compile_backends(sa, mg.PROCESSES[name]["nfinal"] + 2)
    else:
        mg.repatch_standalone_param_cards(sa, mg.PROCESSES[name])
    return drv, eff


def label(name, P):
    drv, eff = backend(name)
    cfg = mg.PROCESSES[name]
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    with mg.CppDriverPipe(drv, eff) as pipe:
        return pipe.compute([(p, None) for p in P], perm=perm, alphas=0.118)


def mdot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def ew():
    alpha, gf, mz = 1 / 132.507, 1.16639e-05, 91.188
    mw = math.sqrt(mz ** 2 / 2 + math.sqrt(mz ** 4 / 4 - math.pi * alpha * mz ** 2 / (math.sqrt(2) * gf)))
    sw2 = 1 - mw ** 2 / mz ** 2
    e = math.sqrt(4 * math.pi * alpha)
    return e, sw2, mz, 2.441404, e / math.sqrt(sw2 * (1 - sw2))


def ffbar_analytic(P, with_y1, eta=1.0):
    """<|M|^2> of e-(row 0) e+(row 1) -> mu+(row 2) mu-(row 3), massless, vector exchanges:
    (s^2/4)[(|A_LL|^2+|A_RR|^2)(1+c)^2 + (|A_LR|^2+|A_RL|^2)(1-c)^2], c the e- / mu- angle."""
    e, sw2, mz, wz, gz = ew()
    s = mdot(P[:, 0] + P[:, 1], P[:, 0] + P[:, 1])
    t = mdot(P[:, 0] - P[:, 3], P[:, 0] - P[:, 3])
    c = 1 + 2 * t / s
    gzl, gzr = gz * (-0.5 + sw2), gz * sw2                 # charged lepton: T3 = -1/2, Q = -1
    pz = 1 / (s - mz ** 2 + 1j * mz * wz)
    py = 1 / (s - mg.NONSM_POINT["MY1"] ** 2 + 1j * mg.NONSM_POINT["MY1"] * mg.NONSM_POINT["WY1"])
    gv, ga = 0.3, 0.1
    gy = {"L": gv - eta * ga, "R": gv + eta * ga}
    gzz = {"L": gzl, "R": gzr}
    A = {}
    for i in "LR":
        for j in "LR":
            A[i + j] = e ** 2 / s + gzz[i] * gzz[j] * pz + (gy[i] * gy[j] * py if with_y1 else 0)
    return s ** 2 / 4 * ((abs(A["LL"]) ** 2 + abs(A["RR"]) ** 2) * (1 + c) ** 2
                         + (abs(A["LR"]) ** 2 + abs(A["RL"]) ** 2) * (1 - c) ** 2)


def hphm_analytic(P):
    """<|M|^2> of e-(row 0) e+(row 1) -> H+(row 2) H-(row 3) through gamma and Z: (1/4) s^2 beta^2 sin^2(theta)
    (|A_L|^2 + |A_R|^2), A_h = e^2 Q_e Q_H / s + g_Z^e(h) g_Z^H / (s - M_Z^2 + i M_Z Gamma_Z), g_Z^H = g (1/2 - sw2)."""
    e, sw2, mz, wz, gz = ew()
    m = mg.NONSM_POINT["mhc"]
    s = mdot(P[:, 0] + P[:, 1], P[:, 0] + P[:, 1])
    t = mdot(P[:, 0] - P[:, 2], P[:, 0] - P[:, 2])
    b2 = 1 - 4 * m ** 2 / s
    bc = 1 - 2 * (m ** 2 - t) / s                          # beta cos(theta)
    b2s2 = b2 - bc ** 2
    pz = 1 / (s - mz ** 2 + 1j * mz * wz)
    gh = gz * (0.5 - sw2)
    AL = -e ** 2 / s + gz * (-0.5 + sw2) * gh * pz
    AR = -e ** 2 / s + gz * sw2 * gh * pz
    return 0.25 * s ** 2 * b2s2 * (abs(AL) ** 2 + abs(AR) ** 2)


def rel(a, b):
    r = np.abs(a / b - 1)
    return {"max": float(r.max()), "median": float(np.median(r))}


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
out = {"pools": {}, "sm_limit": {}, "analytic": {}}
def guarded(key, fn):
    """One check; a failure (a missing pool on this site) is recorded, not fatal to the others."""
    try:
        fn()
    except Exception as e:
        out.setdefault("errors", {})[key] = repr(e)[:300]


def pool_stats(name):
    P, y, path = pool(name)
    out["pools"][name] = {"path": path, "n": int(len(y)), "finite": bool(np.isfinite(y).all()),
                          "n_nonpositive": int((y <= 0).sum()),
                          "log10_range": [float(np.log10(y[y > 0]).min()), float(np.log10(y[y > 0]).max())]}


def sm_limit(zero, sm, bsm):
    P = pool(bsm)[0][: a.n]
    out["sm_limit"][f"{zero}/{sm}"] = rel(label(zero, P), label(sm, P))
    out["sm_limit"][f"{bsm}/{sm}"] = rel(label(bsm, P), label(sm, P))     # how far the point is from the SM


def analytic_ee_mumu():
    P, y, _ = pool("ee_mumu")
    out["analytic"]["ee_mumu (gamma, Z)"] = rel(ffbar_analytic(P, False), y)


def analytic_zp():
    P, y, _ = pool("zp_ee_mumu")
    out["analytic"]["zp_ee_mumu (gamma, Z, Y1), gA sign +"] = rel(ffbar_analytic(P, True, +1), y)
    out["analytic"]["zp_ee_mumu (gamma, Z, Y1), gA sign -"] = rel(ffbar_analytic(P, True, -1), y)
    out["analytic"]["zp_ee_mumu against gamma, Z only"] = rel(ffbar_analytic(P, False), y)


def analytic_hphm():
    P, y, _ = pool("thdm_ee_HpHm")
    out["analytic"]["thdm_ee_HpHm (gamma, Z)"] = rel(hphm_analytic(P), y)


for name in WINDOWS:
    guarded(f"pool {name}", lambda: pool_stats(name))
for zero, sm, bsm in SM_LIMIT:
    guarded(f"sm_limit {zero}", lambda: sm_limit(zero, sm, bsm))
guarded("analytic ee_mumu", analytic_ee_mumu)
guarded("analytic zp", analytic_zp)
guarded("analytic hphm", analytic_hphm)
print("NONSM_VALIDATE " + json.dumps(out), flush=True)
