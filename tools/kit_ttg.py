#!/usr/bin/env python3
"""e+ e- -> t t~ g in the KIT hard phase-space layout (kit_benchmark step 39, D600 (b)): one HDF5 file readable by
tools/kit_hardps.py as dataset kit_fa_ee_ttg__me2, at KIT's u u~ g settings (sqrt s 1 TeV, alpha_s 0.118, 1/alpha
132.507, MadGraph's sm card; m_t from KIT's ee_tt file), exact tree |M|^2 by the process's MadGraph standalone.

    python tools/kit_ttg.py build [--out $DATA_DIR/kit_like/fa_ee_ttg.h5] [--seed 0]    (a CPU job on CC)
    python tools/kit_ttg.py selftest                                                    (kinematics only, local)

Layout (as KIT's): <split>/<stratum>/{momenta (N, 5, 4) float64 (E, px, py, pz) GeV, CM frame, order e+ e- t t~ g;
me2; log10_me2; singular_var; other_var}, root attribute `metadata` (JSON).
  bulk_train 2e5, bulk_test 2e5   flat three-body phase space (uniform in the Dalitz plane, uniform orientation) with
                                  KIT's u u~ g bulk cut applied to the singular pairs: s_tg/s > 0.01 and s_t~g/s > 0.01,
                                  s_ij = 2 p_i.p_j
  soft_x_<lo>_<hi>                gluon x = 2E_g/sqrt s log-uniform per decade 1e-7..1e-2, top direction uniform in the
                                  t t~ rest frame (other_var: the gluon-top opening angle's 1 - cos)
  coll_tg_<lo>_<hi>               the gluon at angle theta to the top, 1 - cos theta log-uniform per decade 1e-7..1e-1, at x
                                  uniform in [0.2, 0.4] (other_var: x): the quasi-collinear limit, regulated by m_t (dead
                                  cone), so |M|^2 is expected flat below theta ~ m_t/E_t
  threshold_beta_<lo>_<hi>        the t t~ pair's velocity beta = sqrt(1 - 4 m_t^2/s_tt) log-uniform per decade 1e-4..1e-1,
                                  top direction uniform in the pair's rest frame (other_var: x = 1 - s_tt/s)
  hard_test (1e5 a stratum) and hard_train (5000) hold the same strata, independent seeds; stratum names as KIT's
  (f'{lo:g}_{hi:g}': soft_x_1e-07_1e-06, soft_x_0.0001_0.001); bulk singular_var min(s_tg, s_t~g)/s (KIT: min_y).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STEM = "fa_ee_ttg"
SQRTS, ALPHA_S, ALPHA_INV = 1000.0, 0.118, 132.507
GEN = "e+ e- > t t~ g"
PDGS = [-11, 11, 6, -6, 21]
N_BULK = {"bulk_train": 200000, "bulk_test": 200000}
N_HARD = {"hard_train": 5000, "hard_test": 100000}   # KIT's N_HARD_TRAIN, N_HARD_TEST (generate.py)
# KIT's particle table (generate.py PARTICLES: Q, spin, mass, T3 of the LH state, B, L, colour dim, C2); the top mass is
# set from KIT's ee_tt file at build
PROPS = {-11: dict(Q=1, spin=0.5, T3=0.5, B=0, L=-1, color=1, C2=0), 11: dict(Q=-1, spin=0.5, T3=-0.5, B=0, L=1, color=1, C2=0),
         6: dict(Q=2 / 3, spin=0.5, T3=0.5, B=1 / 3, L=0, color=3, C2=4 / 3),
         -6: dict(Q=-2 / 3, spin=0.5, T3=-0.5, B=-1 / 3, L=0, color=3, C2=4 / 3),
         21: dict(Q=0, spin=1.0, T3=0, B=0, L=0, color=8, C2=3)}
NAMES = ["e+", "e-", "t", "t~", "g"]
YCUT = 0.01
FAMS = {"soft_x": [(10.0 ** k, 10.0 ** (k + 1)) for k in range(-7, -2)],
        "coll_tg": [(10.0 ** k, 10.0 ** (k + 1)) for k in range(-7, -1)],
        "threshold_beta": [(10.0 ** k, 10.0 ** (k + 1)) for k in range(-4, -1)]}
X_COLL = (0.2, 0.4)


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def _rotations(rng, n):
    """n uniform random rotations (unit quaternions)"""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _assemble(t, tb, g, rng):
    """finals (N, 4) each, rotated uniformly, with the beams along z: (N, 5, 4) in KIT's order"""
    n = len(t)
    R = _rotations(rng, n)
    out = np.zeros((n, 5, 4))
    out[:, 0] = [SQRTS / 2, 0, 0, -SQRTS / 2]     # e+ (KIT: e+ first, along -z as the e- goes +z)
    out[:, 1] = [SQRTS / 2, 0, 0, SQRTS / 2]
    for k, v in ((2, t), (3, tb), (4, g)):
        out[:, k, 0] = v[:, 0]
        out[:, k, 1:] = np.einsum("nij,nj->ni", R, v[:, 1:])
    return out


def from_pair_frame(x, cth, phi, m, rng):
    """gluon x along +z; the t t~ pair recoiling, the top at (cth, phi) in the pair's rest frame"""
    Eg = x * SQRTS / 2
    stt = SQRTS ** 2 * (1 - x)
    ps = np.sqrt(np.clip(stt / 4 - m * m, 0, None))
    Es = np.sqrt(stt) / 2
    sth = np.sqrt(1 - cth ** 2)
    pt = np.stack([Es, ps * sth * np.cos(phi), ps * sth * np.sin(phi), ps * cth], -1)
    pb = pt * np.array([1, -1, -1, -1])
    Ett = SQRTS - Eg
    gam, bg = Ett / np.sqrt(stt), -Eg / np.sqrt(stt)          # boost of the pair along -z
    def boost(p):
        q = p.copy()
        q[:, 0] = gam * p[:, 0] + bg * p[:, 3]
        q[:, 3] = bg * p[:, 0] + gam * p[:, 3]
        return q
    g = np.stack([Eg, 0 * Eg, 0 * Eg, Eg], -1)
    return _assemble(boost(pt), boost(pb), g, rng)


def from_top_angle(x, c, phi, m, rng):
    """gluon x along +z, the top at lab angle acos(c) to it (azimuth phi), its momentum solved from energy conservation
    (unique for c > 0)"""
    Eg = x * SQRTS / 2
    E = SQRTS - Eg
    lo, hi = np.zeros_like(x), np.full_like(x, SQRTS)
    f = lambda p: np.sqrt(m * m + p * p) + np.sqrt(m * m + Eg ** 2 + p * p + 2 * Eg * p * c) - E
    for _ in range(200):
        mid = (lo + hi) / 2
        up = f(mid) > 0
        hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
    p = (lo + hi) / 2
    s = np.sqrt(np.clip(1 - c * c, 0, None))
    t = np.stack([np.sqrt(m * m + p * p), p * s * np.cos(phi), p * s * np.sin(phi), p * c], -1)
    g = np.stack([Eg, 0 * Eg, 0 * Eg, Eg], -1)
    tb = np.stack([SQRTS - Eg - t[:, 0]], -1)
    tb = np.concatenate([tb, -(t[:, 1:] + g[:, 1:])], -1)
    return _assemble(t, tb, g, rng)


def bulk(n, m, rng):
    """flat three-body phase space: (x_t, x_tb) uniform over the Dalitz region, uniform orientation; KIT's bulk cut"""
    out = []
    while sum(len(o) for o in out) < n:
        k = 4 * n
        Et, Eb = rng.uniform(m, SQRTS / 2, (2, k))
        Eg = SQRTS - Et - Eb
        pt, pb = np.sqrt(Et ** 2 - m * m), np.sqrt(Eb ** 2 - m * m)
        ok = Eg > 0
        c = np.where(ok, (pb ** 2 - Eg ** 2 - pt ** 2) / (2 * np.maximum(Eg, 1e-300) * pt), 2.0)   # top-gluon angle
        ok &= np.abs(c) <= 1
        Et, Eg, c, pt = Et[ok], Eg[ok], c[ok], pt[ok]
        phi, s = rng.uniform(0, 2 * np.pi, len(c)), np.sqrt(1 - c * c)
        t = np.stack([Et, pt * s * np.cos(phi), pt * s * np.sin(phi), pt * c], -1)
        g = np.stack([Eg, 0 * Eg, 0 * Eg, Eg], -1)
        tb = np.concatenate([(SQRTS - Eg - Et)[:, None], -(t[:, 1:] + g[:, 1:])], -1)
        ev = _assemble(t, tb, g, rng)
        out.append(ev[cut(ev)])
    return np.concatenate(out)[:n]


def cut(ev):
    s = SQRTS ** 2
    return (2 * dot(ev[:, 2], ev[:, 4]) / s > YCUT) & (2 * dot(ev[:, 3], ev[:, 4]) / s > YCUT)


def stratum(fam, lo, hi, n, m, rng):
    """(momenta, singular_var, other_var) of one hard stratum"""
    u = 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    if fam == "soft_x":
        ev = from_pair_frame(u, rng.uniform(-1, 1, n), phi, m, rng)
        other = 1 - _cos(ev[:, 2], ev[:, 4])
    elif fam == "coll_tg":
        x = rng.uniform(*X_COLL, n)
        ev = from_top_angle(x, 1 - u, phi, m, rng)
        other = x
    else:   # threshold: beta of the pair -> s_tt -> x
        stt = 4 * m * m / (1 - u ** 2)
        x = 1 - stt / SQRTS ** 2
        ev = from_pair_frame(x, rng.uniform(-1, 1, n), phi, m, rng)
        other = x
    return ev, u, other


def _cos(a, b):
    return (a[:, 1:] * b[:, 1:]).sum(-1) / (np.linalg.norm(a[:, 1:], axis=1) * np.linalg.norm(b[:, 1:], axis=1))


def variables(ev, m):
    """the singular variables recomputed from momenta: x, 1 - cos(t, g), beta of the pair"""
    s = SQRTS ** 2
    x = 2 * ev[:, 4, 0] / SQRTS
    stt = dot(ev[:, 2] + ev[:, 3], ev[:, 2] + ev[:, 3])
    return {"x": x, "ctg": 1 - _cos(ev[:, 2], ev[:, 4]), "beta": np.sqrt(np.clip(1 - 4 * m * m / stt, 0, None)),
            "ytg": 2 * dot(ev[:, 2], ev[:, 4]) / s, "ybg": 2 * dot(ev[:, 3], ev[:, 4]) / s}


def check(ev, m, tol=1e-6):
    """momentum conservation, on-shell legs, the massless gluon"""
    tot = ev[:, 2:].sum(1) - ev[:, :2].sum(1)
    assert np.abs(tot).max() < tol * SQRTS, np.abs(tot).max()
    for k, mm in ((2, m), (3, m), (4, 0.0)):
        assert np.abs(np.sqrt(np.abs(dot(ev[:, k], ev[:, k]))) - mm).max() < 1e-3, (k, mm)


def selftest(m=172.5):
    rng = np.random.default_rng(0)
    b = bulk(2000, m, rng)
    check(b, m)
    assert cut(b).all()
    for fam, bins in FAMS.items():
        for lo, hi in bins:
            ev, u, _ = stratum(fam, lo, hi, 500, m, rng)
            check(ev, m)
            v = variables(ev, m)
            got = {"soft_x": v["x"], "coll_tg": v["ctg"], "threshold_beta": v["beta"]}[fam]
            r = np.abs(got / u - 1).max()
            assert r < (1e-6 if fam != "coll_tg" else 1e-3), (fam, lo, r)
    print("SELFTEST OK")


def build(out, seed):
    import h5py
    import mg5_pipeline_final as mg
    from tools import kit_hardps as kh
    from tools.kit_mc_madgraph import standalone_exact
    tt = kh.metadata("ee_tt")
    m = float([p["mass"] for p in tt["particles"] if p["pdg"] == 6][0])
    uug = kh.metadata("ee_uug")
    assert uug["sqrts"] == SQRTS and uug["alpha_s"] == ALPHA_S and uug["alpha_inv"] == ALPHA_INV, uug
    work = os.path.join(os.environ.get("TMPDIR") or os.path.dirname(out), "mg_fa_ee_ttg")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sa = os.path.join(work, "sa")
    mg._mg5_script(f"import model sm\ngenerate {GEN}\noutput standalone_cpp {sa}\nquit\n", out + ".mg5.log")
    mg.patch_param_card_slha(os.path.join(sa, "Cards", "param_card.dat"), {"MT": m, "ymt": m})
    mg.compile_backends(sa, 5)
    backend, _, drv, _ = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and drv, f"no C++ driver in {sa}"
    pipe, exact = standalone_exact(drv, sa, PDGS, GEN, ALPHA_S)
    # the standalone at KIT's settings reproduces KIT's u u~ g |M|^2 elsewhere (kit_mc_madgraph CHECK, 1e-6); here the
    # check is the soft limit: |M|^2 x^2 flat in the deepest two soft strata
    rng = np.random.default_rng(seed)
    meta = {"process": "e+ e- -> t t~ g", "sqrts": SQRTS, "alpha_s": ALPHA_S, "alpha_inv": ALPHA_INV, "generator": "tools/kit_ttg.py",
            "bulk": f"flat three-body phase space, s_ij/s > {YCUT} for every singular pair (t g, t~ g)",
            "particles": [dict(pdg=p, **PROPS[p], mass=(m if abs(p) == 6 else 0.0)) for p in PDGS],
            "particle_names": NAMES,
            "singular_var": "soft: x=2E_g/sqrt(s); coll: 1-cos(t,g); threshold: beta of the t t~ pair; bulk: min(s_tg, s_t~g)/s",
            "targets": {"me2": {"n_loops": 0, "alpha_s_power": 1, "alpha_power": 2}}, "m_t": m, "seed": seed}
    with h5py.File(out + ".part", "w") as F:
        F.attrs["metadata"] = json.dumps(meta)
        def put(split, name, ev, sv=None, ov=None):
            me2 = np.concatenate([exact(ev[i:i + 20000]) for i in range(0, len(ev), 20000)])
            assert np.isfinite(me2).all() and (me2 > 0).all(), (split, name)
            g = F.create_group(f"{split}/{name}")
            g["momenta"], g["me2"], g["log10_me2"] = ev, me2, np.log10(me2)
            if sv is not None:
                g["singular_var"], g["other_var"] = sv, ov
            print(f"{split}/{name}: {len(ev)} events, median log10 |M|^2 {np.median(np.log10(me2)):.3f}", flush=True)
            return me2
        for split, n in N_BULK.items():
            ev = bulk(n, m, rng)
            check(ev, m)
            v = variables(ev, m)
            put(split, "bulk", ev, np.minimum(v["ytg"], v["ybg"]), v["x"])
        for split in ("hard_train", "hard_test"):
            for fam, bins in FAMS.items():
                for lo, hi in bins:
                    ev, u, o = stratum(fam, lo, hi, N_HARD[split], m, rng)
                    check(ev, m)
                    me2 = put(split, f"{fam}_{lo:g}_{hi:g}", ev, u, o)   # KIT's names
                    if fam == "soft_x" and lo < 1e-6:
                        sl = np.polyfit(np.log10(u), np.log10(me2), 1)[0]
                        print(f"  soft slope d log|M|^2 / d log x = {sl:.4f} (expected -2)", flush=True)
                        assert abs(sl + 2) < 0.02, sl
    pipe.__exit__(None, None, None)
    os.replace(out + ".part", out)
    print(f"KIT_TTG_DONE {out}")


def main():   # dup-ok: argparse boilerplate
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=None)
    b.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        import siteconf
        build(a.out or os.path.join(siteconf.DATA_DIR, "kit_like", f"{STEM}.h5"), a.seed)


if __name__ == "__main__":
    main()
