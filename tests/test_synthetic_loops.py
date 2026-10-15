#!/usr/bin/env python
"""CPU validation of the synthetic one-loop amplitudes (tools/synthetic_loops.py; D44).

  (a) special functions: Li2 against mpmath on the complex plane and on both sides of its cut, the bubble's finite
      part against its Feynman-parameter integral (mpmath; skipped where mpmath is missing);
  (b) the born: the loop structure's |M0|^2 equals the version-1 tree's evaluate() to rounding, and the loop name's
      structure keeps the tree's legs, diagrams and numerators;
  (c) Lorentz invariance of V under random rotations x boosts, the mass dimension 8 - 2N, identical-leg swaps;
  (d) reproducibility: structure hashes identical across interpreters and distinct from each other and the borns;
      a 3000-event mixture chunk bit-identical from the same seed;
  (e) the recipe's 19 loop processes: finite on every event, the fraction of negative events, through the pipeline.

    python tests/test_synthetic_loops.py          (a few minutes, one core)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg                  # noqa: E402
from tools import synthetic_amplitudes as syn    # noqa: E402
from tools import synthetic_loops as sl          # noqa: E402

LOOPS = [f"synl_{k:05d}" for k in (0, 6, 8, 10, 11, 16, 1, 4, 9, 12, 13, 15, 18)] + [f"synl2_{k:05d}" for k in range(6)]


def _events(name, n, rng, democratic=0.5):
    cfg = mg.register_synthetic(name)
    lo, hi = cfg["sqrts"]
    ev, _ = mg._candidates(n, lambda k: rng.uniform(lo, hi, k), cfg["m_finals"], cfg["pdg_ids"], rng,
                           mg.FIDUCIAL_CUTS, democratic, 1e-6)
    return np.stack([e[0] for e in ev])


def _lorentz(rng, B, max_rapidity=2.0):
    L = np.zeros((B, 4, 4))
    for b in range(B):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        nn = rng.normal(size=3); nn /= np.linalg.norm(nn)
        y = rng.uniform(0, max_rapidity); g, bg = np.cosh(y), np.sinh(y)
        K = np.eye(4); K[0, 0] = g; K[0, 1:] = K[1:, 0] = bg * nn
        K[1:, 1:] += (g - 1) * np.outer(nn, nn)
        Rot = np.eye(4); Rot[1:, 1:] = Q
        L[b] = K @ Rot
    return L


def check_functions(rng):
    try:
        import mpmath
    except ImportError:
        print("(a) mpmath missing: special functions not checked here")
        return
    z = np.concatenate([rng.normal(size=2000) * 3 + 1j * rng.normal(size=2000) * 3, rng.uniform(-20, 1, 500) + 0j,
                        rng.uniform(1, 30, 300) + 1e-14j, rng.uniform(1, 30, 300) - 1e-14j,
                        np.exp(1j * rng.uniform(0, 2 * np.pi, 300))])
    ref = np.array([complex(mpmath.polylog(2, mpmath.mpc(x.real, x.imag))) for x in z])
    e_li2 = float(np.max(np.abs(sl.li2(z) - ref) / np.maximum(1, np.abs(ref))))
    s, e_b = 1e5, 0.0
    for m in (4.7, 80.419, 172.5):
        for p2 in (-3e5, -1e3, 0.0, 10.0, 3.9 * m * m, 4.004 * m * m, 4.5 * m * m, 1e6):
            pts = [0, 0.5, 1] + ([(1 - np.sqrt(1 - 4 * m * m / p2)) / 2, (1 + np.sqrt(1 - 4 * m * m / p2)) / 2]
                                 if p2 > 4 * m * m else [])
            ref_b = -mpmath.quad(lambda x: mpmath.log(mpmath.mpc(m * m - x * (1 - x) * p2, -1e-30) / s), sorted(pts))
            e_b = max(e_b, abs(sl.bubble(np.array([p2]), m, s)[0] - complex(ref_b)))
    print(f"(a) Li2 against mpmath, {len(z)} points (both sides of the cut): max error {e_li2:.1e}; "
          f"bubble against its Feynman-parameter integral, 24 points: max error {e_b:.1e}")
    assert e_li2 < 1e-13 and e_b < 1e-10, (e_li2, e_b)


def check_born(rng, n=300):
    worst = 0.0
    for nm in LOOPS:
        st, bt = syn.structure(nm), syn.structure(sl.born_name(nm))
        for k in ("pdg_ids", "m_legs", "diagrams", "numerators", "perms", "coupling", "k_s", "sqrts"):
            assert st[k] == bt[k], (nm, k)
        P = _events(nm, n, rng)
        _, born, _ = sl.evaluate_loop(st, P, parts=True)
        worst = max(worst, float(np.max(np.abs(born / syn.evaluate(bt, P) - 1))))
    print(f"(b) born: {len(LOOPS)} loop structures keep their tree's legs, diagrams and numerators; "
          f"|M0|^2 against the tree's evaluate(): max relative deviation {worst:.1e}")
    assert worst < 1e-12, worst


def check_invariance(rng, n=300):
    wl = wd = wp = 0.0
    for nm in LOOPS:
        st = syn.structure(nm)
        P = _events(nm, n, rng)
        a = syn.evaluate(st, P)
        P2 = np.einsum("bij,bkj->bki", _lorentz(rng, n), P)
        rel = lambda b: np.abs(b - a) / np.maximum(np.abs(a), 1e-3 * np.median(np.abs(a)))
        wl = max(wl, float(np.max(rel(syn.evaluate(st, P2)))))
        N = P.shape[1]
        for lam in (0.1, 0.37, 2.9, 10.0):
            wd = max(wd, float(np.max(rel(syn.evaluate(st, lam * P, scale=lam) / lam ** (8 - 2 * N)))))
        for perm in st["perms"][1:]:
            wp = max(wp, float(np.max(rel(syn.evaluate(st, P[:, perm, :])))))
    print(f"(c) Lorentz invariance (rapidity <= 2): max |dV|/|V| {wl:.1e}; mass dimension 8-2N "
          f"(lambda 0.1-10): {wd:.1e}; identical-leg swaps: {wp:.1e} "
          f"(|V| floored at 1e-3 of the process's median, near the zeros of V)")
    assert wl < 1e-8 and wd < 1e-8 and wp < 1e-10, (wl, wd, wp)


def check_reproducible():
    code = ("import sys, json; sys.path.insert(0, %r); from tools import synthetic_amplitudes as s; "
            "print(json.dumps([s.structure(n)['sha'] for n in %r]))" % (ROOT, LOOPS))
    shas = []
    for seed in ("0", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env=dict(os.environ, PYTHONHASHSEED=seed)).stdout
        shas.append(json.loads(out.strip().splitlines()[-1]))
    here = [syn.structure(n)["sha"] for n in LOOPS]
    borns = [syn.structure(sl.born_name(n))["sha"] for n in LOOPS]
    assert shas[0] == shas[1] == here, "structure depends on the interpreter"
    assert len(set(here)) == len(here) and not set(here) & set(borns), "two names drew the same structure"
    cfg = dict(mg.register_synthetic(LOOPS[0])); cfg["sampling"] = {"mode": "mixture"}
    arrs = []
    with tempfile.TemporaryDirectory() as d:
        for k in range(2):
            out = os.path.join(d, f"p{k}.npy")
            mg.build_dataset_variable_energy(3000, *cfg["sqrts"], None, "synthetic", ["synthetic"], None,
                                             cfg, out, rng=np.random.default_rng(7), role="train")
            arrs.append(np.load(out))
    assert np.array_equal(arrs[0], arrs[1]), "pool not reproducible from the seed"
    print(f"(d) reproducibility: {len(LOOPS)} structure hashes identical across interpreters, distinct from each "
          f"other and from their borns; a 3000-event mixture chunk is bit-identical; "
          f"negative events in it: {(arrs[0][:, -1] < 0).mean():.2f}")


def check_pools(rng, n=4000):
    print("(e) the recipe's loop processes (mixture-like candidates, half IR-democratic):")
    for nm in LOOPS:
        st = syn.structure(nm)
        P = _events(nm, n, rng)
        V, born, Vu = sl.evaluate_loop(st, P, parts=True)
        assert np.all(np.isfinite(V)) and np.all(born > 0), nm
        r = V / born
        print(f"    {nm:12s} {st['loop']['type']:3s} 2->{st['nfinal']}  negative {np.mean(V < 0):.2f}  "
              f"V/B median {np.median(r):+8.2f}  10-90% [{np.quantile(r, 0.1):+8.2f}, {np.quantile(r, 0.9):+8.2f}]  "
              f"factorising share of V (median |V_U|/|V|) {np.median(np.abs(Vu) / np.abs(V)):.2f}")


if __name__ == "__main__":
    rng = np.random.default_rng(20261015)
    check_functions(rng)
    check_born(rng)
    check_invariance(rng)
    check_reproducible()
    check_pools(rng)
    print("all checks passed")
