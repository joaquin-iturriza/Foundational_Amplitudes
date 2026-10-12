#!/usr/bin/env python
"""CPU validation of synthetic-amplitude versions 2-6 (tools/synthetic_amplitudes.py; D36, D38, D40), next to
tests/test_synthetic_amplitudes.py (version 1's checks and its pool path). CPU only, a few tens of minutes:

    python tests/test_synthetic_versions.py [--n-names 12] [--quick]

  (0) version 1 and revision 1 of versions 2-5 frozen: the structure hashes of the names behind every pool in use
      (recipes) and of the first drawn names are the recorded ones, and the vectorised evaluator reproduces
      version 1's own on version 1;
  (a) every version and revision: |M|^2 finite and > 0 on fiducial events (half from the IR-democratic splitter);
  (b) Lorentz invariance (random rotation x boost, rapidity <= 2);
  (c) mass dimension 8 - 2N;
  (d) symmetry under the identical-particle permutations, applied to the momenta (not through the average);
  (e) reproducibility: same structure in fresh interpreters (PYTHONHASHSEED 0 / 12345), names distinct;
  (f) v3+: every drawn tree is SM-legal, checked vertex by vertex from its merges, and the legs conserve
      quark generation and lepton family; revision 2: the kept labelled diagrams are the orbit sizes' sum and,
      unless subsampled, every legal diagram;
  (g) v4+: the collinear limit, |M|^2 ~ 1/s_ik as s_ik -> 0 for the pairs carrying a bracket (and, for
      comparison, the v1-v3 behaviour of the same kind of pair), and the soft limit of a gauge leg,
      |M|^2 ~ 1/E_k^2, both on physical (momentum-conserving) sequences;
  (h) the diagram sidecars build and describe every diagram image;
  (i) v6: little-group scaling, every helicity amplitude A_lambda goes to prod_i t_i^(-2 h_i) A_lambda when
      lambda_i -> t_i lambda_i, lambda~_i -> lambda~_i / t_i (complex t_i, momenta unchanged);
  (j) v6: the selection rules, recomputed here from the trees (helicity conserved along massless fermion lines,
      left-handed on W lines and neutrinos, the all-massless counting rule): the surviving configurations are
      exactly those, every one non-zero, and |M|^2 = (1/4) sum over them of (|A(p)|^2 + |A(p~)|^2)/2 with p~
      the spatially inverted momenta, so every other configuration contributes exactly zero; |M|^2 is parity even.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg          # noqa: E402
from tools import synthetic_amplitudes as syn   # noqa: E402

# structure hashes of every v1 name in a recipe (synthetic_pilot, transfer_synthetic: syn_00000..19;
# transfer_finale_synhm: syn5_/syn6_00000..03), recorded before versions 2-5 were added (they match the sidecars
# the v1 pools were built with)
V1_SHA = {
    "syn_00000": "a486a9c1e5f83718", "syn_00001": "6522cc2973422cd2", "syn_00002": "6edc21d181ff0bfb", "syn_00003": "4eff966970921360",
    "syn_00004": "7f1a57abcb8e038b", "syn_00005": "200eb20574990877", "syn_00006": "b8fd666cbdb5d06f", "syn_00007": "d6b3099be942c63c",
    "syn_00008": "12d74b499db7ca54", "syn_00009": "10711c0f660308b4", "syn_00010": "69c21c56e73ac99d", "syn_00011": "0972be2f13c10288",
    "syn_00012": "f3e5fe89ea87b925", "syn_00013": "885d5d0c7f0567d7", "syn_00014": "a93c7da97fd98645", "syn_00015": "bb551701d39a0016",
    "syn_00016": "0bda7fac5e70a332", "syn_00017": "ee9057e94598e273", "syn_00018": "aaf8c4f7ec7a0c17", "syn_00019": "7c1f26b1ee12621f",
    "syn5_00000": "0019597e83810086", "syn5_00001": "3d9cc4859b7c3202", "syn5_00002": "b8f935f4aceb3f46", "syn5_00003": "3b5a0c3cde5f9aa8",
    "syn6_00000": "f7fe4b7bcb16f324", "syn6_00001": "9fa8804f6822e209", "syn6_00002": "10b2ea0c70cb639e", "syn6_00003": "6d42a68b1e286ee8",
}

# structure hashes of revision 1 of versions 2-5 (syn[<n>]v<V>_<k>), recorded 2026-10-12 before revision 2 and
# version 6 were added: the names of recipes/transfer_finale_synhm_v<V>.yaml and the first drawn names
REV1_SHA = {
    "syn5v2_00000": "1979248f4b43f3f3", "syn5v2_00001": "4df7250050510e39", "syn5v2_00002": "7e47d1d4e50fdf4b", "syn5v2_00003": "f366559f166f5b23",
    "syn6v2_00000": "062566669244c3f9", "syn6v2_00001": "19899458b10a821c", "syn6v2_00002": "b378a903139a42bb", "syn6v2_00003": "42cf968caca24a0f",
    "syn5v3_00000": "7b92e3aefe53cdb3", "syn5v3_00001": "b5ceb37bb4b0631a", "syn5v3_00002": "21e49170f5285142", "syn5v3_00003": "0c7b1e23605e13df",
    "syn6v3_00000": "92434bcc4d179621", "syn6v3_00001": "a9787277b6a55bf4", "syn6v3_00002": "c47d790b4b99b4ae", "syn6v3_00003": "91ab16c83b6f15fa",
    "syn5v4_00000": "2c59f2335d5e29fe", "syn5v4_00001": "cf6b8108b9648b71", "syn5v4_00002": "2128fe907a29cd8c", "syn5v4_00003": "cbdb5b5bded65da3",
    "syn6v4_00000": "7e4b6472c8849a56", "syn6v4_00001": "0a719a88287f39af", "syn6v4_00002": "d2c779c20e2a0bf7", "syn6v4_00003": "54593dc7a60b16e3",
    "syn5v5_00000": "a5f361089df7bfe9", "syn5v5_00001": "0c9f8146a0d5f2d3", "syn5v5_00002": "7732459a474e0a70", "syn5v5_00003": "d50365ab9ff35b80",
    "syn6v5_00000": "e4702b2e27bc6ee9", "syn6v5_00001": "a0bba5c1a6a935f7", "syn6v5_00002": "b2b9eed29fb46ccc", "syn6v5_00003": "1d7ec2cca3497e3b",
    "synv2_00000": "2099fd909076ca98", "synv2_00001": "a68652b7b4c56926", "synv2_00002": "a0401e225f317f2a", "synv2_00003": "d6c1365f5446589c",
    "synv3_00000": "095e7102135d8860", "synv3_00001": "eeb2e4c84e61e64f", "synv3_00002": "aff446f9ff7948fb", "synv3_00003": "457adf4031fe30a2",
    "synv4_00000": "282c8203e8ef4757", "synv4_00001": "0b2bb27b41d9fe40", "synv4_00002": "40874c2007cf53d0", "synv4_00003": "9f393d626f0ac73c",
}

DIAG_DIR = None


def _events(name, n, rng, democratic=0.5):
    cfg = mg.register_synthetic(name, DIAG_DIR)
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
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        y = rng.uniform(0, max_rapidity); g, bg = np.cosh(y), np.sinh(y)
        K = np.eye(4); K[0, 0] = g; K[0, 1:] = K[1:, 0] = bg * n
        K[1:, 1:] += (g - 1) * np.outer(n, n)
        Rot = np.eye(4); Rot[1:, 1:] = Q
        L[b] = K @ Rot
    return L


def check_v1_frozen(rng):
    bad = [n for n, s in V1_SHA.items() if syn.structure(n)["sha"] != s]
    assert not bad, f"version 1 changed for {bad}"
    worst = 0.0
    for nm in list(V1_SHA)[:12]:
        P = _events(nm, 200, rng)
        st = syn.structure(nm)
        worst = max(worst, float(np.max(np.abs(syn.evaluate_vec(st, P) / syn.evaluate(st, P) - 1))))
    print(f"(0) version 1 frozen: {len(V1_SHA)} recorded structure hashes unchanged; vectorised evaluator vs "
          f"version 1's own: max rel. difference {worst:.1e}")
    assert worst < 1e-10, worst
    bad = [n for n, s_ in REV1_SHA.items() if syn.structure(n)["sha"] != s_]
    assert not bad, f"revision 1 of versions 2-5 changed for {bad}"
    print(f"    revision 1 of versions 2-5 frozen: {len(REV1_SHA)} recorded structure hashes unchanged")


def check_basic(names, rng, n=300):
    w_lor = w_dim = w_perm = 0.0
    n_ev = 0
    for nm in names:
        st = syn.structure(nm)
        # version 6 at 2->6 labels ~20 events/s on one core: fewer events there
        P = _events(nm, n if st["version"] < 6 or len(st["pdg_ids"]) < 8 else n // 5, rng)
        a = syn.evaluate(st, P)
        assert np.all(np.isfinite(a)) and np.all(a > 0), nm
        n_ev += len(a)
        P2 = np.einsum("bij,bkj->bki", _lorentz(rng, len(P)), P)
        w_lor = max(w_lor, float(np.max(np.abs(syn.evaluate(st, P2) / a - 1))))
        N = P.shape[1]
        for lam in (0.37, 2.9):
            b = syn.evaluate(st, lam * P, scale=lam)
            w_dim = max(w_dim, float(np.max(np.abs(b / (lam ** (8 - 2 * N) * a) - 1))))
        for perm in st["perms"][1:6]:
            w_perm = max(w_perm, float(np.max(np.abs(syn.evaluate(st, P[:, perm]) / a - 1))))
    print(f"    {len(names)} processes, {n_ev} events: finite and > 0; Lorentz max |dM2/M2| {w_lor:.1e}; "
          f"dimension {w_dim:.1e}; identical-leg permutations {w_perm:.1e}")
    assert w_lor < 1e-9 and w_dim < 1e-9 and w_perm < 1e-10, (w_lor, w_dim, w_perm)
    return w_lor, w_dim, w_perm


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
    print(f"(e) reproducibility: {len(names)} structures identical across interpreters, all distinct")


def check_legality(names):
    n_d = n_v = 0
    for nm in names:
        st = syn.structure(nm)
        ver = st["version"]
        if ver < 3:
            continue
        V = syn.sm_vertices(ver >= 5)
        pdgs = st["pdg_ids"]
        assert syn._families(pdgs[2:]) == syn._families(pdgs[:2]), (nm, "flavour")
        for d in st["diagrams"]:
            for trip, is_leg in syn.vertex_list(st, d):
                cls = V.get(tuple(sorted(trip)))
                assert cls is not None, (nm, trip)
                if cls == "HGG":
                    assert any(x == 25 and l for x, l in zip(trip, is_leg)), (nm, trip, "internal H at ggH")
                n_v += 1
            n_d += 1
    print(f"(f) legality: {n_d} diagrams, {n_v} vertices, every one an SM vertex (ggH only on the external H); "
          f"legs conserve quark generation and lepton family")
    rows = []
    for nm in names:
        st = syn.structure(nm)
        if "n_legal" not in st:
            continue
        orbit = sum(d["orbit"] for d in st["diagrams"])
        assert orbit == st["k_all"], nm
        assert st["subsampled"] or st["k_all"] == st["n_legal"], nm
        assert st["subsampled"] or st["k_all"] <= syn.legal_diagram_count(st["pdg_ids"], st["version"] >= 5), nm
        rows.append((nm, st["n_legal"], st["k_all"], st["subsampled"]))
    if rows:
        sub = [r for r in rows if r[3]]
        print(f"    revision 2 / v6: {len(rows)} processes, labelled diagrams = orbit sizes summed; {len(rows) - len(sub)}"
              f" keep every legal diagram (median {int(np.median([r[2] for r in rows if not r[3]] or [0]))}), "
              f"{len(sub)} subsampled: " + ", ".join(f"{r[0]} {r[2]} of {r[1]}" for r in sub))
    return n_d, n_v


def _tree_lines(d, out, N):
    """Fermion lines from the tree, recomputed here (graph walk over the merges): [(end, end, through a W)]."""
    X = {i: out[i] for i in range(N)}
    for c, pr in zip(range(N, N + len(d["merges"])), d["props"]):
        X[c] = pr["x"]
    nodes = {}                                    # edge (cluster id) -> the two vertices it joins
    for v, (a, b) in enumerate(d["merges"]):
        for e in (a, b):
            nodes.setdefault(e, []).append(v)
        nodes.setdefault(N + v, []).append(v)
    for e in d["root"]:
        nodes.setdefault(e, []).append("root")
    edges_at = {}
    for e, vs in nodes.items():
        for v in vs:
            edges_at.setdefault(v, []).append(e)
    ferm = lambda e: abs(X[e]) in (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)
    res, used = [], set()
    for i in range(N):
        if i in used or not ferm(i):
            continue
        e, v, w = i, nodes[i][0], False
        while True:
            w = w or any(abs(X[x]) == 24 for x in edges_at[v])
            e = next(x for x in edges_at[v] if x != e and ferm(x))
            if e < N:
                break
            v = next(u for u in nodes[e] if u != v)
        used.update((i, e))
        res.append((i, e, w))
    return res


def expected_configs(st):
    """The surviving helicity configurations of a version-6 structure, from the rules alone."""
    import itertools
    pdgs, m = st["pdg_ids"], st["m_legs"]
    N = len(pdgs)
    out = [syn._anti(p) if i < 2 else p for i, p in enumerate(pdgs)]
    opts = [(-2, 2) if abs(p) in (21, 22) else (-2, 0, 2) if abs(p) in (23, 24) else (0,) if p == 25 else (-1, 1)
            for p in out]
    scalar = 25 in pdgs or any(abs(pr["pdg"]) == 25 for d in st["diagrams"] for pr in d["props"])
    massless = all(x == 0.0 for x in m) and not scalar
    lines = [_tree_lines(d, out, N) for d in st["diagrams"]]

    def allows(ln, mu):
        for i, j, w in ln:
            if m[i] or m[j]:
                continue
            if mu[i] != -mu[j]:
                return False
            if w or abs(out[i]) in (12, 14, 16) or abs(out[j]) in (12, 14, 16):
                if mu[i if out[i] > 0 else j] != -1:
                    return False
        return True
    res = set()
    for lam in itertools.product(*opts):
        if massless and (sum(h < 0 for h in lam) < 2 or sum(h > 0 for h in lam) < 2):
            continue
        if any(allows(ln, tuple(lam[p] for p in perm)) for perm in st["perms"] for ln in lines):
            res.add(tuple(lam))
    return res


def check_v6(names, rng, n=20):
    """(i) little-group scaling of every helicity amplitude; (j) selection rules and the configuration sum."""
    w_lg = w_sum = w_par = 0.0
    n_conf = n_zero = n_amp = 0
    for nm in names:
        st = syn.structure(nm)
        N = len(st["pdg_ids"])
        P = _events(nm, n, rng)
        A = syn.helicity_amplitudes(st, P)
        t = np.exp(rng.uniform(-0.7, 0.7, N) + 1j * rng.uniform(-np.pi, np.pi, N))
        At = syn.helicity_amplitudes(st, P, lg=t)
        for lam, a in A.items():
            f = np.prod(t ** (-np.array(lam, float)))   # t^(-2h) with 2h = lam
            w_lg = max(w_lg, float(np.max(np.abs(At[lam] / (f * a) - 1))))
        exp = expected_configs(st)
        got = {tuple(c) for c in st["configs"]}
        assert got == exp, (nm, sorted(got ^ exp)[:5])
        assert set(A) == got, nm
        assert all(np.all(np.abs(a) > 0) for a in A.values()), nm
        import itertools
        n_all = int(np.prod([len(syn._hel2_options(p)) for p in st["pdg_ids"]]))
        n_conf += len(got)
        n_zero += n_all - len(got)
        n_amp += len(A) * len(P)
        Pi = P.copy()
        Pi[..., 1:] *= -1                                   # spatial inversion
        Ai = syn.helicity_amplitudes(st, Pi)
        m2 = syn.evaluate(st, P)
        rec = 0.25 * sum(0.5 * (np.abs(A[l]) ** 2 + np.abs(Ai[l]) ** 2) for l in A)
        w_sum = max(w_sum, float(np.max(np.abs(rec / m2 - 1))))
        w_par = max(w_par, float(np.max(np.abs(syn.evaluate(st, Pi) / m2 - 1))))
    print(f"(i) little-group scaling: {n_amp} helicity amplitudes x events, max |A(t)/(t^-2h A) - 1| {w_lg:.1e}")
    print(f"(j) selection rules: {n_conf} surviving configurations equal the rules' set, each non-zero; {n_zero} "
          f"vanish; |M|^2 = (1/4) sum (|A(p)|^2 + |A(p~)|^2)/2 over the survivors to {w_sum:.1e}; parity {w_par:.1e}")
    assert w_lg < 1e-9 and w_sum < 1e-9 and w_par < 1e-9, (w_lg, w_sum, w_par)


def _boost(p, b):
    """Boost 4-vectors p (..., 4) by velocity b (3,)."""
    b2 = float(b @ b)
    if b2 == 0:
        return p
    g = 1 / np.sqrt(1 - b2)
    bp = p[..., 1:] @ b
    E = g * (p[..., 0] - bp)
    v = p[..., 1:] + (((g - 1) * bp / b2 - g * p[..., 0])[..., None]) * b
    return np.concatenate([E[..., None], v], -1)


def _rebalance(fin, recoil):
    """Momentum-conserving event from final momenta `fin` (n, 4): the `recoil` legs are boosted together so the
    final state has no transverse momentum (their invariant mass kept), beams massless along +-z."""
    R = fin[recoil].sum(0)
    T = fin[[i for i in range(len(fin)) if i not in recoil]].sum(0)[1:3]
    M2 = R[0] ** 2 - R[1:] @ R[1:]
    if M2 <= 1e-6 * R[0] ** 2:                 # a lone massless recoiler cannot absorb transverse momentum
        return None
    Rt = np.array([np.sqrt(M2 + T @ T + R[3] ** 2), -T[0], -T[1], R[3]])
    out = fin.copy()
    out[recoil] = _boost(_boost(fin[recoil], R[1:] / R[0]), -Rt[1:] / Rt[0])
    tot = out.sum(0)
    assert abs(tot[1]) < 1e-9 * tot[0] and abs(tot[2]) < 1e-9 * tot[0]
    Ea, Eb = (tot[0] + tot[3]) / 2, (tot[0] - tot[3]) / 2
    return np.concatenate([[[Ea, 0, 0, Ea], [Eb, 0, 0, -Eb]], out], 0)


def _rot(v, axis, th):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(th) + np.cross(axis, v) * np.sin(th) + axis * (axis @ v) * (1 - np.cos(th))


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def limit_slopes(nm, rng, i, k, kind):
    """Slope of log|M|^2 against log s_ik (collinear: k rotated onto i at fixed energy) or log E_k (soft: k scaled
    at fixed direction), on momentum-conserving sequences built from one fiducial event. Final legs i, k (slots)."""
    st = syn.structure(nm)
    P = _events(nm, 1, rng, democratic=0.0)[0]
    fin = P[2:].copy()
    a, b = i - 2, k - 2
    others = [j for j in range(len(fin)) if j not in (a, b)]
    if kind == "collinear":
        axis = np.cross(fin[a, 1:], rng.normal(size=3))
        Ek = fin[b, 0]
        xs, ys = [], []
        for th in np.geomspace(1e-2, 10 ** -5.5, 8):
            f = fin.copy()
            f[b, 1:] = _rot(fin[a, 1:] / np.linalg.norm(fin[a, 1:]) * Ek, axis, th)
            Pe = _rebalance(f, others)
            if Pe is None:
                return None
            q = Pe.copy(); q[:2] *= -1
            xs.append(abs(syn._mink2(q[i] + q[k])))
            ys.append(syn.evaluate(st, Pe[None])[0])
        xs, ys = np.array(xs), np.array(ys)
    else:
        xs, ys = [], []
        for lam in np.geomspace(1e-1, 1e-7, 13):
            f = fin.copy()
            f[b] = fin[b] * lam
            Pe = _rebalance(f, [j for j in range(len(fin)) if j != b])
            if Pe is None:
                return None
            xs.append(f[b, 0])
            ys.append(syn.evaluate(st, Pe[None])[0])
        xs, ys = np.array(xs), np.array(ys)
    # the last decade: a massive resonant line next to the emitter sets an intermediate scale
    # (2 p.k ~ M Gamma) above which the asymptotic power has not set in yet
    return _slope(xs[-3:], ys[-3:])


def check_limits(names, rng):
    """(g) for every v4+ name: each bracket pair's collinear slope (expect -1), each soft-able gauge leg's soft
    slope (expect -2). For v1-v3 names, the slope of their massless two-leg final channels, for comparison."""
    rows = []
    for nm in names:
        st = syn.structure(nm)
        seen = set()
        for d in st["diagrams"]:
            for sp in d.get("spinors", []):
                i, k = sp["legs"]
                if (i, k) in seen or st["nfinal"] < 3:
                    continue
                seen.add((i, k))
                c = limit_slopes(nm, rng, i, k, "collinear")
                soft = [limit_slopes(nm, rng, i if x == k else k, x, "soft") for x in sp["soft"]]
                if c is not None:
                    rows.append((nm, st["version"], (i, k), c, [x for x in soft if x is not None]))
            if st["version"] < 4 and st["nfinal"] >= 3:
                for pr in d["props"]:
                    S = pr["legs"]
                    if pr["mass"] == 0 and len(S) == 2 and 1 not in S and tuple(S) not in seen \
                            and all(st["m_legs"][x] == 0 for x in S):
                        seen.add(tuple(S))
                        c = limit_slopes(nm, rng, *S, "collinear")
                        if c is not None:
                            rows.append((nm, st["version"], tuple(S), c, []))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-names", type=int, default=12, help="extra drawn-multiplicity names per version")
    ap.add_argument("--diagrams-dir", default=None, help="where the sidecars go (default: a temp dir)")
    ap.add_argument("--quick", action="store_true", help="revision 2 and version 6 only (revision 1 as recorded)")
    a = ap.parse_args()
    import tempfile
    DIAG_DIR = a.diagrams_dir or tempfile.mkdtemp(prefix="syn_sidecars_")
    rng = np.random.default_rng(2026)
    check_v1_frozen(rng)
    names = {}
    # (version, name tag, target-shaped): revision 1 of v2-v5 (superseded, still checked), revision 2, version 6
    groups = ([] if a.quick else [(v, f"v{v}", v == 5) for v in (2, 3, 4, 5)]) + \
        [(v, f"v{v}r2", v == 5) for v in (2, 3, 4, 5)] + [(6, "v6", False), (6, "v6t", True)]
    for v, tag, target in groups:
        names[tag] = [f"syn{n}{tag}_{k:05d}" for n in (5, 6) for k in range(4)]
        if not target:
            names[tag] += [f"syn{tag}_{k:05d}" for k in range(a.n_names)]
        elif v == 5:
            names[tag] += [f"syn{n}{tag}_{k:05d}" for n in (5, 6) for k in range(4, 4 + a.n_names // 2)]
        print(f"(a-d) {tag}: " + ", ".join(names[tag][:8]) + f" + {len(names[tag]) - 8} more")
        check_basic(names[tag], rng)
    allv = sum(names.values(), [])
    check_reproducible(allv)
    check_legality(allv)
    v6 = names["v6"] + names["v6t"]
    check_v6(v6, rng)
    v1_names = [] if a.quick else [f"syn_{k:05d}" for k in range(19)] + [f"syn{n}_{k:05d}" for n in (5, 6) for k in range(4)]
    rows = check_limits(v1_names + allv, rng)
    print("(g) limits (slope of log|M|^2 vs log s_ik, collinear; vs log E_k, soft gauge leg):")
    for tag in ["v1"] + list(names):
        r = [x for x in rows if (x[0] in names.get(tag, ()) if tag != "v1" else x[1] == 1)]
        if not r:
            continue
        col = np.array([x[3] for x in r])
        soft = np.array(sum([x[4] for x in r], []))
        msg = (f"    {tag}: {len(r)} massless two-leg final pairs, collinear slope range [{col.min():.3f}, "
               f"{col.max():.3f}] ({int((col < -1.5).sum())} at -2, i.e. 1/s_ik^2)")
        if len(soft):
            msg += f"; {len(soft)} soft gauge legs, slope range [{soft.min():.3f}, {soft.max():.3f}]"
        print(msg)
        if r[0][1] >= 4:
            assert np.all(np.abs(col + 1) < 0.02), r
            assert np.all(np.abs(soft + 2) < 0.02), r
    n_side = 0
    for nm in allv:
        sc = syn.sidecar(nm)
        st = syn.structure(nm)
        n_side += sc["subprocesses"][0]["n_diagrams"]
        assert sc["subprocesses"][0]["n_diagrams"] >= len(st["diagrams"])
    print(f"(h) sidecars: {len(allv)} processes, {n_side} diagram images")
    print("all synthetic-version checks passed")
