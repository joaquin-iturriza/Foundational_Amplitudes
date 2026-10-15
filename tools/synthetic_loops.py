#!/usr/bin/env python
"""Synthetic one-loop amplitudes (site decision D44: "What is still unexplored is loops, so this should be
investigated a bit"; docs/results.tex sec:syn-loops).

A synthetic one-loop process ``synl_<k>`` (or ``synl<n>_<k>``) is the one-loop partner of the version-1 tree
process ``syn_<k>`` (``syn<n>_<k>``): the same legs, diagrams, couplings and helicity numerators
(tools/synthetic_amplitudes.py, version 1), plus loop structure drawn from a seed fixed by the loop name. The
stored target is what the real one-loop pools store (tools/nlo_virtual_pipeline.py, virt_e4): the coupling-stripped
tree-loop interference, at mu^2 = s,

    V = g(sqrt s) <sum_h Re[ A0_h^* A1_h ]>_perm,     so that a loop amplitude A1 = c A0 gives V = Re(c) |M0|^2,

g and the permutation average as the born's. Per helicity h (A0_h = sum_d T_hd, T_hd = c_hd P_hd s^(-k/2) / prod D):

    A1_h = sum_d T_hd [ U + lam_bub sum_{S in d} w_S B(s_S; m_S) ]       factorising: IR dipoles, Sudakov, bubbles
         + lam_box sum_d T'_hd F_d                                       boxes on the diagram's channels
         + lam_rat sum_d T''_hd                                          rational terms

T' and T'' have the diagram's denominators and fresh random numerators drawn like version 1's (degree, monomials,
s-power for the mass dimension). Every loop function is dimensionless and a function of invariant ratios, so V keeps
the born's mass dimension 8 - 2N and Lorentz invariance. What each term mimics:

  U, the universal (process-level) part, as the finite remainder of the one-loop IR structure at mu^2 = s:
    * gamma0                          the constant of a form factor (e+e- -> q qbar: C_F (pi^2 - 8) at mu^2 = s);
    * massless dipoles (i, j):       -1/2 kappa_ij [ L_ij^2 + 2 g_ij L_ij ],  L_ij = ln(-s_ij/s - i0): the double and
                                     single logarithms of Catani's I operator (soft-collinear, collinear), with
                                     L^2 = ln^2|s_ij/s| - pi^2 on a time-like pair (the pi^2 of the analytic
                                     continuation); g = gamma_i / C_i (3/2 for a quark, beta_0 / C_A = 23/18 for a gluon);
    * massive dipoles:                kappa_ij [ -rho_ij L_ij + pi^2 (1 + v^2)/(2 v) on a time-like pair ], v the
                                     pair's relative velocity, rho = artanh(v)/v: the quasi-collinear log of a massive
                                     emitter and the Coulomb singularity 1/v at the pair's threshold; a massless-massive
                                     pair keeps half the double log (one collinear-singular leg);
      kappa_ij = -T_i.T_j: for a QCD loop drawn positive, symmetric, with sum_j kappa_ij = C_i (colour conservation,
      Casimirs of the legs); for an electroweak loop -Q_i Q_j (exact charge conservation, the QED dipoles);
    * electroweak Sudakov logs (EW loops only), L_W = ln(1 + s/M_W^2) (-> ln(s/M_W^2) at high energy, -> 0 where the
      massive bosons decouple): sum_i C_i [ -1/2 L_W^2 + 3/2 f_i L_W ] + sum_{i<j} J_ij ln|s_ij/s| L_W, the leading
      (double), collinear (single) and angular (subleading soft-collinear) logarithms;
    * bubbles on s: w B(s; m) with m among m_t, M_W, M_Z, m_H, m_b: thresholds at sqrt s = 2m.
  B(p^2; m) = 2 - ln(m^2/s) - beta ln((beta + 1)/(beta - 1)), beta = sqrt(1 - 4m^2/(p^2 + i0)): the finite part of the
    scalar bubble, real below threshold, a sqrt(1 - 4m^2/p^2) onset and an imaginary part i pi beta above it; on each
    propagator of each diagram with a loop mass drawn among 0, the line's own mass, m_t, M_W, M_Z (self-energy and
    vertex corrections of that line).
  F_d, the two-mass-easy box function (BDK), on one time-like invariant s_a (s or an s-type line of the diagram) and
    one space-like s_b (an initial-final pair or a t-type line), masses P^2, Q^2 (0, an s-type line, a massive leg):
        F = 1/2 ln^2(s_a/s_b) - Li2(1 - P^2/s_a) - Li2(1 - P^2/s_b) - Li2(1 - Q^2/s_a) - Li2(1 - Q^2/s_b)
            + Li2(1 - P^2 Q^2/(s_a s_b)),
    every invariant continued as x + i0 (Li2(1) = pi^2/6 where a mass is 0): the logs and dilogarithms of Mandelstam
    ratios that do not factorise on the born.
  Rational terms: what is left of a real one-loop amplitude after the integrals (no logs).

Signs: the loop functions change sign over phase space (ln^2 against pi^2, the Sudakov double log against the single
logs, the box against the born), so V changes sign as real virtual corrections do; how often depends on the draws.

Loop type: QCD when the process has two or more coloured legs (probability 0.7), else electroweak. Order vector
[1, 0, k_s, n - k_s] (QCD) or [0, 1, k_s, n - k_s] (EW), the born's coupling powers with the loop in L_QCD or L_EW, as
mg5_pipeline_final.order_vector writes for the real virtuals.

    python tools/synthetic_loops.py --describe synl_00011 synl2_00003
"""
import functools
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools import synthetic_amplitudes as syn  # noqa: E402

LOOP_VERSION = 1
_LSALT = 0x1001
_LNAME = re.compile(r"^synl([2-6])?_(\d+)$")
_HEAVY = (172.5, 80.419, 91.188, 125.0, 4.7)      # loop masses of the s bubbles: t, W, Z, H, b
_LOOP_M = (172.5, 80.419, 91.188)                  # loop masses of the line bubbles (besides 0 and the line's own)
_MW = 80.419
_G_QUARK, _G_GLUON = 1.5, 23.0 / 18.0             # gamma_i / C_i: quark 3/2, gluon beta_0 / C_A (n_f = 5)
_P_QCD = 0.7
_P_BOX = 0.7
_EPS = 1e-12                                       # the +i0 of an invariant, relative to s


def is_loop_name(name):
    return bool(_LNAME.match(str(name)))


def born_name(name):
    m = _LNAME.match(str(name))
    if not m:
        raise KeyError(f"{name!r} is not a synthetic one-loop name (synl[<n>]_<k>)")
    return f"syn{m.group(1) or ''}_{m.group(2)}"


# ----------------------------------------------------------------------------- special functions
_BERN = [1.0, -1.0 / 2, 1.0 / 6, 0.0, -1.0 / 30, 0.0, 1.0 / 42, 0.0, -1.0 / 30, 0.0, 5.0 / 66, 0.0, -691.0 / 2730,
         0.0, 7.0 / 6, 0.0, -3617.0 / 510, 0.0, 43867.0 / 798, 0.0, -174611.0 / 330, 0.0, 854513.0 / 138,
         0.0, -236364091.0 / 2730, 0.0, 8553103.0 / 6]
_LI2_C = [b / np.prod(np.arange(1, n + 2, dtype=float)) for n, b in enumerate(_BERN)]   # B_n / (n+1)!


def li2(z):
    """Complex dilogarithm Li2(z) on the principal branch (cut z > 1, the side set by Im z), vectorised.
    |z| > 1 is mapped by Li2(z) = -Li2(1/z) - pi^2/6 - ln^2(-z)/2, then Re z > 1/2 by
    Li2(z) = -Li2(1-z) + pi^2/6 - ln z ln(1-z), and the rest summed in u = -ln(1-z) with Bernoulli numbers."""
    z = np.asarray(z, np.complex128)
    out = np.empty_like(z)
    flat, res = z.ravel(), out.ravel()
    zero = flat == 0
    res[zero] = 0.0
    w = flat[~zero]
    r = np.zeros_like(w)
    inv = np.abs(w) > 1
    lnmz = np.log(-w[inv])
    r[inv] = -np.pi ** 2 / 6 - 0.5 * lnmz ** 2
    sgn = np.where(inv, -1.0, 1.0)
    w = np.where(inv, 1.0 / w, w)
    ref = w.real > 0.5
    r[ref] += sgn[ref] * (np.pi ** 2 / 6 - np.log(w[ref]) * np.log(1 - w[ref]))
    sgn = np.where(ref, -sgn, sgn)
    w = np.where(ref, 1 - w, w)
    u = -np.log(1 - w)
    acc = np.zeros_like(u)
    for c in reversed(_LI2_C):
        acc = acc * u + c
    r = r + sgn * acc * u
    res[~zero] = r
    return out


def _lnm(x, s):
    """ln(-x/s - i0) for real x: ln|x/s| - i pi theta(x)."""
    return np.log(np.abs(x / s)) - 1j * np.pi * (x > 0)


def bubble(p2, m, s):
    """Finite part of the scalar bubble B0(p^2; m, m) at mu^2 = s (MS-bar), p^2 + i0: 2 - ln(m^2/s) - f, with
    f = beta ln((beta+1)/(beta-1)) below 0, 2 sqrt(r-1) arctan(1/sqrt(r-1)) (r = 4m^2/p^2) between 0 and the threshold
    4m^2, beta [ln((1+beta)/(1-beta)) - i pi] above it; m = 0: 2 - ln(-p^2/s - i0)."""
    p2 = np.asarray(p2, float)
    if m == 0.0:
        return 2.0 - _lnm(p2, s)
    m2 = m * m
    out = np.empty(p2.shape, np.complex128)
    neg = p2 < 0
    above = p2 > 4 * m2
    mid = ~neg & ~above
    b = np.sqrt(1 - 4 * m2 / p2[neg])
    out[neg] = b * np.log((b + 1) / (b - 1))
    b = np.sqrt(1 - 4 * m2 / p2[above])
    out[above] = b * (np.log((1 + b) / (1 - b)) - 1j * np.pi)
    pm = p2[mid]
    safe = np.where(pm > 0, pm, 1.0)
    rr = np.sqrt(np.maximum(4 * m2 / safe - 1, 0.0))
    out[mid] = np.where(pm > 0, 2 * rr * np.arctan(1 / np.where(rr > 0, rr, 1.0)) * (rr > 0), 2.0)
    return 2.0 - np.log(m2 / s) - out


def box2me(sa, sb, P2, Q2, s):
    """The two-mass-easy box function of the module docstring; P2 or Q2 None for a massless corner."""
    c = lambda x: x + 1j * _EPS * s                                    # x + i0
    ca, cb = c(sa), c(sb)
    F = 0.5 * (_lnm(sa, s) - _lnm(sb, s)) ** 2
    zeta2 = np.pi ** 2 / 6
    for M in (P2, Q2):
        if M is None:
            F = F - 2 * zeta2
        else:
            cm = c(M)
            F = F - li2(1 - cm / ca) - li2(1 - cm / cb)
    if P2 is None or Q2 is None:
        F = F + zeta2
    else:
        F = F + li2(1 - c(P2) * c(Q2) / (ca * cb))
    return F


# ----------------------------------------------------------------------------- structure
def _draw_num(rng, N, n_pairs, e_sum):
    """One version-1 numerator term: complex coefficient, degree, 1-3 monomials, the s-power for the dimension."""
    deg = int(rng.integers(0, 3))
    monos = [{"a": float(rng.normal()), "pairs": sorted(int(x) for x in rng.choice(n_pairs, size=deg))}
             for _ in range(int(rng.integers(1, 4)))]
    return {"c": [float(rng.normal()), float(rng.normal())], "deg": deg, "monos": monos,
            "k": 2 * deg - 2 * e_sum - 4 + N}


def _sinkhorn(rng, C):
    """Symmetric positive kappa_ij (zero diagonal) with row sums C_i: a random matrix scaled to the Casimirs."""
    n = len(C)
    K = rng.uniform(0.5, 1.5, (n, n))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 0.0)
    d = np.ones(n)
    for _ in range(500):
        d = np.sqrt(d * np.asarray(C) / (K @ d))
    return d[:, None] * K * d[None, :]


@functools.lru_cache(maxsize=None)
def structure_loop(name):
    m = _LNAME.match(str(name))
    if not m:
        raise KeyError(f"{name!r} is not a synthetic one-loop name (synl[<n>]_<k>)")
    n_forced, idx = (int(m.group(1)) if m.group(1) else 0), int(m.group(2))
    born = syn.structure(born_name(name))
    rng = np.random.default_rng(np.random.SeedSequence([syn._SALT, _LSALT, LOOP_VERSION, n_forced, idx]))
    pdgs, m_legs, n = born["pdg_ids"], born["m_legs"], born["nfinal"]
    N = n + 2
    sign = [-1] * 2 + [1] * n
    cas = [float(syn._props(p)[syn._COL["color_casimir"]]) for p in pdgs]
    Q = [sign[i] * syn._q3(pdgs[i]) / 3.0 for i in range(N)]            # all-outgoing charges
    coloured = [i for i in range(N) if cas[i] > 0]
    typ = "QCD" if len(coloured) >= 2 and rng.random() < _P_QCD else "EW"
    fermion = [abs(p) in syn._FERMIONS for p in pdgs]

    dipoles = []
    if typ == "QCD":
        K = _sinkhorn(rng, [cas[i] for i in coloured])
        for a, b in itertools.combinations(range(len(coloured)), 2):
            i, j = coloured[a], coloured[b]
            gi = _G_QUARK if fermion[i] else _G_GLUON
            gj = _G_QUARK if fermion[j] else _G_GLUON
            dipoles.append({"i": i, "j": j, "kappa": float(K[a, b]), "g": (gi + gj) / 2})
    else:
        charged = [i for i in range(N) if Q[i] != 0]
        for i, j in itertools.combinations(charged, 2):
            dipoles.append({"i": i, "j": j, "kappa": float(-Q[i] * Q[j]), "g": _G_QUARK})
    gamma0 = float(sum(d["kappa"] for d in dipoles) * rng.uniform(-6.0, 0.0))

    sudakov = None
    if typ == "EW":
        C, f = [], []
        for p in pdgs:
            C.append(0.0 if p == 21 else float(rng.uniform(0.0, 1.0)) if p == 22 else float(rng.uniform(0.5, 2.5)))
            f.append(1.0 if abs(p) in syn._FERMIONS else float(rng.uniform(0.0, 2.0)))
        J = [[i, j, float(rng.normal() * np.sqrt(C[i] * C[j]) / 2)]
             for i, j in itertools.combinations(range(N), 2) if C[i] > 0 and C[j] > 0]
        sudakov = {"C": C, "f": f, "J": J}
        gamma0 += float(sum(C) * rng.uniform(-2.0, 1.0))

    s_bub = [{"m": float(_HEAVY[int(rng.integers(len(_HEAVY)))]), "w": float(rng.normal())}
             for _ in range(int(rng.integers(0, 3)))]
    lam = {"bub": float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))),
           "box": float(np.exp(rng.uniform(np.log(0.1), np.log(3.0)))),
           "rat": float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))}

    finals = tuple(range(2, N))
    t_pairs = sorted({syn._canonical({b, j}, N) for b in (0, 1) for j in finals})
    n_pairs = len(born["pairs"])
    H = len(born["numerators"])
    bubbles, boxes = [], []
    for d in born["diagrams"]:
        row = []
        for pr in d["props"]:
            opts = sorted({0.0, *_LOOP_M} | ({float(pr["mass"])} if pr["mass"] > 0 else set()))
            row.append({"m": float(opts[int(rng.integers(len(opts)))]), "w": float(rng.normal())})
        bubbles.append(row)
        e_sum = sum(p["power"] for p in d["props"])
        if rng.random() < _P_BOX:
            s_lines = [tuple(p["legs"]) for p in d["props"] if 1 not in p["legs"]]
            t_lines = [tuple(p["legs"]) for p in d["props"] if 1 in p["legs"]]
            a_opts = [finals] + s_lines
            b_opts = t_pairs + t_lines
            m_opts = s_lines + [(j,) for j in finals if m_legs[j] > 0]
            pick = lambda o: list(o[int(rng.integers(len(o)))])
            mass = lambda: (pick(m_opts) if (m_opts and rng.random() < 0.5) else None)
            boxes.append({"a": pick(a_opts), "b": pick(b_opts), "P": mass(), "Q": mass(),
                          "num": [_draw_num(rng, N, n_pairs, e_sum) for _ in range(H)]})
        else:
            boxes.append(None)
    rational = [[_draw_num(rng, N, n_pairs, sum(p["power"] for p in d["props"])) for d in born["diagrams"]]
                for _ in range(H)]

    st = dict(born)
    st.pop("sha", None)
    st.update({
        "name": name, "born": born["name"], "born_sha": born["sha"], "loop_version": LOOP_VERSION,
        "order": [1, 0, born["k_s"], n - born["k_s"]] if typ == "QCD" else [0, 1, born["k_s"], n - born["k_s"]],
        "loop": {"type": typ, "gamma0": gamma0, "dipoles": dipoles, "sudakov": sudakov, "s_bubbles": s_bub,
                 "lam": lam, "bubbles": bubbles, "boxes": boxes, "rational": rational},
    })
    st["sha"] = hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return st


# ----------------------------------------------------------------------------- evaluation
def _numval(term, s_pair, s_tot):
    poly = np.zeros(s_tot.shape)
    for mono in term["monos"]:
        t = np.full(s_tot.shape, mono["a"])
        for k in mono["pairs"]:
            t = t * s_pair[:, k]
        poly = poly + t
    return complex(*term["c"]) * poly * s_tot ** (-0.5 * term["k"])


def universal(lp, q, s_tot, m_legs, scale=1.0):
    """U per event (complex): gamma0, the dipoles, the Sudakov logs, the s bubbles (module docstring)."""
    U = np.full(s_tot.shape, lp["gamma0"], np.complex128)
    s_ij = lambda i, j: syn._mink2(q[:, i] + q[:, j])
    for dp in lp["dipoles"]:
        i, j, kap = dp["i"], dp["j"], dp["kappa"]
        sij = s_ij(i, j)
        tl = (i < 2) == (j < 2)                                          # both initial or both final: time-like
        mi, mj = m_legs[i] * scale, m_legs[j] * scale
        L = _lnm(sij, s_tot) if tl else np.log(np.abs(sij / s_tot)) + 0j
        if mi == 0 and mj == 0:
            U = U - 0.5 * kap * (L ** 2 + 2 * dp["g"] * L)
        elif mi == 0 or mj == 0:
            U = U - 0.5 * kap * (0.5 * L ** 2 + 2 * dp["g"] * L)
        else:
            x = sij - mi * mi - mj * mj
            v = np.sqrt(np.clip(1 - 4 * mi * mi * mj * mj / (x * x), 1e-24, 1.0))
            v = np.minimum(v, 1 - 1e-15)
            rho = np.arctanh(v) / v
            U = U + kap * (-rho * L + (np.pi ** 2 * (1 + v * v) / (2 * v) if tl else 0.0))
    sk = lp["sudakov"]
    if sk is not None:
        LW = np.log1p(s_tot / (_MW * scale) ** 2)
        for i, (C, f) in enumerate(zip(sk["C"], sk["f"])):
            if C > 0:
                U = U + C * (-0.5 * LW ** 2 + 1.5 * f * LW)
        for i, j, J in sk["J"]:
            U = U + J * np.log(np.abs(s_ij(i, j) / s_tot)) * LW
    for b in lp["s_bubbles"]:
        U = U + b["w"] * bubble(s_tot, b["m"] * scale, s_tot)
    return U


def evaluate_loop(st, P, scale=1.0, parts=False):
    """V of the one-loop structure `st` at momenta P (B, N, 4), rows in pdg_ids order, beams incoming; `scale`
    multiplies every dimensionful constant (the mass-dimension check, as syn.evaluate). parts=True returns
    (V, born |M0|^2, V_factorising) where V_factorising keeps U only (the K-factor-like part)."""
    P = np.asarray(P, np.float64)
    B, N = P.shape[:2]
    sign = np.array([-1.0, -1.0] + [1.0] * (N - 2))
    lp = st["loop"]
    pairs = [tuple(p) for p in st["pairs"]]
    m_legs = st["m_legs"]
    tV, tB, tU = np.zeros(B), np.zeros(B), np.zeros(B)
    for perm in st["perms"]:
        q = P[:, perm, :] * sign[None, :, None]
        s_tot = syn._mink2(q[:, 0] + q[:, 1])
        s_pair = np.stack([syn._mink2(q[:, i] + q[:, j]) for i, j in pairs], axis=1)
        inv = {}

        def sS(S):
            S = tuple(S)
            if S not in inv:
                inv[S] = syn._mink2(q[:, list(S)].sum(1))
            return inv[S]
        dens, bub, Fd = [], [], []
        for d, brow, box in zip(st["diagrams"], lp["bubbles"], lp["boxes"]):
            den = np.ones(B, np.complex128)
            bsum = np.zeros(B, np.complex128)
            for pr, bb in zip(d["props"], brow):
                x = sS(pr["legs"])
                M, G = pr["mass"] * scale, pr["width"] * scale
                D = x - M * M + 1j * M * G
                if pr["power"] == 1.0:
                    den = den * D
                elif M > 0:
                    den = den * np.sqrt(D)
                else:
                    den = den * np.sqrt(np.abs(x))
                bsum = bsum + bb["w"] * bubble(x, bb["m"] * scale, s_tot)
            dens.append(den)
            bub.append(bsum)
            Fd.append(None if box is None else
                      box2me(sS(box["a"]), sS(box["b"]), None if box["P"] is None else sS(box["P"]),
                             None if box["Q"] is None else sS(box["Q"]), s_tot))
        U = universal(lp, q, s_tot, m_legs, scale)
        for h, row in enumerate(st["numerators"]):
            A0 = np.zeros(B, np.complex128)
            A1 = np.zeros(B, np.complex128)
            for di, (term, den) in enumerate(zip(row, dens)):
                T = _numval(term, s_pair, s_tot) / den
                A0 = A0 + T
                A1 = A1 + T * lp["lam"]["bub"] * bub[di]
                box = lp["boxes"][di]
                if box is not None:
                    A1 = A1 + lp["lam"]["box"] * _numval(box["num"][h], s_pair, s_tot) / den * Fd[di]
                A1 = A1 + lp["lam"]["rat"] * _numval(lp["rational"][h][di], s_pair, s_tot) / den
            A1 = A1 + U * A0
            tV = tV + np.real(np.conj(A0) * A1)
            a2 = np.abs(A0) ** 2
            tB = tB + a2
            tU = tU + np.real(U) * a2
    g = st["coupling"] * syn._alpha_s(P, scale) ** st["k_s"] / len(st["perms"])
    return (g * tV, g * tB, g * tU) if parts else g * tV


def _describe(name):
    st = structure_loop(name)
    lp = st["loop"]
    print(f"{name}  loop v{LOOP_VERSION} of {st['born']}  sha={st['sha']}  {st['pdg_ids'][:2]} -> {st['pdg_ids'][2:]}"
          f"  {lp['type']}  order={st['order']}  K={len(st['diagrams'])}  H={len(st['numerators'])}")
    print(f"  gamma0={lp['gamma0']:.2f}  dipoles={[(d['i'], d['j'], round(d['kappa'], 2)) for d in lp['dipoles']]}"
          f"  s-bubbles={[(b['m'], round(b['w'], 2)) for b in lp['s_bubbles']]}  lam={ {k: round(v, 2) for k, v in lp['lam'].items()} }"
          f"  boxes={sum(b is not None for b in lp['boxes'])}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--describe", nargs="*", default=[])
    for nm in ap.parse_args().describe:
        _describe(nm)
