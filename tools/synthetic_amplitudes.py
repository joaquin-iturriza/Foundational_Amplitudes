#!/usr/bin/env python
"""Synthetic matrix elements: the synthetic-amplitude prior (docs/results.tex, hand-off
"priors and representations").

A synthetic process ``syn_<k>`` (or ``syn<n>_<k>`` with n final particles forced) is a fixed
random structure drawn from a seed derived from its name, so the name alone regenerates it:

  * legs: a real SM initial pair (massless, along z) and n = 2..4 SM final particles,
    charge, baryon and lepton number conserved, at the generator masses of
    particle_ids.GLOBAL_PROPERTY_MATRIX, so the model's particle-property inputs are real;
  * K random tree diagrams (random cubic trees over the legs). Every internal edge is a
    propagator over a channel S (a leg subset; s_S = (sum of all-outgoing momenta)^2,
    s-type when S holds only final legs, t-type when it splits the two beams) carrying a
    real SM particle of the channel's charge: gamma/g/Z/H, W/e/mu, u/t, d/b. A massive
    s-type propagator has a Breit-Wigner width Gamma = r M, r log-uniform per particle; a
    t-type one has none. The scalar (PDG 25) takes a random mass per process half of the
    time (``internal_masses``), the others the generator masses. A massless or light (b)
    propagator enters as 1/D_S (scalar-like) or 1/sqrt(D_S) (the 1/s_ij of a gauge-theory
    collinear limit in |M|^2); a massless one only sits on channels the fiducial cuts keep
    away from s_S = 0;
  * per helicity h = 1..H and diagram d a complex coefficient c_hd and a random homogeneous
    polynomial P_hd in the pair invariants s_ij, times the power of s that gives the term
    the amplitude's mass dimension 4 - N:

        |M|^2 = g * < sum_h | sum_d c_hd P_hd(s_ij) s^(-k_hd/2) / prod_{S in d} D_S |^2 >_perm,
        D_S = s_S - M_S^2 + i M_S Gamma_S   (or its square root, |s_S| for a massless one),

    g = (4 pi alpha_s(sqrt s))^k_s (4 pi alpha)^(n - k_s) (alpha_s run per event at one loop,
    as the real pools), averaged over the permutations of
    identical legs (the target must be a function of the particle set). |M|^2 > 0 has mass
    dimension 8 - 2N and is Lorentz invariant by construction (only invariants enter).

The structure also gives the diagram sidecar (``<name>.diagrams.json``, tools/dump_diagrams.py
format), so data.offshell_per_event sees the true propagators. Generation goes through the
normal recipe path: mg5_pipeline_final registers ``syn*`` names into PROCESSES with
``kind: synthetic`` and labels events with ``label_events`` instead of a MadGraph backend.

CLI:
    python tools/synthetic_amplitudes.py --describe syn_00042
    python tools/synthetic_amplitudes.py --recipe recipes/synthetic_pilot.yaml --first 0 --count 20 \
        --n-train 20000 --n-val 2000 --n-test 2000
    python tools/synthetic_amplitudes.py --sidecars recipes/synthetic_pilot.yaml [--out-dir data/diagrams]
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

from particle_ids import PARTICLE_PROPERTIES, PARTICLE_FEATURE_NAMES, _MASSLESS  # noqa: E402

# Bumped whenever a change alters the structure drawn for a name or the amplitude it
# evaluates to: the version enters the structure hash, hence every pool's recipe_id.
SYN_VERSION = 1
_SALT = 0x5E17A                      # seed namespace of synthetic processes
_NAME = re.compile(r"^syn([2-6])?_(\d+)$")
DEFAULT_DIAGRAMS_DIR = os.path.join(_HERE, "data", "diagrams")

ALPHA_S, ALPHA_EW, MZ_REF = 0.118, 1.0 / 132.507, 91.1876      # alpha_s(M_Z), running reference
SQRTS_MAX = 1000.0
_COL = {k: PARTICLE_FEATURE_NAMES.index(k) for k in
        ("charge", "log10_mass_gev", "baryon_number", "lepton_number", "color_casimir")}
_NEUTRINOS = frozenset((12, 14, 16))

# initial pairs (row 0 along +z) and final-state particles with draw weights; the catalog's
# initial states are e-e+, u u~, u d~, d d~ and quark-quark
_INITIAL = [((11, -11), 0.35), ((2, -2), 0.30), ((2, -1), 0.15), ((1, -1), 0.12), ((2, 2), 0.08)]
_FINAL = [(11, 1.0), (-11, 1.0), (13, 1.0), (-13, 1.0), (12, 0.4), (-12, 0.4), (14, 0.4), (-14, 0.4),
          (1, 1.0), (-1, 1.0), (2, 1.0), (-2, 1.0), (3, 0.6), (-3, 0.6), (4, 0.6), (-4, 0.6),
          (5, 0.7), (-5, 0.7), (6, 0.5), (-6, 0.5), (21, 2.5), (22, 1.2), (23, 0.7),
          (24, 0.6), (-24, 0.6), (25, 0.4)]
_N_FINAL_P = {2: 0.3, 3: 0.35, 4: 0.35}
_MAX_FINAL_MASS = 600.0               # keeps the sqrt(s) window [~1.05 threshold, 1000] wide
# propagator particle by the channel's outgoing charge in thirds (massless first)
_PROP = {0: [(22, 0.3), (21, 0.3), (23, 0.2), (25, 0.2)],
         3: [(24, 0.5), (-11, 0.25), (-13, 0.25)], -3: [(-24, 0.5), (11, 0.25), (13, 0.25)],
         2: [(2, 0.6), (6, 0.4)], -2: [(-2, 0.6), (-6, 0.4)],
         -1: [(1, 0.6), (5, 0.4)], 1: [(-1, 0.6), (-5, 0.4)]}
_PROP_HEAVY = {0: [23, 25], 3: [24], -3: [-24], 2: [6], -2: [-6]}   # none for charge 1/3 (b is light)
_WIDTH_RATIO = (1e-3, 0.1)            # Gamma/M, log-uniform per propagator particle
_SCALAR_MASS = (20.0, 700.0)          # random scalar mass (log-uniform), half of the processes
# diagrams per process, at most. N = 7, 8 (2->5, 2->6; reachable only as syn5_<k>, syn6_<k>) added 2026-10-09 for the
# high-multiplicity test (D20), continuing the hand-set rule; every name drawable before draws exactly as before, so
# SYN_VERSION stays 1 and no existing pool changes
_KMAX = {4: 3, 5: 5, 6: 6, 7: 7, 8: 8}
_H = (2, 4)                           # helicity amplitudes summed incoherently (no exact zeros)
_LIGHT = 10.0                         # propagators below this mass (0, b) may enter as 1/sqrt


def is_synthetic(name):
    return bool(_NAME.match(str(name)))


def _props(pdg):
    p = PARTICLE_PROPERTIES.get(int(pdg))
    if p is None:
        raise KeyError(f"PDG {pdg} not in particle_ids.PARTICLE_PROPERTIES")
    return p


def leg_mass(pdg):
    """Generator mass (GeV) of a particle, from the property table (massless sentinel -> 0)."""
    lm = _props(pdg)[_COL["log10_mass_gev"]]
    # rounded back to the generator's literal value (4.7, not 4.700000000000001)
    return 0.0 if abs(lm - _MASSLESS) < 1e-6 else float(f"{10.0 ** lm:.10g}")


def _q3(pdg):
    return int(round(3 * _props(pdg)[_COL["charge"]]))


def _pick(rng, opts):
    items, w = zip(*opts)
    w = np.asarray(w, float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _draw_legs(rng, n):
    """Initial pair + n final particles conserving charge, baryon and lepton number."""
    ini = _pick(rng, _INITIAL)
    qn = lambda pdgs, c: round(sum(_props(p)[_COL[c]] for p in pdgs) * 3)
    order = {p: i for i, (p, _) in enumerate(_FINAL)}
    for _ in range(100000):
        fin = [_pick(rng, _FINAL) for _ in range(n)]
        if all(qn(fin, c) == qn(ini, c) for c in ("charge", "baryon_number", "lepton_number")) \
                and sum(leg_mass(p) for p in fin) <= _MAX_FINAL_MASS:
            return list(ini), sorted(fin, key=lambda p: order[p])   # identical legs adjacent
    raise RuntimeError("no conserving final state drawn")


def _random_tree(rng, N):
    """Random cubic tree over N legs: merge two random clusters until three remain (they
    meet at the root vertex). Returns the merges [(a, b)] over cluster ids (legs 0..N-1,
    merged clusters N, N+1, ...) and each cluster's leg set."""
    legs = {i: frozenset([i]) for i in range(N)}
    alive, merges = list(range(N)), []
    while len(alive) > 3:
        a, b = sorted(rng.choice(len(alive), size=2, replace=False))
        ca, cb = alive[a], alive[b]
        new = N + len(merges)
        merges.append((ca, cb))
        legs[new] = legs[ca] | legs[cb]
        alive = [c for c in alive if c not in (ca, cb)] + [new]
    return merges, legs, alive


def _canonical(S, N):
    """A channel and its complement give the same invariant: keep the side without leg 0."""
    S = frozenset(S)
    return tuple(sorted(frozenset(range(N)) - S if 0 in S else S))


@functools.lru_cache(maxsize=None)
def structure(name):
    """The random structure of synthetic process `name` (a JSON-able dict), drawn from a
    seed fixed by the name and SYN_VERSION."""
    m = _NAME.match(str(name))
    if not m:
        raise KeyError(f"{name!r} is not a synthetic process name (syn_<k> or syn<n>_<k>)")
    n_forced, idx = (int(m.group(1)) if m.group(1) else 0), int(m.group(2))
    rng = np.random.default_rng(np.random.SeedSequence([_SALT, SYN_VERSION, n_forced, idx]))
    n = n_forced or int(_pick(rng, list(_N_FINAL_P.items())))
    ini, fin = _draw_legs(rng, n)
    pdgs = ini + fin
    N = n + 2
    m_legs = [leg_mass(p) for p in pdgs]
    sign = [-1] * 2 + [1] * n                                    # all-outgoing
    cut = [i >= 2 and m_legs[i] == 0.0 and abs(pdgs[i]) not in _NEUTRINOS for i in range(N)]
    guard = [cut[i] or m_legs[i] > 0.0 for i in range(N)]       # keeps an invariant off zero

    def protected(S):
        """A massless or light (M <= _LIGHT) propagator only where the fiducial cuts (or a leg
        mass) keep s_S off 0 (and off a light pole): s-type needs two cut legs or legs
        with masses above _LIGHT (s_S >= 100 GeV^2 either way); t-type (s_S <= 0) a guard on both
        sides of the beams."""
        if 1 not in S:
            return sum(cut[i] for i in S) >= 2 or sum(m_legs[i] for i in S) > _LIGHT
        G = [i for i in S if i >= 2]
        F = [i for i in range(2, N) if i not in S]
        return any(guard[i] for i in G) and any(guard[i] for i in F)

    colored = [_props(p)[_COL["color_casimir"]] > 0 for p in pdgs]
    n_glu = sum(p == 21 for p in fin)
    k_s = min(n, n_glu + int(rng.binomial(n - n_glu, 0.3))) if any(colored) else 0
    # a random scalar mass only without an external H: one PDG must not carry two masses
    m_scalar = float(np.exp(rng.uniform(*np.log(_SCALAR_MASS)))) if rng.random() >= 0.5 else 125.0
    if 25 in fin:
        m_scalar = 125.0
    ratio = {}

    def width(pdg):
        a = abs(pdg)
        if a not in ratio:
            ratio[a] = float(np.exp(rng.uniform(*np.log(_WIDTH_RATIO))))
        return ratio[a]

    diagrams, seen = [], set()
    n_diag = int(rng.integers(1, _KMAX[N] + 1))
    for _ in range(200 * n_diag):
        if len(diagrams) == n_diag:
            break
        merges, legs_of, root = _random_tree(rng, N)
        props, ok = [], True
        for c in range(N, N + len(merges)):
            S = _canonical(legs_of[c], N)
            q = sum(sign[i] * _q3(pdgs[i]) for i in S)
            if q not in _PROP:
                ok = False; break
            pdg = int(_pick(rng, _PROP[q]))
            if leg_mass(pdg) <= _LIGHT and not protected(S):
                if q not in _PROP_HEAVY:            # charge 1/3: no heavy option, redraw the tree
                    ok = False; break
                pdg = int(_PROP_HEAVY[q][int(rng.integers(len(_PROP_HEAVY[q])))])
            M = m_scalar if abs(pdg) == 25 else leg_mass(pdg)
            s_type = 1 not in S
            props.append({"legs": list(S), "pdg": pdg, "mass": M,
                          "width": (width(pdg) * M if (s_type and M > 0) else 0.0),
                          "power": (1.0 if M > _LIGHT else float(rng.choice([0.5, 1.0])))})
        if not ok:
            continue
        key = frozenset((tuple(p["legs"]), p["pdg"]) for p in props)
        if key in seen:
            continue
        seen.add(key)
        qcd = sorted(rng.choice(N - 2, size=k_s, replace=False).tolist())   # QCD vertices
        diagrams.append({"merges": [list(mm) for mm in merges], "root": [int(r) for r in root],
                         "props": props, "qcd_vertices": qcd})
    if not diagrams:
        raise RuntimeError(f"{name}: no diagram with SM-representable channels")

    pairs = list(itertools.combinations(range(N), 2))
    H = int(rng.integers(_H[0], _H[1] + 1))
    numer = []
    for _h in range(H):
        row = []
        for d in diagrams:
            deg = int(rng.integers(0, 3))
            monos = [{"a": float(rng.normal()),
                      "pairs": sorted(int(x) for x in rng.choice(len(pairs), size=deg))}
                     for _ in range(int(rng.integers(1, 4)))]
            e_sum = sum(p["power"] for p in d["props"])
            row.append({"c": [float(rng.normal()), float(rng.normal())], "deg": deg, "monos": monos,
                        # s-power giving the term mass dimension 4 - N: 2 deg - k - 2 e_sum = 4 - N
                        "k": 2 * deg - 2 * e_sum - 4 + N})
        numer.append(row)

    thr = sum(m_legs[2:])
    lo = max(25.0, 1.05 * thr)                                 # the catalog's window rule
    st = {
        "name": name, "version": SYN_VERSION, "pdg_ids": pdgs, "m_legs": m_legs, "nfinal": n,
        "sqrts": [float(np.ceil(lo)), SQRTS_MAX], "k_s": int(k_s),
        "coupling": float((4 * np.pi) ** n * ALPHA_EW ** (n - k_s)),   # times alpha_s(sqrt s)^k_s
        "internal_masses": {25: m_scalar}, "diagrams": diagrams, "pairs": [list(p) for p in pairs],
        "numerators": numer, "perms": _identical_perms(pdgs),
    }
    st["sha"] = hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return st


def _identical_perms(pdgs):
    """Every relabelling of the slots that only exchanges identical particles (initial and
    final kept apart): the group the target is averaged over."""
    N = len(pdgs)
    groups = {}
    for i, p in enumerate(pdgs):
        groups.setdefault((i < 2, p), []).append(i)
    blocks = [g for g in groups.values() if len(g) > 1]
    out = []
    for choice in itertools.product(*[list(itertools.permutations(g)) for g in blocks]):
        perm = list(range(N))
        for g, img in zip(blocks, choice):
            for a, b in zip(g, img):
                perm[a] = b
        out.append(perm)
    return out


def _mink2(v):
    return v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1)


def evaluate(st, P, scale=1.0):
    """|M|^2 of structure `st` at momenta P (B, N, 4) (rows in pdg_ids order, beams incoming).
    `scale` multiplies every dimensionful constant (propagator masses and widths): the
    mass-dimension check is |M|^2(lam P, scale=lam) = lam^(8-2N) |M|^2(P)."""
    P = np.asarray(P, np.float64)
    N = P.shape[1]
    sign = np.array([-1.0, -1.0] + [1.0] * (N - 2))
    pairs = [tuple(p) for p in st["pairs"]]
    total = np.zeros(P.shape[0])
    for perm in st["perms"]:
        q = P[:, perm, :] * sign[None, :, None]
        s_tot = _mink2(q[:, 0] + q[:, 1])
        s_pair = np.stack([_mink2(q[:, i] + q[:, j]) for i, j in pairs], axis=1)
        cache = {}
        dens = []
        for d in st["diagrams"]:
            den = np.ones(P.shape[0], np.complex128)
            for pr in d["props"]:
                S = tuple(pr["legs"])
                if S not in cache:
                    cache[S] = _mink2(q[:, list(S)].sum(1))
                sS = cache[S]
                M, G = pr["mass"] * scale, pr["width"] * scale
                D = sS - M * M + 1j * M * G
                if pr["power"] == 1.0:
                    den = den * D
                elif M > 0:
                    den = den * np.sqrt(D)          # principal branch (cut only where D <= 0, G = 0)
                else:
                    den = den * np.sqrt(np.abs(sS))
            dens.append(den)
        for row in st["numerators"]:
            A = np.zeros(P.shape[0], np.complex128)
            for term, den in zip(row, dens):
                poly = np.zeros(P.shape[0])
                for mono in term["monos"]:
                    t = np.full(P.shape[0], mono["a"])
                    for k in mono["pairs"]:
                        t = t * s_pair[:, k]
                    poly = poly + t
                A = A + complex(*term["c"]) * poly * s_tot ** (-0.5 * term["k"]) / den
            total = total + np.abs(A) ** 2
    # alpha_s runs per event at mu = sqrt(s) as in the real pools (mg5_pipeline_final.compute_alphas,
    # one loop from M_Z, nf = 5); M_Z scales with the other constants
    b0 = (11 * 3 - 2 * 5) / (12 * np.pi)
    s_in = _mink2(P[:, 0] + P[:, 1])
    a_s = ALPHA_S / (1 + ALPHA_S * b0 * np.log(s_in / (MZ_REF * scale) ** 2))
    return st["coupling"] * a_s ** st["k_s"] * total / len(st["perms"])


def label_events(name, events):
    """Labeller for the generation path: events = [(momenta (N,4), pdg)], returns |M|^2."""
    return evaluate(structure(name), np.stack([e[0] for e in events]))


def catalog_entry(name):
    """mg5_pipeline_final.PROCESSES entry of a synthetic process."""
    st = structure(name)
    cfg = {
        "kind": "synthetic", "nfinal": st["nfinal"], "pdg_ids": list(st["pdg_ids"]),
        "m_finals": list(st["m_legs"][2:]), "param_card_patches": {},
        "alphas_power": st["k_s"], "order": [0, 0, st["k_s"], st["nfinal"] - st["k_s"]],
        "internal_masses": dict(st["internal_masses"]), "gen_weight": 1.0,
        "synthetic": {"version": SYN_VERSION, "sha": st["sha"]}, "syn_name": name, "sqrts": list(st["sqrts"]),
        "mg5_generate": [],
    }
    if st["nfinal"] == 2:
        cfg["m_final"] = list(st["m_legs"][2:])
    return cfg


def sidecar(name):
    """The diagram sidecar payload (tools/dump_diagrams.py format) of a synthetic process:
    one subprocess, legs numbered by slot (1, 2 initial), every diagram and its images under
    the identical-leg permutations."""
    st = structure(name)
    pdgs, N = st["pdg_ids"], len(st["pdg_ids"])
    external = [{"number": i + 1, "pdg": int(pdgs[i]), "state": "in" if i < 2 else "out"} for i in range(N)]
    graphs, seen = [], set()
    for perm in st["perms"]:
        for d in st["diagrams"]:
            key = frozenset((tuple(sorted(perm[i] for i in p["legs"])), p["pdg"]) for p in d["props"])
            if key in seen:
                continue
            seen.add(key)
            nodes = [{"kind": "external", "interaction_id": 0, "leg_number": perm[i] + 1,
                      "pdg": int(pdgs[i]), "state": "in" if i < 2 else "out"} for i in range(N)]
            vert = {}                                     # cluster id -> vertex node creating it
            n_vertex = len(d["merges"]) + 1
            orders = [{"QCD": 1} if v in d["qcd_vertices"] else {"QED": 1} for v in range(n_vertex)]
            for v in range(n_vertex):
                nodes.append({"kind": "vertex", "interaction_id": -1, "orders": orders[v]})
            edges = []
            consumer = {}
            for v, (a, b) in enumerate(d["merges"]):
                vert[N + v] = N + v
                consumer[a] = consumer[b] = N + v
            for c in d["root"]:
                consumer[c] = N + len(d["merges"])
            for i in range(N):
                edges.append({"u": i, "v": consumer[i], "pdg": int(pdgs[i]), "external": True,
                              "leg_number": perm[i] + 1})
            for k, p in enumerate(d["props"]):
                c = N + k
                edges.append({"u": vert[c], "v": consumer[c], "pdg": int(p["pdg"]), "external": False,
                              "leg_number": 0})
            graphs.append({"nodes": nodes, "edges": edges, "diagram_type": "tree"})
    tag = " ".join(str(p) for p in pdgs[:2]) + " > " + " ".join(str(p) for p in pdgs[2:])
    return {"process": name, "model": f"synthetic_v{SYN_VERSION}", "mg5_generate": [],
            "synthetic_sha": st["sha"], "n_subprocesses": 1,
            "subprocesses": [{"process_str": f"synthetic {tag}", "external": external,
                              "n_diagrams": len(graphs), "diagrams": graphs}]}


def write_sidecar(name, out_dir=None):
    """Write ``<out_dir>/<name>.diagrams.json`` unless an up-to-date one is there (same
    structure hash); atomic, so concurrent trials cannot read a half-written file."""
    out_dir = out_dir or DEFAULT_DIAGRAMS_DIR
    path = os.path.join(out_dir, f"{name}.diagrams.json")
    sha = structure(name)["sha"]
    try:
        with open(path) as f:
            if json.load(f).get("synthetic_sha") == sha:
                return path
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(sidecar(name), f)
    os.replace(tmp, path)
    return path


def recipe_entries(names, n_train, n_val, n_test):
    return [{"name": nm, "sqrts": [int(structure(nm)["sqrts"][0]), int(structure(nm)["sqrts"][1])],
             "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test)} for nm in names]


def _describe(name):
    st = structure(name)
    print(f"{name}  sha={st['sha']}  {st['pdg_ids'][:2]} -> {st['pdg_ids'][2:]}  m={st['m_legs'][2:]}")
    print(f"  sqrt(s) {st['sqrts']}  k_s={st['k_s']}  g={st['coupling']:.3e}  H={len(st['numerators'])}"
          f"  perms={len(st['perms'])}  scalar mass={st['internal_masses'][25]:.1f}")
    for d in st["diagrams"]:
        print("  diagram: " + ", ".join(
            f"{p['pdg']}{p['legs']}(M={p['mass']:.1f},G={p['width']:.3g},e={p['power']})" for p in d["props"]))


def main():
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--describe", nargs="*", default=[])
    ap.add_argument("--recipe", help="write a recipe YAML listing syn_<first>..syn_<first+count-1>")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--sidecars", help="write the diagram sidecars of every syn* process of a recipe")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    for nm in a.describe:
        _describe(nm)
    if a.recipe:
        names = [f"syn_{k:05d}" for k in range(a.first, a.first + a.count)]
        doc = {"sampling": {"mode": "mixture"},
               "processes": recipe_entries(names, a.n_train, a.n_val, a.n_test)}
        head = (f"# Synthetic-amplitude prior: {a.count} synthetic processes syn_{a.first:05d}.. "
                f"(tools/synthetic_amplitudes.py, v{SYN_VERSION}),\n# mixture sampling as the ladder "
                f"recipes; sqrt(s) windows from each structure (max(25, 1.05 threshold), the catalog's rule).\n")
        with open(a.recipe, "w") as f:
            f.write(head)
            yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False, width=200)
        print(f"wrote {a.recipe} ({a.count} processes)")
    if a.sidecars:
        with open(a.sidecars) as f:
            doc = yaml.safe_load(f)
        procs = doc["processes"] if isinstance(doc, dict) else doc
        for p in procs:
            if is_synthetic(p["name"]):
                print(write_sidecar(p["name"], a.out_dir))


if __name__ == "__main__":
    main()
