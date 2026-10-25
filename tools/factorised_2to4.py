"""The level item's design (ii) (docs/results.tex, the level's hand-off item), its CPU check on the data alone: does a
2->4 probe's |M|^2 factorise onto its 2->3 parent's? Each 2->4 probe (recipes/catalog_2to4_heldout_orders.yaml) is a
catalog 2->3 training process (its "parent") plus one gluon or photon, same beams. Per probe, on its first --n
validation events:
  - the extra particle k: among the final particles of its flavour, the one with the lowest pT (|M|^2 is symmetric
    under exchanging them, so any one may be removed; the softest is the one the factorisation holds best for);
  - the event mapped onto the parent's phase space by the Catani-Seymour initial-initial map (hep-ph/9605323 eq. 5.137
    onwards): the emitting beam a is the one k is closer to (smaller p_a.k), p~_a = x p_a with
    x = (p_a.p_b - p_a.k - p_b.k) / p_a.p_b, every final momentum Lorentz-transformed from K = p_a + p_b - k to
    K~ = p~_a + p_b, so masses and momentum conservation hold for any final state;
  - the parent's exact |M|^2 at the mapped event, from its compiled standalone (mg5amcnlo/<parent>_standalone, the C++
    driver, rows mapped to MadGraph slots by row_to_slot_perm), at the probe event's own alpha_s (the stored
    <pool>.alphas.npy), so the ratio carries one extra alpha_s or alpha and no scale mismatch;
  - the splitting variables: s_ak, s_bk, the smallest s_jk over the other finals (the nearest final emitter, by
    2 p_j.k), pT_k, y_k, 1 - x, all over s_hat; the nearest emitter's PDG id.
Also, as a check of the backend, the parent's own validation pool relabelled on --n-check events (the median and
largest |log ratio| against its stored |M|^2, at its stored alpha_s).
Writes <out>/<probe>.npz (t4, t3, features, emitter) and prints one F24 JSON line per probe. CPU.
    python tools/factorised_2to4.py [--recipe recipes/catalog_2to4_heldout_orders.yaml] [--n 2000] [--out DIR]
"""
import argparse, glob, json, os, sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402


def mdot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def pool(name, role="val"):
    g = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{name}_*_{role}_*amplitudes.npy")))
    g = [f for f in g if os.path.basename(f)[len(name) + 1].isdigit()]    # not a sibling such as <name>_nlo_...
    if len(g) != 1:
        raise SystemExit(f"{len(g)} {role} pools for {name} in $DATA_DIR: {g}")
    X = np.asarray(np.load(g[0], mmap_mode="r"))
    al = g[0][:-4] + ".alphas.npy"
    return X, (np.load(al) if os.path.exists(al) else None), g[0]


def ii_map(P, k_slot):
    """the CS initial-initial map: P (n, N, 4) the 2->4 events, k_slot the removed final row per event -> (n, N-1, 4)"""
    n, N, _ = P.shape
    k = P[np.arange(n), k_slot]
    pa, pb = P[:, 0], P[:, 1]
    sab, sak, sbk = mdot(pa, pb), mdot(pa, k), mdot(pb, k)
    # x = K^2 / s with K = p_a + p_b - k (massless beams): (s - 2 p_a.k - 2 p_b.k + m_k^2) / s, so a massive emission
    # (Z, H) maps as well as a gluon or photon
    x = (sab - sak - sbk + 0.5 * mdot(k, k)) / sab
    a_emits = sak <= sbk
    pa_t = np.where(a_emits[:, None], x[:, None] * pa, pa)
    pb_t = np.where(a_emits[:, None], pb, x[:, None] * pb)
    K = pa + pb - k
    Kt = pa_t + pb_t
    KK = K + Kt
    keep = np.ones((n, N), bool)
    keep[np.arange(n), k_slot] = False
    fin = P[keep].reshape(n, N - 1, 4)[:, 2:]
    fin_t = (fin - 2 * (mdot(fin, KK[:, None]) / mdot(KK, KK)[:, None])[..., None] * KK[:, None]
             + 2 * (mdot(fin, K[:, None]) / mdot(K, K)[:, None])[..., None] * Kt[:, None])
    return np.concatenate([pa_t[:, None], pb_t[:, None], fin_t], 1), x, a_emits


def beam_recoil_map(P, k_slot):
    """for massive beams (a b quark), where the CS map's rescaled beam would leave its mass shell: the other finals are
    kept as they are and the beams absorb k's recoil, p~_a + p~_b = K = p_a + p_b - k, each on its own mass shell (read
    from the event), back to back in K's rest frame along the direction of p_a there. Momentum is conserved and every
    particle is on shell; an event with K^2 < (m_a + m_b)^2 has no such beams and comes back NaN (dropped and counted
    downstream)."""
    n, N, _ = P.shape
    k = P[np.arange(n), k_slot]
    pa, pb = P[:, 0], P[:, 1]
    m_a, m_b = np.sqrt(np.maximum(mdot(pa, pa), 0)), np.sqrt(np.maximum(mdot(pb, pb), 0))
    K = pa + pb - k
    M = np.sqrt(np.maximum(mdot(K, K), 0))
    beta = K[:, 1:] / K[:, :1]

    def boost(p, b):                                         # p boosted by velocity b
        b2 = (b * b).sum(-1, keepdims=True)
        g = 1 / np.sqrt(1 - b2)
        bp = (b * p[:, 1:]).sum(-1, keepdims=True)
        return np.concatenate([g * (p[:, :1] + bp),
                               p[:, 1:] + ((g - 1) * bp / np.where(b2 > 0, b2, 1) + g * p[:, :1]) * b], 1)
    pa_rest = boost(pa, -beta)
    u = pa_rest[:, 1:] / np.linalg.norm(pa_rest[:, 1:], axis=1, keepdims=True)
    Ea = (M ** 2 + m_a ** 2 - m_b ** 2) / (2 * M)
    with np.errstate(invalid="ignore"):
        q = np.sqrt(Ea ** 2 - m_a ** 2)                       # NaN below threshold
    a_t = boost(np.concatenate([Ea[:, None], q[:, None] * u], 1), beta)
    b_t = boost(np.concatenate([(M - Ea)[:, None], -q[:, None] * u], 1), beta)
    keep = np.ones((n, N), bool)
    keep[np.arange(n), k_slot] = False
    fin = P[keep].reshape(n, N - 1, 4)[:, 2:]
    x = mdot(K, K) / mdot(pa + pb, pa + pb)
    return np.concatenate([a_t[:, None], b_t[:, None], fin], 1), x, mdot(pa, k) <= mdot(pb, k)


def me2(proc, mom, alphas):
    cfg = mg.PROCESSES[proc]
    sa = f"{mg.WORK_DIR}/{proc}_standalone"
    backend, _, driver, _ = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and driver, f"no compiled C++ backend at {sa}"
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    with mg.CppDriverPipe(driver, sa) as pipe:
        return np.asarray(pipe.compute([(m, np.asarray(cfg["pdg_ids"])) for m in mom], perm=perm, alphas=alphas), float)


V3_EXTRA = {"g": 21, "a": 22, "z": 23, "h": 25}


def v3_probes():
    """Every catalog_v3 2->4 class (with pools) built as a 2->3 class plus one g, a, Z or H (the census's `why` =
    '<parent>+<x>'). Massless beams take the CS map, b beams the beam-recoil map; a parent with no compiled
    standalone is compiled (datagen.ensure_backend). A class whose parent is not in the catalog is listed (V3_SKIP). Returns [(probe, parent, extra pdg)]."""
    import json
    gen = set(json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"])   # classes with pools
    out = []
    for kind, pdg in V3_EXTRA.items():
        c = []
        for n, cfg in sorted(mg.PROCESSES.items()):
            why = str(cfg.get("why", ""))
            if n not in gen or cfg.get("nfinal") != 4 or not why.endswith("+" + kind):
                continue
            par = why[: -len(kind) - 1]
            why_skip = "parent not in the catalog" if par not in mg.PROCESSES else None
            if why_skip:
                print("V3_SKIP " + json.dumps({"probe": n, "parent": par, "kind": kind, "why": why_skip}), flush=True)
                continue
            c.append((n, par, pdg))
        print(f"V3_KIND {kind}: {len(c)} read", flush=True)
        out += c
    return out


def to_parent_order(Pm, ids_kept, ids_par):
    """rows of the mapped events (beams, then the kept finals in the probe's order) permuted to the parent's PDG order"""
    fin_kept, fin_par = list(ids_kept[2:]), list(ids_par[2:])
    used, perm = set(), [0, 1]
    for q in fin_par:
        j = next(j for j, r in enumerate(fin_kept) if r == q and j not in used)
        used.add(j); perm.append(2 + j)
    assert list(ids_kept[:2]) == list(ids_par[:2]), (ids_kept, ids_par)
    return Pm[:, perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=os.path.join(ROOT, "recipes", "catalog_2to4_heldout_orders.yaml"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-check", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "zz_factorised_2to4"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--v3", action="store_true",
                    help="every catalog_v3 2->4 class built as a 2->3 class plus g, a, Z or H (the skipped listed)")
    ap.add_argument("--skip-done", action="store_true", help="skip a probe whose npz is already in --out (a rerun)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.v3:
        jobs = v3_probes()
    else:
        jobs = [(e["name"], mg.PROCESSES[e["name"]]["parent"], None) for e in yaml.safe_load(open(a.recipe))["processes"]]
    for p, parent, extra in jobs:
        if a.only and p not in a.only:
            continue
        if a.skip_done and os.path.exists(os.path.join(a.out, p + ".npz")):
            continue
        cfg = mg.PROCESSES[p]
        pcfg = mg.PROCESSES[parent]
        ids4, ids3 = [int(i) for i in cfg["pdg_ids"]], [int(i) for i in pcfg["pdg_ids"]]
        if extra is None:
            extra = ids4[-1]
            assert ids4[:-1] == ids3 and extra in (21, 22), (p, ids4, ids3)
        X, al, f4 = pool(p)
        X = X[: a.n]
        N = len(ids4)
        P = X[:, : 4 * N].reshape(-1, N, 4).astype(np.float64)
        t4 = X[:, -1].astype(np.float64)
        alphas = None if al is None else np.asarray(al[: a.n], np.float64)
        cand = [j for j in range(2, N) if ids4[j] == extra]
        pt = np.sqrt(P[:, cand, 1] ** 2 + P[:, cand, 2] ** 2)
        k_slot = np.asarray(cand)[pt.argmin(1)]
        b_beam = any(abs(q) == 5 for q in ids4[:2])
        Pm, x, a_emits = (beam_recoil_map if b_beam else ii_map)(P, k_slot)
        if a.v3 and not os.path.isdir(f"{mg.WORK_DIR}/{parent}_standalone"):
            import datagen
            datagen.ensure_backend(parent)
        if a.v3:                                     # the kept finals in the parent's PDG order
            Pm = np.stack([to_parent_order(Pm[i:i + 1], [ids4[j] for j in range(N) if j != int(k_slot[i])], ids3)[0]
                           for i in range(len(Pm))])
        # events the map cannot place (the beam-recoil map below threshold) never reach the driver, which dies on NaN
        fin = np.all(np.isfinite(Pm.reshape(len(Pm), -1)), 1)
        t3 = np.full(len(Pm), np.nan)
        try:
            if fin.any():
                t3[fin] = me2(parent, Pm[fin], None if alphas is None else alphas[fin])
        except Exception as e:                       # listed, never dropped silently
            print("F24 " + json.dumps({"probe": p, "parent": parent, "error": repr(e)[:300]}), flush=True)
            continue
        n = len(P)
        k = P[np.arange(n), k_slot]
        s = mdot(P[:, 0] + P[:, 1], P[:, 0] + P[:, 1])
        sjk = np.stack([2 * mdot(P[:, j], k) for j in range(2, N)], 1)
        sjk[np.arange(n), k_slot - 2] = np.inf
        near = sjk.argmin(1)
        ptk = np.sqrt(k[:, 1] ** 2 + k[:, 2] ** 2)
        yk = 0.5 * np.log((k[:, 0] + k[:, 3]) / np.maximum(k[:, 0] - k[:, 3], 1e-300))
        feats = np.stack([np.log(2 * mdot(P[:, 0], k) / s), np.log(2 * mdot(P[:, 1], k) / s),
                          np.log(sjk.min(1) / s), np.log(ptk ** 2 / s), yk, np.log(np.maximum(1 - x, 1e-300)),
                          np.log(s)], 1)
        emitter = np.asarray(ids4[2:])[near]
        # the backend check: the parent's own pool relabelled at its stored alpha_s
        # (a parent compiled here for the check has no pool: no backend check, recorded as null)
        try:
            Xp, alp, _ = pool(parent)
        except SystemExit:
            Xp = None
        if Xp is not None:
            Xp = Xp[: a.n_check]
            Pp = Xp[:, : 4 * len(ids3)].reshape(-1, len(ids3), 4).astype(np.float64)
            lr = np.log(np.abs(me2(parent, Pp, None if alp is None else np.asarray(alp[: a.n_check], np.float64)))
                        / np.abs(Xp[:, -1]))
        np.savez_compressed(os.path.join(a.out, p + ".npz"), t4=t4, t3=t3, feats=feats, emitter=emitter,
                            a_emits=a_emits, x=x, b_beam=b_beam)
        ok = (t3 > 0) & np.isfinite(t3)
        r = np.log(t4[ok]) - np.log(t3[ok])
        rec = dict(unmapped=int((~fin).sum()), probe=p, parent=parent, extra=extra, map="beam_recoil" if b_beam else "cs_ii", n=int(n), t3_bad=int((~ok).sum()), pool=os.path.basename(f4),
                   var_log_t4=float(np.var(np.log(t4))), var_r=float(np.var(r)), mean_r=float(np.mean(r)),
                   check_median_log_ratio=None if Xp is None else float(np.median(lr)),
                   check_max_abs_log_ratio=None if Xp is None else float(np.abs(lr).max()),
                   has_alphas=al is not None)
        print("F24 " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
