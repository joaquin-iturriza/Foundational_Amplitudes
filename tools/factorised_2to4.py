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
    x = (sab - sak - sbk) / sab
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


def me2(proc, mom, alphas):
    cfg = mg.PROCESSES[proc]
    sa = f"{mg.WORK_DIR}/{proc}_standalone"
    backend, _, driver, _ = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and driver, f"no compiled C++ backend at {sa}"
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    with mg.CppDriverPipe(driver, sa) as pipe:
        return np.asarray(pipe.compute([(m, np.asarray(cfg["pdg_ids"])) for m in mom], perm=perm, alphas=alphas), float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=os.path.join(ROOT, "recipes", "catalog_2to4_heldout_orders.yaml"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-check", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "zz_factorised_2to4"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    probes = [e["name"] for e in yaml.safe_load(open(a.recipe))["processes"]]
    for p in probes:
        if a.only and p not in a.only:
            continue
        cfg = mg.PROCESSES[p]
        parent = cfg["parent"]
        pcfg = mg.PROCESSES[parent]
        ids4, ids3 = [int(i) for i in cfg["pdg_ids"]], [int(i) for i in pcfg["pdg_ids"]]
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
        Pm, x, a_emits = ii_map(P, k_slot)
        t3 = me2(parent, Pm, alphas)
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
        Xp, alp, _ = pool(parent)
        Xp = Xp[: a.n_check]
        Pp = Xp[:, : 4 * len(ids3)].reshape(-1, len(ids3), 4).astype(np.float64)
        lr = np.log(np.abs(me2(parent, Pp, None if alp is None else np.asarray(alp[: a.n_check], np.float64)))
                    / np.abs(Xp[:, -1]))
        np.savez_compressed(os.path.join(a.out, p + ".npz"), t4=t4, t3=t3, feats=feats, emitter=emitter,
                            a_emits=a_emits, x=x)
        ok = (t3 > 0) & np.isfinite(t3)
        r = np.log(t4[ok]) - np.log(t3[ok])
        rec = dict(probe=p, parent=parent, extra=extra, n=int(n), t3_bad=int((~ok).sum()), pool=os.path.basename(f4),
                   var_log_t4=float(np.var(np.log(t4))), var_r=float(np.var(r)), mean_r=float(np.mean(r)),
                   check_median_log_ratio=float(np.median(lr)), check_max_abs_log_ratio=float(np.abs(lr).max()),
                   has_alphas=al is not None)
        print("F24 " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
