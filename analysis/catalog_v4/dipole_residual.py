"""The per-event soft-dipole factor against the network's zero-shot residual (plan v3_prep_explore, the dipole design).

Per probe read (tools/zero_shot_parent.py .npz from @902b96c5 on, which saves the scored events' pool indices `idx` and
the pool path `pool`; older reads carry neither and are refused: their events are neither the pool's first rows nor
matchable by value, 2026-10-31). The read's t is the control's target (log|M|^2 with the emission scales and couplings
taken out) and t - logF its log|M|^2 up to a per-process constant; the alignment is checked: over the scored events,
log|M|^2 of the pool rows at idx against t - logF must have Pearson r >= 0.999 and a spread about their median
difference below 0.05, else the probe is refused and counted:
  u   the residual (q - t) minus its mean over the probe's scored events
  D   per event, the sum over each final gluon (photon) j of the colour-blind antenna sum over pairs (a, b) of the
      other coloured (charged) legs, s_ab / (s_aj s_bj), s_xy = 2 p_x.p_y with incoming momenta reversed, GeV^-2;
      a function of the momenta alone, no fit, the same for every process
  E   the placebo: log of the summed energy of the same legs (a per-event scale with no soft structure)
  logF the read's per-event target factor (the target is t, log|M|^2 is t - logF): the second placebo
  r2_D, r2_E, r2_F   the squared Pearson correlation of u with log D, E and logF
  r2_tD, r2_rawD     the gate: r^2 of the control's target t, and of log|M|^2 (t - logF), on log D (no network)
A probe with no final gluon or photon has no D and is listed apart. Runs where the pools live (CC-IN2P3), on the reads'
directories; writes one JSON line per probe and a summary.
    python3 analysis/catalog_v4/dipole_residual.py <reads dir> [<reads dir> ...] --out <json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.common.readers import split_pool  # noqa: E402

QUARKS = {1, 2, 3, 4, 5, 6}
CHARGE = {1: -1 / 3, 2: 2 / 3, 3: -1 / 3, 4: 2 / 3, 5: -1 / 3, 6: 2 / 3, 11: -1, 13: -1, 15: -1, 24: 1}


def coloured(pid):
    a = abs(int(pid))
    return a in QUARKS or a == 21


def charged(pid):
    return CHARGE.get(abs(int(pid)), 0) != 0


def mdot(p, q):
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1)


def factors(mom, pdg, n_in=2):
    """log D and E per event; None when the process has no final gluon or photon."""
    mom = mom.copy()
    mom[:, :n_in] *= -1                                   # all outgoing
    ids = pdg[0]
    emitters = [j for j in range(n_in, len(ids)) if abs(ids[j]) in (21, 22)]
    if not emitters:
        return None
    D = np.zeros(len(mom))
    legs_all = set()
    for j in emitters:
        partner = coloured if abs(ids[j]) == 21 else charged
        legs = [a for a in range(len(ids)) if a != j and partner(ids[a])]
        legs_all.update(legs + [j])
        for x in range(len(legs)):
            for y in range(x + 1, len(legs)):
                a, b = legs[x], legs[y]
                sab = 2 * mdot(mom[:, a], mom[:, b])
                saj = 2 * mdot(mom[:, a], mom[:, j])
                sbj = 2 * mdot(mom[:, b], mom[:, j])
                D += np.abs(sab / (saj * sbj))
    E = np.log(np.abs(mom[:, sorted(legs_all), 0]).sum(1))
    return np.log(np.maximum(D, 1e-300)), E


def pool_rows(z):
    """The scored events' rows in the read's pool file. A read saves `pool_row` (zero_shot_parent from the dipole
    repair on); before that its `idx` are positions in the experiment's role-contiguous array (train|val|test, each
    role shuffled by one rng seeded 42, experiment._init_data: _role_perm), not pool rows (the first job,
    FA-ccin2p3-3a2a388b, failed on idx 10000 in a 10000-row pool). Rebuilt here from the role counts: the read scores its
    whole role, so n_role = len(idx), the role starts at min(idx), and a test read's val role is taken of the same size;
    a wrong guess is caught by the alignment check below."""
    if "pool_row" in z.files and len(z["pool_row"]):
        return np.asarray(z["pool_row"])
    idx = np.asarray(z["idx"]).astype(np.int64)
    n, start = len(idx), int(idx.min())
    if str(z["split"]) == "val":
        counts = [start, n]
    else:
        counts = [start - n, n, n]
    if counts[0] < 0:
        return None
    rng = np.random.default_rng(seed=42)
    perm = [rng.permutation(c) for c in counts][-1]
    pos = idx - start
    if pos.max() >= n:
        return None
    return perm[pos]


# catalog_v4's masses and widths (fa-catalog-reference: MT 172.5, WT 1.4915; the loop-induced pools' corrected Z, W;
# b massless in v4, so a b line is a massless propagator; WH the sm card's 6.382339e-03); alpha at the G_mu scheme's 1/132.5, alpha_s(M_Z) 0.118
MW = {23: (91.188, 2.441404), 24: (80.419, 2.0476), 25: (125.0, 6.382339e-03), 6: (172.5, 1.4915)}
LOG_G2 = {"QCD": np.log(4 * np.pi * 0.118), "QED": np.log(4 * np.pi / 132.5)}


def edge_sides(diag):  # dup-ok: norm_surrogate.py parses its arguments at import, so it cannot be imported
    """Per internal edge of a tree diagram: (|pdg|, the external leg numbers on one side of it)."""
    nodes, edges = diag["nodes"], diag["edges"]
    adj = {i: [] for i in range(len(nodes))}
    for k, e in enumerate(edges):
        adj[e["u"]].append((e["v"], k))
        adj[e["v"]].append((e["u"], k))
    out = []
    for k, e in enumerate(edges):
        if e["external"]:
            continue
        seen, stack, legs = {e["u"]}, [e["u"]], []
        while stack:
            x = stack.pop()
            if nodes[x]["kind"] == "external":
                legs.append(nodes[x]["leg_number"])
            for y, kk in adj[x]:
                if kk != k and y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append((abs(int(e["pdg"])), legs))
    return out


def log_prop(mom, pdg, sidecar):
    """log P per event: log sum_d C_d prod_{internal S} 1/|D_S|^k over the sidecar's tree diagrams (norm_surrogate.py's
    prop, the process constant left out): D_S = s_S - M^2 + i M Gamma, k = 2 for a massive line (Breit-Wigner), 1 for a
    massless one (|s_S|), s_S from the legs on one side of the edge (all outgoing); C_d = prod over the diagram's
    vertices of g^2 per coupling power (g^2 = 4 pi alpha_s or 4 pi alpha). None when the sidecar is missing or has no
    subprocess with the pool's legs."""
    if not os.path.exists(sidecar):
        return None
    sd = json.load(open(sidecar))
    ids = [int(x) for x in pdg[0]]
    n = len(ids)
    perm, sub = None, None
    for s_ in sd["subprocesses"]:
        want = [int(l["pdg"]) for l in sorted(s_["external"], key=lambda l: l["number"])]
        if want == ids:
            perm, sub = list(range(n)), s_
            break
        if sorted(want[:2]) == sorted(ids[:2]) and sorted(want[2:]) == sorted(ids[2:]):
            pm, used = [], set()
            for w in want:
                k = next(k for k in range(n) if ids[k] == w and k not in used and (k < 2) == (len(pm) < 2))
                pm.append(k)
                used.add(k)
            perm, sub = pm, s_
            break
    if sub is None:
        return None
    Q = mom[:, perm].copy()
    Q[:, :2] *= -1
    shat = mdot(mom[:, 0] + mom[:, 1], mom[:, 0] + mom[:, 1])
    diags = [d for d in sub["diagrams"] if d.get("diagram_type", "tree") == "tree"]
    logc = np.array([sum(LOG_G2.get(o, 0.0) * v for nd in d["nodes"] for o, v in (nd.get("orders") or {}).items())
                     for d in diags])
    if not len(diags):
        return None
    tot = np.zeros(len(mom))
    for d, lc in zip(diags, logc):
        term = np.full(len(mom), np.exp(lc - logc.max()))
        for pid, legs in edge_sides(d):
            k = Q[:, [l - 1 for l in legs]].sum(1)
            sS = mdot(k, k)
            if pid in MW:
                M, G = MW[pid]
                term = term / ((sS - M * M) ** 2 + (M * G) ** 2)
            else:
                term = term / np.maximum(np.abs(sS), 1e-12 * shat)
        tot += term
    return np.log(np.maximum(tot, 1e-300))


def eta2(u, x, bins=20):
    """The share of var(u) explained by u's mean in `bins` equal-count bins of x (a flexible, monotone-free fit)."""
    if np.std(x) == 0 or np.var(u) == 0:
        return 0.0
    b = np.minimum((np.argsort(np.argsort(x)) * bins) // len(x), bins - 1)
    m = np.bincount(b, u, bins) / np.maximum(np.bincount(b, None, bins), 1)
    return float(np.var(m[b]) / np.var(u))


def tail_share(u, x):
    """(share of sum u^2 in the bottom decile of x, in the top decile); 0.10 each when the error is flat in x."""
    lo, hi = np.quantile(x, [0.1, 0.9])
    tot = float((u ** 2).sum()) or 1.0
    return float((u[x <= lo] ** 2).sum() / tot), float((u[x >= hi] ** 2).sum() / tot)


def joint_features(mom, t, logF, logP, logD):
    """The event's full kinematics with the candidate factors: log s_ij/s_hat per leg pair (all outgoing), log s_hat,
    each final leg's log energy over sqrt(s_hat) and |cos theta| in the partonic frame, logF, log P, log D (0 where
    the process has none), the control's target t."""
    Q = mom.copy()
    Q[:, :2] *= -1
    n = mom.shape[1]
    tot = mom[:, 0] + mom[:, 1]
    shat = mdot(tot, tot)
    cols = [np.log(np.maximum(np.abs(mdot(Q[:, i], Q[:, j])) * 2 / shat, 1e-300))
            for i in range(n) for j in range(i + 1, n)]
    cols.append(np.log(shat))
    for i in range(2, n):
        cols.append(np.log(np.maximum(mom[:, i, 0], 1e-300) / np.sqrt(shat)))
        cols.append(np.abs(mom[:, i, 3]) / np.maximum(np.linalg.norm(mom[:, i, 1:], axis=1), 1e-300))
    cols += [logF, logP, logD if logD is not None else np.zeros(len(t)), t]
    return np.stack(cols, 1)


# the leave-class-out read (design of 2026-10-31, after the joint read's dead end): leg-order-free features, column
# groups for the drop-group read; species counted over the final legs
SPECIES = (("g", {21}), ("a", {22}), ("q", {1, 2, 3, 4}), ("b", {5}), ("l", {11, 13, 15}), ("nu", {12, 14, 16}),
           ("W", {24}), ("Z", {23}), ("H", {25}), ("t", {6}))
FACTORS = ("t", "logF", "logP", "logD")


def lof_features(mom, pdg, t, logF, logP, logD, n_in=2):
    """(X, names): the sorted log s_ij/s_hat over all leg pairs (incoming reversed), the sorted final legs'
    log E/sqrt(s_hat) and |cos theta| (partonic frame), log s_hat, the final legs' species counts, and the factors
    t, logF, log P, log D (0 where none). No column depends on the order the legs are listed in."""
    Q = mom.copy()
    Q[:, :n_in] *= -1
    n = mom.shape[1]
    tot = mom[:, 0] + mom[:, 1]
    shat = mdot(tot, tot)
    sij = np.stack([np.log(np.maximum(np.abs(mdot(Q[:, i], Q[:, j])) * 2 / shat, 1e-300))
                    for i in range(n) for j in range(i + 1, n)], 1)
    fin = mom[:, n_in:]
    lE = np.log(np.maximum(fin[..., 0], 1e-300) / np.sqrt(shat)[:, None])
    cth = np.abs(fin[..., 3]) / np.maximum(np.linalg.norm(fin[..., 1:], axis=-1), 1e-300)
    ids = np.abs(pdg[0, n_in:])
    counts = np.array([sum(int(i in s) for i in ids) for _, s in SPECIES], dtype=np.float64)
    cols = [np.sort(sij, 1), np.sort(lE, 1), np.sort(cth, 1), np.log(shat)[:, None], np.tile(counts, (len(t), 1)),
            np.stack([t, logF, logP, logD if logD is not None else np.zeros(len(t))], 1)]
    names = ([f"sij{k}" for k in range(sij.shape[1])] + [f"lE{k}" for k in range(lE.shape[1])]
             + [f"cth{k}" for k in range(cth.shape[1])] + ["lshat"] + [f"n_{k}" for k, _ in SPECIES] + list(FACTORS))
    return np.concatenate(cols, 1), names


def own_ceiling(X, u, seed=0):
    """The per-pool own-label ceiling: out-of-fold R^2 of u on X in 5 folds of the pool's own events (K = 4/5 of n),
    the map's settings (300 iterations, defaults)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    p = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, random_state=seed), X, u,
                          cv=KFold(5, shuffle=True, random_state=seed))
    return 1 - float(np.var(u - p) / np.var(u))


def joint_r2(X, u, seed=0):
    """(out-of-fold R^2 of a gradient-boosted fit of u on X, the same on u permuted): 5 folds, 100 iterations."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    cv = KFold(5, shuffle=True, random_state=seed)
    fit = lambda y: 1 - float(np.var(y - cross_val_predict(HistGradientBoostingRegressor(max_iter=100, random_state=seed),
                                                              X, y, cv=cv)) / np.var(y))
    return fit(u), fit(np.random.default_rng(seed).permutation(u))


def r2(u, x):
    if np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(u, x)[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="read directories; restrict to cells with --probes")
    ap.add_argument("--probes", default="", help="a file of probe names, one per line (empty: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prop", action="store_true", help="the propagator-product read (log P; placebos E, logF, log D)")
    ap.add_argument("--joint", action="store_true", help="with --prop: the joint out-of-fold fit of u (R2_joint, R2_null)")
    ap.add_argument("--dump", default="", help="with --prop: the leave-class-out read's stage 1: per probe, 500 events "
                    "(seed 0) of lof_features and u to <dump>/<probe>.npz")
    ap.add_argument("--ceiling", action="store_true", help="with --dump: the own-label ceiling on all the probe's events")
    ap.add_argument("--exclude", nargs="*", default=[], help="with --ceiling: feature columns left out of the ceiling's "
                    "fit (t: the scored event's own target, a label; the t-free re-read of 2026-10-31)")
    ap.add_argument("--sidecars", default="data/diagrams", help="the diagram sidecars' directory (--prop)")
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        keep = set(open(a.probes).read().split()) if a.probes else None
        for f in sorted(glob.glob(os.path.join(d, "v4_*.npz"))):   # catalog_v4 probes only (older v3_ reads share the dir)
            name = os.path.basename(f)[:-4]
            if keep is not None and name not in keep:
                continue
            z = np.load(f)
            t = z["t"].astype(np.float64)
            if len(z["q"]) != len(t) or len(z["logF"]) != len(t):
                rows.append({"dir": d, "probe": name, "error": "no q or no logF in the read"})
                continue
            if "idx" not in z.files or len(z["idx"]) != len(t) or not str(z["pool"]):
                rows.append({"dir": d, "probe": name, "error": "no event indices in the read (a read before @902b96c5)"})
                continue
            pr = np.load(str(z["pool"]), mmap_mode="r")
            n = (pr.shape[1] - 1) // 5
            rows_at = pool_rows(z)
            if rows_at is None or rows_at.max() >= len(pr):
                rows.append({"dir": d, "probe": name, "error": "event indices do not map to the pool's rows"})
                continue
            mom, pdg, amp = split_pool(np.asarray(pr[rows_at]), n)
            raw = t - z["logF"].astype(np.float64)
            diff = np.log(np.abs(amp)) - raw
            if np.corrcoef(np.log(np.abs(amp)), raw)[0, 1] < 0.999 or np.median(np.abs(diff - np.median(diff))) >= 0.05:
                rows.append({"dir": d, "probe": name, "error": "pool rows at idx do not align with the read"})
                continue
            u = z["q"].astype(np.float64) - t
            u -= u.mean()
            logF = z["logF"].astype(np.float64)
            rec = {"dir": d, "probe": name}
            if a.prop:     # the propagator-product read: log P with E, logF and log D (when there is one) as placebos
                logP = log_prop(mom, pdg, os.path.join(a.sidecars, name + ".diagrams.json"))
                if logP is None:
                    rows.append({**rec, "error": "no sidecar subprocess for the pool's legs"})
                    continue
                fx = factors(mom, pdg)
                logD, E = fx if fx is not None else (None, np.log(np.abs(mom[:, 2:, 0]).sum(1)))
                ok = np.isfinite(logP) & np.isfinite(E) & np.isfinite(logF)
                if logD is not None:
                    ok &= np.isfinite(logD)
                uk, lu2 = u[ok], np.log(np.maximum(u[ok] ** 2, 1e-300))
                X = {"P": logP[ok], "E": E[ok], "F": logF[ok]}
                if logD is not None:
                    X["D"] = logD[ok]
                rec.update({"n": int(ok.sum()), "var_u": float(np.var(uk)), "has_emitter": fx is not None,
                            "r2_tP": r2(t[ok], logP[ok]), "r2_rawP": r2(raw[ok], logP[ok]),
                            # the ceiling: what one per-event scalar explains of u
                            "eta2_t": eta2(uk, t[ok]), "eta2_raw": eta2(uk, raw[ok]), "r2_ut": r2(uk, t[ok])})
                rec["ceiling"] = max(rec["eta2_t"], rec["eta2_raw"])
                for k, x in X.items():
                    rec[f"r2_{k}"] = r2(uk, x)          # linear, as the dipole read
                    rec[f"eta2_{k}"] = eta2(uk, x)      # the mean read
                    rec[f"r2size_{k}"] = r2(lu2, x)     # the size read: log u^2 on x
                rec["tail_lo_P"], rec["tail_hi_P"] = tail_share(uk, logP[ok])
                if a.dump:       # the leave-class-out read's stage 1 (design of 2026-10-31)
                    Xf, names = lof_features(mom[ok], pdg[ok], t[ok], logF[ok], logP[ok],
                                             None if logD is None else logD[ok])
                    if a.ceiling:
                        keep = [i for i, nm in enumerate(names) if nm not in a.exclude]
                        rec["ceiling_own"] = own_ceiling(Xf[:, keep], uk)
                        rec["ceiling_K"] = int(len(uk) * 4 // 5)
                    pick = np.sort(np.random.default_rng(0).permutation(len(uk))[:500])
                    os.makedirs(a.dump, exist_ok=True)
                    np.savez(os.path.join(a.dump, name + ".npz"), X=Xf[pick].astype(np.float32), u=uk[pick],
                             names=np.array(names), var_u=float(np.var(uk)), n=len(uk),
                             ceiling_own=rec.get("ceiling_own", np.nan))
                if a.joint:      # the joint read (design of 2026-10-31): u on the event's full kinematics together
                    X = joint_features(mom[ok], t[ok], logF[ok], logP[ok], None if logD is None else logD[ok])
                    rec["R2_joint"], rec["R2_null"] = joint_r2(X, uk)
                    rec["R2_joint_minus_null"] = rec["R2_joint"] - rec["R2_null"]
                rows.append(rec)
                continue
            fx = factors(mom, pdg)
            if fx is None:
                rows.append({**rec, "no_emitter": True})
                continue
            logD, E = fx
            ok = np.isfinite(logD) & np.isfinite(E) & np.isfinite(logF)
            rows.append({**rec, "n": int(ok.sum()), "var_u": float(np.var(u[ok])),
                         "r2_D": r2(u[ok], logD[ok]), "r2_E": r2(u[ok], E[ok]), "r2_F": r2(u[ok], logF[ok]),
                         "r2_tD": r2(t[ok], logD[ok]), "r2_rawD": r2(raw[ok], logD[ok])})
    with open(a.out, "w") as fh:
        json.dump(rows, fh)
    per = {}
    for r in rows:
        if ("r2_P" if a.prop else "r2_D") in r:
            per.setdefault(r["dir"], []).append(r)
    keys = ("eta2_P", "r2size_P", "ceiling", "r2_tP", "r2_rawP", "eta2_E", "eta2_F", "r2size_E", "r2size_F", "r2_P",
            "tail_lo_P", "tail_hi_P") + (("R2_joint", "R2_null", "R2_joint_minus_null", "var_u") if a.joint else ()) \
        + (("ceiling_own",) if a.ceiling else ()) if a.prop else ("r2_D", "r2_E", "r2_F", "r2_tD", "r2_rawD")
    main_k = keys[0]
    rng = np.random.default_rng(0)

    def med(v, k):
        return round(float(np.median([x[k] for x in v])), 3)

    def boot(v, k):  # the 16-84% range of the median over pools resampled with replacement, 1000 draws
        a = np.array([x[k] for x in v])
        m = [np.median(rng.choice(a, len(a))) for _ in range(1000)]
        return [round(float(np.quantile(m, 0.16)), 3), round(float(np.quantile(m, 0.84)), 3)]
    print(json.dumps({d: {"n": len(v), **{f"median_{k}": med(v, k) for k in keys},
                          f"boot_{main_k}": boot(v, main_k), "refused": sum(1 for r in rows if r["dir"] == d and "error" in r)}
                      for d, v in per.items()}))


if __name__ == "__main__":
    main()
