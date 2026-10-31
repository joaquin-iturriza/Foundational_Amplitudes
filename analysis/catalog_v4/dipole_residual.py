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


def r2(u, x):
    if np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(u, x)[0, 1] ** 2)


def pool_rows(z):
    """The pool-file rows of a saved read's events. A saved z['pool_row'] wins. Else z['idx'] are positions in the
    role-contiguous train|val|test array and z['split'] the role; each role was shuffled by one default_rng(42)
    drawing rng.permutation(n) for train, val, test in turn (experiment._init_data), so position i of a role holds
    pool row perm[i]. The role sizes come from z['role_counts'] when saved, else from the idx: the scored role is whole
    (n = len(idx)) and starts after the earlier roles (idx.min()); a test read with earlier roles of unknown split
    is refused."""
    if "pool_row" in z.files and len(z["pool_row"]):
        return np.asarray(z["pool_row"]).astype(np.int64)
    idx = np.asarray(z["idx"]).astype(np.int64).reshape(-1)
    role = str(z["split"])
    order = ("train", "val", "test")
    if "role_counts" in z.files and len(z["role_counts"]) == 3:
        cnt = dict(zip(order, (int(c) for c in z["role_counts"])))
    else:
        lo = int(idx.min()) if len(idx) else 0
        if role == "val":
            cnt = {"train": lo, "val": len(idx), "test": 0}
        elif role == "test" and lo == 0:
            cnt = {"train": 0, "val": 0, "test": len(idx)}
        elif role == "train" and lo == 0:
            cnt = {"train": len(idx), "val": 0, "test": 0}
        else:
            raise ValueError("test/train sizes of the earlier roles are not in the read (no role_counts)")
    rng = np.random.default_rng(seed=42)
    perms = {r: rng.permutation(cnt[r]) for r in order}
    start = sum(cnt[r] for r in order[:order.index(role)])
    return perms[role][idx - start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="read directories; restrict to cells with --probes")
    ap.add_argument("--probes", default="", help="a file of probe names, one per line (empty: all)")
    ap.add_argument("--out", required=True)
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
            if ("idx" not in z.files and "pool_row" not in z.files) or len(z["idx"] if "idx" in z.files else z["pool_row"]) != len(t) or not str(z["pool"]):
                rows.append({"dir": d, "probe": name, "error": "no event indices in the read (a read before @902b96c5)"})
                continue
            pr = np.load(str(z["pool"]), mmap_mode="r")
            n = (pr.shape[1] - 1) // 5
            mom, pdg, amp = split_pool(np.asarray(pr[pool_rows(z)]), n)
            raw = t - z["logF"].astype(np.float64)
            diff = np.log(np.abs(amp)) - raw
            if np.corrcoef(np.log(np.abs(amp)), raw)[0, 1] < 0.999 or np.median(np.abs(diff - np.median(diff))) >= 0.05:
                rows.append({"dir": d, "probe": name, "error": "pool rows at idx do not align with the read"})
                continue
            fx = factors(mom, pdg)
            if fx is None:
                rows.append({"dir": d, "probe": name, "no_emitter": True})
                continue
            u = z["q"].astype(np.float64) - t
            u -= u.mean()
            logF = z["logF"].astype(np.float64)
            logD, E = fx
            ok = np.isfinite(logD) & np.isfinite(E) & np.isfinite(logF)
            rows.append({"dir": d, "probe": name, "n": int(ok.sum()), "var_u": float(np.var(u[ok])),
                         "r2_D": r2(u[ok], logD[ok]), "r2_E": r2(u[ok], E[ok]), "r2_F": r2(u[ok], logF[ok]),
                         "r2_tD": r2(t[ok], logD[ok]), "r2_rawD": r2(raw[ok], logD[ok])})
    with open(a.out, "w") as fh:
        json.dump(rows, fh)
    per = {}
    for r in rows:
        if "r2_D" in r:
            per.setdefault(r["dir"], []).append(r)
    rng = np.random.default_rng(0)

    def med(v, k):
        return round(float(np.median([x[k] for x in v])), 3)

    def boot(v, k):  # the 16-84% range of the median over pools resampled with replacement, 1000 draws
        a = np.array([x[k] for x in v])
        m = [np.median(rng.choice(a, len(a))) for _ in range(1000)]
        return [round(float(np.quantile(m, 0.16)), 3), round(float(np.quantile(m, 0.84)), 3)]
    print(json.dumps({d: {"n": len(v), **{f"median_{k}": med(v, k) for k in ("r2_D", "r2_E", "r2_F", "r2_tD", "r2_rawD")},
                          "boot_r2_D": boot(v, "r2_D"), "refused": sum(1 for r in rows if r["dir"] == d and "error" in r)}
                      for d, v in per.items()}))


if __name__ == "__main__":
    main()
