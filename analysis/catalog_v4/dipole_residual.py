"""The per-event soft-dipole factor against the network's zero-shot residual (plan v3_prep_explore, the dipole design).

Per probe read (tools/zero_shot_parent.py .npz of the test split: the scored events are the first rows of the probe's
frozen pool of the read's split (val or test), in order (--eval-subsample); the alignment is checked, not assumed: the pool rows' log|M|^2 must equal
the read's t to float32 precision on every event, else the probe is refused and listed):
  u   the residual (q - t) minus its mean over the probe's scored events
  D   per event, the sum over each final gluon (photon) j of the colour-blind antenna sum over pairs (a, b) of the
      other coloured (charged) legs, s_ab / (s_aj s_bj), s_xy = 2 p_x.p_y with incoming momenta reversed, GeV^-2;
      a function of the momenta alone, no fit, the same for every process
  E   the placebo: log of the summed energy of the same legs (a per-event scale with no soft structure)
  logF the read's per-event emission-scale factor (the control's target is t - logF): the second placebo
  r2_D, r2_E, r2_F   the squared Pearson correlation of u with log D, E and logF
  r2_tD, r2_rawD     the gate: r^2 of the control's target t - logF, and of the raw t, on log D (no network)
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
from analysis.common.readers import find_pool, split_pool  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="read directories; restrict to cells with --probes")
    ap.add_argument("--probes", default="", help="a file of probe names, one per line (empty: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        keep = set(open(a.probes).read().split()) if a.probes else None
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            name = os.path.basename(f)[:-4]
            if keep is not None and name not in keep:
                continue
            z = np.load(f)
            t = z["t"].astype(np.float64)
            if len(z["q"]) != len(t) or len(z["logF"]) != len(t):
                rows.append({"dir": d, "probe": name, "error": "no q or no logF in the read"})
                continue
            role = str(z["split"]) if "split" in z.files else "val"   # reads before --split scored val
            pool = find_pool(name, role=role, any_window=True)
            if pool is None:
                rows.append({"dir": d, "probe": name, "error": "no test pool"})
                continue
            pr = np.load(pool, mmap_mode="r")
            n = (pr.shape[1] - 1) // 5
            mom, pdg, amp = split_pool(np.asarray(pr[:len(t)]), n)
            lt = np.log(np.abs(amp)).astype(np.float32)
            if len(amp) == len(t) and not np.allclose(lt, t.astype(np.float32), rtol=0, atol=1e-4):
                # the same rows in another order (the loader's shuffle): a permutation only when every read value
                # meets exactly one pool row (unique values, one to one), else refused
                key = {v: i for i, v in enumerate(lt.tolist())}
                perm = [key.get(v, -1) for v in t.astype(np.float32).tolist()]
                if len(key) == len(lt) and min(perm) >= 0 and len(set(perm)) == len(perm):
                    mom, pdg = mom[perm], pdg[perm]
                    lt = lt[perm]
            if len(amp) != len(t) or not np.allclose(lt, t.astype(np.float32), rtol=0, atol=1e-4):
                rows.append({"dir": d, "probe": name, "error": "pool rows do not align with the read's t"})
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
                         "r2_tD": r2((t - logF)[ok], logD[ok]), "r2_rawD": r2(t[ok], logD[ok])})
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
