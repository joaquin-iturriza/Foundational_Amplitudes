"""The per-event soft-dipole factor against the network's zero-shot residual (plan v3_prep_explore, the dipole design).

Per probe read (tools/zero_shot_parent.py .npz with the scored events' pool indices `idx` and the pool path `pool`):
  u   the residual (q - t) minus its mean over the probe's scored events
  D   per event, the sum over each final gluon (photon) j of the colour-blind antenna sum over pairs (a, b) of the
      other coloured (charged) legs, s_ab / (s_aj s_bj), s_xy = 2 p_x.p_y with incoming momenta reversed, GeV^-2;
      a function of the momenta alone, no fit, the same for every process
  E   the placebo: log of the summed energy of the same legs (a per-event scale with no soft structure)
  r2_D, r2_E   the squared Pearson correlation of u with log D and with E
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            name = os.path.basename(f)[:-4]
            if "idx" not in z.files or len(z["idx"]) == 0 or len(z["q"]) == 0:
                rows.append({"dir": d, "probe": name, "error": "no event indices or no q in the read"})
                continue
            pool = str(z["pool"])
            pr = np.load(pool)
            n = (pr.shape[1] - 1) // 5
            mom, pdg, _ = split_pool(pr[np.asarray(z["idx"])], n)
            fx = factors(mom, pdg)
            if fx is None:
                rows.append({"dir": d, "probe": name, "no_emitter": True})
                continue
            u = (z["q"] - z["t"]).astype(np.float64)
            u -= u.mean()
            logD, E = fx
            ok = np.isfinite(logD) & np.isfinite(E)
            rows.append({"dir": d, "probe": name, "n": int(ok.sum()), "var_u": float(np.var(u[ok])),
                         "r2_D": r2(u[ok], logD[ok]), "r2_E": r2(u[ok], E[ok])})
    with open(a.out, "w") as fh:
        json.dump(rows, fh)
    per = {}
    for r in rows:
        if "r2_D" in r:
            per.setdefault(r["dir"], []).append(r)
    print(json.dumps({d: {"n": len(v), "median_r2_D": round(float(np.median([x["r2_D"] for x in v])), 3),
                          "median_r2_E": round(float(np.median([x["r2_E"] for x in v])), 3)} for d, v in per.items()}))


if __name__ == "__main__":
    main()
