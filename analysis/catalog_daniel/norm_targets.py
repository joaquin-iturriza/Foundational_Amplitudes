"""D83, step 1 of the 2026-10-23 plan, at home: every candidate target scored on the pools themselves, no training.
Candidates, per event, from data/norm_targets_stats.json (norm_targets_stats.py):
    raw      y = log|M|^2                                   (GeV units)
    s0       y + (N - 4) log s0, s0 = 289 GeV^2              (the fixed reference of the Adim arm)
    shat     y + (N - 4) log s_hat                           (dimensionless, Lambda^2 = s_hat)
    pairs    y + (N - 4) Lp                                  (Lambda^2 = the pair scale)
    kt       y + Lk                                          (Lambda^2 = the event's k_T clustering scales)
    kt_ga    y + Lkg                                         (the same, gluons and photons only)
    kt_mT    y + Lkm                                         (every final particle, m_T: generic)
    qperp    y + Lq                                          (Lorentz invariant, physics_scales.py: the trainable one)
    qperp_m0 y + Lq0                                         (the same, massive particles at s_hat)
each also with the coupling orders divided out (+ord: - q log 4 pi alpha_s - e log 4 pi alpha at alpha_s = 0.118, the
orders from classes_default.json; a 2->4 probe takes its parent's plus one for the added g or photon).
Three reads per candidate:
  range   where the held-out processes' mean targets fall against the training processes' (fraction above the
          training maximum, the median distance to the training median in units of the training spread)
  level   a level model from composition alone (ridge on the counts of each kind of particle, incoming and outgoing
          apart, and N; the interatomic-potential reference-energy convention), fitted on the 1352 training processes
          and scored on the 43 held-out processes that relabel none and on the 2->4 probes: (mu_hat - mu)^2 in log
          units, median; cross-fitted on the training processes by relabelling class (GroupKFold over exact_key)
  shape   the median within-process standard deviation of the target
Output: the table printed, data/norm_targets.json.
    .venv-driver/bin/python analysis/catalog_daniel/norm_targets.py
"""
import json, os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
KIND = {21: "g", 22: "a", 23: "Z", 24: "W", 25: "H", 6: "t", 5: "b", 1: "q", 2: "q", 3: "q", 4: "q",
        11: "l", 13: "l", 15: "l", 12: "v", 14: "v", 16: "v"}
KINDS = sorted(set(KIND.values()))
LS0 = np.log(289.0)
A_S, A_EM = 0.118, 1 / 132.507


def comp(pdg):
    ini, fin = pdg[:2], pdg[2:]
    return np.array([len(pdg)] + [sum(KIND[abs(c)] == k for c in ini) for k in KINDS]
                    + [sum(KIND[abs(c)] == k for c in fin) for k in KINDS], float)


def cand(r, name, ord_):
    """per candidate: (mean, var) of the target over the process's events"""
    m, C, N = np.array(r["mean"]), np.array(r["cov"]), r["N"]
    vecs = {"raw": [1, 0, 0, 0, 0, 0, 0, 0], "s0": [1, 0, 0, 0, 0, 0, 0, 0], "shat": [1, N - 4, 0, 0, 0, 0, 0, 0],
            "pairs": [1, 0, N - 4, 0, 0, 0, 0, 0], "kt": [1, 0, 0, 1, 0, 0, 0, 0], "kt_ga": [1, 0, 0, 0, 1, 0, 0, 0], "kt_mT": [1, 0, 0, 0, 0, 1, 0, 0],
            "qperp": [1, 0, 0, 0, 0, 0, 1, 0], "qperp_m0": [1, 0, 0, 0, 0, 0, 0, 1]}
    out = {}
    for k, a in vecs.items():
        a = np.array(a, float)
        mu = float(a @ m) + ((N - 4) * LS0 if k == "s0" else 0.0)
        out[k] = (mu, float(a @ C @ a))
        if ord_ is not None:
            q, e = ord_
            out[k + "+ord"] = (mu - q * np.log(4 * np.pi * A_S) - e * np.log(4 * np.pi * A_EM), out[k][1])
    return out


def main():
    st = json.load(open(os.path.join(HERE, "data", "norm_targets_stats.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))

    def orders(s, n):
        if s in ("train", "heldout"):
            o = meta[n]["orders"][0]; return o.get("QCD", 0), o.get("QED", 0)
        par = "dc_" + n.split("_", 1)[1].rsplit("_", 1)[0]
        if par not in meta:
            return None
        o = meta[par]["orders"][0]; add = n.rsplit("_", 1)[1]
        return o.get("QCD", 0) + (add == "g"), o.get("QED", 0) + (add == "a")

    rows = {}
    for s in ("train", "heldout", "2to4", "2to4c30"):
        for n, r in st[s].items():
            if r.get("signed"):
                continue
            rows[(s, n)] = (comp(r["pdg"]), cand(r, n, orders(s, n)))
    keys = list(next(iter(rows.values()))[1])
    tr = [k for k in rows if k[0] == "train"]
    sets = {"43 new": [k for k in rows if k[0] == "heldout" and k[1] not in twins],
            "2to4 +g": [k for k in rows if k[0] == "2to4" and k[1].endswith("_g")],
            "2to4 +a": [k for k in rows if k[0] == "2to4" and k[1].endswith("_a")],
            "2to4 +g, pT>=30": [k for k in rows if k[0] == "2to4c30" and k[1].endswith("_g")]}
    Xtr = np.array([rows[k][0] for k in tr])
    groups = np.array([meta[k[1]].get("exact_key", k[1]) for k in tr])
    res = {}
    for c in keys:
        if any(c not in rows[k][1] for k in tr):
            continue
        ytr = np.array([rows[k][1][c][0] for k in tr])
        lo, med, hi = ytr.min(), np.median(ytr), ytr.max()
        iqr = np.subtract(*np.quantile(ytr, [0.75, 0.25]))
        cf = np.zeros_like(ytr)
        for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
            cf[b] = Ridge(alpha=1e-2).fit(Xtr[a], ytr[a]).predict(Xtr[b])
        mdl = Ridge(alpha=1e-2).fit(Xtr, ytr)
        r = {"train": [float(lo), float(med), float(hi)], "crossfit_level": float(np.median((cf - ytr) ** 2)),
             "shape_sd_train": float(np.median([np.sqrt(rows[k][1][c][1]) for k in tr]))}
        for sname, ks in sets.items():
            ks = [k for k in ks if c in rows[k][1]]
            if not ks:
                continue
            y = np.array([rows[k][1][c][0] for k in ks]); X = np.array([rows[k][0] for k in ks])
            r[sname] = {"n": len(ks), "above_max": float(np.mean(y > hi)), "below_min": float(np.mean(y < lo)),
                        "dist_iqr": float(np.median((y - med) / iqr)),
                        "level": float(np.median((mdl.predict(X) - y) ** 2)),
                        "shape_sd": float(np.median([np.sqrt(rows[k][1][c][1]) for k in ks]))}
        res[c] = r
        print(f"{c:10s} train [{lo:6.1f} {med:6.1f} {hi:6.1f}] crossfit level {r['crossfit_level']:.3g} sd {r['shape_sd_train']:.2f} | " +
              " | ".join(f"{s}: above {r[s]['above_max']:.2f} dist {r[s]['dist_iqr']:+.2f} level {r[s]['level']:.3g} sd {r[s]['shape_sd']:.2f}"
                         for s in sets if s in r))
    json.dump(res, open(os.path.join(HERE, "data", "norm_targets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
