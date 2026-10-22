"""The process-to-statistics map g (D81, design A): predict a process's mean mu_p and spread sigma_p of log|M|^2 from
the process alone, so a process with no data can be normalized (and scored on its own). Fitted on the 1352 training
processes of the arXiv:2606.23791 split, scored on its 139 held-out ones, and cross-fitted on the training ones by
relabelling class (GroupKFold over weak_key: a class is out of the fit together).

Features (no momenta): the particle count N (the mass dimension of |M|^2 is 8 - 2N, so at fixed sqrt(s) the level is
linear in N), the coupling powers of the amplitude (QCD, QED: |M|^2 goes as alpha_s^QCD alpha^QED), and the counts of
each kind of external particle, incoming and outgoing apart. Maps, each on mu and on log sigma:
  shared      one pooled mean and spread (data.shared_standardization)
  affine_N    mean linear in N, one spread (Jonas; data.affine_multiplicity_std)
  phys_lin    ridge regression on all the features (the physics skeleton)
  phys_gbr    phys_lin plus gradient-boosted trees on its residuals (the learned correction)
Scored per process: the level error (mu_hat - mu)^2 / sigma^2 and the spread ratio sigma_hat / sigma, the two parts of
a zero-shot error that the normalization sets (fig:zero_shot_parts). Inputs: data/pool_stats.json (pool_stats.py),
classes_default.json, heldout_twins.json. Runs in .venv-driver (scikit-learn).
    .venv-driver/bin/python analysis/catalog_daniel/stats_net.py   -> figs/stats_net_{a,b}, data/stats_net.json
"""
import json, os, sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

KIND = {"g": "g", "a": "a", "z": "Z", "w+": "W", "w-": "W", "h": "H", "t": "t", "t~": "t", "b": "b", "b~": "b"}
for q in "udsc":
    KIND[q] = KIND[q + "~"] = "q"
for l in ("e", "mu", "ta"):
    KIND[l + "+"] = KIND[l + "-"] = "l"
for v in ("ve", "vm", "vt"):
    KIND[v] = KIND[v + "~"] = "v"
KINDS = ["g", "q", "b", "t", "a", "Z", "W", "H", "l", "v"]
METHODS = ("shared", "affine_N", "phys_lin", "phys_gbr")


UP, DOWN = set("uct"), set("dsb")


def flavour_flow(ini, fin):
    """The flavour structure the counts by kind lose (2026-10-22: s s~ > d u~ W+ looked like processes with a very
    different mean): up/down-type quarks and antiquarks on each side, an initial quark-antiquark pair of one flavour
    (it can annihilate), a flavour carried from the initial state into the final one, and the charge in. Flavour
    names never enter, so what is learned on d carries to s."""
    q = lambda p: p.rstrip("~") in UP | DOWN
    up = lambda p: p.rstrip("~") in UP
    anti = lambda p: p.endswith("~")
    f = []
    for side in (ini, fin):
        f += [sum(q(p) and up(p) and not anti(p) for p in side), sum(q(p) and up(p) and anti(p) for p in side),
              sum(q(p) and not up(p) and not anti(p) for p in side), sum(q(p) and not up(p) and anti(p) for p in side)]
    f.append(float(len(ini) == 2 and q(ini[0]) and q(ini[1]) and ini[0].rstrip("~") == ini[1].rstrip("~")
                   and anti(ini[0]) != anti(ini[1])))
    f.append(float(sum(p in fin for p in ini if q(p))))
    f.append(float(sum(1 for p in fin if q(p) and (p[:-1] if anti(p) else p + "~") in fin) // 2))
    return f


CC = {"w+": "w-", "w-": "w+", "e+": "e-", "e-": "e+", "mu+": "mu-", "mu-": "mu+", "ta+": "ta-", "ta-": "ta+"}


def conj(p):
    """Charge conjugate of one MadGraph particle name."""
    if p in CC:
        return CC[p]
    if p in ("g", "a", "z", "h"):
        return p
    return p[:-1] if p.endswith("~") else p + "~"


def features(meta):
    """|M|^2 is the same for a process and its charge conjugate (CP violation aside), so both get the features of
    whichever of the two gives the smaller feature vector: s s~ > d u~ w+ and its twin d d~ > c s~ w- (relabelled
    and conjugated) then look alike."""
    ini, fin = meta["process"].split(" > ")
    ini, fin = ini.split(), fin.split()
    a = _features(meta, ini, fin)
    b = _features(meta, [conj(p) for p in ini], [conj(p) for p in fin])
    return min(a, b, key=tuple)


def _features(meta, ini, fin):
    o = meta["orders"][0]
    f = [len(ini) + len(fin), o.get("QCD", 0), o.get("QED", 0)]
    f += [sum(KIND[p] == k for p in ini) for k in KINDS] + [sum(KIND[p] == k for p in fin) for k in KINDS]
    if FLOW:
        f += flavour_flow(ini, fin)
    if PROPS:
        f += prop_pool(ini) + prop_pool(fin)
    if NODIAG:
        f.append(np.log(max(meta.get("n_diagrams") or 1, 1)))
    return np.array(f, float)


FLOW = "--no-flow" not in sys.argv
PROPS = "--props" in sys.argv          # the model's own particle table (particle_ids.py), pooled per side
NODIAG = "--ndiag" in sys.argv         # the number of Feynman diagrams (classes_default.json)
MG_PDG = {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5, "t": 6, "e-": 11, "ve": 12, "mu-": 13, "vm": 14, "ta-": 15, "vt": 16,
          "g": 21, "a": 22, "z": 23, "w+": 24, "h": 25}


def pdg(p):
    if p in MG_PDG:
        return MG_PDG[p]
    if p.endswith("~"):
        return -MG_PDG[p[:-1]]
    return -MG_PDG[{"e+": "e-", "mu+": "mu-", "ta+": "ta-", "w-": "w+"}[p]]


def prop_pool(side):
    """Sum and sum of absolute values of the 9 particle properties over one side's particles."""
    from particle_ids import PARTICLE_PROPERTIES
    v = np.array([PARTICLE_PROPERTIES[pdg(p)] for p in side], float)
    return list(v.sum(0)) + list(np.abs(v).sum(0))


def fit_predict(method, Xtr, ytr, Xte):
    """ytr: (n, 2) columns mu, log sigma. Returns (m, 2)."""
    if method == "shared":
        return np.tile(ytr.mean(0), (len(Xte), 1))
    if method == "affine_N":
        A = np.stack([np.ones(len(Xtr)), Xtr[:, 0]], 1)
        c = np.linalg.lstsq(A, ytr[:, 0], rcond=None)[0]
        return np.stack([c[0] + c[1] * Xte[:, 0], np.full(len(Xte), ytr[:, 1].mean())], 1)
    out = []
    for j in range(2):
        lin = Ridge(alpha=1.0).fit(Xtr, ytr[:, j])
        p = lin.predict(Xte)
        if method == "phys_gbr":
            r = ytr[:, j] - lin.predict(Xtr)
            p = p + GradientBoostingRegressor(n_estimators=800, max_depth=3, learning_rate=0.03,
                                              subsample=0.8, random_state=0).fit(Xtr, r).predict(Xte)
        out.append(p)
    return np.stack(out, 1)


def scores(pred, y):
    lvl = (pred[:, 0] - y[:, 0]) ** 2 / np.exp(2 * y[:, 1])
    spr = np.exp(pred[:, 1] - y[:, 1])
    return lvl, spr


def main():
    st = json.load(open(os.path.join(HERE, "data", "pool_stats.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    twins = json.load(open(os.path.join(HERE, "heldout_twins.json")))
    tr, ho = sorted(st["train"]), sorted(st["heldout"])
    Xtr = np.array([features(meta[n]) for n in tr]); Xho = np.array([features(meta[n]) for n in ho])
    Y = lambda names, s: np.array([[st[s][n]["mu"], np.log(st[s][n]["sd"])] for n in names])
    ytr, yho = Y(tr, "train"), Y(ho, "heldout")
    groups = np.array([meta[n]["weak_key"] for n in tr])
    res = {"heldout": {}, "crossfit": {}}
    for m in METHODS:
        cf = np.zeros_like(ytr)
        for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
            cf[b] = fit_predict(m, Xtr[a], ytr[a], Xtr[b])
        res["crossfit"][m] = [list(map(float, v)) for v in scores(cf, ytr)]
        pho = fit_predict(m, Xtr, ytr, Xho)
        res["heldout"][m] = [list(map(float, v)) for v in scores(pho, yho)]
        res.setdefault("heldout_pred", {})[m] = pho.tolist()      # (mu_hat, log sigma_hat) per held-out process
    res["heldout_names"], res["relabel"] = ho, [n in twins for n in ho]
    res["split"] = [meta[n]["split"] for n in ho]
    json.dump(res, open(os.path.join(HERE, "data", "stats_net.json"), "w"))

    cols = dict(zip(METHODS, (ps.C.grey, ps.C.blue, ps.C.green, ps.C.vermillion)))
    labs = {"shared": "one shared normalization", "affine_N": r"mean linear in $N$",
            "phys_lin": "physics features, linear", "phys_gbr": "physics features + trees"}
    figs = ps.panels(2)
    for i, ((fig, ax), xl) in enumerate(zip(figs, (r"level error $(\hat\mu-\mu)^2/\sigma^2$",
                                                    r"spread ratio $\hat\sigma/\sigma$"))):
        for m in METHODS:
            v = np.sort(np.array(res["heldout"][m][i]))
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=cols[m], label=labs[m])
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("fraction of held-out processes")
    ps.shared_legend(figs[0][0], figs[0][1], ncol=2)
    ps.save_panels(figs, os.path.join(HERE, "figs", "stats_net"))

    rel = np.array(res["relabel"]); sp = np.array(res["split"])
    for part, sel in (("all 139", np.ones(len(ho), bool)), ("96 relabels", rel), ("43 no relabel", ~rel),
                      ("interpolation", sp == "interpolation"), ("extrapolation", sp == "extrapolation")):
        print(f"held-out, {part}:")
        for m in METHODS:
            l, s = (np.array(v)[sel] for v in res["heldout"][m])
            print(f"  {m:9s} level error median {np.median(l):.3g} (90% {np.quantile(l, .9):.3g}); spread ratio "
                  f"median {np.median(s):.2f}, 10-90% {np.quantile(s, .1):.2f} to {np.quantile(s, .9):.2f}")
    zp = os.path.join(HERE, "data", "heldout_preds.npz")
    if os.path.exists(zp):
        z = np.load(zp)
        for m in ("affine_N", "phys_lin", "phys_gbr"):
            mse = []
            for n, (mu, lsd) in zip(ho, res["heldout_pred"][m]):
                t = z[f"{n}__t"].astype(float)
                mse.append(np.mean((mu + np.exp(lsd) * z[f"{n}__h"].astype(float) - t) ** 2))
            mse = np.array(mse)
            print(f"hp58 on its own with {m}: MSE(log|M|^2) median 43 new {np.median(mse[~rel]):.3g} (90% "
                  f"{np.quantile(mse[~rel], .9):.3g}), 96 relabels {np.median(mse[rel]):.3g}")
    print("cross-fitted on the training processes (by relabelling class):")
    for m in METHODS:
        l, s = (np.array(v) for v in res["crossfit"][m])
        print(f"  {m:9s} level error median {np.median(l):.3g} (90% {np.quantile(l, .9):.3g}); spread ratio "
              f"median {np.median(s):.2f}, 10-90% {np.quantile(s, .1):.2f} to {np.quantile(s, .9):.2f}")


if __name__ == "__main__":
    main()
