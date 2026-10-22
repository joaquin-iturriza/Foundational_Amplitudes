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


def coupling_features(meta):
    """From the process's tree diagrams: the largest coupling product sum_vertices log c^2, the log of the sum over
    diagrams of the products, the number of diagrams, and the massive internal lines of the leading diagram. The same for
    a process and its conjugate, so it is added outside the conjugation choice."""
    r = _DC["dc_" + meta["folder"]]
    l = np.array(r["logc2"])
    i = int(np.argmax(l))
    f = [float(l.max()), float(l.max() + np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
         float(r["massive_internal"][i]), float(max(r["massive_internal"]))]
    return [f[j] for j in COUP_COLS]


COUP_COLS = [int(c) for c in os.environ.get("G_COUP_COLS", "0,1,2,3,4").split(",") if c != ""]


def features(meta):
    """|M|^2 is the same for a process and its charge conjugate (CP violation aside), so both get the features of
    whichever of the two gives the smaller feature vector: s s~ > d u~ w+ and its twin d d~ > c s~ w- (relabelled
    and conjugated) then look alike."""
    ini, fin = meta["process"].split(" > ")
    ini, fin = ini.split(), fin.split()
    a = _features(meta, ini, fin)
    b = _features(meta, [conj(p) for p in ini], [conj(p) for p in fin])
    f = min(a, b, key=tuple)
    if COUP and _DC:
        f = np.concatenate([f, coupling_features(meta)])
    return f


def _features(meta, ini, fin):
    o = meta["orders"][0]
    f = [len(ini) + len(fin), o.get("QCD", 0), o.get("QED", 0)]
    f += [sum(KIND[p] == k for p in ini) for k in KINDS] + [sum(KIND[p] == k for p in fin) for k in KINDS]
    if FLOW:
        f += flavour_flow(ini, fin)
    if PROPS:
        f += prop_pool(ini) + prop_pool(fin)
    if RES:
        f += higgs_and_resonances(ini, fin)
    if NODIAG:
        f.append(np.log(max(meta.get("n_diagrams") or 1, 1)))
    return np.array(f, float)


FLOW = "--no-flow" not in sys.argv
PROPS = "--props" in sys.argv          # the model's own particle table (particle_ids.py), pooled per side
NODIAG = "--ndiag" in sys.argv
COUP = "--coup" in sys.argv            # the diagrams' coupling products as features (diagram_couplings.py)
OFFSET = "--offset" in sys.argv        # ... or the leading one as a fixed offset of the mean: |M|^2 scales as c^2
_DC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagram_couplings.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagram_couplings.json")) else {}
RES = "--res" in sys.argv              # Higgs coupling strength and final-state resonances: tried, no cross-fit gain         # the number of Feynman diagrams (classes_default.json)
MG_PDG = {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5, "t": 6, "e-": 11, "ve": 12, "mu-": 13, "vm": 14, "ta-": 15, "vt": 16,
          "g": 21, "a": 22, "z": 23, "w+": 24, "h": 25}


def pdg(p):
    if p in MG_PDG:
        return MG_PDG[p]
    if p.endswith("~"):
        return -MG_PDG[p[:-1]]
    return -MG_PDG[{"e+": "e-", "mu+": "mu-", "ta+": "ta-", "w-": "w+"}[p]]


MASS = {"t": 172.5, "t~": 172.5, "w+": 80.4, "w-": 80.4, "z": 91.2, "h": 125.0, "b": 4.7, "b~": 4.7, "ta+": 1.78,
        "ta-": 1.78}
LEP = {"e-": "ve", "mu-": "vm", "ta-": "vt"}


def higgs_and_resonances(ini, fin):
    """The two failures of the cross-fit (2026-10-22): q q~ > b b~ h sat 8 in log|M|^2 above its truth (the Higgs
    attaches only through the b Yukawa coupling) and q q~ > t b~ w+ 7 below it (b~ w+... can come from an on-shell top).
    Per Higgs, log10 of the heaviest other particle it can couple to (0 without one: light quarks and leptons are
    massless here); counts of final pairs that an on-shell t, W or Z/H can decay to."""
    allp = ini + fin
    nh = sum(p == "h" for p in allp)
    heavy = max([MASS[p] for p in allp if p in MASS and p != "h"] or [1.0])
    up = lambda p: p.rstrip("~") in UP
    qk = lambda p: p.rstrip("~") in UP | DOWN
    pairs = [(a, b) for i, a in enumerate(fin) for b in fin[i + 1:]]
    top = sum({a, b} in ({"b", "w+"}, {"b~", "w-"}) for a, b in pairs)
    def wpair(a, b):
        if qk(a) and qk(b):   # up quark with a down antiquark (W+) or its conjugate (W-)
            return up(a) != up(b) and a.endswith("~") != b.endswith("~") and \
                (up(a) and not a.endswith("~")) == (not up(b) and b.endswith("~"))
        for l, v in LEP.items():
            lp = l[:-1] + "+"
            if {a, b} in ({l, v + "~"}, {lp, v}):
                return True
        return False
    w = sum(wpair(a, b) for a, b in pairs)
    conjp = lambda p: conj(p)
    z = sum(b == conjp(a) and a not in ("g", "a", "z", "h") and not a.startswith("w") for a, b in pairs)
    f = [nh * np.log10(heavy), float(nh > 0 and heavy < 5), float(top), float(w), float(z)]
    return [f[i] for i in RES_COLS]


RES_COLS = [int(c) for c in os.environ.get("G_RES_COLS", "0,1,2,3,4").split(",") if c != ""]


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
    off = (lambda names: np.array([max(_DC[n]["logc2"]) for n in names])) if OFFSET else (lambda names: np.zeros(len(names)))
    otr, oho = off(tr), off(ho)
    ytr[:, 0] -= otr; yho[:, 0] -= oho             # scored on mu - offset: the same level error as on mu
    groups = np.array([meta[n]["weak_key"] for n in tr])
    res = {"heldout": {}, "crossfit": {}}
    for m in METHODS:
        cf = np.zeros_like(ytr)
        for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
            cf[b] = fit_predict(m, Xtr[a], ytr[a], Xtr[b])
        res["crossfit"][m] = [list(map(float, v)) for v in scores(cf, ytr)]
        pho = fit_predict(m, Xtr, ytr, Xho)
        res["heldout"][m] = [list(map(float, v)) for v in scores(pho, yho)]
        res.setdefault("heldout_pred", {})[m] = (pho + np.stack([oho, 0 * oho], 1)).tolist()      # (mu_hat, log sigma_hat) per held-out process
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
