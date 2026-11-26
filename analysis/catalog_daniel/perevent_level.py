"""llm_compare D566: a new family's level from a per-event quantity built from the process and its diagrams, against
the per-process level (one number per dataset, tied to how its pool was sampled). Candidates, each a prediction of
log|M|^2 per event:
  shared     the training processes' mean of log|M|^2 (no level model)
  regressor  the per-process diagram-descriptor level, frozen (data/level_file_B_diag.json, norm_explore3 B|diag|sampled)
  A          log P (the diagrams' propagator product at the event's invariants, couplings C_d; catalog_v4
             dipole_residual.log_prop) + the level of r = log|M|^2 - log P placed by the content-only 2x64 network
             (stats_net features, no diagram descriptors: the diagrams enter through P per event)
  A0         log P + the training processes' mean of r (no level model)
  B          a per-event regressor g(process features, diagram descriptors, log s_hat, Lmin - log s_hat, E, log P)
             of log|M|^2 (HistGradientBoosting), cross-fitted by relabelling class on the training processes;
             with --model mlp the same map as norm_explore3's level network instead (2x64 MLP, inputs and target
             standardized, 3 seeds averaged, batches of 4096 events), written as candidate B_mlp
Per process, err = prediction - log|M|^2 over its events, sigma_p = std of log|M|^2: (i) level error
mean(err)^2 / sigma_p^2 (and mean(err)^2 in nat^2); (ii) sampling shift |mean_w(err) - mean(err)| / sigma_p, w flat in
log s_hat (20 bins between the 1st and 99th percentile, equal weight per non-empty bin) or toward the singular regions
(the fifth with the smallest Lmin - log s_hat carrying half the weight); (iii) std(err) / sigma_p. Medians over the 43
new held-out processes (per family: classes_default.json test_group) and, cross-fitted, the 1352 training processes.
CPU, on CC (pool cache and diagram sidecars there), as a CPU job:
    python analysis/catalog_daniel/perevent_level.py [--events 2000]
Existing-data mode, locally (--surrogate): (A)'s level error (i) and spread (iii) from the per-process moments of
norm_surrogate.py's "prop" in data/surrogate.json (mean, std, correlation with log|M|^2 over the same 2000 events; prop =
the leading coupling product, MadGraph's averaging, symmetry and colour, and the diagrams' propagators at the event's
invariants), with the factor's coefficient 1 and norm_surrogate_more.py's fitted 0.57 / 0.66 (2->2 / 2->3); no shifts.
Output: analysis/catalog_daniel/data/perevent_level.json (perevent_level_mlp.json with --model mlp); summary on stdout.
"""
import argparse, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v4"))
import stats_net as sn  # noqa: E402
from dipole_residual import log_prop, mdot  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

MASSLESS = {1, 2, 3, 4, 21, 22}
CANDS = ("shared", "regressor", "A0", "A", "B")


def events(x):
    """log|M|^2, log s_hat, Lmin (log of the smallest pair invariant), E (norm_moments2.py's soft-emission sum), the
    momenta and the signed PDG ids of a pool's rows."""
    n = (x.shape[1] - 1) // 5
    P = x[:, : 4 * n].reshape(len(x), n, 4)
    pdg = x[:, 4 * n: 5 * n].astype(int)
    pin = P[:, 0] + P[:, 1]
    s = mdot(pin, pin)
    lS = np.full((len(x), n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            if j < 2:
                continue
            q = P[:, i] + P[:, j]
            v = 2 * np.abs(mdot(P[:, i], P[:, j])) if i < 2 else np.abs(mdot(q, q))
            lS[:, i, j] = lS[:, j, i] = np.log(np.maximum(v, 1e-12 * s))
    Lmin = np.nanmin(lS.reshape(len(x), -1), 1)
    E = np.zeros(len(x))
    for k in range(2, n):
        if abs(pdg[0, k]) in MASSLESS:
            E += np.log(s) - np.nanmin(lS[:, k, :], 1)
    return np.log(x[:, -1]), np.log(s), Lmin, E, P, pdg


def weights(L, q):
    lo, hi = np.percentile(L, [1, 99])
    b = np.clip(((L - lo) / max(hi - lo, 1e-9) * 20).astype(int), 0, 19)
    c = np.bincount(b, minlength=20)
    wf = 1.0 / c[b]
    sing = q <= np.percentile(q, 20)
    ws = np.where(sing, 0.5 / sing.sum(), 0.5 / (~sing).sum())
    return wf / wf.sum(), ws


def reads(err, y, wf, ws):
    sd, m = float(y.std()), float(err.mean())
    return {"e": m * m / sd ** 2, "bias2": m * m, "shift_flat": abs(float(wf @ err) - m) / sd,
            "shift_sing": abs(float(ws @ err) - m) / sd, "spread": float(err.std()) / sd, "bias": m}


def mlp(r):  # norm_explore3.py's level network
    return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                        early_stopping=True, validation_fraction=0.15, random_state=r)


class Net:  # norm_explore3's level network on events: scaled inputs and target, seeds averaged
    # defaults = step 17's network; l2 is sklearn's alpha / batch (norm_explore3's 1e-2 / 200, its batch)
    def __init__(self, width=64, depth=2, l2=1e-2 / 200, lr=3e-3, batch=4096, seeds=3):
        self.hp = dict(width=int(width), depth=int(depth), l2=float(l2), lr=float(lr), batch=int(batch))
        self.seeds = seeds

    def fit(self, X, y):
        self.sx, self.m, self.s = StandardScaler().fit(X), y.mean(), y.std()
        Z, t = self.sx.transform(X), (y - self.m) / self.s
        self.nets = []
        h = self.hp
        for r in range(self.seeds):
            # shortcut: 200 epochs cap (2.7M events an epoch, ~660 updates), early stopping usually ends it first
            net = MLPRegressor(hidden_layer_sizes=(h["width"],) * h["depth"], alpha=h["l2"] * h["batch"],
                               learning_rate_init=h["lr"], max_iter=200, batch_size=h["batch"],
                               early_stopping=True, validation_fraction=0.15, random_state=r)
            self.nets.append(net.fit(Z, t))
            print(f"  mlp seed {r}: {net.n_iter_} epochs, best val R2 {net.best_validation_score_:.4f}", flush=True)
        return self

    def predict(self, X):
        Z = self.sx.transform(X)
        return self.m + self.s * np.mean([n.predict(Z) for n in self.nets], 0)



# step 20's search spaces (DyHPO single-fidelity GP): network around step 17's 2x64, l2 and lr +/- 1 and 1/2 decade;
# tree around sklearn's defaults and results.tex l.7820's 800 rounds at 0.03
SPACE = {"mlp": [{"name": "width", "type": "int_log", "low": 32, "high": 128},
                 {"name": "depth", "type": "int_uniform", "low": 1, "high": 3},
                 {"name": "l2", "type": "float_log", "low": 5e-6, "high": 5e-4},
                 {"name": "lr", "type": "float_log", "low": 1e-3, "high": 9e-3},
                 {"name": "batch", "type": "categorical", "choices": [1024, 4096, 16384]}],
         "hgb": [{"name": "learning_rate", "type": "float_log", "low": 0.02, "high": 0.3},
                 {"name": "max_leaf_nodes", "type": "int_log", "low": 8, "high": 255},
                 {"name": "max_iter", "type": "int_log", "low": 100, "high": 1000},
                 {"name": "l2_regularization", "type": "float_log", "low": 1e-3, "high": 10.0}]}
HEAVY = {"h": "h", "w+": "w", "w-": "w", "z": "z", "t": "t", "t~": "t", "ta+": "ta", "ta-": "ta", "a": "a",
         "e+": "l", "e-": "l", "mu+": "l", "mu-": "l", "ve": "v", "ve~": "v", "vm": "v", "vm~": "v", "vt": "v", "vt~": "v"}


def content_sig(process):
    """The final state's non-QCD content as a sorted multiset ('h h z'; '(jets)' for none): the new families are
    defined by it (33 of the 43 new carry a signature no training process has), so folds that hold out whole
    signatures make the cross-fit an extrapolation, where relabelling-class folds measure a lookup (step 17's tree:
    cross-fit 0.036, new 3.64)."""
    return " ".join(sorted(HEAVY[x] for x in process.split(">")[1].split() if x in HEAVY)) or "(jets)"


def level_net(Xtr, ytr, grp, Xho):
    """cross-fitted predictions on the training processes, and the full fit's on the held-out, 3 seeds averaged"""
    cf = np.zeros(len(Xtr))
    for i, j in GroupKFold(5).split(Xtr, ytr, grp):
        sc = StandardScaler().fit(Xtr[i])
        cf[j] = np.mean([mlp(r).fit(sc.transform(Xtr[i]), ytr[i]).predict(sc.transform(Xtr[j])) for r in range(3)], 0)
    sc = StandardScaler().fit(Xtr)
    ho = np.mean([mlp(r).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xho)) for r in range(3)], 0)
    return cf, ho


def from_surrogate():
    S = {k: v for k, v in json.load(open(os.path.join(HERE, "data", "surrogate.json"))).items() if "error" not in v}
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    lev = json.load(open(os.path.join(HERE, "data", "level_file_B_diag.json")))["levels"]
    tr = sorted(n for n in S if meta[n]["split"] == "train")
    new = sorted(n for n in S if meta[n]["split"] != "train" and n not in rel)
    grp = np.array([meta[n]["weak_key"] for n in tr])
    beta = {4: 0.5702, 5: 0.6595}  # data/surrogate_more.json factor|prop.beta
    out = {"n": {"train": len(tr), "new": len(new)}}
    sd = {n: S[n]["y_std"] for n in S}
    My = np.mean([S[n]["y_mean"] for n in tr])
    out["shared"] = {n: (My - S[n]["y_mean"]) ** 2 / sd[n] ** 2 for n in new}
    out["regressor"] = {n: (lev[n] - S[n]["y_mean"]) ** 2 / sd[n] ** 2 for n in new}
    spread = {}
    for tag, b in (("A", None), ("Abeta", beta)):
        bb = {n: (1.0 if b is None else b[S[n]["N"]]) for n in S}
        m = {n: S[n]["y_mean"] - bb[n] * S[n]["prop"][0] for n in S}
        Mr = np.mean([m[n] for n in tr])
        out[tag + "0"] = {n: (Mr - m[n]) ** 2 / sd[n] ** 2 for n in new}
        Xtr = np.array([sn.features(meta[n]) for n in tr]); Xho = np.array([sn.features(meta[n]) for n in new])
        cf, ho = level_net(Xtr, np.array([m[n] - Mr for n in tr]), grp, Xho)
        out[tag] = {n: (h + Mr - m[n]) ** 2 / sd[n] ** 2 for n, h in zip(new, ho)}
        out[tag + "_crossfit_train_median"] = float(np.median([(c + Mr - m[n]) ** 2 / sd[n] ** 2 for n, c in zip(tr, cf)]))
        ps, c = {n: S[n]["prop"][1] for n in S}, {n: S[n]["prop"][2] for n in S}
        spread[tag] = float(np.median([np.sqrt(max(sd[n] ** 2 + (bb[n] * ps[n]) ** 2 - 2 * bb[n] * c[n] * sd[n] * ps[n],
                                                   0)) / sd[n] for n in new]))
    summ = {k: float(np.median(list(v.values()))) for k, v in out.items() if isinstance(v, dict) and k != "n"}
    fams = {}
    for n in new:
        fams.setdefault(meta[n]["test_group"], []).append(n)
    fam = {k: {g: float(np.median([out[k][n] for n in ns])) for g, ns in fams.items()}
           for k, v in out.items() if isinstance(v, dict) and k != "n"}
    res = {"n": out["n"], "e_new_median": summ, "spread_new_median": spread, "families": fam,
           "crossfit_train_median": {k: out[k] for k in out if k.endswith("_crossfit_train_median")}}
    json.dump(res, open(os.path.join(HERE, "data", "perevent_level_surrogate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


MMAP_ROLE = {}


def mmap_pools(dirs):
    """{process: f(k) -> its first k train-role events as pool rows (4P momenta, P PDG ids, |M|^2)} from memmap base
    caches, which hold every pool's first events in pool order, momenta boosted to the incoming pair's frame (the
    invariants used here do not change)"""
    out = {}
    for d in dirs:
        m = json.load(open(os.path.join(d, "meta.json")))
        mom, pdg, amp = (np.load(os.path.join(d, x + ".npy"), mmap_mode="r") for x in ("mom", "pdg", "amp"))
        segs = {}
        for sg in m["segs"]:  # the train role, else the first role the cache holds for the process
            if sg["name"] not in segs or sg["role"] == "train":
                segs[sg["name"]] = sg
        for sg in segs.values():
            MMAP_ROLE[sg["name"]] = sg["role"]
            def rows(k, sg=sg, mom=mom, pdg=pdg, amp=amp):  # bound now: the next cache rebinds the names
                k, P = min(k, sg["n"]), sg["P"]
                mo = np.asarray(mom[sg["p0"]: sg["p0"] + k * P], np.float64).reshape(k, 4 * P)
                pd = np.asarray(pdg[sg["p0"]: sg["p0"] + k * P], np.float64).reshape(k, P)
                return np.column_stack([mo, pd, np.asarray(amp[sg["ev0"]: sg["ev0"] + k], np.float64)])
            out[sg["name"]] = rows
    return out


# D566 attempt C: the regressor's descriptors and network, with a sampling-independent target. Per training process,
# log|M|^2 = c_p + b_p . (x - x0) fitted on its events (x per event: the features below; x0 and the feature scales are
# constants of the method, the medians and std over all training events, never a statistic of the dataset scored);
# c_p, b_p are properties of the matrix element where the form holds (a conditional-mean fit does not move when the
# events are drawn with another density over x), and the network maps (process, diagrams) -> (c_p, b_p). Prediction
# per event: c^ + b^ . (x - x0).
ANCH = {"C_s": ("L",), "C_sqE": ("L", "q", "E"), "C_sqEP": ("L", "q", "E", "logP")}
# D566 attempt D (--anchor process): the same form at a process-intrinsic anchor x0_p = a ridge map (RidgeCV on the
# standardized process + diagram descriptors) of the training processes' median x, cross-fitted by relabelling class on
# them and fitted on all of them for the held-out: a function of the process alone, so c_p is the level at the
# process's own typical point (little leverage) and no statistic of the scored dataset enters. *_own (x0_p = the pool's
# own median, a statistic of the scored pool) is a diagnostic of the anchor map's error only.
ANCH_D = {"D_s": ("L",), "D_sqE": ("L", "q", "E"), "D_s_own": ("L",), "D_sqE_own": ("L", "q", "E")}
SUBSETS = ("all", "s_lo", "s_hi", "q_hard", "q_soft")
NENS = 10


def subsets(d):
    """the same process sampled another way, exactly: an unweighted pool restricted to a window is the pool of that
    window. s_lo / s_hi: sqrt(s_hat) below / above the pool's median (another collider window); q_hard: the smallest
    pair invariant over s_hat above its median (a harder cut); q_soft: its lowest fifth (the singular region)"""
    L, q = d["L"], d["q"]
    return {"all": np.ones(len(L), bool), "s_lo": L <= np.median(L), "s_hi": L > np.median(L),
            "q_hard": q >= np.median(q), "q_soft": q <= np.percentile(q, 20)}


def anchor_fit(X, y, lam=1e-3):
    """(c, b) of y = c + b . X (X already anchored and scaled): ridge on b only, so a feature constant in the process
    gets b = 0 and leaves c alone"""
    xm, ym = X.mean(0), y.mean()
    Z = X - xm
    b = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(X.shape[1]), Z.T @ (y - ym))
    return np.concatenate([[ym - xm @ b], b])


def anchored(D, tr, ho, new, rel, meta, grp, pfeat, lev, events, nboot=30, mode="global"):
    ANCH = globals()["ANCH"] if mode == "global" else ANCH_D
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(500, 2)); assert np.allclose(anchor_fit(np.column_stack([Xs, np.ones(500)]),
                                                                 1.5 + Xs @ [2.0, -1.0], 1e-9), [1.5, 2, -1, 0], atol=1e-6)
    out = {"events": events, "mmap_roles": dict(sorted(MMAP_ROLE.items())), "n": {"train": len(tr), "new": len(new), "relabels": len(ho) - len(new)},
           "features": {k: list(v) for k, v in ANCH.items()}}
    pred = {"shared": {}, "regressor": {}, "oracle_const": {}}
    My = np.mean([D[n]["y"].mean() for n in tr])
    for n in D:
        k = len(D[n]["y"])
        pred["shared"][n] = np.full(k, My)
        pred["regressor"][n] = np.full(k, lev[n])
        pred["oracle_const"][n] = np.full(k, D[n]["y"].mean())   # the best any one number per pool can do on the pool
    Xp_tr = np.array([pfeat(n, True) for n in tr]); Xp_ho = np.array([pfeat(n, True) for n in ho])

    def net_fit(Ttr, seeds):  # the regressor's network on a (vector) per-process target: cross-fit train, full fit held out
        tm, ts = Ttr.mean(0), Ttr.std(0) + 1e-12
        cf = np.zeros_like(Ttr)
        for i, j in GroupKFold(5).split(Xp_tr, Ttr, grp):
            sc = StandardScaler().fit(Xp_tr[i])
            cf[j] = np.mean([mlp(r).fit(sc.transform(Xp_tr[i]), (Ttr[i] - tm) / ts).predict(sc.transform(Xp_tr[j]))
                             for r in seeds], 0).reshape(len(j), -1) * ts + tm
        sc = StandardScaler().fit(Xp_tr)
        hp = np.mean([mlp(r).fit(sc.transform(Xp_tr), (Ttr - tm) / ts).predict(sc.transform(Xp_ho))
                      for r in seeds], 0).reshape(len(ho), -1) * ts + tm
        return {**dict(zip(tr, cf)), **dict(zip(ho, hp))}

    def anchor_map(med):  # x0_p from the process alone: cross-fit on train, full fit for the held out
        a = np.zeros((len(D), med.shape[1]))
        it = {n: i for i, n in enumerate(D)}
        itr, iho = [it[n] for n in tr], [it[n] for n in ho]
        def fit(i):
            sc = StandardScaler().fit(Xp_tr[i])
            return sc, RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xp_tr[i]), med[i])
        for i, j in GroupKFold(5).split(Xp_tr, med, grp):
            sc, r = fit(i)
            a[np.array(itr)[j]] = r.predict(sc.transform(Xp_tr[j])).reshape(len(j), -1)
        sc, r = fit(np.arange(len(tr)))
        a[iho] = r.predict(sc.transform(Xp_ho)).reshape(len(ho), -1)
        return {n: a[it[n]] for n in D}

    X, T = {}, {}
    for tag, cols in ANCH.items():
        allx = np.concatenate([np.column_stack([D[n][c] for c in cols]) for n in tr])
        x0, sx = np.median(allx, 0), allx.std(0)
        own = {n: np.median(np.column_stack([D[n][c] for c in cols]), 0) for n in D}
        if mode == "global":
            x0p = {n: x0 for n in D}
        elif tag.endswith("_own"):
            x0p = own
        else:
            x0p = anchor_map(np.array([own[n] for n in tr]))
            out.setdefault("anchor_err", {})[tag] = {   # the map's miss of the pool's own median, in units of sx
                g: np.median(np.abs((np.array([x0p[n] for n in ns]) - np.array([own[n] for n in ns])) / sx), 0).tolist()
                for g, ns in (("new", new), ("train_crossfit", tr))}
        out.setdefault("anchor", {})[tag] = {"x0": x0.tolist(), "scale": sx.tolist()}
        X[tag] = {n: (np.column_stack([D[n][c] for c in cols]) - x0p[n]) / sx for n in D}
        T[tag] = {n: anchor_fit(X[tag][n], D[n]["y"]) for n in D}
        pred[tag + "_fit"] = {n: T[tag][n][0] + X[tag][n] @ T[tag][n][1:] for n in D}   # the form's own floor
    sub = {n: subsets(D[n]) for n in D}

    def e_of(p_, n, idx=None):
        y, p = D[n]["y"], p_[n]
        S = sub[n] if idx is None else subsets({"L": D[n]["L"][idx], "q": D[n]["q"][idx]})
        if idx is not None:
            y, p = y[idx], p[idx]
        sd = D[n]["y"].std()
        return {s: float((p[m] - y[m]).mean() ** 2 / sd ** 2) for s, m in S.items()}

    # ten 3-seed ensembles (seeds 3k..3k+2), as norm_explore4's read of the regressor (results.tex l.8510-8514: its
    # 0.045 is one draw, ten ensembles median 0.087, 0.016 to 0.19); regressor_ens = the same network on the pool mean
    ens = {c: [] for c in ["regressor_ens"] + list(ANCH) + [t + "_trueb" for t in ANCH]
           + ([t + "_bcls" for t in ANCH] if mode != "global" else [])}
    # attempt D: slopes shrunk fully to the mean true slope of the training processes with the same multiplicity
    bcls = {t: {k: np.mean([T[t][n][1:] for n in tr if meta[n]["n_out"] == k], 0) for k in {meta[n]["n_out"] for n in tr}}
            for t in ANCH}
    ybar = np.array([[D[n]["y"].mean()] for n in tr])
    for k in range(NENS):
        seeds = range(3 * k, 3 * k + 3)
        P = {"regressor_ens": {n: np.full(len(D[n]["y"]), v[0]) for n, v in net_fit(ybar, seeds).items()}}
        for tag in ANCH:
            Th = net_fit(np.array([T[tag][n] for n in tr]), seeds)
            P[tag] = {n: Th[n][0] + X[tag][n] @ Th[n][1:] for n in D}
            P[tag + "_trueb"] = {n: Th[n][0] + X[tag][n] @ T[tag][n][1:] for n in D}   # level predicted, slopes true
            if mode != "global":
                P[tag + "_bcls"] = {n: Th[n][0] + X[tag][n] @ bcls[tag][meta[n]["n_out"]] for n in D}
        for c, pc in P.items():
            ens[c].append({g: {s: float(np.median([e_of(pc, n)[s] for n in ns])) for s in SUBSETS}
                           for g, ns in (("new", new), ("train_crossfit", tr))})
            if k == 0:
                pred[c] = pc
        print(f"ensemble {k}: " + ", ".join(f"{c} {ens[c][-1]['new']['all']:.3g}" for c in ens), flush=True)
    out["ensembles"] = {c: {g: {s: {"median": float(np.median([e[g][s] for e in v])),
                                    "min": float(min(e[g][s] for e in v)), "max": float(max(e[g][s] for e in v)),
                                    "all": [e[g][s] for e in v]} for s in SUBSETS} for g in ("new", "train_crossfit")}
                        for c, v in ens.items()}
    cands = list(pred)
    R = {c: {n: e_of(pred[c], n) for n in D} for c in cands}
    for c in cands:
        for n in D:
            R[c][n]["spread"] = float((pred[c][n] - D[n]["y"]).std() / D[n]["y"].std())
    rel_ho = [n for n in ho if n in rel]
    summ = {c: {g: {s: float(np.median([R[c][n][s] for n in ns])) for s in SUBSETS + ("spread",)}
                for g, ns in (("new", new), ("relabels", rel_ho), ("train_crossfit", tr))} for c in cands}
    # event-bootstrap noise of the new-set medians (ensemble 0's predictions fixed, events resampled, subsets redrawn)
    bs = {c: {s: [] for s in SUBSETS} for c in cands}
    for _ in range(nboot):
        idx = {n: rng.integers(0, len(D[n]["y"]), len(D[n]["y"])) for n in new}
        for c in cands:
            ee = [e_of(pred[c], n, idx[n]) for n in new]
            for s in SUBSETS:
                bs[c][s].append(np.median([x[s] for x in ee]))
    for c in cands:
        summ[c]["new"]["boot_sd"] = {s: float(np.std(bs[c][s])) for s in SUBSETS}
    e_sh, e_rg = summ["shared"]["new"]["all"], summ["regressor"]["new"]["all"]
    print(f"check: shared e median {e_sh:.3g} (2026-10-22: 47), regressor {e_rg:.3g} (level_source.py: 0.044)")
    assert abs(e_sh / 47 - 1) < 0.10 and abs(e_rg / 0.044 - 1) < 0.10, (e_sh, e_rg)
    fams = {}
    for n in new:
        fams.setdefault(meta[n]["test_group"], []).append(n)
    out["summary"] = summ
    out["families"] = {c: {g: {s: float(np.median([R[c][n][s] for n in ns])) for s in SUBSETS} | {"n": len(ns)}
                           for g, ns in fams.items()} for c in cands}
    out["per_process_new"] = {meta[n]["process"]: {c: R[c][n] for c in cands} for n in new}
    json.dump(out, open(os.path.join(HERE, "data", "perevent_anchored" + ("_C" if mode == "global" else "_D") + ".json"), "w"),
              indent=1)
    for c in cands:
        print(c, {g: {s: round(v, 4) for s, v in summ[c][g].items() if s != "boot_sd"} for g in ("new", "relabels")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2000)
    ap.add_argument("--surrogate", action="store_true")
    ap.add_argument("--model", choices=("hgb", "mlp"), default="hgb")
    ap.add_argument("--mmap", nargs="+", help="read each process's first train-role events from these memmap base "
                    "caches (data_mmap: <key>/<depth> dirs) instead of $SCRATCH/amp_data_cache's pools")
    ap.add_argument("--anchored", action="store_true", help="D566 attempt C, output data/perevent_anchored_C.json")
    ap.add_argument("--anchor", choices=("global", "process"), default="global", help="--anchored's x0: one constant "
                    "(attempt C) or a process-intrinsic x0_p (attempt D, output data/perevent_anchored_D.json)")
    ap.add_argument("--search", type=int, default=0, help="step 20: DyHPO GP trials on the content-signature "
                    "cross-fit (1 network seed per fit), then the best read as candidate B[_mlp]_tuned")
    a = ap.parse_args()
    if a.surrogate:
        return from_surrogate()
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    lev = json.load(open(os.path.join(HERE, "data", "level_file_B_diag.json")))["levels"]
    cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
    files = sorted(os.listdir(cache))
    mm = mmap_pools(a.mmap) if a.mmap else {}
    D, missing = {}, []
    for split, f in (("train", "catalog_daniel_train"), ("heldout", "catalog_daniel_heldout")):
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f + ".yaml")))["processes"]:
            n = p["name"]
            g = [x for x in files if x.startswith(n + "_") and "_train_" in x and x.endswith("amplitudes.npy")]
            if a.mmap:
                if n not in mm:
                    missing.append(n); continue
                x = mm[n](a.events)
            elif not g:
                missing.append(n); continue
            else:
                x = np.asarray(np.load(os.path.join(cache, g[0]), mmap_mode="r")[: a.events], dtype=np.float64)
            y, L, Lmin, E, P, pdg = events(x)
            lp = log_prop(P, pdg, os.path.join(ROOT, "data", "diagrams", n + ".diagrams.json"))
            if lp is None or not np.isfinite(lp).all():
                missing.append(n); continue
            D[n] = {"split": split, "y": y, "L": L, "q": Lmin - L, "E": E, "logP": lp}
    tr = sorted(n for n in D if D[n]["split"] == "train")
    ho = sorted(n for n in D if D[n]["split"] == "heldout")
    new = [n for n in ho if n not in rel]
    print(f"{len(tr)} train, {len(ho)} held-out ({len(new)} new) pools read; missing or no sidecar: {missing}")
    # a site without catalog_daniel's train-role pools (CC after D131/D134) must stop here, not in the cross-fit
    assert not missing, f"{len(missing)} pools missing or without sidecar under {cache}: run where the pools are"
    grp_p = np.array([meta[n]["weak_key"] for n in tr])

    def pfeat(n, diag):
        f = list(sn.features(meta[n]))
        if diag:  # norm_explore3.py's diagram descriptors
            l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
            f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
                  float(mi[int(np.argmax(l))]), float(max(mi))]
        return np.array(f, float)

    if a.anchored:
        return anchored(D, tr, ho, new, rel, meta, grp_p, pfeat, lev, a.events, mode=a.anchor)
    global CANDS
    if a.model == "mlp":
        CANDS = CANDS[:-1] + ("B_mlp",)
    if a.search:
        CANDS = CANDS[:-1] + (CANDS[-1] + "_tuned",)
    B = CANDS[-1]
    pred = {c: {} for c in CANDS}
    My = np.mean([D[n]["y"].mean() for n in tr])
    r_mean = {n: float((D[n]["y"] - D[n]["logP"]).mean()) for n in D}
    Mr = np.mean([r_mean[n] for n in tr])
    Xc_tr = np.array([pfeat(n, False) for n in tr]); Xc_ho = np.array([pfeat(n, False) for n in ho])
    cf, hf = level_net(Xc_tr, np.array([r_mean[n] - Mr for n in tr]), grp_p, Xc_ho)
    lvA = {**dict(zip(tr, cf + Mr)), **dict(zip(ho, hf + Mr))}
    for n in D:
        d = D[n]
        pred["shared"][n] = np.full(len(d["y"]), My)
        pred["regressor"][n] = np.full(len(d["y"]), lev[n])
        pred["A0"][n] = d["logP"] + Mr
        pred["A"][n] = d["logP"] + lvA[n]

    def rows(ns):
        X = np.concatenate([np.column_stack([np.repeat(pfeat(n, True)[None], len(D[n]["y"]), 0), D[n]["L"],
                                             D[n]["q"], D[n]["E"], D[n]["logP"]]) for n in ns])
        return X, np.concatenate([D[n]["y"] for n in ns])

    def hgb(learning_rate=0.1, max_leaf_nodes=63, max_iter=300, l2_regularization=0.0, seeds=None):  # step 17's
        return HistGradientBoostingRegressor(max_iter=int(max_iter), learning_rate=float(learning_rate),
                                             max_leaf_nodes=int(max_leaf_nodes),
                                             l2_regularization=float(l2_regularization), random_state=0)

    make = Net if a.model == "mlp" else hgb

    def crossfit(hp, groups, seeds=3):
        out = {}
        for i, j in GroupKFold(5).split(tr, groups=groups):
            m = make(**hp, seeds=seeds).fit(*rows([tr[k] for k in i]))
            for k in j:
                out[tr[k]] = m.predict(rows([tr[k]])[0])
        return out

    def cv_e(p):  # median level error of the cross-fitted training processes
        return float(np.median([reads(p[n] - D[n]["y"], D[n]["y"], *weights(D[n]["L"], D[n]["q"]))["e"] for n in tr]))

    best, search = {}, None
    if a.search:
        sys.path.insert(0, os.path.join(ROOT, "sweep"))
        from dyhpo_sampler import DyHPOSampler
        sig = np.array([content_sig(meta[n]["process"]) for n in tr])
        # proxy check: step 17's model on this cross-fit, to set beside its new-set level error (tree 3.64, net 1.63)
        search = {"objective": "median level error, cross-fit by content signature (5 folds, 1 seed)",
                  "n_signatures": int(len(set(sig))), "default_sig_cv": cv_e(crossfit({}, sig, 1)), "trials": []}
        print("default (step 17) on the signature cross-fit:", search["default_sig_cv"], flush=True)
        out_dir = os.path.join(os.environ["SCRATCH"], "perevent_search_" + a.model)
        os.makedirs(out_dir, exist_ok=True)
        S = DyHPOSampler(SPACE[a.model], {"t_steps": [1]}, n_candidates=2000, seed=42, output_path=out_dir,
                         n_startup=3, surrogate="gp")
        for t in range(a.search):
            idx, hp, ts = S.suggest()
            v = cv_e(crossfit(hp, sig, 1))
            S.observe(idx, ts, v)
            search["trials"].append({"hp": {k: (v_ if isinstance(v_, str) else float(v_)) for k, v_ in hp.items()},
                                     "sig_cv_e": v})
            print(f"trial {t}: {hp} -> {v:.4g}", flush=True)
        best = min(search["trials"], key=lambda r: r["sig_cv_e"])["hp"]
        search["best"] = best
    pred[B] = crossfit(best, grp_p)
    m = make(**best).fit(*rows(tr))
    for n in ho:
        pred[B][n] = m.predict(rows([n])[0])

    R = {c: {} for c in CANDS}
    rng = np.random.default_rng(0)
    for n in D:
        wf, ws = weights(D[n]["L"], D[n]["q"])
        for c in CANDS:
            R[c][n] = reads(pred[c][n] - D[n]["y"], D[n]["y"], wf, ws)
        if n in new:  # the shift's noise at this event count: its std over 30 bootstrap resamples of the events
            bs = [rng.integers(0, len(D[n]["y"]), len(D[n]["y"])) for _ in range(30)]
            for c in CANDS:
                v = []
                for i in bs:
                    w1, w2 = weights(D[n]["L"][i], D[n]["q"][i])
                    v.append(reads(pred[c][n][i] - D[n]["y"][i], D[n]["y"][i], w1, w2)["shift_flat"])
                R[c][n]["shift_flat_sd"] = float(np.std(v))

    def med(c, ns, k):
        return float(np.median([R[c][n][k] for n in ns]))

    fams = {}
    for n in new:
        fams.setdefault(meta[n]["test_group"], []).append(n)
    ggtbw = [n for n in new if n in ("dc_gg_tbxwm", "dc_gg_btxwp")]
    rel_ho = [n for n in ho if n in rel]
    summary = {c: {s: {k: med(c, ns, k) for k in ("e", "bias2", "shift_flat", "shift_sing", "spread")}
                   for s, ns in (("new", new), ("relabels", rel_ho), ("train_crossfit", tr),
                                 ("new_no_ggtbw", [n for n in new if n not in ggtbw]))} for c in CANDS}
    for c in CANDS:
        summary[c]["new"]["shift_flat_sd"] = med(c, new, "shift_flat_sd")
        summary[c]["families"] = {g: {k: med(c, ns, k) for k in ("e", "shift_flat", "shift_sing")} | {"n": len(ns)}
                                  for g, ns in fams.items()}
    # the references reproduce the 2026-10-22 and step-15 numbers (design check): shared 47, regressor 0.044
    e_sh, e_rg = summary["shared"]["new"]["e"], summary["regressor"]["new"]["e"]
    print(f"check: shared e median {e_sh:.3g} (2026-10-22: 47), regressor {e_rg:.3g} (level_source.py: 0.044)")
    assert abs(e_sh / 47 - 1) < 0.10 and abs(e_rg / 0.044 - 1) < 0.10, (e_sh, e_rg)
    out = {"events": a.events, "model": a.model, "missing": missing, "search": search,
           "n": {"train": len(tr), "new": len(new), "relabels": len(rel_ho)},
           "summary": summary,
           "per_process_new": {meta[n]["process"]: {c: R[c][n] for c in CANDS} | {"family": meta[n]["test_group"]}
                               for n in new}}
    json.dump(out, open(os.path.join(HERE, "data", "perevent_level%s%s.json" % ("_mlp" if a.model == "mlp" else "",
                                                                         "_search" if a.search else "")), "w"), indent=1)
    for c in CANDS:
        print(c, {s: {k: round(v, 4) for k, v in summary[c][s].items()} for s in ("new", "relabels", "train_crossfit")})
        for g, f in summary[c]["families"].items():
            print(f"   {str(g)[:40]:40s}", {k: round(v, 4) for k, v in f.items()})


if __name__ == "__main__":
    main()
