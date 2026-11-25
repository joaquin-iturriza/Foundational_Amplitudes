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
             of log|M|^2 (HistGradientBoosting), cross-fitted by relabelling class on the training processes
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
Output: analysis/catalog_daniel/data/perevent_level.json; its summary on stdout.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2000)
    ap.add_argument("--surrogate", action="store_true")
    a = ap.parse_args()
    if a.surrogate:
        return from_surrogate()
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    lev = json.load(open(os.path.join(HERE, "data", "level_file_B_diag.json")))["levels"]
    cache = os.path.join(os.environ["SCRATCH"], "amp_data_cache")
    files = sorted(os.listdir(cache))
    D, missing = {}, []
    for split, f in (("train", "catalog_daniel_train"), ("heldout", "catalog_daniel_heldout")):
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f + ".yaml")))["processes"]:
            n = p["name"]
            g = [x for x in files if x.startswith(n + "_") and "_train_" in x and x.endswith("amplitudes.npy")]
            if not g:
                missing.append(n); continue
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
    grp_p = np.array([meta[n]["weak_key"] for n in tr])

    def pfeat(n, diag):
        f = list(sn.features(meta[n]))
        if diag:  # norm_explore3.py's diagram descriptors
            l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
            f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
                  float(mi[int(np.argmax(l))]), float(max(mi))]
        return np.array(f, float)

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

    def hgb():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)

    for i, j in GroupKFold(5).split(tr, groups=grp_p):
        Xi, yi = rows([tr[k] for k in i])
        m = hgb().fit(Xi, yi)
        for k in j:
            pred["B"][tr[k]] = m.predict(rows([tr[k]])[0])
    m = hgb().fit(*rows(tr))
    for n in ho:
        pred["B"][n] = m.predict(rows([n])[0])

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
    out = {"events": a.events, "missing": missing, "n": {"train": len(tr), "new": len(new), "relabels": len(rel_ho)},
           "summary": summary,
           "per_process_new": {meta[n]["process"]: {c: R[c][n] for c in CANDS} | {"family": meta[n]["test_group"]}
                               for n in new}}
    json.dump(out, open(os.path.join(HERE, "data", "perevent_level.json"), "w"), indent=1)
    for c in CANDS:
        print(c, {s: {k: round(v, 4) for k, v in summary[c][s].items()} for s in ("new", "relabels", "train_crossfit")})
        for g, f in summary[c]["families"].items():
            print(f"   {str(g)[:40]:40s}", {k: round(v, 4) for k, v in f.items()})


if __name__ == "__main__":
    main()
