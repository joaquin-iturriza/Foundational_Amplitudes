"""D83, the CPU exploration of a data-free catalog normalization (the user: "try many things out"). Every target is
    y' = log|M|^2 - a (4 - N) log s_hat - c E - o_p
with E the soft/collinear proximity of norm_moments.py and o_p a per-process offset that needs no data of the process,
built from: log C_p, the leading diagram's coupling product, or log sum_d C_d over its tree diagrams
(diagram_couplings.py); log IDEN, MadGraph's averaging over initial spins and colours and its symmetry factor for
identical final particles, by which the stored |M|^2 is divided; log of the product of the final particles' colour
dimensions (the colour sum). a and c are fixed (a = 1, c = 0) or fitted once for the whole catalog on the training
processes, together with one reference scale, by least squares on the process levels. Per target:
  between   the std of the training processes' levels (what one shared standardization leaves the network to learn)
  within    the median within-process std
  new/rel   on the 43 new and the 96 relabelled held-out processes, the median level error (m_p - M)^2 / sigma_p^2
            against the shared mean M, a network that learned no level
  samp      the median over training processes of |level reweighted flat in log s_hat - level as sampled| / sigma_p:
            how much the level moves when the pool's s_hat sampling changes
Then a small network (sklearn MLPRegressor, the process features of stats_net.py) on the best target's residual
level, cross-fitted by relabelling class, scored the same way. Input data/norm_moments.json (CC), output
data/norm_explore.json and figs/norm_explore_{a,b}.
    python analysis/catalog_daniel/norm_explore.py
"""
import itertools, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402

SPIN = {21: 2, 22: 2, 23: 3, 24: 3, 25: 1}            # fermions 2
COLOR = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 21: 8}  # else 1


def iden(pdg_signed_in, pdg_signed_out):
    """MadGraph's IDEN: initial spin x colour averaging times n! for each set of identical final particles."""
    v = 1.0
    for p in pdg_signed_in:
        v *= SPIN.get(abs(p), 2) * COLOR.get(abs(p), 1)
    from collections import Counter
    for c in Counter(pdg_signed_out).values():
        v *= math.factorial(c)
    return v


def signed_pdgs(meta):
    import stats_net as sn
    ini, fin = meta["process"].split(" > ")
    return [sn.pdg(p) for p in ini.split()], [sn.pdg(p) for p in fin.split()]


def main():
    mo = json.load(open(os.path.join(HERE, "data", "norm_moments.json")))
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    tr, ho = sorted(mo["train"]), sorted(mo["heldout"])
    allp = tr + ho
    R = {**mo["train"], **mo["heldout"]}
    off = {}
    for n in allp:
        lc = np.array(dc[n]["logc2"])
        ins, outs = signed_pdgs(meta[n])
        off[n] = {"lead": float(lc.max()), "sum": float(lc.max() + np.log(np.exp(lc - lc.max()).sum())),
                  "iden": math.log(iden(ins, outs)),
                  "col": float(sum(math.log(COLOR.get(abs(p), 1)) for p in outs))}
    k = {n: 4 - R[n]["N"] for n in allp}

    def level_and_sd(n, a, c, o, ls0=0.0):
        w = np.array([1.0, -a * k[n], -c])
        mean = float(np.dot(R[n]["mean"], w)) - o + a * k[n] * ls0
        sd = float(np.sqrt(max(w @ np.array(R[n]["cov"]) @ w, 1e-12)))
        bm, bc = np.array(R[n]["bin_mean"]), np.array(R[n]["bin_count"])
        flat = float((bm[bc > 0] @ w).mean()) - o + a * k[n] * ls0
        return mean, sd, flat

    def score(a, c, ofun, fit=False):
        """fit: a, c and a reference (a k log s0 term) fitted on the training levels by least squares."""
        if fit:
            X = np.array([[k[n] * R[n]["mean"][1], k[n], R[n]["mean"][2], 1.0] for n in tr])
            yv = np.array([R[n]["mean"][0] - ofun(n) for n in tr])
            coef = np.linalg.lstsq(X, yv, rcond=None)[0]
            a, kls0, c = coef[0], -coef[1], coef[2]
            ls0 = kls0 / a if abs(a) > 1e-9 else 0.0
        else:
            ls0 = None
        rows = {n: level_and_sd(n, a, c, ofun(n), 0.0 if ls0 is None else ls0) for n in allp}
        m = np.array([rows[n][0] for n in tr]); s = np.array([rows[n][1] for n in tr])
        if ls0 is None:          # a fixed: the reference scale is still one number, the mean level absorbs it only
            pass                 # for a = 0; with a = 1 pick the s0 that makes the training levels closest
        if not fit and a != 0:
            kk = np.array([k[n] for n in tr], float)
            g = np.linspace(0, np.log(1e8), 801)
            best = min(g, key=lambda q: np.std(m + a * kk * q))
            rows = {n: level_and_sd(n, a, c, ofun(n), best) for n in allp}
            m = np.array([rows[n][0] for n in tr]); ls0 = best
        M = float(m.mean())
        lv = {n: (rows[n][0] - M) ** 2 / rows[n][1] ** 2 for n in ho}
        new = [n for n in ho if n not in rel]
        samp = np.median([abs(rows[n][2] - rows[n][0]) / rows[n][1] for n in tr])
        return dict(a=float(a), c=float(c), sqrt_s0=float(np.exp((ls0 or 0) / 2)), between=float(m.std()),
                    within=float(np.median([rows[n][1] for n in tr])),
                    new=float(np.median([lv[n] for n in new])), new90=float(np.quantile([lv[n] for n in new], .9)),
                    rel=float(np.median([lv[n] for n in ho if n in rel])), samp=float(samp),
                    lv=[lv[n] for n in new], levels={n: rows[n][0] - M for n in allp},
                    sds={n: rows[n][1] for n in allp})

    offs = {"none": lambda n: 0.0, "C_lead": lambda n: off[n]["lead"], "C_sum": lambda n: off[n]["sum"],
            "C_lead-iden": lambda n: off[n]["lead"] - off[n]["iden"],
            "C_lead-iden+col": lambda n: off[n]["lead"] - off[n]["iden"] + off[n]["col"],
            "C_sum-iden+col": lambda n: off[n]["sum"] - off[n]["iden"] + off[n]["col"]}
    res = {}
    for (oname, ofun), (dname, a, c, fit) in itertools.product(
            offs.items(), (("raw", 0.0, 0.0, False), ("shat", 1.0, 0.0, False), ("fit", 1.0, 0.0, True))):
        res[f"{oname}|{dname}"] = score(a, c, ofun, fit)
    print(f"{'target':28s} {'a':>6s} {'c':>6s} {'sqrt s0':>8s} {'between':>8s} {'within':>7s} {'new':>7s} {'new90':>7s} "
          f"{'rel':>6s} {'samp':>6s}")
    for kname, v in sorted(res.items(), key=lambda kv: kv[1]["new"]):
        print(f"{kname:28s} {v['a']:6.2f} {v['c']:6.2f} {v['sqrt_s0']:8.3g} {v['between']:8.2f} {v['within']:7.2f} "
              f"{v['new']:7.3g} {v['new90']:7.3g} {v['rel']:6.3g} {v['samp']:6.3f}")

    # a small network on the best target's residual level (cross-fitted by relabelling class)
    best = min(res, key=lambda q: res[q]["new"])
    import stats_net as sn
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    X = {n: sn.features(meta[n]) for n in allp}
    Xtr = np.array([X[n] for n in tr]); ytr = np.array([res[best]["levels"][n] for n in tr])
    grp = np.array([meta[n]["weak_key"] for n in tr])

    def mlp():
        return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                            early_stopping=True, validation_fraction=0.15, random_state=0)

    sc = StandardScaler().fit(Xtr)
    cf = np.zeros(len(tr))
    for i, j in GroupKFold(5).split(Xtr, ytr, grp):
        s2 = StandardScaler().fit(Xtr[i]); cf[j] = np.mean([mlp().set_params(random_state=r).fit(s2.transform(Xtr[i]), ytr[i])
                                                             .predict(s2.transform(Xtr[j])) for r in range(3)], 0)
    sdtr = np.array([res[best]["sds"][n] for n in tr])
    cf_lvl = (cf - ytr) ** 2 / sdtr ** 2
    nets = [mlp().set_params(random_state=r).fit(sc.transform(Xtr), ytr) for r in range(3)]
    new = [n for n in ho if n not in rel]
    pho = {n: float(np.mean([m_.predict(sc.transform(X[n][None]))[0] for m_ in nets])) for n in ho}
    lv = {n: (pho[n] - res[best]["levels"][n]) ** 2 / res[best]["sds"][n] ** 2 for n in ho}
    nn = dict(target=best, crossfit=float(np.median(cf_lvl)), crossfit90=float(np.quantile(cf_lvl, .9)),
              new=float(np.median([lv[n] for n in new])), new90=float(np.quantile([lv[n] for n in new], .9)),
              rel=float(np.median([lv[n] for n in ho if n in rel])), lv=[lv[n] for n in new])
    print(f"network on {best}: cross-fit level error median {nn['crossfit']:.3g} (90% {nn['crossfit90']:.3g}); "
          f"43 new {nn['new']:.3g} (90% {nn['new90']:.3g}); 96 relabels {nn['rel']:.3g}")
    worst = sorted(new, key=lambda n: -lv[n])[:6]
    print("worst new with the network:", "; ".join(f"{meta[n]['process']} {lv[n]:.3g}" for n in worst))
    json.dump({"targets": {q: {kk: vv for kk, vv in v.items() if kk not in ("levels", "sds")} for q, v in res.items()},
               "network": nn}, open(os.path.join(HERE, "data", "norm_explore.json"), "w"))

    figs = ps.panels(2)
    (f1, a1), (f2, a2) = figs
    show = [("none|raw", ps.C.grey, r"$\log|\mathcal{M}|^2$"), ("C_lead|shat", ps.C.blue, "A"),
            (best, ps.C.vermillion, "best data-free")]
    for q, col, lab in show:
        v = res[q]
        l = np.sort(v["lv"]); a2.plot(l, np.arange(1, len(l) + 1) / len(l), color=col, label=lab)
        lev = np.sort([v["levels"][n] for n in tr]); a1.plot(lev, np.arange(1, len(lev) + 1) / len(lev), color=col, label=lab)
    l = np.sort(nn["lv"]); a2.plot(l, np.arange(1, len(l) + 1) / len(l), color=ps.C.green, label="best + network")
    a1.set_xlabel(r"level $m_p - M$"); a1.set_ylabel("fraction of processes below")
    a2.set_xscale("log"); a2.set_xlabel(r"level error $(m_p-M)^2/\sigma_p^2$"); a2.set_ylabel("fraction of the 43 below")
    ps.legend_strip(a2, os.path.join(HERE, "figs", "norm_explore_legend"), ncol=4)
    ps.save_panels(figs, os.path.join(HERE, "figs", "norm_explore"))
    print("best data-free target:", best)


if __name__ == "__main__":
    main()
