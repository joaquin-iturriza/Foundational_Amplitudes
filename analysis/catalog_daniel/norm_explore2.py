"""D83 exploration, second pass (norm_moments2.py): per-event scales that see the singular regions. A target is
    y' = log|M|^2 - sum_k c_{N,k} X_k - o_p - b_N
X_k per-event scales (L = log s_hat, Lg the mean log pair invariant, Lmin the smallest, Lnn the nearest-neighbour mean,
E the soft/collinear proximity), c_{N,k} either fixed by the mass dimension (4 - N on one scale) or one set per
particle count N fitted on the training pools by the within-process regression (each process's covariance, summed;
a level difference between processes does not enter it), o_p the data-free per-process offset of norm_explore.py
(leading coupling product, MadGraph's IDEN, final colour sum) or none, b_N one intercept per N or none (fitted on the
training levels). Scored as norm_explore.py (between, within, new, rel, samp), then the small network on the best
target's residual level, cross-fitted by relabelling class. Output: data/norm_explore2.json, figs/norm_explore2_{a,b}.
    python analysis/catalog_daniel/norm_explore2.py
"""
import itertools, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402
from norm_explore import iden, signed_pdgs, COLOR  # noqa: E402


def main():
    mo = json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))
    V = mo["vars"]
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    tr, ho = sorted(mo["train"]), sorted(mo["heldout"])
    R = {**mo["train"], **mo["heldout"]}
    new = [n for n in ho if n not in rel]
    Ns = sorted({R[n]["N"] for n in R})
    off, off_lead = {}, {}
    for n in R:
        ins, outs = signed_pdgs(meta[n])
        off[n] = max(dc[n]["logc2"]) - math.log(iden(ins, outs)) + sum(math.log(COLOR.get(abs(p), 1)) for p in outs)
        off_lead[n] = max(dc[n]["logc2"])          # A as launched (job_catalog_norm.sh): the coupling product only

    def coefs(xs, mode):
        """per N: the coefficient vector on the scales xs"""
        out = {}
        for N in Ns:
            if mode == "dim":                  # 4 - N on the first scale, nothing else
                out[N] = np.array([4 - N] + [0.0] * (len(xs) - 1))
                continue
            ix = [V.index(x) for x in xs]
            Cxx = sum(np.array(R[n]["cov"])[np.ix_(ix, ix)] * R[n]["n"] for n in tr if R[n]["N"] == N)
            Cxy = sum(np.array(R[n]["cov"])[ix, 0] * R[n]["n"] for n in tr if R[n]["N"] == N)
            out[N] = np.linalg.solve(Cxx + 1e-9 * np.eye(len(ix)), Cxy)
        return out

    def score(xs, mode, use_off, use_b, ls0=0.0):
        """ls0: log s0 (GeV^2) of the dimension factor (a dim-mode target on L): y - (4 - N)(L - ls0)"""
        cN = coefs(xs, mode) if xs else {N: np.zeros(0) for N in Ns}
        ix = [V.index(x) for x in xs]
        rows = {}
        for n in R:
            w = np.zeros(len(V)); w[0] = 1.0
            for c, i in zip(cN[R[n]["N"]], ix):
                w[i] -= c
            o = off_lead[n] if use_off == "lead" else (off[n] if use_off else 0.0)
            shift = (4 - R[n]["N"]) * ls0 if (mode == "dim" and xs[:1] == ["L"]) else 0.0
            mean = float(np.dot(R[n]["mean"], w)) - o + shift
            sd = float(np.sqrt(max(w @ np.array(R[n]["cov"]) @ w, 1e-12)))
            bm, bc = np.array(R[n]["bin_mean"]), np.array(R[n]["bin_count"])
            flat = float((bm[bc > 0] @ w).mean()) - o + shift
            rows[n] = [mean, sd, flat]
        if use_b:
            for N in Ns:
                bN = np.mean([rows[n][0] for n in tr if R[n]["N"] == N])
                for n in R:
                    if R[n]["N"] == N:
                        rows[n][0] -= bN; rows[n][2] -= bN
        m = np.array([rows[n][0] for n in tr]); M = float(m.mean())
        lv = {n: (rows[n][0] - M) ** 2 / rows[n][1] ** 2 for n in ho}
        return dict(xs=xs, mode=mode, off=use_off, bN=use_b, coefs={str(N): cN[N].tolist() for N in Ns},
                    between=float(m.std()), within=float(np.median([rows[n][1] for n in tr])),
                    new=float(np.median([lv[n] for n in new])), new90=float(np.quantile([lv[n] for n in new], .9)),
                    rel=float(np.median([lv[n] for n in ho if n in rel])),
                    samp=float(np.median([abs(rows[n][2] - rows[n][0]) / rows[n][1] for n in tr])),
                    lv=[lv[n] for n in new], levels={n: rows[n][0] - M for n in R}, sds={n: rows[n][1] for n in R})

    sets = [[], ["L"], ["Lg"], ["Lmin"], ["Lnn"], ["E"], ["L", "E"], ["Lg", "Lmin"], ["L", "Lg", "Lmin", "Lnn", "E"]]
    res = {}
    for xs, use_off, use_b in itertools.product(sets, (False, True), (False, True)):
        modes = ["fit"] if len(xs) > 1 else (["dim", "fit"] if xs else ["none"])
        for mode in modes:
            key = f"{'+'.join(xs) or '-'}|{mode}|{'off' if use_off else 'nooff'}|{'bN' if use_b else '-'}"
            res[key] = score(xs, mode, use_off, use_b)
    print(f"{'target':36s} {'between':>8s} {'within':>7s} {'new':>7s} {'new90':>7s} {'rel':>6s} {'samp':>6s}  coefs")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["new"])[:24]:
        print(f"{k:36s} {v['between']:8.2f} {v['within']:7.2f} {v['new']:7.3g} {v['new90']:7.3g} {v['rel']:6.3g} "
              f"{v['samp']:6.3f}  " + " ".join(f"N{N}:" + ",".join(f"{c:.2f}" for c in cs) for N, cs in v["coefs"].items()))
    print("lowest sampling sensitivity:")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["samp"])[:6]:
        print(f"  {k:36s} samp {v['samp']:.3f} within {v['within']:.2f} new {v['new']:.3g}")

    import stats_net as sn
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    X = {n: sn.features(meta[n]) for n in R}
    grp = np.array([meta[n]["weak_key"] for n in tr])

    def mlp(r):
        return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                            early_stopping=True, validation_fraction=0.15, random_state=r)

    def network(key):
        v = res[key]
        Xtr = np.array([X[n] for n in tr]); ytr = np.array([v["levels"][n] for n in tr])
        sd = np.array([v["sds"][n] for n in tr])
        cf = np.zeros(len(tr))
        for i, j in GroupKFold(5).split(Xtr, ytr, grp):
            s2 = StandardScaler().fit(Xtr[i])
            cf[j] = np.mean([mlp(r).fit(s2.transform(Xtr[i]), ytr[i]).predict(s2.transform(Xtr[j])) for r in range(3)], 0)
        sc = StandardScaler().fit(Xtr)
        nets = [mlp(r).fit(sc.transform(Xtr), ytr) for r in range(3)]
        lv = {n: (np.mean([m_.predict(sc.transform(X[n][None]))[0] for m_ in nets]) - v["levels"][n]) ** 2
              / v["sds"][n] ** 2 for n in ho}
        cfl = (cf - ytr) ** 2 / sd ** 2
        return dict(target=key, crossfit=float(np.median(cfl)), crossfit90=float(np.quantile(cfl, .9)),
                    new=float(np.median([lv[n] for n in new])), new90=float(np.quantile([lv[n] for n in new], .9)),
                    rel=float(np.median([lv[n] for n in ho if n in rel])), lv=[lv[n] for n in new],
                    worst=sorted(((meta[n]["process"], lv[n]) for n in new), key=lambda t: -t[1])[:6])

    res["L|dim|lead|-"] = score(["L"], "dim", "lead", False, math.log(289.0))   # A as launched: s0 = (17 GeV)^2
    res["L|dim|off|-|s0=289"] = score(["L"], "dim", True, False, math.log(289.0))
    top = [k for k, _ in sorted(res.items(), key=lambda kv: kv[1]["new"])[:3]] + ["-|none|nooff|-", "L|dim|lead|-"]
    nets = {k: network(k) for k in top}
    for k, nn in nets.items():
        print(f"network on {k}: cross-fit {nn['crossfit']:.3g} (90% {nn['crossfit90']:.3g}); 43 new {nn['new']:.3g} "
              f"(90% {nn['new90']:.3g}); 96 relabels {nn['rel']:.3g}; worst " +
              "; ".join(f"{p} {e:.3g}" for p, e in nn["worst"][:4]))
    json.dump({"targets": {q: {kk: vv for kk, vv in v.items() if kk not in ("levels", "sds")} for q, v in res.items()},
               "networks": nets}, open(os.path.join(HERE, "data", "norm_explore2.json"), "w"))

    best = top[0]
    bestnet = min(nets, key=lambda q: nets[q]["new"])
    figs = ps.panels(2)
    (f1, a1), (f2, a2) = figs
    show = [("-|none|nooff|-", ps.C.grey, "B"), ("L|dim|lead|-", ps.C.blue, "A"),
            ("L|dim|off|-", ps.C.vermillion, "A + IDEN, colour")]
    for q, col, lab in show:
        v = res[q]
        l = np.sort(v["lv"]); a2.plot(l, np.arange(1, len(l) + 1) / len(l), color=col, label=lab)
        lev = np.sort([v["levels"][n] for n in tr]); a1.plot(lev, np.arange(1, len(lev) + 1) / len(lev), color=col, label=lab)
    for q, col, ls, lab in (("-|none|nooff|-", ps.C.grey, "--", "B + network"),
                            ("L|dim|off|-", ps.C.vermillion, "--", "A + IDEN, colour + network")):
        l = np.sort(nets[q]["lv"]); a2.plot(l, np.arange(1, len(l) + 1) / len(l), ls, color=col, label=lab)
    a1.set_xlabel(r"level $m_p - M$"); a1.set_ylabel("fraction of processes below")
    a2.set_xscale("log"); a2.set_xlabel(r"level error $(m_p-M)^2/\sigma_p^2$"); a2.set_ylabel("fraction of the 43 below")
    ps.legend_strip(a2, os.path.join(HERE, "figs", "norm_explore2_legend"), ncol=3)
    ps.save_panels(figs, os.path.join(HERE, "figs", "norm_explore2"))
    print("best data-free:", best, "| best with network:", bestnet)


if __name__ == "__main__":
    main()
