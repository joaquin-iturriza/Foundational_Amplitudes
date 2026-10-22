"""P1 (the user, 2026-10-22: physics priors for a process's statistics without true data): how well the momentum-only
surrogates of norm_surrogate.py place each held-out process's true mean and std of log|M|^2. Per surrogate, two
linear maps fitted on the 1352 training processes (each surrogate's per-process mean -> true mean, log surrogate std
-> log true std), optionally with one intercept per multiplicity; scored on the 43 new and 96 relabels: level error
(predicted - true mean)^2 / true var (as g's 0.32 and the k-event mean's ~1/k), |log std error|, and the median
per-event correlation of the surrogate with the truth (what it carries of the shape). Reference: a constant (the
training mean / mean log std), g (stats_net.json), and the k = 3 few-shot mean and std (expected level error 1/3).
Also P2 in multiplicity: the maps fitted on 2->2 only and scored on 2->3 (train and held-out).
Input data/surrogate.json (CC: norm_surrogate.py $SCRATCH/zz_surrogate.json). Output data/surrogate_score.json.
    python analysis/catalog_daniel/norm_surrogate_score.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["none", "dim", "dimid", "eik", "pt", "prop", "propeik"]


def main():
    S = {k: v for k, v in json.load(open(os.path.join(HERE, "data", "surrogate.json"))).items() if "error" not in v}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    held = set(json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))["heldout"])
    tr = sorted(n for n in S if n not in held); ho = sorted(n for n in S if n in held)
    new = [n for n in ho if n not in rel]; rl = [n for n in ho if n in rel]
    out = {"n_train": len(tr), "n_new": len(new), "n_rel": len(rl)}

    def feats(n, k):
        if k == "none":
            return 0.0, 0.0
        m, s_, _ = S[n][k]
        return m, np.log(max(s_, 1e-6))

    def fit_score(k, fit_on, test_sets, perN):
        Xm = np.array([feats(n, k)[0] for n in fit_on]); Xs = np.array([feats(n, k)[1] for n in fit_on])
        ym = np.array([S[n]["y_mean"] for n in fit_on]); ys = np.log([S[n]["y_std"] for n in fit_on])
        Ns = sorted({S[n]["N"] for n in fit_on})
        def design(names, which):
            x = np.array([feats(n, k)[which] for n in names])
            cols = [x] if k != "none" else []
            if perN:
                cols += [np.array([S[n]["N"] == N for n in names], float) for N in Ns]
            else:
                cols += [np.ones(len(names))]
            return np.stack(cols, 1)
        cm = np.linalg.lstsq(design(fit_on, 0), ym, rcond=None)[0]
        cs = np.linalg.lstsq(design(fit_on, 1), ys, rcond=None)[0]
        res = {}
        for lab, names in test_sets.items():
            if not names:
                continue
            if perN and any(S[n]["N"] not in Ns for n in names):
                # an unseen multiplicity: its intercept is the mean of the seen ones (no fitted value)
                pass
            pm = design(names, 0) @ cm if not (perN and any(S[n]["N"] not in Ns for n in names)) else None
            ps_ = design(names, 1) @ cs if pm is not None else None
            if pm is None:
                continue
            tm = np.array([S[n]["y_mean"] for n in names]); ts = np.array([S[n]["y_std"] for n in names])
            lev = (pm - tm) ** 2 / ts ** 2; lse = np.abs(ps_ - np.log(ts))
            corr = [S[n][k][2] for n in names] if k != "none" else [0.0]
            res[lab] = dict(level=float(np.median(lev)), level90=float(np.quantile(lev, .9)),
                            logstd=float(np.median(lse)), corr=float(np.median(corr)))
        return res

    for perN in (False, True):
        for k in KEYS:
            r = fit_score(k, tr, {"new": new, "rel": rl}, perN)
            key = f"{k}|{'perN' if perN else 'one'}"
            out[key] = r
            print(f"{key:16s} 43 new: level {r['new']['level']:.3g} (90% {r['new']['level90']:.3g}), |dlog std| "
                  f"{r['new']['logstd']:.3g}, corr {r['new']['corr']:.2f} | 96 rel: level {r['rel']['level']:.3g}, "
                  f"|dlog std| {r['rel']['logstd']:.3g}", flush=True)
    # P2, multiplicity: fit on 2->2 (training processes), test on 2->3 (every process)
    f22 = [n for n in tr if S[n]["N"] == 4]; t23 = [n for n in S if S[n]["N"] == 5]
    for k in KEYS:
        r = fit_score(k, f22, {"2to3": t23}, False)
        out[f"{k}|22to23"] = r
        if r:
            print(f"{k:8s} fit 2->2, test 2->3 ({len(t23)}): level {r['2to3']['level']:.3g} (90% {r['2to3']['level90']:.3g}), "
                  f"|dlog std| {r['2to3']['logstd']:.3g}", flush=True)
    # g, for reference (stats_net.json: held-out predictions of mean and log std)
    try:
        from heldout_delta import G
        sn = json.load(open(os.path.join(HERE, "data", "stats_net.json")))
        g = {n: v for n, v in zip(sn["heldout_names"], sn["heldout_pred"][G])}
        for lab, names in (("new", new), ("rel", rl)):
            names = [n for n in names if n in g]
            lev = [(g[n][0] - S[n]["y_mean"]) ** 2 / S[n]["y_std"] ** 2 for n in names]
            lse = [abs(g[n][1] - np.log(S[n]["y_std"])) for n in names]
            out[f"g|{lab}"] = dict(level=float(np.median(lev)), logstd=float(np.median(lse)))
            print(f"g {lab}: level {np.median(lev):.3g}, |dlog std| {np.median(lse):.3g}")
    except Exception as e:  # noqa: BLE001
        print("g reference skipped:", e)
    json.dump(out, open(os.path.join(HERE, "data", "surrogate_score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
