"""D83 exploration, fifth pass: few-shot calibration of a pretrained model's raw output on a new process. For hp58 (its raw
output h, per-process standardized units), A and B (their prediction q in log|M|^2), the held-out process's k labelled
events fit
  level   q' = q + c                          (one number)
  affine  q' = a + b q                        (two numbers)
  affine+ q' = a + b q + c1 L + c2 Lmin       (L = log s_hat, Lmin = log of the smallest pair invariant; four numbers)
on k = 3, 10, 30, 100 events drawn at random (5 draws), or the k with the smallest / largest s_hat (a biased calibration
sample), scored as MSE(log|M|^2) on the remaining events. The per-event invariants come from the val momenta, which the
npz files do not hold; the s_hat ordering uses A's logF, so it is real only for 2->3 (at 2->2 logF does not depend on
s_hat and the "biased" picks are the pool's first events); affine+ uses the truth-independent proxies available in the npz (q itself and, for A, its
logF, which is -(4-N) log(s_hat/s0) - log C_p, i.e. log s_hat up to a process constant) and is reported for A only.
Median and 90% over the 43 new and the 96 relabels. Output data/norm_explore5.json.
    python analysis/catalog_daniel/norm_explore5.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fit(cols, t):
    A = np.stack(cols, 1)
    return np.linalg.lstsq(A, t, rcond=None)[0]


def main():
    tw = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    models = {}
    z = dict(np.load(os.path.join(HERE, "data", "heldout_preds.npz")))
    models["hp58"] = {p: (z[p + "__h"].astype(float), z[p + "__t"].astype(float), None)
                      for p in {k.split("__")[0] for k in z if k.endswith("__t")}}
    for arm in ("A", "B"):
        z = dict(np.load(os.path.join(HERE, "data", f"catalog_norm_{arm}.npz")))
        models[arm] = {p: (z[p + "__q"].astype(float), z[p + "__t"].astype(float),
                           z[p + "__logF"].astype(float) if f"{p}__logF" in z and z[p + "__logF"].size else None)
                       for p in {k.split("__")[0] for k in z if k.endswith("__t")}}
    rng = np.random.default_rng(0)
    out = {}
    for name, M in models.items():
        for k in (3, 10, 30, 100):
            for pick in ("random", "low_shat", "high_shat"):
                for kind in ("level", "affine", "affine+"):
                    if kind == "affine+" and name != "A":
                        continue
                    if pick != "random" and name != "A":
                        continue          # s_hat known per event only through A's logF
                    res = {False: [], True: []}
                    for p, (q, t, lf) in M.items():
                        n = len(t)
                        draws = 5 if pick == "random" else 1
                        errs = []
                        for _ in range(draws):
                            if pick == "random":
                                idx = rng.choice(n, k, replace=False)
                            else:
                                # logF = log(s_hat/s0) - log C_p at N=5 (rises with s_hat); constant at N=4, where the order is the pool's (random)
                                o = np.argsort(lf) if pick == "low_shat" else np.argsort(-lf)
                                idx = o[:k]
                            rest = np.setdiff1d(np.arange(n), idx)
                            if kind == "level":
                                c = np.mean(t[idx] - q[idx]); pr = q[rest] + c
                            elif kind == "affine":
                                if k < 3:
                                    continue
                                c = fit([np.ones(k), q[idx]], t[idx]); pr = c[0] + c[1] * q[rest]
                            else:
                                if k < 5:
                                    continue
                                c = fit([np.ones(k), q[idx], lf[idx]], t[idx])
                                pr = c[0] + c[1] * q[rest] + c[2] * lf[rest]
                            errs.append(np.mean((pr - t[rest]) ** 2))
                        if errs:
                            res[p in tw].append(float(np.median(errs)))
                    if not res[False]:
                        continue
                    key = f"{name}|{kind}|{pick}|k={k}"
                    out[key] = dict(new=float(np.median(res[False])), new90=float(np.quantile(res[False], .9)),
                                    rel=float(np.median(res[True])), rel90=float(np.quantile(res[True], .9)))
                    v = out[key]
                    print(f"{key:32s} 43 new {v['new']:.3g} (90% {v['new90']:.3g}) | 96 rel {v['rel']:.3g} (90% {v['rel90']:.3g})",
                          flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "norm_explore5.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
