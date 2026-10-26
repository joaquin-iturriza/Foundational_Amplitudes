"""Why the s_hat factor lowers the flat-in-log-s_hat level shift on catalog_v3 (0.30 to 0.10, par. The sampling
sensitivity on catalog_v3) but raised it on Daniel's catalog (0.30 to 0.39, par. More data-free normalizations).
The shift of a target y' = log|M|^2 + (N - 4) log s_hat under the flat reweighting is, to first order, the
within-process slope of y' on log s_hat times how far the reweighting moves the mean log s_hat; the factor adds N - 4
to the slope of log|M|^2. So per class: the within-process slope b of log|M|^2 on log s_hat, the slope of
log(least pair invariant / s_hat) on log s_hat (-1 if the cuts that bound the singular regions are fixed in GeV, 0 if
they scale with s_hat), dL (the flat weights' mean log s_hat minus the pool's) and the signed shifts in units of the
pool's std, raw and with the factor, by multiplicity N (all legs).
Two modes:
    site run ccin2p3 FA -- python analysis/catalog_v3/prep_shat_slope.py pools > analysis/catalog_v3/data/prep_shat_slope_raw.txt
        the 200 training classes of prep_shape.py / prep_sampling_pools.py (seed 0), their validation pools (CPU)
    python3 analysis/catalog_v3/prep_shat_slope.py
        reads that file and Daniel's per-process moments (analysis/catalog_daniel/data/norm_moments2.json, the same
        flat definition: 20 bins between the 1st and 99th percentile of log s_hat) -> data/prep_shat_slope.json
"""
import json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def pools():
    import yaml
    from physics_scales import minkowski_dots
    from prep_sampling_pools import pool, w_flat
    tr = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]
          if p["name"].startswith("v3_") and "_nlo" not in p["name"] and "_loop" not in p["name"]]
    pick = np.random.default_rng(0).permutation(len(tr))[:200]
    for i in sorted(pick):
        name, sq = tr[i]["name"], tr[i].get("sqrts")
        f = pool(name, sq)
        if f is None:
            print("SLOPE_SKIP", name, flush=True); continue
        x = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)
        N = (x.shape[1] - 1) // 5
        P = x[:, : 4 * N].reshape(len(x), N, 4); amp = x[:, -1]
        if np.any(amp <= 0) or not np.all(np.isfinite(amp)):
            print("SLOPE_SKIP", name, "signed", flush=True); continue
        G = minkowski_dots(P)
        L = np.log(np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1]))
        iu = [(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) != (0, 1)]
        Lmin = np.stack([np.log(np.clip(2 * np.abs(G[:, i, j]), 1e-300, None)) for i, j in iu], 1).min(1)
        y = np.log(amp)
        w = w_flat(L)
        vL = L.var()
        r = {"name": name, "N": int(N), "sdL": float(np.sqrt(vL)), "dL": float(np.sum(w * L) - L.mean()),
             "b": float(np.cov(y, L)[0, 1] / vL), "b_lmin": float(np.cov(Lmin - L, L)[0, 1] / vL)}
        for k, yy in (("raw", y), ("dless", y + (N - 4) * L)):
            r[f"shift_{k}"] = float((np.sum(w * yy) - yy.mean()) / yy.std())
        print("SLOPE", json.dumps(r), flush=True)


def daniel():
    d = json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "data", "norm_moments2.json")))
    V = d["vars"]; iy, iL, im = V.index("y"), V.index("L"), V.index("Lmin")
    out = []
    for name, r in d["train"].items():
        N, C, m = r["N"], np.array(r["cov"]), np.array(r["mean"])
        bm, ok = np.array(r["bin_mean"]), np.array(r["bin_count"]) > 0
        row = {"name": name, "N": N, "sdL": float(np.sqrt(C[iL, iL])), "dL": float(bm[ok][:, iL].mean() - m[iL]),
               "b": float(C[iy, iL] / C[iL, iL]), "b_lmin": float(C[im, iL] / C[iL, iL] - 1)}
        for k, a in (("raw", 0), ("dless", N - 4)):
            w = np.zeros(len(V)); w[iy] = 1; w[iL] = a
            row[f"shift_{k}"] = float(((bm[ok] @ w).mean() - m @ w) / np.sqrt(w @ C @ w))
        out.append(row)
    return out


def summarise(rows):
    def med(rs, k, f=lambda v: v):
        return round(float(np.median([f(r[k]) for r in rs])), 3)
    s = {"n": len(rows), "abs_raw": med(rows, "shift_raw", abs), "abs_dless": med(rows, "shift_dless", abs), "byN": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["N"]].append(r)
    for N, rs in sorted(by.items()):
        s["byN"][str(N)] = {"n": len(rs), "share": round(len(rs) / len(rows), 3), "b": med(rs, "b"),
                            "b_plus_factor": round(med(rs, "b") + N - 4, 3), "b_lmin": med(rs, "b_lmin"),
                            "dL": med(rs, "dL"), "sdL": med(rs, "sdL"),
                            "abs_raw": med(rs, "shift_raw", abs), "abs_dless": med(rs, "shift_dless", abs),
                            "neg_b": round(float(np.mean([r["b"] < 0 for r in rs])), 3)}
    return s


def main():
    v3 = [json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(HERE, "data", "prep_shat_slope_raw.txt"))
          if l.startswith("SLOPE {")]
    out = {"catalog_v3": summarise(v3), "daniel": summarise(daniel())}
    json.dump(out, open(os.path.join(HERE, "data", "prep_shat_slope.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    pools() if sys.argv[1:] == ["pools"] else main()
