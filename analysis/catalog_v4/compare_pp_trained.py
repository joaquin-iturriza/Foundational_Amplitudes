"""catalog_v4 against training on pp, on the 13 TeV pp pools (catalog_coverage; the user, 2026-10-26: "what mattered the
most was checking how training on catalog 4 compares to training on pp collisions on the datasets generated like pp
collisions"). At no compute: the reads already home.
  catalog_v4, Muon      v4_dimensionless_muon, _seed2 (seeds 42, 43; analysis/catalog_v4/compare_v3.py)
  catalog_v3, Muon      dimensionless_muon_best, _seed2 (the same design on catalog_v3's windows)
  pp-trained, AdamW     dimensionless_pp (FA-ccin2p3-e786bd74, its last training classes read again by
                        FA-ccin2p3-982b5a14): the dimensionless arm with every hadronic training tree class on its pp
                        events, one seed (analysis/catalog_v3/pp_compare.py, data/pp_rows.json)
  catalog_v3, AdamW     dimensionless: the same arm on catalog_v3's windows, one seed (data/pp_rows.json)
Same events: every read scores the first 2000 rows of the class's pp validation pool; the Muon reads were matched to the
pool by 40 sorted log|M|^2 quantiles (compare_v3.collect), the AdamW reads by all 2000 (pp_compare.collect); this script
checks that the share of events inside catalog_v3's coverage, from the two collections, is the same on every class and
that the truth's variance agrees between the Muon and AdamW reads of each class (float32 storage). Inside / outside:
one definition, pp_compare.support_masks (catalog_v3's fiducial cuts in the partonic frame and the class's sqrt(s_hat)
window), used by both collections.
Groups as compare_v3.pp: the class's role in both catalogs (trained in both, held out in both, the role changed, dropped
in catalog_v4), b classes apart; every class in one group (asserted).
  (a) direct: per class the ratio of catalog_v4's MSE (geometric mean of its seeds) to the pp-trained arm's; the
      optimizers differ (Muon against AdamW)
  (b) each against its own optimizer's catalog_v3 window arm: catalog_v4 / catalog_v3 (Muon) and pp-trained /
      catalog_v3 (AdamW), per class, medians; the share of the pp-trained arm's gain catalog_v4 recovers,
      log(median catalog_v4 ratio) / log(median pp-trained ratio), signed (negative: catalog_v4 loses where pp gains)
  (c) the optimizer alone on pp: catalog_v3 Muon / catalog_v3 AdamW
Per group the median over classes of each run's MSE and of each per-class ratio, with the share of classes where the
numerator is lower (in both seeds for catalog_v4).
    python3 analysis/catalog_v4/compare_pp_trained.py   -> data/compare_pp_trained.json, figs/compare_pp_trained.{png,pdf}
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import compare_v3 as cv

KEYS = ("mse", "mse_inside", "mse_outside")
V4, V3M = cv.RUNS["v4"], cv.RUNS["v3"]


def rows():
    site = json.load(open(os.path.join(HERE, "data", "compare_v3_site.json")))
    p = cv.pp(site)
    adam = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "pp_rows.json")))
    R = {k: {r: p["_rows"][k][r] for r in V4 + V3M} for k in KEYS}
    for arm in ("dimensionless_pp", "dimensionless"):
        for k in KEYS:
            R[k][arm] = {n: v[k] for n, v in adam[arm]["pp"].items() if v.get(k) is not None}
    # the same events: the inside share from the two collections, and the truth's spread in each read
    fi = {n: s["inside"].count("1") / len(s["inside"]) for n, s in site["pp"].items()}
    d_frac = max(abs(fi[n] - adam[a]["pp"][n]["frac_inside"]) for n in fi for a in ("dimensionless_pp", "dimensionless"))
    d_var = 0.0
    for n in fi:
        for r in (V4[0], V3M[0]):
            t = np.load(os.path.join(cv.SR, r, "PP", n + ".npz"))["t"].astype(float)
            for a in ("dimensionless_pp", "dimensionless"):
                d_var = max(d_var, abs(np.var(t) - adam[a]["pp"][n]["var_t"]) / adam[a]["pp"][n]["var_t"])
    assert d_frac == 0.0, d_frac
    assert d_var < 1e-3, d_var
    return R, p, {"max_frac_inside_diff": d_frac, "max_var_t_rel_diff": d_var, "n_classes": len(fi)}


def gm(R, k, runs, n):
    return float(np.sqrt(np.prod([R[k][r][n] for r in runs]))) if len(runs) == 2 else R[k][runs[0]][n]


def ratio(R, k, num, den, ps_, both=None):
    """per class num / den (each a run tuple, a pair a geometric mean of seeds): median, 10/90% quantiles, the share
    where num is lower (both seeds when num is a pair)"""
    ps_ = [n for n in ps_ if all(n in R[k][r] for r in num + den)]
    if not ps_:
        return None
    x = np.array([gm(R, k, num, n) / gm(R, k, den, n) for n in ps_])
    low = np.array([all(R[k][r][n] < gm(R, k, den, n) for r in num) for n in ps_])
    return {"n": len(ps_), "ratio": float(np.median(x)), "q10_q90": [float(q) for q in np.quantile(x, [0.1, 0.9])],
            "num_lower_pct": float(100 * low.mean()), "gap": float(np.median(np.abs(np.log10(x)))), "num_med": float(np.median([gm(R, k, num, n) for n in ps_])),
            "den_med": float(np.median([gm(R, k, den, n) for n in ps_]))}


def main():
    R, p, chk = rows()
    role, b = p["_role"], p["_b"]
    groups = {"train": lambda n: role[n]["v3"] == "train" and role[n]["v4"] == "train",
              "heldout": lambda n: role[n]["v3"] == "heldout" and role[n]["v4"] == "heldout",
              # trained in catalog_v3 (so on pp by the pp-trained arm) and held out in catalog_v4 by its class merges
              "role_changed": lambda n: role[n]["v3"] != role[n]["v4"] and role[n]["v4"] != "absent",
              # the b-Higgs classes catalog_v4 dropped
              "dropped_in_v4": lambda n: role[n]["v4"] == "absent"}
    # the seed spread of catalog_v4's two runs per group, median over classes of |log10 seed 43 / seed 42|, the scale a
    # gap is read against (the AdamW arms have one seed each: no spread of their own)
    out = {"check": chk}
    for bk, bsel in (("no_b", lambda n: not b[n]), ("b", lambda n: b[n])):
        for g, gsel in groups.items():
            ps_ = [n for n in role if bsel(n) and gsel(n)]
            if not ps_:
                continue
            e = out.setdefault(bk, {}).setdefault(g, {"n": len(ps_)})
            for k in KEYS:
                f = {"direct": ratio(R, k, V4, ("dimensionless_pp",), ps_),
                     "v4_over_v3_muon": ratio(R, k, V4, V3M, ps_),
                     "pp_over_v3_adamw": ratio(R, k, ("dimensionless_pp",), ("dimensionless",), ps_),
                     "muon_over_adamw_v3": ratio(R, k, V3M, ("dimensionless",), ps_),
                     "v4_seed_spread": float(np.median([abs(np.log10(R[k][V4[1]][n] / R[k][V4[0]][n])) for n in ps_
                                                        if n in R[k][V4[0]] and n in R[k][V4[1]]])),
                     "med": {r: float(np.median([R[k][r][n] for n in ps_ if n in R[k][r]]))
                             for r in V4 + V3M + ("dimensionless_pp", "dimensionless")}}
                a, c = f["v4_over_v3_muon"], f["pp_over_v3_adamw"]
                # signed: negative where catalog_v4 loses against its reference while pp training gains; read next to
                # both ratios (the AdamW arms have one seed each, so no spread to gate it on)
                f["recovered"] = float(np.log(a["ratio"]) / np.log(c["ratio"])) if a and c and c["ratio"] != 1 else None
                e[k] = f
    out["check"]["n_grouped"] = sum(e["n"] for bk in ("no_b", "b") for e in out[bk].values())
    assert out["check"]["n_grouped"] == chk["n_classes"], out["check"]["n_grouped"]
    json.dump(out, open(os.path.join(HERE, "data", "compare_pp_trained.json"), "w"), indent=1)
    plot(out)
    scatter(R, [n for n in role if not b[n] and groups["train"](n)], [n for n in role if not b[n] and groups["heldout"](n)])
    print("check", chk)
    for bk in ("no_b", "b"):
        for g, e in out[bk].items():
            for k in KEYS:
                f = e[k]
                print(f"{bk:4s} {g:7s} {k:11s} n={e['n']:3d} " + " ".join(
                    f"{x} {f[x]['ratio']:.3g} ({f[x]['num_lower_pct']:.0f}% lower)" for x in
                    ("direct", "v4_over_v3_muon", "pp_over_v3_adamw", "muon_over_adamw_v3") if f[x]) +
                      f" gap direct {f['direct']['gap']:.3g} v4 spread {f['v4_seed_spread']:.3g} recovered {f['recovered']}" + " med " + " ".join(f"{r} {v:.3g}" for r, v in f["med"].items()))
    claims = {"n_classes": chk["n_classes"]}
    for bk in ("no_b", "b"):
        for g, e in out[bk].items():
            for k in KEYS:
                f = e[k]
                c = claims.setdefault(bk, {}).setdefault(g, {}).setdefault(k, {"n": f["direct"]["n"]})
                for x in ("direct", "v4_over_v3_muon", "pp_over_v3_adamw", "muon_over_adamw_v3"):
                    c[x] = f[x]["ratio"]
                    c[x + "_lower_pct"] = f[x]["num_lower_pct"]
                    c[x + "_gap"] = f[x]["gap"]
                c["recovered"] = f["recovered"]
                c["v4_seed_spread"] = f["v4_seed_spread"]
                c["med_v4"] = float(np.sqrt(f["med"][V4[0]] * f["med"][V4[1]]))
                c["med_pp"] = f["med"]["dimensionless_pp"]
                c["med_v3_muon"] = float(np.sqrt(f["med"][V3M[0]] * f["med"][V3M[1]]))
                c["med_v3_adamw"] = f["med"]["dimensionless"]
    print(json.dumps(claims))


def plot(out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    ks = ("mse", "mse_inside", "mse_outside")
    series = ((("dimensionless",), ps.C.blue, "o", "catalog v3, AdamW"),
              (("dimensionless_pp",), ps.C.green, "D", "pp-trained, AdamW"),
              (V3M, ps.C.sky, "^", "catalog v3, Muon"),
              (V4, ps.C.vermillion, "s", "catalog v4, Muon"))
    for ax, g, lab in zip(axs, ("train", "heldout"), ("trained in both", "held out in both")):
        x = np.arange(len(ks))
        for j, (runs, col, mk, name) in enumerate(series):
            dx = (j - 1.5) * 0.14
            for s, r in enumerate(runs):
                y = [out["no_b"][g][k]["med"][r] for k in ks]
                ax.plot(x + dx, y, mk, color=col, mfc=col if s == 0 else "none",
                        label=name + (f", seed {42 + s}" if len(runs) == 2 else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(["whole pool", "inside", "outside"])
        ax.set_yscale("log")
        ax.set_ylabel("median MSE, 13 TeV pp")
        ax.set_ylim(0.08, 10)                            # one range for both panels
        ax.set_xlabel(f"classes {lab}")
    ps.shared_legend(fig, axs[0], ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", "compare_pp_trained"))


def scatter(R, train, held):
    """per class (no b), catalog_v4's MSE (geometric mean of its seeds) against the pp-trained arm's, on the pp events
    inside and outside catalog_v3's coverage: (a) classes trained in both catalogs, (b) held out in both"""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    lo, hi = np.inf, 0
    for ax, ps_ in zip(axs, (train, held)):
        for k, col, mk, lab in (("mse_inside", ps.C.blue, "o", "inside v3 coverage"),
                                ("mse_outside", ps.C.vermillion, "s", "outside v3 coverage")):
            q = [n for n in ps_ if all(n in R[k][r] for r in V4 + ("dimensionless_pp",))]
            x = [R[k]["dimensionless_pp"][n] for n in q]
            y = [gm(R, k, V4, n) for n in q]
            ax.plot(x, y, mk, color=col, mfc="none", ms=3, alpha=0.6, label=lab)
            lo, hi = min(lo, min(x), min(y)), max(hi, max(x), max(y))
    for ax in axs:
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo / 2, hi * 2); ax.set_ylim(lo / 2, hi * 2)
        ax.set_xlabel("MSE, pp-trained arm")
        ax.set_ylabel("MSE, catalog v4")
    ps.shared_legend(fig, axs[0], ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", "compare_pp_trained_scatter"))


if __name__ == "__main__":
    main()
