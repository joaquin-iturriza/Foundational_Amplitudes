"""The Muon test (D144, D146): does Muon with decoupled weight decay beat AdamW on catalog_v3's short pretraining and
its zero-shot reads? Four short pretrainings of the dimensionless arm (scripts/job_catalog_v3_short.sh, 1397 pools,
4264 steps, the same HPs): AdamW (dimensionless, dimensionless_seed2) and MuonAdamW (Muon on muP's 32 hidden matrices,
decoupled weight decay 0.1, update RMS matched to AdamW's, AdamW on the rest; dimensionless_muon,
dimensionless_muon_seed2), seeds 42 and 43 (model init and batch order; pools unchanged).
  collect (on the site): each run's result.json (best-checkpoint val_loss_no_reg, its curve, every training pool's
           val_loss_no_reg at that checkpoint on its 200 validation events) and the shared standardization's std, as JSON:
    site run --item muon_test ccin2p3 FA -- python analysis/catalog_v3/muon_compare.py collect > analysis/catalog_v3/data/muon_pretrain.json
  main: from data/muon_pretrain.json and the zero-shot predictions (data/short_reads/<arm>/<axis>/<probe>.npz, as
        short_reads.py: mse = mean (q - t)^2 in log|M|^2 on 2000 validation events):
    pretraining   the aggregate per run, in the standardized target and times std^2 (log|M|^2 units); per pool the
                  ratio AdamW / Muon within each seed, its median over pools and the share of pools Muon wins in both seeds;
                  the seed spread, the median over pools of abs log10 of seed 43 over seed 42, per optimizer
    zero-shot     per axis and run the median MSE over probes; per probe the ratio AdamW / Muon of the two seeds'
                  geometric means, its median over the axis's probes, the share of probes Muon wins in both seeds, and
                  the seed spread as above, so a gap is read against the seeds' own scatter
Figure muon_compare (2x2): (a) the validation curves; (b) per training pool, Muon against AdamW, seed 42, with AdamW
seed 43 against seed 42 as the seed reference; (c) the median zero-shot MSE per axis, one marker per run; (d) per
held-out probe, as (b). The last line printed is the numbers docs/results.tex quotes, as one JSON object.
    python3 analysis/catalog_v3/muon_compare.py   -> data/muon_compare.json, figs/muon_compare.{png,pdf}
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
RUNS = {("AdamW", 42): "dimensionless", ("AdamW", 43): "dimensionless_seed2",
        ("Muon", 42): "dimensionless_muon", ("Muon", 43): "dimensionless_muon_seed2"}
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
AXNAME = {"H1": "new comb.", "H2": r"$2\to4$", "H2p5": r"$2\to5$", "H3": "top", "H5": "one loop",
          "H6": r"$\alpha_s$", "H7E": "1-10 TeV", "H7c": r"$p_T\geq30$"}


def collect():
    out = {}
    for arm in RUNS.values():
        d = os.path.join(ROOT, "runs", "catalog_v3_short", arm)
        r = json.load(open(os.path.join(d, "result.json")))
        std = float(json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0])
        out[arm] = {"val_loss_no_reg": r["val_loss"], "best_step": r["best_step"], "validate_every": r["validate_every"],
                    "val_curve": r["val_curve"], "prepd_std": std, "proc": r["proc_val_losses_no_reg"]}
    json.dump(out, sys.stdout)


def zs(arm, axis):
    rows = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "short_reads", arm, axis, "*.npz"))):
        z = np.load(f)
        if "q" in z.files:
            rows[os.path.basename(f)[:-4]] = float(np.mean((z["q"].astype(float) - z["t"].astype(float)) ** 2))
    return rows


def compare(a42, a43, m42, m43):
    """per key: AdamW / Muon of the seeds' geometric means; share Muon wins in both seeds; seed spreads"""
    ks = sorted(set(a42) & set(a43) & set(m42) & set(m43))
    A42, A43, M42, M43 = (np.array([d[k] for k in ks]) for d in (a42, a43, m42, m43))
    ratio = np.sqrt(A42 * A43) / np.sqrt(M42 * M43)
    return {"n": len(ks), "ratio_median": float(np.median(ratio)),
            "ratio_q10_q90": [float(x) for x in np.quantile(ratio, [0.1, 0.9])],
            "muon_wins_both_pct": float(100 * np.mean((M42 < A42) & (M43 < A43))),
            "adamw_wins_both_pct": float(100 * np.mean((A42 < M42) & (A43 < M43))),
            "seed_spread_adamw": float(np.median(np.abs(np.log10(A43 / A42)))),
            "seed_spread_muon": float(np.median(np.abs(np.log10(M43 / M42)))),
            "gap": float(np.median(np.abs(np.log10(ratio))))}


def main():
    pre = json.load(open(os.path.join(HERE, "data", "muon_pretrain.json")))
    res = {"pretrain": {}, "zero_shot": {}}
    for (opt, seed), arm in RUNS.items():
        p = pre[arm]
        res["pretrain"][arm] = {"val_loss_no_reg": p["val_loss_no_reg"], "best_step": p["best_step"],
                                "mse_log": p["val_loss_no_reg"] * p["prepd_std"] ** 2}
    pa = res["pretrain"]
    res["pretrain"]["ratio_seed42"] = pa["dimensionless"]["val_loss_no_reg"] / pa["dimensionless_muon"]["val_loss_no_reg"]
    res["pretrain"]["ratio_seed43"] = (pa["dimensionless_seed2"]["val_loss_no_reg"]
                                       / pa["dimensionless_muon_seed2"]["val_loss_no_reg"])
    res["pretrain"]["pools"] = compare(*(pre[RUNS[k]]["proc"] for k in
                                         (("AdamW", 42), ("AdamW", 43), ("Muon", 42), ("Muon", 43))))
    reads = {arm: {a: zs(arm, a) for a in AXES} for arm in RUNS.values()}
    for a in AXES:
        r = {arm: reads[arm][a] for arm in RUNS.values()}
        c = compare(*(r[RUNS[k]] for k in (("AdamW", 42), ("AdamW", 43), ("Muon", 42), ("Muon", 43))))
        c["median_mse"] = {arm: float(np.median(list(r[arm].values()))) for arm in RUNS.values()}
        res["zero_shot"][a] = c
    allp = {arm: {f"{a}/{k}": v for a in AXES for k, v in reads[arm][a].items()} for arm in RUNS.values()}
    res["zero_shot"]["all"] = compare(*(allp[RUNS[k]] for k in (("AdamW", 42), ("AdamW", 43), ("Muon", 42), ("Muon", 43))))
    json.dump(res, open(os.path.join(HERE, "data", "muon_compare.json"), "w"), indent=1)
    plot(pre, reads)
    for a in AXES + ("all",):
        z = res["zero_shot"][a]
        print(f"{a:5s} n={z['n']:4d} AdamW/Muon median {z['ratio_median']:.2f} Muon wins both {z['muon_wins_both_pct']:.0f}% "
              f"AdamW wins both {z['adamw_wins_both_pct']:.0f}% seed spread A {z['seed_spread_adamw']:.3f} "
              f"M {z['seed_spread_muon']:.3f} gap {z['gap']:.3f}")
    r2 = lambda x: float(f"{x:.3g}")
    out = {"pre": {k: r2(v["val_loss_no_reg"]) for k, v in pa.items() if isinstance(v, dict) and "val_loss_no_reg" in v},
           "pre_ratio42": r2(pa["ratio_seed42"]), "pre_ratio43": r2(pa["ratio_seed43"]),
           "pools": {k: (r2(v) if isinstance(v, float) else v) for k, v in pa["pools"].items() if k != "ratio_q10_q90"},
           "zs": {a: {"ratio": r2(z["ratio_median"]), "wins": round(z["muon_wins_both_pct"]),
                      "loses": round(z["adamw_wins_both_pct"]), "n": z["n"],
                      "spread_a": r2(z["seed_spread_adamw"]), "spread_m": r2(z["seed_spread_muon"]),
                      **({"mse": {k: r2(v) for k, v in z["median_mse"].items()}} if "median_mse" in z else {})}
                  for a, z in res["zero_shot"].items()}}
    print(json.dumps(out))


def plot(pre, reads):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a, b), (c, d) = axs
    sty = {("AdamW", 42): (ps.C.blue, "-", "o"), ("AdamW", 43): (ps.C.blue, "--", "s"),
           ("Muon", 42): (ps.C.vermillion, "-", "^"), ("Muon", 43): (ps.C.vermillion, "--", "v")}
    for k, arm in RUNS.items():
        col, ls, mk = sty[k]
        p = pre[arm]
        a.plot(p["validate_every"] * (np.arange(len(p["val_curve"])) + 1), p["val_curve"], color=col, ls=ls,
               label=f"{k[0]}, seed {k[1]}")
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel(r"val loss (no reg.)")
    ps.legend(a, "upper right")

    def scatter(ax, get, xl, yl):
        A42, A43, M42 = (get(RUNS[k]) for k in (("AdamW", 42), ("AdamW", 43), ("Muon", 42)))
        ks = sorted(set(A42) & set(A43) & set(M42))
        x = np.array([A42[k] for k in ks])
        ax.plot(x, [A43[k] for k in ks], "o", mfc="none", color=ps.C.grey, label="AdamW seed 43")
        ax.plot(x, [M42[k] for k in ks], "^", mfc="none", color=ps.C.vermillion, label="Muon seed 42")
        y = np.concatenate([[A43[k] for k in ks], [M42[k] for k in ks]])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", label="equal")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ps.legend(ax, "upper left")
        ps.make_room(ax, max_growth=4)

    scatter(b, lambda arm: pre[arm]["proc"], "pool val loss, AdamW seed 42", "pool val loss")
    xs = np.arange(len(AXES))
    for i, (k, arm) in enumerate(RUNS.items()):
        col, ls, mk = sty[k]
        c.plot(xs + (i - 1.5) * 0.12, [np.median(list(reads[arm][ax].values())) for ax in AXES], mk, color=col,
               mfc="none" if k[1] == 43 else col, label=f"{k[0]}, seed {k[1]}")
    c.set_yscale("log"); c.set_xticks(xs); c.set_xticklabels([str(i + 1) for i in xs])
    c.set_xlabel("held-out axis"); c.set_ylabel(r"median zero-shot MSE$(\log|\mathcal{M}|^2)$")
    ps.make_room(c, max_growth=4)
    scatter(d, lambda arm: {f"{ax}/{k}": v for ax in AXES for k, v in reads[arm][ax].items()},
            r"zero-shot MSE, AdamW seed 42", r"zero-shot MSE")
    ps.save(fig, os.path.join(HERE, "figs", "muon_compare"))


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else main()
