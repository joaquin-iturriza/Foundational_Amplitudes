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
  collect-ft: the small-D fine-tunes (tools/catalog_ft.py, scripts/job_catalog_ft.sh), one CATFT JSON line per cell in
           each job's log, gathered into one list (144 cells: 4 probes x 4 D x {AdamW, Muon parent} x {AdamW, Muon
           fine-tune} x {the parent's standardization kept, refit on the probe}, and the same 16 from scratch with
           AdamW), all at the weight-average fix (ddfc0314; the pre-fix refit jobs 6e5d2ed2 ce1dba1c 54fd017f be7d9ff7
           and parent-stats jobs 483f4bfc 8fe3a758 3a2f3a89 b03e4c9a are not read):
    for r in 70adfc03 8103a0bb 13e57de7 e5265e31 fe864fda f620ed40 60177afb 29e8ad2f 00d080ba; do
        site logs FA-ccin2p3-$r -n 200000; done | python3 analysis/catalog_v3/muon_compare.py collect-ft \
        > analysis/catalog_v3/data/muon_ft.json
  fine-tune read (in main, from data/muon_ft.json): each cell's best-checkpoint MSE in log|M|^2 (val_loss_no_reg times
        the std^2 of the target standardization the cell trained in, on the probe's first 2000 validation events, the
        zero-shot read's events), one seed per cell. The zero-shot reference of a (parent, probe, D) is the
        parent-stats cell's step-0 validation (the parent's weights, the cell's own events and target transform; equal
        to the zero-shot read on the tree probes; on the sign-changing one-loop probe the signedlog scale is the 1%
        quantile of |M|^2 over the D training events, so the target and its step 0 move slightly with D). Ratios, with
        medians over the 16 cells and counts: fine-tune / zero-shot (below, within 0.1%, above), refit / parent stats,
        AdamW parent / Muon parent, AdamW fine-tune / Muon fine-tune, scratch / fine-tune, and per D over the 4 probes.
        Figure muon_ft (2x2, one panel per probe): MSE against D, the eight fine-tunes and scratch, each parent's step
        0 dotted.
    python3 analysis/catalog_v3/muon_compare.py   -> data/muon_compare.json, figs/muon_compare.{png,pdf}, figs/muon_ft.{png,pdf}
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


FT_PROBES = {"v3_dg_dgwpwm": ("H1", r"$dg\to dgW^+W^-$"), "v3_usx_cxuwpz": ("H2", r"$u\bar s\to \bar c uW^+Z$"),
             "v3_gg_attx": ("H3", r"$gg\to\gamma t\bar t$"), "v3_udx_agwp_nlo": ("H5", r"$u\bar d\to\gamma gW^+$, one loop")}
FT_D = (10, 32, 100, 316)
FT_COMBOS = (("dimensionless", "AdamW"), ("dimensionless", "MuonAdamW"), ("dimensionless_muon", "AdamW"),
             ("dimensionless_muon", "MuonAdamW"))
FT_NAME = {"dimensionless": "AdamW", "dimensionless_muon": "Muon", "AdamW": "AdamW", "MuonAdamW": "Muon"}


def collect_ft():
    cells = [json.loads(l[len("CATFT "):]) for l in sys.stdin if l.startswith("CATFT {")]
    json.dump(sorted(cells, key=lambda c: (c["kind"], c["parent"], c["optimizer"], c["probe"], c["D"])), sys.stdout,
              indent=0)


STATS = ("parent", "refit")          # fine_tune.target_stats=parent_shared (catft_parentstats) or own (catft)
KIND_STATS = {"catft_parentstats": "parent", "catft": "refit"}
FT_ARMS = [(st, pa, op) for st in STATS for pa, op in FT_COMBOS]


def ft_read():
    """per arm (stats, parent, optimizer) or ("scratch",): {(probe, D): best-checkpoint mse_log}; the zero-shot reference
    per (parent, probe, D): the parent-stats cell's step-0 validation (the parent's weights before any update, the
    cell's own validation events and target transform), checked against the zero-shot read (short_reads) on the tree
    probes, where the transform does not depend on D"""
    cells = json.load(open(os.path.join(HERE, "data", "muon_ft.json")))
    assert not any("error" in c for c in cells), [c for c in cells if "error" in c]
    tab, z0, step0_refit = {}, {}, {}
    for c in cells:
        k = (c["probe"], c["D"])
        if c["kind"] == "scratch":
            tab.setdefault(("scratch",), {})[k] = c["mse_log"]
            continue
        st = KIND_STATS[c["kind"]]
        tab.setdefault((st, c["parent"], c["optimizer"]), {})[k] = c["mse_log"]
        if st == "parent":
            k0 = (c["parent"],) + k
            if k0 in z0:      # the same parent weights on the same events: both fine-tune optimizers start equal
                assert abs(z0[k0] / c["mse_log_step0"] - 1) < 1e-4, (k0, z0[k0], c["mse_log_step0"])
            z0[k0] = c["mse_log_step0"]
        else:
            step0_refit[(c["parent"], c["optimizer"]) + k] = c["mse_log_step0"]
    read = {}
    for parent in ("dimensionless", "dimensionless_muon"):
        for probe, (ax, _) in FT_PROBES.items():
            z = np.load(os.path.join(HERE, "data", "short_reads", parent, ax, probe + ".npz"))
            t = z["t"].astype(float)
            read[(parent, probe)] = float(np.mean((z["q"].astype(float) - t) ** 2))
            if probe != "v3_udx_agwp_nlo":
                for D in FT_D:
                    assert abs(z0[(parent, probe, D)] / read[(parent, probe)] - 1) < 1e-3, (parent, probe, D)
    return tab, z0, read, step0_refit


def ft_compare(tab, z0, read, step0_refit):
    keys = [(p, D) for p in FT_PROBES for D in FT_D]
    for k in FT_ARMS + [("scratch",)]:
        assert set(tab[k]) == set(keys), (k, set(keys) - set(tab[k]))
    name = lambda arm: "_".join([arm[0]] + [FT_NAME[x] for x in arm[1:]])

    def ratio(num, den, ks=keys):
        r = np.array([num[k] / den[k] for k in ks])
        return {"median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()),
                "den_lower": int(np.sum(r > 1)), "n": len(r)}

    out = {"vs_zero_shot": {}, "refit_vs_parent": {}, "parent": {}, "ft_opt": {}, "vs_scratch": {}, "cells": {}}
    # 1. does the fine-tune end below its parent's zero-shot (step 0)? below: under it by more than 0.1%; at: within
    for arm in FT_ARMS:
        r = np.array([tab[arm][(p, D)] / z0[(arm[1], p, D)] for p, D in keys])
        d = {"median": float(np.median(r)), "below": int(np.sum(r < 0.999)), "at": int(np.sum(abs(r - 1) <= 1e-3)),
             "above": int(np.sum(r > 1.001)), "n": len(r), "max": float(r.max())}
        for D in FT_D:
            rD = np.array([tab[arm][(p, D)] / z0[(arm[1], p, D)] for p in FT_PROBES])
            d[f"D{D}"] = {"median": float(np.median(rD)), "below": int(np.sum(rD < 0.999))}
        out["vs_zero_shot"][name(arm)] = d
    # 2. refit over parent stats, same parent and fine-tune optimizer
    for pa, op in FT_COMBOS:
        d = ratio(tab[("refit", pa, op)], tab[("parent", pa, op)])
        for D in FT_D:
            ks = [(p, D) for p in FT_PROBES]
            d[f"D{D}"] = ratio(tab[("refit", pa, op)], tab[("parent", pa, op)], ks)["median"]
        out["refit_vs_parent"][f"{FT_NAME[pa]}_{FT_NAME[op]}"] = d
    # 3. AdamW parent over Muon parent, per target statistics and fine-tune optimizer
    for st in STATS:
        for op in ("AdamW", "MuonAdamW"):
            d = ratio(tab[(st, "dimensionless", op)], tab[(st, "dimensionless_muon", op)])
            for D in FT_D:
                d[f"D{D}"] = ratio(tab[(st, "dimensionless", op)], tab[(st, "dimensionless_muon", op)],
                                   [(p, D) for p in FT_PROBES])["median"]
            out["parent"][f"{st}_{FT_NAME[op]}"] = d
    # 4. AdamW fine-tune over Muon fine-tune
    for st in STATS:
        for pa in ("dimensionless", "dimensionless_muon"):
            out["ft_opt"][f"{st}_{FT_NAME[pa]}"] = ratio(tab[(st, pa, "AdamW")], tab[(st, pa, "MuonAdamW")])
    # 5. scratch over fine-tune
    for arm in FT_ARMS:
        d = ratio(tab[("scratch",)], tab[arm])
        for D in FT_D:
            d[f"D{D}"] = ratio(tab[("scratch",)], tab[arm], [(p, D) for p in FT_PROBES])["median"]
        out["vs_scratch"][name(arm)] = d
    # the zero-shot reference itself: per parent and probe at each D, and the zero-shot read
    out["zs_ratio_median"] = float(np.median([z0[("dimensionless", p, D)] / z0[("dimensionless_muon", p, D)]
                                              for p, D in keys]))
    for parent in ("dimensionless", "dimensionless_muon"):
        for p in FT_PROBES:
            out["cells"][f"{FT_NAME[parent]}/{p}"] = {
                "zero_shot_read": read[(parent, p)], **{f"step0_D{D}": z0[(parent, p, D)] for D in FT_D},
                **{f"refit_start_D{D}": step0_refit[(parent, "AdamW", p, D)] for D in FT_D}}
    for k, v in tab.items():
        out["cells"][name(k) if k[0] != "scratch" else "scratch"] = {f"{p}/{D}": v[(p, D)] for p, D in keys}
    return out


def plot_ft(tab, z0):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    col = {"dimensionless": ps.C.blue, "dimensionless_muon": ps.C.vermillion}
    ls = {"parent": "-", "refit": "--"}
    mk = {"AdamW": "o", "MuonAdamW": "s"}
    lab = {"parent": "parent's standardization", "refit": "refit"}
    for ax, (probe, (axis, label)) in zip(axs.flat, FT_PROBES.items()):
        for st, pa, op in FT_ARMS:
            ax.plot(FT_D, [tab[(st, pa, op)][(probe, D)] for D in FT_D], ls=ls[st], marker=mk[op], color=col[pa],
                    mfc=col[pa] if st == "parent" else "none",
                    label=f"{FT_NAME[pa]} parent, {FT_NAME[op]} fine-tune, {lab[st]}")
        ax.plot(FT_D, [tab[("scratch",)][(probe, D)] for D in FT_D], ls="-", marker="D", color=ps.C.grey,
                label="scratch, AdamW")
        for pa in ("dimensionless", "dimensionless_muon"):
            ax.plot(FT_D, [z0[(pa, probe, D)] for D in FT_D], ls=":", color=col[pa],
                    label=f"{FT_NAME[pa]} parent zero-shot (step 0)")
        ys = [v[(probe, D)] for v in tab.values() for D in FT_D] + [z0[(p, probe, D)] for p in
                                                                    ("dimensionless", "dimensionless_muon") for D in FT_D]
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(min(ys) / 1.5, max(ys) * 2.5)
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, label, loc="upper right")
    ps.shared_legend(fig, axs.flat[0], ncol=2)
    ps.save(fig, os.path.join(HERE, "figs", "muon_ft"))


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
    tab, z0, read, step0_refit = ft_read()
    res["fine_tune"] = ft_compare(tab, z0, read, step0_refit)
    json.dump(res, open(os.path.join(HERE, "data", "muon_compare.json"), "w"), indent=1)
    plot(pre, reads)
    plot_ft(tab, z0)
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
    ft = res["fine_tune"]
    rr = lambda v: {kk: (r2(vv) if isinstance(vv, float) else rr(vv) if isinstance(vv, dict) else vv)
                    for kk, vv in v.items()}
    out["ft"] = {sec: {k: rr(v) for k, v in ft[sec].items()}
                 for sec in ("vs_zero_shot", "refit_vs_parent", "parent", "ft_opt", "vs_scratch")}
    out["ft"]["zs_ratio_median"] = r2(ft["zs_ratio_median"])
    out["ft"]["zs"] = {k: rr(v) for k, v in ft["cells"].items() if "zero_shot_read" in v}
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
    {"collect": collect, "collect-ft": collect_ft}.get(sys.argv[1] if sys.argv[1:] else "", main)()
