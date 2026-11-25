"""Plan selective_rescale (D552) step 3: where does rescale below x=0.1 cost e+e- -> u u~ g bulk? Rescale below x=0.1
changes the input only of momenta with x_i = 2 p_i.Q/s < 0.1 (Q the incoming sum), i.e. only of events whose softest
final-state x is below 0.1 (2.3% of events). Its searched-best u u~ g val MSE(log|M|^2) is 1.33x (1e4) and 2.90x (1e5)
that of no rescale (par:selres_stage2). Here each arm's validation squared error is split by event: min final-state
x < 0.1 (input changed by the rule) against the rest (input identical in all three arms but the full rescale), and
binned in log10(min x).

Arms per size D (1e4, 1e5), each its search's chosen trial (seed 42) + seeds 43/44, all on CC:
  off       no rescale      ftmech_d539_uug_<D>_off_b2048 chosen trial (d539_read), + _s43/_s44
  rescale   full rescale    ftmech_d539_uug_<D>_eps1e-8_b2048 chosen trial, + _s43/_s44
  below_x0p1  rescale below x=0.1   data/stage2_chosen.json uug_<D>_softbelow_x0p1, + selres_s2_uug_<D>_softbelow_x0p1_s4x
Per run: sigma_stage/preds_val.npz (the final model, after the sigma fit), rows matched to the val pool named in the
run log by the saved pool_row, checked against the raw |M|^2 (float32 as saved, rel < 1e-5; repeated rows counted); residual = (pred - truth) x prepd_std.
Group MSE ratios over the three seeds: zng4g_d1e3_read.ratio (geometric means, 2-sigma from the seeds' log spread).

    python3 analysis/selective_rescale/uug_split.py [--cached]   -> data/uug_split.json, figs/uug_split.{png,pdf}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "finetune_mechanism"))
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from zng4g_d1e3_read import driven, ratio  # noqa: E402

X0 = 0.1          # model.input_soft_x0 of the rule
EPS = 1e-8        # model.input_soft_eps (D539)
SEEDS = (42, 43, 44)
ARMS = ("off", "rescale", "below_x0p1")
RAW = {"off": "off", "rescale": "rescale", "below_x0p1": "below0.1"}   # arm labels of the site read (uug_split_raw.json)
BIN = 10          # bins per decade of min x

REMOTE = r'''
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.getcwd()); import siteconf
eta = np.array([1., -1., -1., -1.]); out = {}
for a in sys.argv[1:]:
    lab, d = a.split("#"); r = {}
    d = os.path.join(siteconf.PROJECT_DIR, "runs", d)
    try:
        log = open(os.path.join(d, "out_0.log")).read()
        m = re.search(r"\[val\] (\S+): (\d+) events.*? from (\S+\.npy)", log)
        r["val_path"] = m.group(3)
        pool = np.asarray(np.load(m.group(3), mmap_mode="r")).astype(np.float64)
        n = (pool.shape[1] - 1) // 5
        p = pool[:, :4 * n].reshape(-1, n, 4); Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
        x = np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:]
        z = np.load(os.path.join(d, "sigma_stage", "preds_val.npz"))
        raw = np.asarray(z["raw_truth"]).reshape(-1).astype(np.float64)
        row = np.asarray(z["pool_row"]).astype(int)   # each prediction's row in the val pool (experiment.py)
        rel = np.abs(pool[row, -1] - raw) / np.abs(raw)
        assert rel.max() < 1e-5, f"predictions do not match their pool rows (max rel {rel.max():.2g})"
        amb = len(row) - len(np.unique(row))   # rows predicted twice
        std = json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0]
        res = (np.asarray(z["pred"])[:, 0].astype(np.float64) - np.asarray(z["truth"])[:, 0]) * std
        mx = x[row].min(1); soft = x[row].argmin(1)
        b = np.floor(np.log10(mx) * BIN).astype(int)
        h = {}
        for k, bb, e in zip(soft, b, res):
            t = h.setdefault(f"{k}|{bb}", [0, 0.0, 0.0]); t[0] += 1; t[1] += e; t[2] += e * e
        r.update(n=int(len(res)), n_pool=int(len(pool)), n_ambiguous=int(amb), max_rel=float(rel.max()), std=std,
                 mse=float((res ** 2).mean()), pdg=[int(v) for v in pool[0, 4 * n + 2:5 * n]], hist=h)
    except Exception as e:
        r["error"] = repr(e)[:300]
    out[lab] = r
print("UUGSPLIT " + json.dumps(out))
'''.replace("BIN", str(BIN))


def runs():
    ch = json.load(open(os.path.join(HERE, "data", "stage2_chosen.json")))
    out = {}
    for d in ("1e4", "1e5"):
        for a, sweep in (("off", f"ftmech_d539_uug_{d}_off_b2048"), ("rescale", f"ftmech_d539_uug_{d}_eps1e-8_b2048")):
            out[f"{d}|{a}|42"] = min(driven(sweep), key=lambda t: t[2])[4].split("/runs/", 1)[1]
            for s in SEEDS[1:]:
                out[f"{d}|{a}|{s}"] = f"{sweep}_s{s}"
        out[f"{d}|below0.1|42"] = ch[f"uug_{d}_softbelow_x0p1"]["run_dir"].split("/runs/", 1)[1]
        for s in SEEDS[1:]:
            out[f"{d}|below0.1|{s}"] = f"selres_s2_uug_{d}_softbelow_x0p1_s{s}"
    return out


def group(h, sel):   # (n, sum e^2) over the bins whose (softest index, bin) passes sel
    n = sum(v[0] for k, v in h.items() if sel(*map(int, k.split("|"))))
    return n, sum(v[2] for k, v in h.items() if sel(*map(int, k.split("|"))))


def main():
    cache = os.path.join(HERE, "data", "uug_split_raw.json")
    rd = runs()
    raw = json.load(open(cache)) if "--cached" in sys.argv else site_python(REMOTE, [f"'{k}#{v}'" for k, v in rd.items()],
                                                                             "UUGSPLIT", site="ccin2p3")
    json.dump(raw, open(cache, "w"))
    bad = {k: v["error"] for k, v in raw.items() if "error" in v}
    assert not bad, bad
    lb = int(np.floor(np.log10(X0) * BIN))   # bins below X0 have index < lb (log10 0.1 * 10 = -10 exactly)
    groups = {"changed": lambda k, b: b < lb, "rest": lambda k, b: b >= lb,
              "rest_x0p1_0p2": lambda k, b: lb <= b < lb + 3, "rest_ge_x0p2": lambda k, b: b >= lb + 3}
    res = {"run_dirs": rd, "pdg": raw[next(iter(raw))]["pdg"], "cells": {}}
    for d in ("1e4", "1e5"):
        c = {}
        for a in ARMS:
            rr = [raw[f"{d}|{RAW[a]}|{s}"] for s in SEEDS]
            c[a] = {"mse": [r["mse"] for r in rr], "n": [r["n"] for r in rr], "n_ambiguous": [r["n_ambiguous"] for r in rr]}
            for g, f in groups.items():
                ns = [group(r["hist"], f) for r in rr]
                c[a][g] = {"n": ns[0][0], "mse": [s / n for n, s in ns],
                           "se_share": [s / (r["mse"] * r["n"]) for (n, s), r in zip(ns, rr)]}
        for a in ("below_x0p1", "rescale"):
            c[a]["over_off"] = {g: ratio(np.array(c[a][g]["mse"]), np.array(c["off"][g]["mse"]))
                                for g in groups}
            c[a]["over_off"]["total"] = ratio(np.array(c[a]["mse"]), np.array(c["off"]["mse"]))
            # how much of the arm's extra total squared error (gmean over seeds) sits in each group
            tot = np.exp(np.mean(np.log(c[a]["mse"]))) - np.exp(np.mean(np.log(c["off"]["mse"])))
            c[a]["extra_share"] = {g: float((np.exp(np.mean(np.log(c[a][g]["mse"]))) - np.exp(np.mean(np.log(c["off"][g]["mse"]))))
                                            * c[a][g]["n"] / c[a]["n"][0] / tot) for g in ("changed", "rest")}
        # per-bin seed-pooled MSE against min x, per arm, for the figure (bins merged to 0.25 dec where sparse)
        c["bins"] = {}
        for a in ARMS:
            pooled = {}
            for s in SEEDS:
                for k, v in raw[f"{d}|{RAW[a]}|{s}"]["hist"].items():
                    b = int(k.split("|")[1]); t = pooled.setdefault(b, [0, 0.0, []])
                    t[0] += v[0]; t[1] += v[2]
            per_seed = {b: [sum(v[2] for k, v in raw[f"{d}|{RAW[a]}|{s}"]["hist"].items() if int(k.split("|")[1]) == b)
                            / max(1, sum(v[0] for k, v in raw[f"{d}|{RAW[a]}|{s}"]["hist"].items() if int(k.split("|")[1]) == b))
                            for s in SEEDS] for b in pooled}
            c["bins"][a] = {str(b): {"n": pooled[b][0] // 3, "mse_seeds": per_seed[b]} for b in sorted(pooled)}
        res["cells"][d] = c
    # input level: the log-x span the rule and the full rescale give the trained soft range [x_min, X0] (wrappers._soft_rescale)
    xmin = json.load(open(os.path.join(HERE, "data", "input_check.json")))["pools"]["uug"]["x_min"]
    lx = lambda x, x0: x0 * np.log1p(x / EPS) / np.log1p(x0 / EPS)   # x' = x F(x) / F(x0)
    res["soft_range_map"] = {"x_min": xmin, "dec_in": float(np.log10(X0 / xmin)),
                             "below_x0p1": {"xp_min": float(lx(xmin, X0)), "dec_out": float(np.log10(X0 / lx(xmin, X0)))},
                             "rescale": {"xp_min": float(lx(xmin, 1.0)), "xp_x0": float(lx(X0, 1.0)),
                                         "dec_out": float(np.log10(lx(X0, 1.0) / lx(xmin, 1.0))),
                                         "dec_all": float(np.log10(1.0 / lx(xmin, 1.0)))}}
    json.dump(res, open(os.path.join(HERE, "data", "uug_split.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line
    plot(res)


def display_bins(B, nmin=50, wmin=2):
    """The 0.1-dec bins merged from the lowest x upward until each holds >= nmin events and spans >= wmin fine bins
    (every event kept; a short last group joins the one before it)."""
    fine = sorted(int(b) for b in B)
    out, cur = [], []
    for b in fine:
        cur.append(b)
        if sum(B[str(c)]["n"] for c in cur) >= nmin and cur[-1] - cur[0] + 1 >= wmin:
            out.append(cur); cur = []
    if cur:
        out[-1] += cur
    return out


def plot(res):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    lab = {"off": "no rescale", "rescale": r"rescale, all $x$", "below_x0p1": r"rescale below $x=0.1$"}
    cols = {"off": ps.C.blue, "rescale": ps.C.vermillion, "below_x0p1": ps.sequence(3)[0]}
    ls = {"1e4": "-", "1e5": "--"}
    fig, axs = ps.figure(ncols=2, nrows=2)
    (am, ar), (an, cell) = axs
    for d in ("1e4", "1e5"):
        B = res["cells"][d]["bins"]
        groups = display_bins(B["off"])
        med = {}
        for a in ARMS:
            xs, m, lo, hi, ns = [], [], [], [], []
            for g in groups:
                n = [B[a][str(b)]["n"] for b in g if str(b) in B[a]]
                mse = [sum(B[a][str(b)]["mse_seeds"][s] * B[a][str(b)]["n"] for b in g if str(b) in B[a]) / sum(n)
                       for s in range(3)]
                xs.append(10 ** (np.average([b + 0.5 for b in g], weights=[B["off"][str(b)]["n"] for b in g]) / BIN))
                m.append(np.median(mse)); lo.append(min(mse)); hi.append(max(mse)); ns.append(sum(n))
            med[a] = np.array(m)
            am.plot(xs, m, ls[d], marker="o", ms=3, color=cols[a])
            am.fill_between(xs, lo, hi, color=cols[a], alpha=0.15, lw=0)
            if a != "off":
                ar.plot(xs, med[a] / med["off"], ls[d], marker="o", ms=3, color=cols[a])
            else:
                an.plot(xs, ns, ls[d], marker="o", ms=3, color="0.4")
    for ax in (am, ar, an):
        ax.set_xscale("log"); ax.axvline(X0, color="k", lw=0.8, ls=":")
        ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
        ax.set_xlabel(r"softest final-state $x=2p\cdot Q/s$")
    am.set_yscale("log"); am.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ar.set_yscale("log"); ar.axhline(1, color="0.5", lw=0.8); ar.set_ylabel("MSE over no rescale")
    an.set_yscale("log"); an.set_ylabel("val events per bin")
    for ax in (am, ar, an):
        ps.process_label(ax, r"$e^+e^-\to u\bar ug$")
    for a in ARMS:
        am.plot([], [], "o", color=cols[a], ms=3, label=lab[a])
    for d in ("1e4", "1e5"):
        am.plot([], [], ls[d], color="0.3", label=rf"$D_{{\rm train}}=10^{d[-1]}$")
    ps.legend_cell(cell, am)
    ps.save(fig, os.path.join(HERE, "figs", "uug_split"))


if __name__ == "__main__":
    main()
