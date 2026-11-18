#!/usr/bin/env python3
"""zng_uq_benchmark: our runs and the lloca-experiments-uq networks on the same 1e6 test events per multiplicity.

    python analysis/zng_uq/benchmark.py collect [--theirs <study dir>] [--out analysis/zng_uq/data/benchmark.json]  (CC)
    python analysis/zng_uq/benchmark.py plot [--data analysis/zng_uq/data/benchmark.json]                         (local)

collect (on the site with the data): per multiplicity n = 1..4,
  ours   runs/kit/kit_<A|A_ft>_uq_z<g..>_me2_n500000_t12500*/kit_benchmark/{predictions.h5, sigma_ln.npy} (kit_predict),
         truth from $DATA_DIR/zng_uq/uq_<proc>.h5 bulk_test (their test events, in their order)
  theirs <study>/z<n>g/<net>/preds_test_0.npz (their experiment_uq.py: log_truth, log_mu, log_sigma2_tot, index);
         their test indices must equal our split_idx/test, event by event, else the run is refused.
Every metric is their experiments/uq/metrics.py definition, in ln|M|^2: Delta = expm1(mu - truth), pull =
(truth - mu) / sigma. Written per arm: median, q90, q99 |Delta|, pull mean and std, coverage of +-1 sigma, and the
|Delta| and pull histograms on fixed bins (the figure's data).
"""
import argparse
import glob
import sys
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROCS = {1: "zg", 2: "zgg", 3: "zggg", 4: "zgggg"}
NETS = {"slim": "LGATr-slim", "lloca": "LLoCa-Transformer", "transformer": "Transformer", "mlp": "MLP"}
OURS = {"A": "from scratch", "A_ft": "fine-tuned"}
SP = ", Suprio's protocol"
# (mode, run-dir suffix, label, val every n steps): the 12,500-step runs at kit_hpo_uug_T hp1677, then B2 (D484): the
# same network under Suprio's protocol (Adam 1e-3, batch 1024, RLROP, early stopping, <= 2e5 steps, validation every 1000)
ARMS = [(m, "t12500", lab, None) for m, lab in OURS.items()] + \
       [(m, "t200000_suprio_protocol", lab + SP, 1000) for m, lab in OURS.items()]
# B1 (D484): our protocol at HPs searched per multiplicity (sweep/sweep_config_tp_zng_hpo_<p>.yaml, DyHPO, 8 trials):
# scratch = each search's lowest-val trial (curve test passes only at n=1: analysis/zng_uq/data/b1_search.json), scored
# on the test events by job_kit.sh KIT_SCORE_DIR; fine-tuned from 1262 at the passing trial's HPs (n=1 only)
SH = ", searched HPs"
ARMS += [("A_ft", "t12500_b1_hp1612", OURS["A_ft"] + SH, None)]
B1_SCR = {1: "runs/tp_zng_hpo_zg/trial_1612", 2: "runs/tp_zng_hpo_zgg/trial_0353", 3: "runs/tp_zng_hpo_zggg/trial_0494",
          4: "runs/tp_zng_hpo_zgggg/trial_0238"}
ZS = "pretraining only, zero-shot"
DBINS = np.logspace(-8, 1, 91)          # |Delta| bins
PBINS = np.linspace(-6, 6, 121)         # pull bins


def metrics(truth, mu, sigma):
    """Their point_metrics and gaussian_metrics (experiments/uq/metrics.py), the numbers this study reads."""
    ad = np.abs(np.expm1(mu - truth))
    out = {"n": int(len(truth)), "median_abs_delta": float(np.median(ad)),
           "q90_abs_delta": float(np.quantile(ad, 0.9)), "q99_abs_delta": float(np.quantile(ad, 0.99)),
           "mse_log": float(np.mean((mu - truth) ** 2)),
           "hist_abs_delta": np.histogram(np.clip(ad, DBINS[0], DBINS[-1]), DBINS)[0].tolist()}
    if sigma is not None:
        pull = (truth - mu) / sigma
        out.update({"pull_mean": float(np.mean(pull)), "pull_std": float(np.std(pull)),
                    "cov_0.68": float(np.mean(np.abs(pull) <= 0.994457883)),
                    "sigma_median": float(np.median(sigma)),
                    "hist_pull": np.histogram(np.clip(pull, PBINS[0], PBINS[-1]), PBINS)[0].tolist()})
    return out


def collect(theirs, out):
    import h5py
    data = {"bins_abs_delta": DBINS.tolist(), "bins_pull": PBINS.tolist(), "runs": {}}
    root = os.path.join(os.environ.get("PROJECT_DIR", "."), "runs", "kit")
    for n, p in PROCS.items():
        with h5py.File(os.path.join(os.environ["DATA_DIR"], "zng_uq", f"uq_{p}.h5"), "r") as f:
            truth = np.log(10.0) * f["bulk_test/bulk/log10_me2"][:]
            test_idx = f["split_idx/test"][:]
            # the UQ repo standardises log|M|^2 with the training events' mean/std (README_uq; first 5e5 = our n_train)
            std_train = float(np.std(np.log(10.0) * f["bulk_train/bulk/log10_me2"][:500000]))
        b1 = [(os.path.join(os.environ.get("PROJECT_DIR", "."), B1_SCR[n]), OURS["A"] + SH)]
        for d, label in [(d, label) for mode, suffix, label, _ in ARMS
                         for d in sorted(glob.glob(os.path.join(root, f"kit_{mode}_uq_{p}_me2_n500000_{suffix}")))] + b1:
                if "smoke" in d:
                    continue
                kb = os.path.join(d, "kit_benchmark")
                if not os.path.exists(os.path.join(kb, "predictions.h5")):
                    continue
                with h5py.File(os.path.join(kb, "predictions.h5"), "r") as f:
                    mu = np.log(10.0) * f["bulk_test/bulk/log10_me2"][:]
                sg = os.path.join(kb, "sigma_ln.npy")
                sigma = np.load(sg)[-len(mu):] if os.path.exists(sg) else None    # bulk_test is the last block
                assert f"{label}|{n}" not in data["runs"], f"{d}: a second run of {label}, n={n}; name the one to read"
                data["runs"][f"{label}|{n}"] = dict(metrics(truth, mu, sigma), run=d, arm=label, n_gluons=n,
                                                       std_train_ln=std_train)
                data["runs"][f"{label}|{n}"]["mse_std"] = data["runs"][f"{label}|{n}"]["mse_log"] / std_train ** 2
        # the catalog_v4 pretraining zero-shot (scripts/job_zng_zeroshot.sh, tools/zero_shot_parent.py): t, q, s in the
        # scored target's ln units (a target factor divides both, so q - t is the ln residual of |M|^2)
        zf = os.path.join(root, "zng_zeroshot_1262_n1000000", "preds", f"kit_uq_{p}__me2.npz")
        if os.path.exists(zf):
            z = np.load(zf)
            assert len(z["t"]) == len(truth), f"zero-shot {p}: {len(z['t'])} events, not the {len(truth)} test events"
            data["runs"][f"{ZS}|{n}"] = dict(metrics(z["t"].astype(np.float64), z["q"].astype(np.float64),
                                                     z["s"].astype(np.float64) if len(z["s"]) else None),
                                             run=zf, arm=ZS, n_gluons=n, std_train_ln=std_train)
            data["runs"][f"{ZS}|{n}"]["mse_std"] = data["runs"][f"{ZS}|{n}"]["mse_log"] / std_train ** 2
        for net, label in NETS.items():
            f = os.path.join(theirs, f"z{n}g", net, "preds_test_0.npz") if theirs else ""
            if not os.path.exists(f):
                continue
            z = np.load(f)
            assert np.array_equal(z["index"], test_idx), f"{f}: their test events are not ours"
            assert np.allclose(z["log_truth"], truth, rtol=0, atol=1e-9), f"{f}: their truth differs from ours"
            data["runs"][f"{label}|{n}"] = dict(metrics(z["log_truth"], z["log_mu"], np.sqrt(z["log_sigma2_tot"])),
                                                run=f, arm=label, n_gluons=n)
    json.dump(data, open(out, "w"), indent=1)
    for k, v in data["runs"].items():
        print(k, {a: v[a] for a in ("median_abs_delta", "pull_std") if a in v})


PROC_LAB = {1: r"$q\bar q\to Zg$", 2: r"$q\bar q\to Zgg$", 3: r"$q\bar q\to Zggg$", 4: r"$q\bar q\to Zgggg$"}


def plot(path, curves=None):
    """(a) median |Delta| against n; (b) |Delta| on Z+4g; (c) pulls per n against N(0,1); (d) val/train mu-MSE curves per n (curve test)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    col = {"from scratch": ps.C.blue, "fine-tuned": ps.C.vermillion, ZS: "k"}
    col.update({a + SP: col[a] for a in OURS.values()})
    col.update({a + SH: col[a] for a in OURS.values()})
    ls = lambda a: "--" if a.endswith(SP) else (":" if a.endswith(SH) else "-")
    # legend labels: ", Suprio" = trained under Suprio's protocol
    lab = lambda a: {ZS: "zero-shot", NETS["lloca"]: "Suprio's network"}.get(a, a.replace("from scratch", "scratch").replace(SP, ", Suprio").replace(SH, ", searched"))
    col.update({NETS["lloca"]: ps.C.green})
    arms = [a for a in [x[2] for x in ARMS] + [OURS["A"] + SH, ZS] + list(NETS.values()) if any(v["arm"] == a for v in d["runs"].values())]
    figs = os.path.join(HERE, "figs")
    os.makedirs(figs, exist_ok=True)
    ours = lambda a, key: sorted((v["n_gluons"], v[key]) for v in d["runs"].values() if v["arm"] == a)
    # (a)
    fig, (axl, cell) = ps.figure(2)
    for a in arms:
        axl.plot(*zip(*ours(a, "median_abs_delta")), "o", linestyle=ls(a), color=col[a], label=lab(a))
    axl.set_ylabel(r"median $|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2-1$")
    axl.set_yscale("log"); axl.set_xlabel("number of gluons $n$"); axl.set_xticks([1, 2, 3, 4])
    # the legend in a cell of its own: eight entries in the panel covered the zero-shot curve
    ps.legend_cell(cell, axl)
    ps.save(fig, os.path.join(figs, "zng_median_delta"))
    # (b)
    fig, (ax, cell) = ps.figure(2)
    for a in [a for a in arms if f"{a}|4" in d["runs"]]:
        h = np.asarray(d["runs"][f"{a}|4"]["hist_abs_delta"], float)
        ax.stairs(h / h.sum(), d["bins_abs_delta"], color=col[a], linestyle=ls(a), label=lab(a))
    ax.set_xscale("log"); ax.set_xlabel(r"$|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2-1$")
    # the legend in a cell of its own: in the panel it covered Suprio's peak
    ax.set_ylabel("fraction of test events"); ps.legend_cell(cell, ax); ps.process_label(ax, PROC_LAB[4])
    ps.save(fig, os.path.join(figs, "zng_delta_z4g"))
    # (c)
    b = np.asarray(d["bins_pull"]); x = np.linspace(b[0], b[-1], 481)
    fig, axs = ps.figure(2, 3)
    for n, ax in zip((1, 2, 3, 4), axs.flat):
        for a in [a for a in arms if f"{a}|{n}" in d["runs"]]:
            h = np.asarray(d["runs"][f"{a}|{n}"]["hist_pull"], float)
            ax.stairs(h / h.sum(), b, color=col[a], linestyle=ls(a), label=lab(a))
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * (b[1] - b[0]), "k:", label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel(r"pull $(\ln|\mathcal{M}|^2-\mu)/\sigma$"); ax.set_ylabel("fraction of test events")
        ps.process_label(ax, PROC_LAB[n])
    # the legend in the two spare cells of a 2x3 (in the n=1 panel it covered the data), split across them
    h, l = axs[0][0].get_legend_handles_labels()
    k = (len(l) + 1) // 2
    for cell, sl in ((axs[2][0], slice(0, k)), (axs[2][1], slice(k, None))):
        cell.set_axis_off(); cell.legend(h[sl], l[sl], loc="center", frameon=False)
    ps.save(fig, os.path.join(figs, "zng_pulls"))
    # (d)
    if curves:
        c = json.load(open(curves))
        for suffix, name in (("t12500", "zng_curves"), ("t200000_suprio_protocol", "zng_curves_suprio")):
            fig, axs = ps.figure(2, 2)
            for n, ax in zip((1, 2, 3, 4), axs.flat):
                for m, _, a, every in [x for x in ARMS if x[1] == suffix]:
                    r = c.get(f"kit_{m}_uq_{PROCS[n]}_me2_n500000_{suffix}")
                    if not r:
                        continue
                    # the 12,500-step runs: evaluations evenly over 12,500 steps; B2: one every `every` steps
                    st = np.arange(1, len(r["val"]) + 1) * (every or 12500 / len(r["val"]))
                    a0 = lab(a)
                    ax.plot(st, r["val"], color=col[a], label=a0 + ", validation")
                    ax.plot(st, [np.nan if x is None else x for x in r["train"]], color=col[a], linestyle=":",
                            label=a0 + ", training")
                ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel(r"MSE$(\mu)$")
                ps.process_label(ax, PROC_LAB[n])
            ps.legend(axs[0][0], "lower left")
            ps.save(fig, os.path.join(figs, name))


PARENT = "runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262/sigma_stage"
PARENT_SITE = r'''
import json, sys
import numpy as np
z = np.load(sys.argv[1] + "/preds_val.npz")
names = list(z["names"]); out = {}
for p in ("v4_bbx_gz", "v4_bbx_ggz", "v4_bbx_gggz"):
    m = z["process_id"] == names.index(p)
    r = np.log(z["raw_pred"][m, 0]) - np.log(z["raw_truth"][m, 0])
    out[p] = {"n": int(m.sum()), "median_abs_delta": float(np.median(np.abs(np.expm1(r)))), "mse_ln": float(np.mean(r ** 2))}
print("PARENTBBX " + json.dumps(out))
'''


def parent(out):
    """1262's own level on its b b~ -> Z+1..3g training processes (massless b: the same tree |M|^2 as d d~ -> Z+ng):
    median |Delta| and MSE of ln|M|^2 on its validation events (sigma_stage/preds_val.npz, mu frozen = the best)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from analysis.common.readers import site_python
    d = site_python(PARENT_SITE, [PARENT], "PARENTBBX")
    json.dump(d, open(out, "w"), indent=1)
    print(json.dumps(d))


def summary(path, curves):
    """The numbers docs/results.tex quotes, one JSON line (verify_claims): per arm (scr, ft, zs) and n the median |Delta|,
    MSE of standardised ln|M|^2, pull std, 68% coverage; per trained run the curve test."""
    d = json.load(open(path))
    short = {"from scratch": "scr", "fine-tuned": "ft", ZS: "zs", "from scratch" + SP: "scr_sp", "fine-tuned" + SP: "ft_sp"}
    out = {}
    for v in d["runs"].values():
        if v["arm"] in short:
            o = out.setdefault(short[v["arm"]], {}).setdefault(str(v["n_gluons"]), {})
            o.update(median=v["median_abs_delta"], mse_std=v["mse_std"], mse_ln=v["mse_log"],
                     pull_std=v.get("pull_std"), cov68=v.get("cov_0.68"), sigma_median=v.get("sigma_median"))
    if curves and os.path.exists(curves):
        c = json.load(open(curves))
        for m, sfx, a in (("A", "t12500", "scr"), ("A_ft", "t12500", "ft"), ("A", "t200000_suprio_protocol", "scr_sp"),
                          ("A_ft", "t200000_suprio_protocol", "ft_sp")):
            for n, p in PROCS.items():
                r = c.get(f"kit_{m}_uq_{p}_me2_n500000_{sfx}")
                if r:
                    out[a][str(n)].update(separation=r["separation"], well=r["well"], best_step=r["best_step"])
        out["n_well"] = sum(r["well"] for k, r in c.items() if k.endswith("t12500"))
        out["n_well_sp"] = sum(r["well"] for k, r in c.items() if k.endswith("suprio_protocol"))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["collect", "plot", "summary", "parent"])
    ap.add_argument("--theirs", default=os.path.join(os.environ.get("WORK", ""), "ext", "uq_runs", "runs", "prod_het"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "benchmark.json"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "benchmark.json"))
    ap.add_argument("--curves", default=os.path.join(HERE, "data", "zng_curves.json"))
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.theirs, a.out)
    elif a.cmd == "parent":
        parent(os.path.join(HERE, "data", "parent_bbx.json"))
    elif a.cmd == "plot":
        plot(a.data, a.curves)
    else:
        summary(a.data, a.curves)


if __name__ == "__main__":
    main()
