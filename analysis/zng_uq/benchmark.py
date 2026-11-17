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
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROCS = {1: "zg", 2: "zgg", 3: "zggg", 4: "zgggg"}
NETS = {"slim": "LGATr-slim", "lloca": "LLoCa-Transformer", "transformer": "Transformer", "mlp": "MLP"}
OURS = {"A": "from scratch", "A_ft": "fine-tuned"}
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
            # their paper standardises log|M|^2 with the training set's mean/std (first 5e5 = our n_train)
            std_train = float(np.std(np.log(10.0) * f["bulk_train/bulk/log10_me2"][:500000]))
        for mode, label in OURS.items():
            for d in sorted(glob.glob(os.path.join(root, f"kit_{mode}_uq_{p}_me2_n500000_t12500*"))):
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


def plot(path, curves=None, paper=os.path.join(HERE, "data", "paper_mse.json")):
    """(a) median |Delta| and standardised-log MSE against n, the latter beside their paper's six networks (their own
    test set); (b) |Delta| on Z+4g; (c) pulls per n against N(0,1); (d) val/train mu-MSE curves per n (curve test)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    col = {"from scratch": ps.C.blue, "fine-tuned": ps.C.vermillion}
    figs = os.path.join(HERE, "figs")
    os.makedirs(figs, exist_ok=True)
    ours = lambda a, key: sorted((v["n_gluons"], v[key]) for v in d["runs"].values() if v["arm"] == a)
    # (a)
    fig, (axl, axr) = ps.figure(2)
    for a in OURS.values():
        axl.plot(*zip(*ours(a, "median_abs_delta")), "o-", color=col[a], label=a)
        axr.plot(*zip(*ours(a, "mse_std")), "o-", color=col[a])
    pm = json.load(open(paper))["mse"]
    for net, c in zip(sorted(pm, key=lambda k: pm[k]["Z+4g"]), [ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky,
                                                                  ps.C.grey, ps.C.yellow]):
        axr.plot([1, 2, 3, 4], [pm[net][f"Z+{n}g"] for n in (1, 2, 3, 4)], "s--", color=c, label=net.replace("MLP-I", "MLP on invariants") + ", paper")
    axl.set_ylabel(r"median $|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2-1$")
    axr.set_ylabel(r"MSE of standardised $\log|\mathcal{M}|^2$")
    for ax in (axl, axr):
        ax.set_yscale("log"); ax.set_xlabel("number of gluons $n$"); ax.set_xticks([1, 2, 3, 4])
    ps.legend(axl, "upper left"); ps.legend(axr, "upper left")
    ps.save(fig, os.path.join(figs, "zng_median_delta"))
    # (b)
    fig, ax = ps.figure()
    for a in OURS.values():
        h = np.asarray(d["runs"][f"{a}|4"]["hist_abs_delta"], float)
        ax.stairs(h / h.sum(), d["bins_abs_delta"], color=col[a], label=a)
    ax.set_xscale("log"); ax.set_xlabel(r"$|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2-1$")
    ax.set_ylabel("fraction of test events"); ps.legend(ax, "upper left"); ps.process_label(ax, PROC_LAB[4])
    ps.save(fig, os.path.join(figs, "zng_delta_z4g"))
    # (c)
    b = np.asarray(d["bins_pull"]); x = np.linspace(b[0], b[-1], 481)
    fig, axs = ps.figure(2, 2)
    for n, ax in zip((1, 2, 3, 4), axs.flat):
        for a in OURS.values():
            h = np.asarray(d["runs"][f"{a}|{n}"]["hist_pull"], float)
            ax.stairs(h / h.sum(), b, color=col[a], label=a)
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * (b[1] - b[0]), "k:", label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel(r"pull $(\ln|\mathcal{M}|^2-\mu)/\sigma$"); ax.set_ylabel("fraction of test events")
        ps.process_label(ax, PROC_LAB[n])
    ps.legend(axs[0][0], "upper left")
    ps.save(fig, os.path.join(figs, "zng_pulls"))
    # (d)
    if curves:
        c = json.load(open(curves))
        fig, axs = ps.figure(2, 2)
        for n, ax in zip((1, 2, 3, 4), axs.flat):
            for m, a in OURS.items():
                r = c[f"kit_{m}_uq_{PROCS[n]}_me2_n500000_t12500"]
                st = np.arange(1, len(r["val"]) + 1) * 12500 / len(r["val"])   # evaluations evenly over 12,500 steps
                ax.plot(st, r["val"], color=col[a], label=a + ", validation")
                ax.plot(st, r["train"], color=col[a], linestyle="--", label=a + ", training")
            ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel(r"MSE$(\mu)$")
            ps.process_label(ax, PROC_LAB[n])
        ps.legend(axs[0][0], "lower left")
        ps.save(fig, os.path.join(figs, "zng_curves"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["collect", "plot"])
    ap.add_argument("--theirs", default=os.path.join(os.environ.get("WORK", ""), "ext", "uq_runs", "runs", "prod_het"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "benchmark.json"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "benchmark.json"))
    ap.add_argument("--curves", default=os.path.join(HERE, "data", "zng_curves.json"))
    a = ap.parse_args()
    collect(a.theirs, a.out) if a.cmd == "collect" else plot(a.data, a.curves)


if __name__ == "__main__":
    main()
