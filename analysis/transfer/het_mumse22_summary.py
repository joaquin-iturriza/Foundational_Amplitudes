"""D89: the 147 2->2 Delta cells of D47 (het loss at each MSE cell's own HPs) retrained with the main run's checkpoint picked
on the validation mu MSE instead of the validation NLL (scripts/job_het_delta.sh --mumse, training.heterosc_select=mu_mse; the
sigma stage after it still selects on the NLL, mu frozen). Same HPs, data, steps and seeds as D47, so the selection is the
only difference (D85 did the same for the 24 2->3 and 2->4 cells, het_mumse_summary.py). Per cell: median |Delta| on the
test split for the MSE cell (delta_plots.pick), the NLL-selected het retrain (D47) and the mu-MSE-selected one, and the
pull std of both het retrains. The cell that blew up in D47 (tp3_uu64fte_ee_nnbar_d8 trial 114) is reported apart.
Inputs: het_mumse22_data/ccin2p3.npz (collect_het_delta.py on $SCRATCH/zz_delta_het_mumse), het_delta_data/ccin2p3.npz
(D47), delta_data/, delta_audit.json. Outputs: the summary printed, het_mumse22_summary.json, figs/het_mumse22.{png,pdf}: the ECDF of the rerun over the first run (each at its mu-MSE minimum) and of the NLL
pick over the mu-MSE minimum within the first run, at 2->2 and at 2->3/2->4.
    .venv-driver/bin/python analysis/transfer/het_mumse22_summary.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, pick, tag  # noqa: E402

BLEW = "tp3_uu64fte_ee_nnbar_d8__trial_0114"


def arrs(path, suf):
    with np.load(path, allow_pickle=True) as z:
        return {k[: -len(suf)]: z[k] for k in z.files if k.endswith(suf)}


def main():
    mse = load()
    new_d = arrs(os.path.join(HERE, "het_mumse22_data", "ccin2p3.npz"), "__delta")
    new_p = arrs(os.path.join(HERE, "het_mumse22_data", "ccin2p3.npz"), "__pull")
    old_d = arrs(os.path.join(HERE, "het_delta_data", "ccin2p3.npz"), "__delta")
    old_p = arrs(os.path.join(HERE, "het_delta_data", "ccin2p3.npz"), "__pull")
    rows = {tag((r.get("paper_8k") or {}).get("run_dir", "?")): r for r in
            json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"] if r.get("paper_8k")}
    med = lambda a: float(np.median(np.abs(a[np.isfinite(a)])))
    sd = lambda a: float(np.std(a[np.isfinite(a)]))
    pts, missing = [], []
    for k in sorted(new_d):
        t = k[len("runs__"):] if k.startswith("runs__") else k
        r = rows.get(t) or rows.get(k)
        if r is None or k not in old_d:
            missing.append(k); continue
        pts.append(dict(tag=t, probe=r["probe"], arm=r["arm"], mse=med(mse[pick(mse, r["paper_8k"])]),
                        het_nll=med(old_d[k]), het_mumse=med(new_d[k]), pull_nll=sd(old_p[k]), pull_mumse=sd(new_p[k])))
    blew = [p for p in pts if p["tag"] == BLEW]
    q = [p for p in pts if p["tag"] != BLEW]
    a = np.array([p["het_nll"] / p["mse"] for p in q]); b = np.array([p["het_mumse"] / p["mse"] for p in q])
    c = np.array([p["het_mumse"] / p["het_nll"] for p in q])
    pn = np.array([p["pull_nll"] for p in q]); pm = np.array([p["pull_mumse"] for p in q])
    s = dict(n=len(q), nll_over_mse=float(np.median(a)), nll_below_mse=int((a < 1).sum()),
             mumse_over_mse=float(np.median(b)), mumse_below_mse=int((b < 1).sum()),
             mumse_over_nll=float(np.median(c)), mumse_below_nll=int((c < 1).sum()),
             mumse_over_nll_q=[float(x) for x in np.quantile(c, [0.1, 0.25, 0.75, 0.9])],
             pull_nll_q=[float(x) for x in np.quantile(pn, [0.25, 0.5, 0.75])],
             pull_mumse_q=[float(x) for x in np.quantile(pm, [0.25, 0.5, 0.75])],
             pull_mumse_in=int(((pm > 0.9) & (pm < 1.2)).sum()), pull_nll_in=int(((pn > 0.9) & (pn < 1.2)).sum()))
    per = {}
    for p in q:
        per.setdefault(p["probe"], []).append(p["het_mumse"] / p["het_nll"])
    s["mumse_over_nll_by_probe"] = {k: float(np.median(v)) for k, v in sorted(per.items())}
    print(json.dumps(s, indent=1))
    print("blew up in D47:", blew, "missing:", missing)
    json.dump(dict(summary=s, cells=pts, blew_up=blew, missing=missing),
              open(os.path.join(HERE, "het_mumse22_summary.json"), "w"), indent=1)

    # the figure: what the selection does inside one run against how far two runs of one config land apart, from the
    # runs' best checkpoints (each run's validation mu-MSE minimum, CLAUDE.md "Reported values"; never the last step),
    # runs' validation mu-MSE curves (het_mumse22_curves.json: D47 and this rerun; het_mumse24_curves.json: D82 and D85)
    c22 = json.load(open(os.path.join(HERE, "het_mumse22_curves.json")))
    c24 = json.load(open(os.path.join(HERE, "het_mumse24_curves.json")))
    sel22 = [x["nllsel_a"] / x["min_a"] for x in c22]; run22 = [x["min_b"] / x["min_a"] for x in c22]
    sel24 = [x["nllsel_a"] / x["min_a"] for x in c24]; run24 = [x["min_b"] / x["min_a"] for x in c24]
    s.update(run22_q=[float(x) for x in np.quantile(run22, [.1, .25, .5, .75, .9])], run22_2x=int(sum(r > 2 or r < .5 for r in run22)),
             sel22_q=[float(x) for x in np.quantile(sel22, [.1, .25, .5, .75, .9])],
             run24_q=[float(x) for x in np.quantile(run24, [.1, .25, .5, .75, .9])],
             sel24_q=[float(x) for x in np.quantile(sel24, [.1, .25, .5, .75, .9])])
    json.dump(dict(summary=s, cells=pts, blew_up=blew, missing=missing),
              open(os.path.join(HERE, "het_mumse22_summary.json"), "w"), indent=1)
    fig, ax = ps.figure()
    ecdf = lambda v: (np.sort(v), np.arange(1, len(v) + 1) / len(v))
    for v, col, ls, lab in ((run22, ps.C.blue, "-", r"$2\to2$, two runs"),
                            (sel22, ps.C.blue, "--", r"$2\to2$, NLL pick"),
                            (run24, ps.C.vermillion, "-", r"$2\to3,4$, two runs"),
                            (sel24, ps.C.vermillion, "--", r"$2\to3,4$, NLL pick")):
        x, y = ecdf(np.array(v))
        ax.step(x, y, where="post", color=col, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"validation $\mu$ MSE ratio"); ax.set_ylabel("fraction of cells"); ax.set_yticks([0, 0.5, 1])
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "het_mumse22"))


if __name__ == "__main__":
    main()
