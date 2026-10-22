"""D85: the 24 2->3 / 2->4 Delta cells of D82 (het loss at the het sweeps' shared HPs) retrained with the main run's
checkpoint picked on the validation mu-MSE instead of the validation NLL (scripts/job_het_delta.sh --shared-mumse,
training.heterosc_select=mu_mse; the sigma stage after it still selects on the NLL, with mu frozen). Same HPs, data,
steps and seeds as D82, so the selection is the only difference. Median |Delta| on the test split, one point per cell,
against the MSE cell (delta_plots.pick), next to D82's NLL-selected retrain (fig:het_shared).
Inputs: het_mumse_data/zz_delta_hetshared_mumse.npz (lines 1, 2, 5, 6 of het_shared_jobs.txt) and
het_mumse_data/zz_delta_hetshared_mumse_clean.npz (lines 3 and 4: two duplicate jobs trained those cells at once in the
first root, so they were rerun into a clean one), collect_het_delta.py on each root; het_shared_data/, delta_data/,
delta_audit.json, het_shared_jobs.txt.
    python analysis/transfer/het_mumse_summary.py   -> figs/het_mumse, het_mumse_summary.json
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, pick, tag  # noqa: E402

NFIN = {"ee_ddbarg": 3, "ee_ttbarg": 3, "uubar_Zgg": 3, "uubar_Zggg": 4}


def deltas(path):
    with np.load(path) as z:
        return {k[: -len("__delta")]: z[k] for k in z.files if k.endswith("__delta")}


def main():
    mse = load()
    nll = {}
    for f in glob.glob(os.path.join(HERE, "het_shared_data", "*.npz")):
        nll.update(deltas(f))
    first = deltas(os.path.join(HERE, "het_mumse_data", "zz_delta_hetshared_mumse.npz"))
    clean = deltas(os.path.join(HERE, "het_mumse_data", "zz_delta_hetshared_mumse_clean.npz"))
    lines = [l.split()[1:] for l in open(os.path.join(HERE, "het_shared_jobs.txt")) if not l.startswith("#")]
    src = {tag(c): (clean if i in (2, 3) else first) for i, l in enumerate(lines) for c in l}
    rows = {tag((r.get("paper_8k") or {}).get("run_dir", "?")): r for r in
            json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"] if r.get("paper_8k")}
    med = lambda a: float(np.median(np.abs(a[np.isfinite(a)])))
    pts, missing = [], []
    for t, d in src.items():
        if t not in d or t not in nll:
            missing.append(t); continue
        r = rows[t]
        pts.append(dict(tag=t, probe=r["probe"], arm=r["arm"], mse=med(mse[pick(mse, r["paper_8k"])]),
                        het_nll=med(nll[t]), het_mumse=med(d[t])))
    for p in pts:
        print(f"{p['tag']:42s} MSE {p['mse']:.4g}  het, NLL-selected {p['het_nll']:.4g}  het, mu-MSE-selected "
              f"{p['het_mumse']:.4g}  mu-MSE/NLL {p['het_mumse'] / p['het_nll']:.2f}")
    out = {"cells": pts, "missing": missing}
    for nf in (3, 4):
        q = [p for p in pts if NFIN[p["probe"]] == nf]
        a = np.array([p["het_nll"] / p["mse"] for p in q]); b = np.array([p["het_mumse"] / p["mse"] for p in q])
        c = np.array([p["het_mumse"] / p["het_nll"] for p in q])
        s = dict(n=len(q), nll_over_mse=float(np.median(a)), nll_below_mse=int((a < 1).sum()),
                 mumse_over_mse=float(np.median(b)), mumse_below_mse=int((b < 1).sum()),
                 mumse_over_nll=float(np.median(c)), mumse_below_nll=int((c < 1).sum()))
        out[f"2->{nf}"] = s
        print(f"2->{nf}: {s['n']} cells; het/MSE median: NLL-selected {s['nll_over_mse']:.2f} "
              f"({s['nll_below_mse']}/{s['n']} below MSE), mu-MSE-selected {s['mumse_over_mse']:.2f} "
              f"({s['mumse_below_mse']}/{s['n']} below MSE); mu-MSE/NLL median {s['mumse_over_nll']:.2f}, "
              f"lower in {s['mumse_below_nll']}/{s['n']}")
    print("missing:", missing)
    json.dump(out, open(os.path.join(HERE, "het_mumse_summary.json"), "w"), indent=1)

    fig, ax = ps.figure()
    for nf, col, lab in ((3, ps.C.vermillion, r"$2\to3$"), (4, ps.C.green, r"$2\to4$")):
        q = [p for p in pts if NFIN[p["probe"]] == nf]
        ax.plot([p["mse"] for p in q], [p["het_mumse"] for p in q], "o", color=col, label=lab + r", $\mu$-MSE selected")
        ax.plot([p["mse"] for p in q], [p["het_nll"] for p in q], "o", color=col, mfc="none", label=lab + ", NLL selected")
    v = [p[k] for p in pts for k in ("mse", "het_nll", "het_mumse")]
    lim = (min(v) / 1.5, max(v) * 1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.plot(lim, lim, "--", color="grey", label="equal")
    ax.set_xlabel(r"median $|\Delta|$, MSE"); ax.set_ylabel(r"median $|\Delta|$, het")
    ps.shared_legend(fig, ax, ncol=2)
    ps.save(fig, os.path.join(HERE, "figs", "het_mumse"))


if __name__ == "__main__":
    main()
