"""D82: the 24 other 2->3 / 2->4 Delta cells retrained with the het loss at the het sweeps' shared HPs
(scripts/job_het_delta.sh --shared; D75's sweeps, fig:het23), against the MSE cell (delta_plots.pick) and the het
retrain at the MSE cell's own HPs (D47). Median |Delta| on the test split, one point per cell, both axes on the same
limits so the diagonal is the plot's. Inputs: het_shared_data/*.npz (collect_het_delta.py on
$SCRATCH/zz_delta_hetshared), het_delta_data/, delta_data/, delta_audit.json, het_shared_jobs.txt.
    python analysis/transfer/het_shared_summary.py   -> figs/het_shared
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, pick, tag  # noqa: E402

NFIN = {"ee_ddbarg": 3, "ee_ttbarg": 3, "uubar_Zgg": 3, "uubar_Zggg": 4}


def main():
    mse, hetd = load(), load("het_delta_data")
    hs = {}
    for f in glob.glob(os.path.join(HERE, "het_shared_data", "*.npz")):
        with np.load(f) as z:
            hs.update({k[: -len("__delta")]: z[k] for k in z.files if k.endswith("__delta")})
    cells = [c for l in open(os.path.join(HERE, "het_shared_jobs.txt")) if not l.startswith("#") for c in l.split()[1:]]
    rows = {tag((r.get("paper_8k") or {}).get("run_dir", "?")): r for r in
            json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"] if r.get("paper_8k")}
    med = lambda a: float(np.median(np.abs(a[np.isfinite(a)])))
    pts, missing = [], []
    for c in cells:
        t = tag(c)
        if t not in hs:
            missing.append(t); continue
        r = rows[t]
        pts.append(dict(tag=t, probe=r["probe"], arm=r["arm"], mse=med(mse[pick(mse, r["paper_8k"])]),
                        het_mse=med(hetd[t]) if t in hetd else None, het_shared=med(hs[t])))
    for p in pts:
        print(f"{p['tag']:42s} MSE {p['mse']:.4g}  het@MSE-HPs {p['het_mse'] or float('nan'):.4g}  het@shared "
              f"{p['het_shared']:.4g}  shared/MSE {p['het_shared'] / p['mse']:.2f}")
    for nf in (3, 4):
        q = [p for p in pts if NFIN[p["probe"]] == nf]
        r_ = np.array([p["het_shared"] / p["mse"] for p in q])
        r2 = np.array([p["het_mse"] / p["mse"] for p in q if p["het_mse"]])
        print(f"2->{nf - 1 + 1 if nf == 3 else 4}: {len(q)} cells, het@shared / MSE median {np.median(r_):.2f}, lower than MSE in "
              f"{int((r_ < 1).sum())}/{len(r_)}; het@MSE-HPs / MSE median {np.median(r2):.2f}")
    print("missing (lost to the full disk):", missing)
    json.dump({"cells": pts, "missing": missing}, open(os.path.join(HERE, "het_shared_summary.json"), "w"), indent=1)

    fig, ax = ps.figure()
    for nf, col, lab in ((3, ps.C.vermillion, r"$2\to3$"), (4, ps.C.green, r"$2\to4$")):
        q = [p for p in pts if NFIN[p["probe"]] == nf]
        ax.plot([p["mse"] for p in q], [p["het_shared"] for p in q], "o", color=col, label=lab + ", shared HPs")
        ax.plot([p["mse"] for p in q if p["het_mse"]], [p["het_mse"] for p in q if p["het_mse"]], "o", color=col,
                mfc="none", label=lab + r", the MSE cell's HPs")
    v = [p[k] for p in pts for k in ("mse", "het_mse", "het_shared") if p[k]]
    lim = (min(v) / 1.5, max(v) * 1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.plot(lim, lim, "--", color="grey", label="equal")
    ax.set_xlabel(r"median $|\Delta|$, MSE"); ax.set_ylabel(r"median $|\Delta|$, het")
    ps.shared_legend(fig, ax, ncol=2)
    ps.save(fig, os.path.join(HERE, "figs", "het_shared"))


if __name__ == "__main__":
    main()
