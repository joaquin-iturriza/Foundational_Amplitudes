"""The het retrains of the Delta cells (D47) against the MSE cells they retrain, one point per cell of delta_audit.json's
paper_8k grid that has both. Left: the median |Delta| of the het model's mu against the MSE model's (test split, raw
|M|^2). Right: the pull std against the fraction of events with |pull| < 1 (the sigma stage's sigma, D70); a unit
Gaussian sits at (1, 0.683). Colour: 2->2 or 2->3. Inputs: analysis/transfer/delta_data/*.npz (the MSE cells),
het_delta_data/*.npz (collect_het_delta.py).
    python analysis/transfer/het_delta_summary.py   -> figs/het_delta_summary
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, tag  # noqa: E402

# the probes with three final-state particles (2->3)
THREE = {"ee_ddbarg", "ee_ttbarg", "uubar_Zgg", "uubar_Zggg"}


def main():
    mse = load()
    het = {}
    for f in sorted(glob.glob(os.path.join(HERE, "het_delta_data", "*.npz"))):
        with np.load(f) as z:
            het.update({k: z[k] for k in z.files if k != "_meta"})
    rows = json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"]
    seen, pts = set(), []
    for r in rows:
        t = tag((r.get("paper_8k") or {}).get("run_dir", "?"))
        if t in seen or t + "__delta" not in het:
            continue
        seen.add(t)
        m = next((mse[c] for c in (t, t + "_rerun", t + "_tf32on", t + "_tf32off") if c in mse), None)
        d, p = het[t + "__delta"], het[t + "__pull"]
        pts.append(dict(probe=r["probe"], arm=r["arm"], three=r["probe"] in THREE,
                        mse=None if m is None else float(np.median(np.abs(m[np.isfinite(m)]))),
                        het=float(np.median(np.abs(d[np.isfinite(d)]))), std=float(p.std()),
                        f1=float(np.mean(np.abs(p) < 1))))
    fig, (a1, a2) = ps.figure(ncols=2)
    for three, c, lab in ((False, ps.C.blue, r"$2\to2$"), (True, ps.C.vermillion, r"$2\to3$")):
        q = [x for x in pts if x["three"] == three]
        qq = [x for x in q if x["mse"] is not None]
        a1.plot([x["mse"] for x in qq], [x["het"] for x in qq], "o", color=c, label=lab)
        a2.plot([x["std"] for x in q], [x["f1"] for x in q], "o", color=c)
    lo = min(min(x["mse"] for x in pts if x["mse"]), min(x["het"] for x in pts))
    hi = max(max(x["mse"] for x in pts if x["mse"]), max(x["het"] for x in pts))
    a1.plot([lo, hi], [lo, hi], "--", color="grey", label="equal")
    a2.plot([1], [0.683], "*", color="k", label="unit Gaussian")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"median $|\Delta|$, MSE"); a1.set_ylabel(r"median $|\Delta|$, het")
    a2.set_xlabel("pull std"); a2.set_ylabel(r"fraction $|{\rm pull}|<1$")
    ps.legend(a1, "upper left"); ps.legend(a2, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "het_delta_summary"))
    r = np.array([x["het"] / x["mse"] for x in pts if x["mse"]])
    print(f"{len(pts)} cells, {len(r)} with an MSE counterpart; het/MSE median |Delta|: median {np.median(r):.2f}, "
          f"range {r.min():.2f} to {r.max():.2f}, het lower in {int((r < 1).sum())}")
    for three in (False, True):
        q = [x for x in pts if x["three"] == three]
        rr = np.array([x["het"] / x["mse"] for x in q if x["mse"]])
        print(f"{'2->3' if three else '2->2'}: het/MSE median |Delta| median {np.median(rr):.2f} ({rr.min():.2f} to "
              f"{rr.max():.2f}, het lower in {int((rr < 1).sum())}/{len(rr)})")
        print(f"{'2->3' if three else '2->2'}: {len(q)} cells, pull std {min(x['std'] for x in q):.2f} to "
              f"{max(x['std'] for x in q):.2f}, |pull|<1 {min(x['f1'] for x in q):.2f} to {max(x['f1'] for x in q):.2f}")
    for x in sorted(pts, key=lambda x: -x["std"])[:12]:
        print(f"  {x['probe']:16s} {x['arm']:24s} std {x['std']:.2f} |p|<1 {x['f1']:.2f} het/mse "
              f"{(x['het'] / x['mse']) if x['mse'] else float('nan'):.2f}")


if __name__ == "__main__":
    main()
