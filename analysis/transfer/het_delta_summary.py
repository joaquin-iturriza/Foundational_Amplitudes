"""The het retrains of the Delta cells (D47) against the MSE cells they retrain, one point per cell of delta_audit.json's
paper_8k grid that has both. Left: the MSE of ln|M|^2 (from Delta) of the het model's mu against the MSE model's (test split, raw
|M|^2). Right: the pull std against the fraction of events with |pull| < 1 (the sigma stage's sigma, D70); a unit
Gaussian sits at (1, 0.683). Colour: 2->2, 2->3 or 2->4 (uubar_Zggg). Inputs: analysis/transfer/delta_data/*.npz (the MSE cells),
het_delta_data/*.npz (collect_het_delta.py).
    python analysis/transfer/het_delta_summary.py   -> figs/het_delta_summary
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, pick, tag  # noqa: E402

# pull thresholds for the right panel: the fraction of events beyond k sigma, the whole shape of |pull| (a std and one
# fraction are two numbers of it and do not define a Gaussian)
KS = np.linspace(0, 5, 26)
# final-state multiplicity of the probes with more than two (the rest are 2->2)
NFIN = {"ee_ddbarg": 3, "ee_ttbarg": 3, "uubar_Zgg": 3, "uubar_Zggg": 4}
GROUPS = ((2, r"$2\to2$"), (3, r"$2\to3$"), (4, r"$2\to4$"))
# het retrains that blew up: drawn apart at their best checkpoint (CLAUDE.md, reported values), left out of the ranges
BLOWUP = {"tp3_uu64fte_ee_nnbar_d8__trial_0114": "step 1360, best checkpoint step 400"}


EXCLUDED = [0]       # events msel could not take (non-finite Delta or a predicted |M|^2 of exactly 0), over all calls


def msel(d):
    """MSE of ln|M|^2 from Delta = r - 1, the metric of every other per-cell scatter in the paper."""
    # ln|1+d| = ln|M_pred|^2 - ln|M_true|^2 for either sign of the prediction, so a flipped-sign event (d <= -1) counts
    ok = np.isfinite(d) & (d != -1)       # d == -1 is a predicted |M|^2 of exactly 0: ln undefined, the only exclusion
    EXCLUDED[0] += int((~ok).sum())
    return float(np.mean(np.log(np.abs(1 + d[ok])) ** 2))


def main():
    mse = load()
    het = {}
    for f in sorted(glob.glob(os.path.join(HERE, "het_delta_data", "*.npz"))):
        with np.load(f) as z:
            het.update({k: z[k] for k in z.files if k != "_meta"})
    rows = json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"]
    seen, pts, reruns = set(), [], 0
    for r in rows:
        t = tag((r.get("paper_8k") or {}).get("run_dir", "?"))
        if t in seen or t + "__delta" not in het:
            continue
        seen.add(t)
        c = pick(mse, r.get("paper_8k") or {})        # the MSE evaluation the Delta plots draw for this cell
        m = mse.get(c)
        reruns += c.endswith("_rerun")
        d, p = het[t + "__delta"], het[t + "__pull"]
        pts.append(dict(tag=t, probe=r["probe"], arm=r["arm"], nf=NFIN.get(r["probe"], 2), blowup=t in BLOWUP,
                        mse=None if m is None else msel(m), het=msel(d), std=float(p.std()),
                        f1=float(np.mean(np.abs(p) < 1)),
                        tail=np.array([np.mean(np.abs(p) > k) for k in KS])))
    print(f"MSE baselines that are reruns of a pruned trial: {reruns}")
    print(f"events excluded from the MSE of ln|M|^2 (non-finite Delta or predicted |M|^2 = 0): {EXCLUDED[0]}")
    bad = [x for x in pts if x["blowup"]]
    pts = [x for x in pts if not x["blowup"]]
    fig, (a1, a2) = ps.figure(ncols=2)
    for (nf, lab), c in zip(GROUPS, (ps.C.blue, ps.C.vermillion, ps.C.green)):
        q = [x for x in pts if x["nf"] == nf]
        qq = [x for x in q if x["mse"] is not None]
        a1.plot([x["mse"] for x in qq], [x["het"] for x in qq], "o", color=c, label=lab)
        T = np.array([x["tail"] for x in q])
        a2.plot(KS, np.median(T, 0), "-", color=c)
        a2.fill_between(KS, np.percentile(T, 10, 0), np.percentile(T, 90, 0), color=c, alpha=0.25, lw=0)
    for x in bad:
        if x["mse"]:
            a1.plot([x["mse"]], [x["het"]], "x", color="k", label="diverged")
    lo = min(min(x["mse"] for x in pts if x["mse"]), min(x["het"] for x in pts))
    hi = max(max(x["mse"] for x in pts if x["mse"]), max(x["het"] for x in pts))
    lim = (lo / 1.5, hi * 1.5)          # the same limits on both axes: "equal" is the plot box's diagonal
    a1.set_xlim(*lim); a1.set_ylim(*lim)
    a1.plot(lim, lim, "--", color="grey", label=r"$y=x$")
    from math import erfc
    a2.plot(KS, [erfc(k / np.sqrt(2)) for k in KS], "--", color="k")
    # one legend strip above both panels (a corner legend would make room by growing the left panel's y range and
    # break its equal limits); the right panel's entries are drawn as proxies on the left
    a1.plot([], [], "--", color="k", label=r"$\mathcal{N}(0,1)$ (right)")
    a1.plot([], [], "-", color="grey", label="median (right)")
    a1.fill_between([], [], [], color="grey", alpha=0.25, lw=0, label=r"$10$ to $90\%$ (right)")
    a2.set_yscale("log"); a2.set_ylim(1e-5, 1.5)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$, MSE loss"); a1.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$, heteroscedastic")
    a2.set_xlabel(r"$k$"); a2.set_ylabel(r"fraction with $|{\rm pull}|>k$")
    ps.shared_legend(fig, a1, ncol=4)
    ps.save(fig, os.path.join(HERE, "figs", "het_delta_summary"))
    r = np.array([x["het"] / x["mse"] for x in pts if x["mse"]])
    print(f"{len(pts)} cells, {len(r)} with an MSE counterpart; het/MSE MSE of ln|M|^2: median {np.median(r):.2f}, "
          f"range {r.min():.2f} to {r.max():.2f}, het lower in {int((r < 1).sum())}")
    for nf, _ in GROUPS:
        q = [x for x in pts if x["nf"] == nf]
        rr = np.array([x["het"] / x["mse"] for x in q if x["mse"]])
        print(f"2->{nf}: het/MSE MSE of ln|M|^2 median {np.median(rr):.2f} ({rr.min():.2f} to "
              f"{rr.max():.2f}, het lower in {int((rr < 1).sum())}/{len(rr)})")
        print(f"2->{nf}: {len(q)} cells, pull std {min(x['std'] for x in q):.2f} to "
              f"{max(x['std'] for x in q):.2f}, |pull|<1 {min(x['f1'] for x in q):.2f} to {max(x['f1'] for x in q):.2f}")
        st = np.array([x["std"] for x in q])
        print(f"   pull std quartiles {np.percentile(st, 25):.2f} {np.median(st):.2f} {np.percentile(st, 75):.2f}; "
              f"within 0.9 to 1.2: {int(((st >= 0.9) & (st <= 1.2)).sum())}/{len(st)}")
    print("per probe: cells, median het/MSE, median pull std, median |pull|<1")
    for pr in sorted({x["probe"] for x in pts}):
        q = [x for x in pts if x["probe"] == pr]
        rr = [x["het"] / x["mse"] for x in q if x["mse"]]
        print(f"  {pr:16s} {len(q):3d} {np.median(rr):.2f} {np.median([x['std'] for x in q]):.2f} "
              f"{np.median([x['f1'] for x in q]):.2f}")
    for x in bad:
        print(f"blew up: {x['tag']} ({BLOWUP[x['tag']]}): het/MSE {(x['het'] / x['mse']) if x['mse'] else float('nan'):.2f}, "
              f"pull std {x['std']:.2f}, |pull|<1 {x['f1']:.2f}")
    out = dict(cells=len(pts) + len(bad), reruns=reruns,     # the one blown-up cell: het/MSE MSE of ln|M|^2, pull std
               blowup_ratio=[x["het"] / x["mse"] for x in bad if x["mse"]], blowup_pull_std=[x["std"] for x in bad])
    for nf, _ in GROUPS:
        q = [x for x in pts if x["nf"] == nf]
        rr = np.array([x["het"] / x["mse"] for x in q if x["mse"]])
        st = np.array([x["std"] for x in q])
        f1 = np.array([x["f1"] for x in q])
        g = dict(n=len(q), ratio_median=np.median(rr), ratio_min=rr.min(), ratio_max=rr.max(),
                 lower=int((rr < 1).sum()), std_q1=np.percentile(st, 25), std_median=np.median(st),
                 std_q3=np.percentile(st, 75), std_min=st.min(), std_max=st.max(),
                 within=int(((st >= 0.9) & (st <= 1.2)).sum()), f1_min=f1.min(), f1_max=f1.max())
        if nf == 2:
            # the per-probe medians of the 2->2 probes, and the cells outside 0.9 to 1.2 (below, above)
            pm = [np.median([x["het"] / x["mse"] for x in q if x["probe"] == pr and x["mse"]])
                  for pr in {x["probe"] for x in q}]
            lo, hi = [x for x in q if x["std"] < 0.9], sorted([x for x in q if x["std"] > 1.2], key=lambda x: -x["std"])
            g.update(probe_ratio_min=min(pm), probe_ratio_max=max(pm), below_n=len(lo),
                     below_min=min(x["std"] for x in lo), below_max=max(x["std"] for x in lo), above_n=len(hi),
                     above_std=[x["std"] for x in hi], above_f1_min=min(x["f1"] for x in hi),
                     above_f1_max=max(x["f1"] for x in hi),
                     above_max_abs_pull=[float(np.abs(het[x["tag"] + "__pull"]).max()) for x in hi])
        out[f"2to{nf}"] = g
    # the jobs and cells of the two retrain batches (one line per job after the header, its cells after the GPU-h)
    for site, f in (("cclx", "het_delta_jobs_cclx.txt"), ("jz", "het_delta_jobs_jz.txt")):
        jobs = [l.split()[1:] for l in open(os.path.join(HERE, f)) if l.strip() and not l.startswith("#")]
        out[f"jobs_{site}"], out[f"cells_{site}"] = len(jobs), sum(map(len, jobs))
    print(json.dumps(out, default=float))
    print("largest pull std:")
    for x in sorted(pts, key=lambda x: -x["std"])[:12]:
        print(f"  {x['probe']:16s} {x['arm']:24s} std {x['std']:.2f} |p|<1 {x['f1']:.2f} het/mse "
              f"{(x['het'] / x['mse']) if x['mse'] else float('nan'):.2f}")


if __name__ == "__main__":
    main()
