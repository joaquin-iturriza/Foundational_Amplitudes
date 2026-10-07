"""Transfer study, equal-total-data rungs against the ladder at fine-tuning (8k cells). Per cell (rung, probe, D) with
results on both sides: the best trial of each DyHPO search (loss = MSE of log|M|^2 at the best checkpoint,
val_loss * prepd_std^2), its lr and lambda, and the equal-data trial nearest the ladder's best point in (log lr,
log lambda): the closest the equal-data search came to running the ladder's HPs. The two searches drew different
candidate pools, so the comparison is best against best. Rungs 2, 6, 7, 9 compare hp73 parents on both sides; on 3, 4,
5, 8 the ladder parent is another point of the rung's pretraining search (hp185, hp15, hp47, hp47).
    python analysis/transfer/eqd_vs_ladder.py   -> figs/eqd_vs_ladder_lr, figs/eqd_vs_ladder_loss; table on stdout
"""
import math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

CLEAN = {2, 6, 7, 9}
PAT = re.compile(r"^tp3_eqd_r(\d)fte_(.+)_d(\d+)$")


def trials(name, T=None):
    """(lr, lambda, loss) of a cell's trials at horizon T (default: the cell's most common one; a cell runs at one
    horizon, its step count set by D)."""
    ts = [t for t in S.get(name, []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if T is None and ts:
        Ts = [t["T"] for t in ts]
        T = max(set(Ts), key=Ts.count)
    out = []
    for t in ts:
        lam = (t.get("hps") or {}).get("lambda")
        if t.get("T") != T or not lam:
            continue
        out.append((t["lr"], lam, t["val_loss"] * t["prepd_std"] ** 2))
    return out, T


rows = []
for name in sorted(S):
    m = PAT.match(name)
    if not m:
        continue
    r, p, k = int(m.group(1)), m.group(2), int(m.group(3))
    e, T = trials(name)
    l, _ = trials(f"tp3_r{r}fte_{p}_d{k}", T)
    if len(e) < 5 or len(l) < 5:   # a cell is a finished 5-trial search on both sides
        continue
    be, bl = min(e, key=lambda q: q[2]), min(l, key=lambda q: q[2])
    near = min(e, key=lambda q: math.hypot(math.log10(q[0] / bl[0]), math.log10(q[1] / bl[1]) / 2))
    rows.append(dict(r=r, p=p, k=k, ne=len(e), nl=len(l), lr_e=be[0], lr_l=bl[0], lam_e=be[1], lam_l=bl[1],
                     L_e=be[2], L_l=bl[2], L_near=near[2],
                     dist=math.hypot(math.log10(near[0] / bl[0]), math.log10(near[1] / bl[1]) / 2)))

print(f"{len(rows)} cells with 5 trials on both sides")
for lab, sel in (("clean (rungs 2,6,7,9)", lambda x: x["r"] in CLEAN), ("other parents (3,4,5,8)", lambda x: x["r"] not in CLEAN)):
    R = [x for x in rows if sel(x)]
    if not R:
        continue
    dlr = np.array([math.log10(x["lr_e"] / x["lr_l"]) for x in R])
    dlam = np.array([math.log10(x["lam_e"] / x["lam_l"]) for x in R])
    ratio = np.array([x["L_e"] / x["L_l"] for x in R])
    regret = np.array([x["L_near"] / x["L_e"] for x in R])
    print(f"\n{lab}: {len(R)} cells")
    print(f"  best lr, log10(eqd/ladder): median {np.median(dlr):+.2f}, IQR [{np.percentile(dlr,25):+.2f}, {np.percentile(dlr,75):+.2f}]"
          f", |.|<0.5 in {np.mean(abs(dlr) < 0.5):.0%}")
    print(f"  best lambda, log10(eqd/ladder): median {np.median(dlam):+.2f}, IQR [{np.percentile(dlam,25):+.2f}, {np.percentile(dlam,75):+.2f}]")
    print(f"  best loss eqd/ladder: median {np.median(ratio):.2f}, IQR [{np.percentile(ratio,25):.2f}, {np.percentile(ratio,75):.2f}]"
          f", eqd better in {np.mean(ratio < 1):.0%}, within x1.25 either way in {np.mean(abs(np.log(ratio)) < math.log(1.25)):.0%}")
    print(f"  eqd trial nearest the ladder's best / eqd best: median {np.median(regret):.2f}, "
          f"IQR [{np.percentile(regret,25):.2f}, {np.percentile(regret,75):.2f}]; median distance {np.median([x['dist'] for x in R]):.2f} dec")
    for r in sorted({x["r"] for x in R}):
        rr = np.array([x["L_e"] / x["L_l"] for x in R if x["r"] == r])
        print(f"    rung {r}: {len(rr):3d} cells, loss eqd/ladder median {np.median(rr):.2f}")

base = os.path.join(ROOT, "analysis", "transfer", "figs", "eqd_vs_ladder")
cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.sky, ps.C.purple, ps.C.yellow, ps.C.grey]
rungs = sorted({x["r"] for x in rows})
col = {r: cols[i % len(cols)] for i, r in enumerate(rungs)}
fig, (a1, a2) = ps.figure(ncols=2)
for r in rungs:
    R = [x for x in rows if x["r"] == r]
    a1.scatter([x["lr_l"] for x in R], [x["lr_e"] for x in R], color=col[r], label=f"rung {r}")
    a2.scatter([10 ** (x["k"] / 2) for x in R], [x["L_e"] / x["L_l"] for x in R], color=col[r], label=f"rung {r}")
lo, hi = 1e-4, 3e-2
a1.plot([lo, hi], [lo, hi], color="0.5", ls="--", label="equal")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lo, hi); a1.set_ylim(lo, hi)
a1.set_xlabel("best lr, ladder"); a1.set_ylabel("best lr, equal data")
a2.axhline(1, color="0.5", ls="--")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel(r"fine-tune events $D$"); a2.set_ylabel(r"best loss, equal data / ladder")
ps.legend(a1, "upper left")
ps.save(fig, base)
