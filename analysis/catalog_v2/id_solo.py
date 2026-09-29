"""Intrinsic dimension of the solo references' learned per-event representation (tools/measure_id.py, all cells in
analysis/id/*.jsonl), each cell at its best trial's best checkpoint, and the compute-scaling exponent of the same
process, against the number of independent kinematic invariants 3 n_fs - 4 (final-state momenta, minus 4 for
momentum conservation, minus 3 for the overall rotation, plus sqrt(s), which varies in every pool here).
Sets: catalog pools at bs 16384 (solo16k, 63 ... 4000 steps) and bs 1024 (solob1k; the signed pools from solob1kv,
33 ... 1072 steps); the old set's pools with the catalog setup at the old best HPs (solo16kflatold, 4 ... 4000);
the old set's pools swept (solo16kflat, 63 ... 4000) once measured.
Per process and set: ID = the mean over its step counts, uncertainty = the standard deviation over step counts
(larger than each measurement's own resampling spread, which is quoted in the printout); alpha = the floor-aware
fit A C^-alpha + L_inf over its step counts (CLAUDE.md, Scaling fits), uncertainty = the range of leave-one-point-
out refits.
    python analysis/catalog_v2/id_solo.py        writes analysis/catalog_v2/id_solo (png + pdf)
(a) ID per process, grouped by final-state multiplicity, one marker per set, the invariant count 3 n_fs - 4 as a
dashed segment per group; (b) alpha against ID, the bound alpha = 4 / ID dashed (fig:alphamult's 4/DOF with the
measured ID in place of DOF).
    python analysis/catalog_v2/id_solo.py --inv   writes id_solo_inv: (b) against 4 / ID instead (the bound is the
        diagonal; x uncertainty 4 sd / ID^2, first-order propagation of the spread over step counts)"""
import collections, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from solo_b1k import solo_mse
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
PROCS = ["ee_aa", "uubar_uubar", "ee_uu", "ee_ddbar", "ee_bb_nlo",
         "ee_uug", "udbar_WpZZ", "uubar_ZaZ_nlo", "udbar_Wgg_nlo", "uubar_ddbara_nlo",
         "ee_uugg", "udbar_WpZaa"]
SETS = [  # key, label, marker, filled, colour, heads, batch, steps
    ("solo16k", "catalog pool, bs 16384", "o", True, ps.C.vermillion, 8, 16384, [63, 125, 250, 500, 1000, 2000, 4000]),
    ("b1k", "catalog pool, bs 1024", "s", False, ps.C.blue, 8, 1024, [33, 67, 134, 268, 536, 1072]),
    ("solo16kflat", "old pool, swept", "D", True, ps.C.green, 8, 16384, [63, 125, 250, 500, 1000, 2000, 4000]),
    ("solo16kflatold", "old pool, old best HPs", "^", False, ps.C.grey, 8, 16384, [4, 126, 400, 1265, 4000])]
INV = "--inv" in sys.argv
dof = lambda p: 3 * (NP[p] - 2) - 4

ID = collections.defaultdict(dict)                   # (set, process) -> {steps: (id_mean, id_std)}
for f in sorted(glob.glob(os.path.join(ROOT, "analysis", "id", "*.jsonl"))):
    if os.path.basename(f).startswith("test"): continue
    for l in open(f):
        r = json.loads(l)
        if "id_mean" not in r: continue
        name = r["run_dir"].rstrip("/").split("/")[-1] if r["sweep"].startswith("solo16kflatold") else r["sweep"]
        m = re.match(r"(solo16kflatold|solo16kflat|solo16k|solob1kv|solob1k)_t(\d+)_(.+)", name)
        if m:
            key = "b1k" if m.group(1).startswith("solob1k") else m.group(1)
            ID[(key, m.group(3))][int(m.group(2))] = (r["id_mean"], r["id_std"])

LOSS = {"b1k": solo_mse()}                           # set -> {"process|steps": best val loss}
for key, fn in (("solo16k", "solo16k.json"), ("solo16kflat", "solo16kflat.json"), ("solo16kflatold", "solo16kflatold.json")):
    path = os.path.join(HERE, fn)
    if os.path.exists(path):
        LOSS[key] = {k: (min(v) if isinstance(v, list) else v) for k, v in json.load(open(path)).items() if v}

def alpha(key, p, h, bs, T):
    t = [S for S in T if f"{p}|{S}" in LOSS.get(key, {})]
    if len(t) < 4: return None
    x = np.array([flops_per_step(h, NP[p], bs) * S for S in t], float); y = np.array([LOSS[key][f"{p}|{S}"] for S in t])
    f = fit_power_law_with_floor(x, y)
    if f is None: return None
    loo = [fit_power_law_with_floor(x[np.arange(len(x)) != i], y[np.arange(len(x)) != i]) for i in range(len(x))] if len(x) >= 5 else []
    a = [r[1] for r in loo if r]
    return f[1], (min(a) if a else f[1]), (max(a) if a else f[1])

fig, (axA, axB) = ps.figure(ncols=2)
xpos = {p: i + 0.6 * (NP[p] - 2 - 2) for i, p in enumerate(PROCS)}     # a gap between multiplicity groups
off = np.linspace(-0.27, 0.27, len(SETS))
for n in (2, 3, 4):
    xs = [xpos[p] for p in PROCS if NP[p] - 2 == n]
    axA.plot([min(xs) - 0.4, max(xs) + 0.4], [3 * n - 4] * 2, color="black", ls="--")
print(f"{'process':17s} 3n-4 | " + " | ".join(f"{s[1]}: ID [resampling sd], alpha [LOO]" for s in SETS))
for j, (key, lab, mk, filled, col, h, bs, T) in enumerate(SETS):
    xa, ya, ea, xb, yb, exb, eyb = [], [], [], [], [], [], []
    for p in PROCS:
        v = ID.get((key, p))
        if not v: continue
        ids = np.array([m for m, _ in v.values()]); m, s = ids.mean(), ids.std()
        xa.append(xpos[p] + off[j]); ya.append(m); ea.append(s)
        a = alpha(key, p, h, bs, T)
        if a:
            xb.append(4 / m if INV else m); yb.append(a[0]); exb.append(4 * s / m**2 if INV else s); eyb.append([a[0] - a[1], a[2] - a[0]])
    if xa:
        axA.errorbar(xa, ya, yerr=ea, fmt=mk, color=col, mfc=col if filled else "none", capsize=2, label=lab)
    if xb:
        axB.errorbar(xb, yb, xerr=exb, yerr=np.array(eyb).T, fmt=mk, color=col, mfc=col if filled else "none", capsize=2, label=lab)
for p in PROCS:
    cells = []
    for key, *_rest in SETS:
        v = ID.get((key, p)); h, bs, T = _rest[4], _rest[5], _rest[6]
        if not v: cells.append("-"); continue
        ids = np.array([m for m, _ in v.values()]); rs = np.mean([sd for _, sd in v.values()])
        a = alpha(key, p, h, bs, T)
        cells.append(f"{ids.mean():.1f}+-{ids.std():.1f} [{rs:.2f}]" + (f", {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]" if a else ""))
    print(f"{p:17s} {dof(p):4d} | " + " | ".join(cells))
axA.set_xticks([xpos[p] for p in PROCS], [LABEL.get(p, p) for p in PROCS], rotation=0)
axA.set_xticks([np.mean([xpos[p] for p in PROCS if NP[p] - 2 == n]) for n in (2, 3, 4)], [r"$2\to2$", r"$2\to3$", r"$2\to4$"])
axA.set_ylabel("intrinsic dimension")
axA.plot([], [], color="black", ls="--", label=r"$3n_{\rm fs}-4$")
ps.legend(axA, "upper left")
if INV:
    axB.plot([0, 3.3], [0, 3.3], color="black", ls="--", label=r"$\alpha=4/{\rm ID}$")
    axB.set_xlim(0, 2.4); axB.set_ylim(0, 3.3); axB.set_xlabel(r"$4\,/\,$intrinsic dimension")
else:
    g = np.geomspace(1.7, 14, 100); axB.plot(g, 4 / g, color="black", ls="--", label=r"$\alpha=4/{\rm ID}$")
    axB.set_xscale("log"); axB.set_ylim(0, 3.3); axB.set_xlabel("intrinsic dimension")
axB.set_ylabel(r"$\alpha$ in $A\,C^{-\alpha}+L_\infty$")
ps.legend(axB, "upper right")
ps.save(fig, os.path.join(HERE, "id_solo_inv" if INV else "id_solo"))
