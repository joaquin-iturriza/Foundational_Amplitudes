"""Intrinsic dimension of the solo references' learned per-event representation (tools/measure_id.py, all cells in
analysis/id/solo_all.jsonl), each cell at its best trial's best checkpoint, against the number of independent
kinematic invariants 3 n_fs - 4 (final-state momenta, minus 4 for momentum conservation, minus 3 for the overall
rotation, plus sqrt(s), which varies in every pool here). Sets: catalog pools at bs 16384 (solo16k) and bs 1024
(solob1k, the signed pools from solob1kv), and the old set's pools with the catalog setup at the old best HPs
(solo16kflatold). (a) ID against steps, bs 16384, one line per process, dashed at 3 n_fs - 4 = 2, 5, 8;
(b) every cell against 3 n_fs - 4 (small horizontal jitter), ID = 3 n_fs - 4 dashed.
    python analysis/catalog_v2/id_solo.py        writes analysis/catalog_v2/id_solo (png + pdf)"""
import collections, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import plot_style as ps
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
SET = {"solo16k": "catalog pool, bs 16384", "solob1k": "catalog pool, bs 1024", "solob1kv": "catalog pool, bs 1024",
       "solo16kflatold": "old pool, old best HPs"}
D = collections.defaultdict(dict)
for l in open(os.path.join(ROOT, "analysis", "id", "solo_all.jsonl")):
    r = json.loads(l)
    name = r["run_dir"].rstrip("/").split("/")[-1] if r["sweep"].startswith("solo16kflatold") else r["sweep"]
    m = re.match(r"(solo16kflatold|solo16k|solob1kv|solob1k)_t(\d+)_(.+)", name)
    if m: D[(SET[m.group(1)], m.group(3))][int(m.group(2))] = r["id_mean"]
PROCS = ["ee_aa", "uubar_uubar", "ee_uu", "ee_ddbar", "ee_uug", "udbar_WpZZ", "ee_uugg", "udbar_WpZaa",
         "uubar_ZaZ_nlo", "ee_bb_nlo", "udbar_Wgg_nlo", "uubar_ddbara_nlo"]
dof = lambda p: 3 * (NP[p] - 2) - 4
COL = {2: ps.C.blue, 3: ps.C.orange, 4: ps.C.vermillion}
fig, (axA, axB) = ps.figure(ncols=2)
for p in PROCS:
    k = ("catalog pool, bs 16384", p)
    t = sorted(D[k]); axA.plot(t, [D[k][x] for x in t], color=COL[NP[p] - 2], marker="o")
for n, c in COL.items():
    axA.axhline(3 * n - 4, color=c, ls="--")
axA.set_xscale("log"); axA.set_xlabel("optimizer steps (bs 16384)"); axA.set_ylabel("intrinsic dimension")
H = [axA.plot([], [], color=c, marker="o")[0] for c in COL.values()] + [axA.plot([], [], color="black", ls="--")[0]]
ps.legend(axA, "upper left", handles=H, labels=[r"$2\to2$", r"$2\to3$", r"$2\to4$", r"$3n_{\rm fs}-4$"], ncol=2)
MK = {"catalog pool, bs 16384": ("o", True), "catalog pool, bs 1024": ("s", False), "old pool, old best HPs": ("^", True)}
rng = np.random.default_rng(0)
for s, (mk, filled) in MK.items():
    xs, ys, cs = [], [], []
    for p in PROCS:
        for t, m in D.get((s, p), {}).items():
            xs.append(dof(p) * (1 + 0.05 * rng.standard_normal())); ys.append(m); cs.append(COL[NP[p] - 2])
    axB.scatter(xs, ys, marker=mk, facecolors=cs if filled else "none", edgecolors=cs, label=s)
g = np.array([1.5, 9]); axB.plot(g, g, color="black", ls="--", label=r"ID $=3n_{\rm fs}-4$")
axB.set_xlabel(r"kinematic invariants $3n_{\rm fs}-4$"); axB.set_ylabel("intrinsic dimension")
ps.legend(axB, "upper left")
ps.save(fig, os.path.join(HERE, "id_solo"))
