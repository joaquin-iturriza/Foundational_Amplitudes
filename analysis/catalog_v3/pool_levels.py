"""catalog_v3's levels before training: does each held-out axis sit inside the training set's level range? A zero-shot
prediction has to place a process's level (the median of log|M|^2 over its events); a held-out process whose level lies
outside every training process of its multiplicity asks the level model to extrapolate, not interpolate. From the pool
audit (pool_audit.json, the val split of every pool) and the roles (recipes/catalog_v3_roles.json): per pool its
multiplicity N and its median log10|M|^2 (GeV units, mass dimension 8 - 2N, so compared only at equal N). Trees: the
training classes against H1, H2, H3; one-loop pools (stage 2a only, the rest waits on the mass-scheme call): training
against H5. Per axis: the share of held-out pools outside the training range at their N, and the median distance from
the nearest training level at their N in units of that N's training spread.
    python3 analysis/catalog_v3/pool_levels.py   -> data/pool_levels.json, figs/pool_levels.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def main():
    au = json.load(open(os.path.join(HERE, "pool_audit.json")))["pools"]
    ro = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))
    pts = {}
    for name, r in au.items():
        v = r["val"]
        if v.get("missing") or not v.get("q"):
            continue
        N = len(v["pdg"])
        if name in ro["trees"]:
            t = ro["trees"][name]; grp = "tree train" if t["role"] == "train" else t["axis"]
        elif name in ro["loops"]:
            t = ro["loops"][name]; grp = "loop train" if t["role"] == "train" else "H5"
        else:
            continue
        pts.setdefault(grp, []).append((N, v["q"][1], name))
    out = {}
    for axis, ref in (("H1", "tree train"), ("H2", "tree train"), ("H3", "tree train"), ("H5", "loop train")):
        tr = {}
        for N, m, _ in pts.get(ref, []):
            tr.setdefault(N, []).append(m)
        outside, dist, nolayer = [], [], 0
        for N, m, n in pts.get(axis, []):
            if N not in tr:
                nolayer += 1; continue
            a = np.array(tr[N]); sd = a.std() or 1.0
            o = m < a.min() or m > a.max()
            outside.append(n) if o else None
            dist.append(float(np.min(np.abs(a - m)) / sd))
        k = len(pts.get(axis, []))
        out[axis] = {"pools": k, "no_training_at_N": nolayer, "outside": len(outside),
                     "outside_names": outside, "median_nearest_in_sd": float(np.median(dist)) if dist else None,
                     "max_nearest_in_sd": float(np.max(dist)) if dist else None}
        print(axis, json.dumps({x: y for x, y in out[axis].items() if x != "outside_names"}), outside[:8])
    out["counts"] = {g: len(v) for g, v in pts.items()}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "pool_levels.json"), "w"), indent=1)

    import plot_style as ps
    from matplotlib.ticker import MaxNLocator
    fig, (axL, axR) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    for ax, groups in ((axL, (("tree train", ps.C.blue), ("H1", ps.C.vermillion), ("H2", ps.C.green),
                               ("H3", ps.C.orange))),
                       (axR, (("loop train", ps.C.blue), ("H5", ps.C.vermillion)))):
        for g, c in groups:
            a = pts.get(g, [])
            if not a:
                continue
            off = {"tree train": -0.24, "loop train": -0.12, "H1": -0.08, "H2": 0.08, "H3": 0.24, "H5": 0.12}[g]
            ax.plot([N + off + rng.uniform(-0.05, 0.05) for N, _, _ in a], [m for _, m, _ in a], ".", color=c,
                    label=f"{g} ({len(a)})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(r"particles $N$"); ax.set_ylabel(r"median $\log_{10}|\mathcal{M}|^2$")
        ps.legend(ax, "upper right"); ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "pool_levels"))


if __name__ == "__main__":
    main()
