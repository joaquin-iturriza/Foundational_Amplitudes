"""D99, the first GPU test of a candidate that passed the CPU proxy (norm_targets.py): DlessInOrdQ (the coupling-order
design run with each extra particle's own Lorentz-invariant emission scale, physics_scales.qperp_logs, in place of
s_hat; scripts/job_catalog_norm.sh DlessInOrdQ short: inputs, encoder, steps, HPs and seed as DlessInOrd) against
DlessInOrd on every held-out set both were read on; DlessInOrdQs (the soft minimum, tau = 0.3, qperp_soft) in the table
where it was scored (the 43 new, the 96 relabels, the 2->4 probes). Per process the error d = q - t of log|M|^2: the MSE, the offset
(mean of d) and the shape (variance of d); medians over processes. One seed each.
Inputs: data/catalog_short_<arm>_sigstage.npz, catalog_2to4{,c30,E100,E100p}_short_<arm>.npz, catalog_loop_short_<arm>.npz,
catalog_top_short_<arm>{,_notop}.npz where present. Output: the table printed, data/qperp_ab.json, figs/qperp_ab.{png,pdf}.
    .venv-driver/bin/python analysis/catalog_daniel/qperp_ab.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ARMS = ("DlessInOrd", "DlessInOrdQ", "DlessInOrdQs")
LOOP = ("_nlo_", "_loop_")


def sets():
    tw = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    return [("43 new", "catalog_short_{}_sigstage", lambda p: p not in tw),
            ("96 relabels", "catalog_short_{}_sigstage", lambda p: p in tw),
            ("2to3 at 100 TeV", "catalog_2to4E100_short_{}", lambda p: True),
            ("2to4 +g", "catalog_2to4_short_{}", lambda p: p.endswith("_g")),
            ("2to4 +a", "catalog_2to4_short_{}", lambda p: p.endswith("_a")),
            ("2to4 +g, pT>=30", "catalog_2to4c30_short_{}", lambda p: p.endswith("_g")),
            ("2to4 +a, pT>=30", "catalog_2to4c30_short_{}", lambda p: p.endswith("_a")),
            ("2to4 +g, 100 TeV", "catalog_2to4E100p_short_{}", lambda p: p.endswith("_g")),
            ("2to4 +a, 100 TeV", "catalog_2to4E100p_short_{}", lambda p: p.endswith("_a")),
            ("loop trees", "catalog_loop_short_{}", lambda p: not any(s in p for s in LOOP)),
            ("one-loop", "catalog_loop_short_{}", lambda p: any(s in p for s in LOOP)),
            ("top, trained with", "catalog_top_short_{}", lambda p: True),
            ("top, never seen", "catalog_top_short_{}_notop", lambda p: True)]


def per_process(f, sel):
    path = os.path.join(HERE, "data", f + ".npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    ps = [p for p in sorted({k.split("__")[0] for k in z.files if k.endswith("__q")}) if sel(p)]
    return {p: z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in ps}


def main():
    res, per = {}, {}
    for lab, f, sel in sets():
        row = {}
        for a in ARMS:
            d = per_process(f.format(a), sel)
            if d is None:
                continue
            per[(lab, a)] = {p: float(np.mean(x ** 2)) for p, x in d.items()}
            row[a] = dict(n=len(d), mse=float(np.median([np.mean(x ** 2) for x in d.values()])),
                          offset=float(np.median([x.mean() for x in d.values()])),
                          shape=float(np.median([x.var() for x in d.values()])))
        for b, key in ((ARMS[1], "ratio"), (ARMS[2], "ratio_soft")):
            if ARMS[0] in row and b in row:
                common = sorted(set(per[(lab, ARMS[0])]) & set(per[(lab, b)]))
                r = np.array([per[(lab, b)][p] / per[(lab, ARMS[0])][p] for p in common])
                row[key] = [float(x) for x in np.quantile(r, [0.1, 0.5, 0.9])]
        res[lab] = row
        print(f"{lab:18s} " + "  ".join(f"{a}: n={v['n']} MSE {v['mse']:.3g} off {v['offset']:+.2f} shape {v['shape']:.3g}"
                                        for a, v in row.items() if a in ARMS)
              + (f"  Q/Ord per process 10/50/90% {np.round(row['ratio'], 2).tolist()}" if "ratio" in row else "")
              + (f"  Qs/Ord {np.round(row['ratio_soft'], 2).tolist()}" if "ratio_soft" in row else ""))
    json.dump(res, open(os.path.join(HERE, "data", "qperp_ab.json"), "w"), indent=1)
    plot(per)


def plot(per):
    import plot_style as ps

    def ecdf(ax, v, **kw):
        x = np.sort(np.asarray(v)); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)
    fig, (axL, axR) = ps.figure(ncols=2)
    col = {ARMS[0]: ps.C.blue, ARMS[1]: ps.C.vermillion}
    lab = {ARMS[0]: r"$\hat s$", ARMS[1]: "emission scales"}
    for a in ARMS[:2]:             # the soft-minimum arm (DlessInOrdQs) was scored on the 43 new and the 2->4 probes only
        v = [x for s in ("2to4 +g", "2to4 +a", "2to4 +g, pT>=30", "2to4 +a, pT>=30") for x in per.get((s, a), {}).values()]
        ecdf(axL, v, color=col[a], label=lab[a] + r", $2\to4$")
        ecdf(axR, list(per.get(("43 new", a), {}).values()), color=col[a], label=lab[a] + ", 43 new")
        ecdf(axR, list(per.get(("one-loop", a), {}).values()), color=col[a], ls="--", label=lab[a] + ", one loop")
    for ax in (axL, axR):
        ax.set_xscale("log"); ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$ per process"); ax.set_ylabel("fraction of processes")
        ax.set_yticks([0, 0.5, 1])
    ps.legend(axL, "upper left"); ps.legend(axR, "upper left")
    ps.make_room(axL); ps.make_room(axR)
    ps.save(fig, os.path.join(HERE, "figs", "qperp_ab"))


if __name__ == "__main__":
    main()
