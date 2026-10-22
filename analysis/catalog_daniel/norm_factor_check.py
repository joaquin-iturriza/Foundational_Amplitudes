"""D82, the CPU check of normalization A (the user: "Do the cpu check"): with one standardization shared by every
process, the network has to learn each process's level itself; how large is that job under each target?
  B   y = log|M|^2
  A1  y - log C_p                           (the leading diagram's coupling product, diagram_couplings.py)
  A2  y - (4 - N) log(s_hat / s0)           (the mass dimension of |M|^2, per event)
  A   y - log C_p - (4 - N) log(s_hat / s0)
s0 is one reference scale for the whole catalog, chosen on the training processes to make their levels closest
(a linear-in-N term in disguise, so it is also reported at s0 = (1 TeV)^2). Per variant: the spread (std) of the
process means over the 1352 training processes, the median within-process std, and for the held-out processes the
level error (m_p - M)^2 / sigma_p^2 against the shared mean M, i.e. what a network that learned nothing about levels
would be off by (g, the statistics map, reached 0.32 on the 43 new; stats_net.py). Input: data/norm_factor_stats.json
(norm_factor_stats.py, CC). Output: figs/norm_factor_{a,b}, data/norm_factor_check.json.
    python analysis/catalog_daniel/norm_factor_check.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402


def moments(r, k, c, ls0):
    """mean and std of y - c - k (L - ls0) from the stored moments."""
    m = r["my"] - c - k * (r["mL"] - ls0)
    v = r["vy"] + k * k * r["vL"] - 2 * k * r["cyL"]
    return m, np.sqrt(max(v, 1e-12))


def table(st, dc, var, ls0):
    out = {}
    for split in ("train", "heldout"):
        rows = []
        for n, r in st[split].items():
            k = (4 - r["N"]) if var in ("A2", "A") else 0
            c = max(dc[n]["logc2"]) if var in ("A1", "A") else 0.0
            rows.append((n, *moments(r, k, c, ls0), r["n"]))
        out[split] = rows
    return out


def summary(t, rel):
    m = np.array([x[1] for x in t["train"]]); s = np.array([x[2] for x in t["train"]]); w = np.array([x[3] for x in t["train"]])
    M = np.average(m, weights=w)
    S = np.sqrt(np.average(s ** 2 + (m - M) ** 2, weights=w))     # the shared standardization's scale
    ho = t["heldout"]
    lvl = np.array([(x[1] - M) ** 2 / x[2] ** 2 for x in ho])
    new = np.array([x[0] not in rel for x in ho])
    return dict(between=float(m.std()), within=float(np.median(s)), shared_sd=float(S),
                lvl_new=float(np.median(lvl[new])), lvl_new90=float(np.quantile(lvl[new], .9)),
                lvl_rel=float(np.median(lvl[~new])), lvl=lvl.tolist(), new=new.tolist())


def main():
    st = json.load(open(os.path.join(HERE, "data", "norm_factor_stats.json")))
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    res = {}
    grid = np.linspace(np.log(1.0), np.log(1e8), 401)            # s0 from 1 GeV^2 to (10 TeV)^2
    for var in ("B", "A1", "A2", "A"):
        if var in ("A2", "A"):
            best = min(grid, key=lambda g: np.std([x[1] for x in table(st, dc, var, g)["train"]]))
            for tag, ls0 in (("", best), ("_1TeV", np.log(1e6))):
                res[var + tag] = summary(table(st, dc, var, ls0), rel) | {"sqrt_s0": float(np.exp(ls0 / 2))}
        else:
            res[var] = summary(table(st, dc, var, 0.0), rel)
    for k, v in res.items():
        print(f"{k:8s} level spread across training processes {v['between']:.2f}, within-process std median "
              f"{v['within']:.2f}, shared std {v['shared_sd']:.2f}; held-out level error with no level learned: "
              f"43 new median {v['lvl_new']:.3g} (90% {v['lvl_new90']:.3g}), 96 relabels {v['lvl_rel']:.3g}"
              + (f"; sqrt(s0) = {v['sqrt_s0']:.3g} GeV" if "sqrt_s0" in v else ""))
    json.dump(res, open(os.path.join(HERE, "data", "norm_factor_check.json"), "w"))

    lab = {"B": r"$\log|\mathcal{M}|^2$", "A1": r"$-\log C_p$", "A2": r"$-(4-N)\log(\hat s/s_0)$", "A": "both"}
    col = {"B": ps.C.grey, "A1": ps.C.blue, "A2": ps.C.green, "A": ps.C.vermillion}
    figs = ps.panels(2)
    (f1, a1), (f2, a2) = figs
    for var in ("B", "A1", "A2", "A"):
        t = table(st, dc, var, np.log(res[var].get("sqrt_s0", 1.0) ** 2))
        m = np.sort([x[1] for x in t["train"]]); m = m - np.median(m)
        a1.plot(m, np.arange(1, len(m) + 1) / len(m), color=col[var], label=lab[var])
        v = res[var]
        l = np.sort(np.array(v["lvl"])[np.array(v["new"])])
        a2.plot(l, np.arange(1, len(l) + 1) / len(l), color=col[var], label=lab[var])
    a1.set_xlabel(r"level $m_p$ minus the median"); a1.set_ylabel("fraction of processes below")
    a2.set_xscale("log"); a2.set_xlabel(r"level error $(m_p-M)^2/\sigma_p^2$"); a2.set_ylabel("fraction of the 43 below")
    ps.legend_strip(a1, os.path.join(HERE, "figs", "norm_factor_legend"), ncol=4)
    ps.save_panels(figs, os.path.join(HERE, "figs", "norm_factor"))


if __name__ == "__main__":
    main()
