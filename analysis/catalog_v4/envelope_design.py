#!/usr/bin/env python3
"""Step 49 (catalog_coverage, Joaquin 2026-11-04: "sample in a way that covers the range the pp beams cover, but
wider, and with enough data so that performance is good"): per 13 TeV pp class, how far catalog_v4's sampling is from
the envelope criterion, and what candidate sampling changes would close it. Design input only, nothing generated.

    python3 analysis/catalog_v4/envelope_design.py     data/pp_support_reads.json -> data/envelope_design.json

Densities are each sample's normalised counts (count / n) in the 2D occupancy of pp_support.py (log10 sqrt(s_hat) in
0.1 dex x ln|M|^2 in 2 nat), the pp sample being the class's 13 TeV MadEvent train+test events (1.2e4), the catalog
its catalog_v4 pool (1e5). Criterion at fraction f: the share of pp events in bins where catalog density < f x pp
density (a bin with no catalog event counts as outside the support). Reported per class at f in F, and pooled.
Candidates (mixtures, each normalised; w the weight of the added component):
  current        catalog_v4 as generated
  train_mix25    what the share-0.25 training mixture sees (0.75 catalog + 0.25 the class's default-sampled events,
                 here the pp sample standing in for them; information, not a catalog change)
  sqrts_pp_w     (1-w) catalog + w x a component with pp's sqrt(s_hat) marginal widened by a Gaussian of 0.1 dex in
                 log10 sqrt(s_hat), and the catalog's own ln|M|^2 at that sqrt(s_hat) (catalog conditional per
                 sqrt(s_hat) row; a row with no catalog event takes the nearest row that has one)
  sqrts_pp_unw_w (1-w) catalog + w x the same sqrt(s_hat) component with ln|M|^2 drawn as pp's conditional, widened
                 by one 2-nat bin each side (an |M|^2-unweighted component: candidates accepted with probability
                 proportional to |M|^2 over their proposal density, at the widened sqrt(s_hat))
  unw_w_sS       (1-w) catalog + w x an |M|^2-unweighted component (pp's conditional ln|M|^2 at each sqrt(s_hat), not
                 widened) at pp's sqrt(s_hat) marginal widened by a Gaussian of S dex: the proposed design, for
                 w in W2 and S in SIG
The last is an upper bound for that component: it assumes the candidates reach every bin pp reaches; where the
candidates' support ends (the 2->4 tail above the catalog's max, step 46/48) only a new candidate generator does.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
F = (0.1, 0.25, 1 / 3, 0.5, 1.0)
W = (0.25, 0.5)
W2 = (0.5, 0.6, 0.7, 0.8)
SIG = (0.05, 0.1)


def widen_rows(m, sig_bins=1.0):
    """m (n_s, n_l) smeared along axis 0 by a Gaussian of sig_bins bins (0.1 dex each), mass kept"""
    k = np.arange(-4, 5)
    g = np.exp(-0.5 * (k / sig_bins) ** 2); g /= g.sum()
    out = np.zeros_like(m)
    for j, gj in zip(k, g):
        sh = np.roll(m, j, axis=0)
        if j > 0: sh[:j] = 0
        if j < 0: sh[j:] = 0
        out += gj * sh
    return out / out.sum()


def catalog_conditional(c):
    """per sqrt(s_hat) row the catalog's ln|M|^2 distribution; empty rows take the nearest filled row"""
    rows = c.sum(1)
    filled = np.where(rows > 0)[0]
    cond = np.zeros_like(c)
    for i in range(c.shape[0]):
        j = filled[np.argmin(np.abs(filled - i))]
        cond[i] = c[j] / c[j].sum()
    return cond


def gap(cat, pp, f):
    """share of pp mass in bins with cat < f pp (cat, pp normalised)"""
    m = pp > 0
    return float(pp[m][cat[m] < f * pp[m]].sum())


def one(r):
    pp = np.array(r["default"]["hist2d"], float); c = np.array(r["catalog"]["hist2d"], float)
    if pp.sum() == 0 or c.sum() == 0:
        return None
    pp /= pp.sum(); c /= c.sum()
    marg = widen_rows(pp.sum(1, keepdims=True))[:, 0]
    comp_cat = marg[:, None] * catalog_conditional(c)
    ppc = pp / np.where(pp.sum(1, keepdims=True) > 0, pp.sum(1, keepdims=True), 1)
    ppc = ppc + np.roll(ppc, 1, 1) + np.roll(ppc, -1, 1)
    ppc = ppc / np.where(ppc.sum(1, keepdims=True) > 0, ppc.sum(1, keepdims=True), 1)
    # rows with no pp event: nearest pp row's conditional
    filled = np.where(pp.sum(1) > 0)[0]
    for i in range(ppc.shape[0]):
        if ppc[i].sum() == 0:
            ppc[i] = ppc[filled[np.argmin(np.abs(filled - i))]]
    comp_unw = marg[:, None] * ppc
    comp_unw /= comp_unw.sum()
    ppc0 = pp / np.where(pp.sum(1, keepdims=True) > 0, pp.sum(1, keepdims=True), 1)
    for i in range(ppc0.shape[0]):
        if ppc0[i].sum() == 0:
            ppc0[i] = ppc0[filled[np.argmin(np.abs(filled - i))]]
    cands = {"current": c, "train_mix25": 0.75 * c + 0.25 * pp}
    for w in W:
        cands[f"sqrts_pp_{w}"] = (1 - w) * c + w * comp_cat
        cands[f"sqrts_pp_unw_{w}"] = (1 - w) * c + w * comp_unw
    for S in SIG:
        m_s = widen_rows(pp.sum(1, keepdims=True), S / 0.1)[:, 0]
        comp = m_s[:, None] * ppc0
        comp /= comp.sum()
        for w in W2:
            cands[f"unw_{w}_s{S}"] = (1 - w) * c + w * comp
    out = {k: {f"{f:.3g}": gap(v, pp, f) for f in F} for k, v in cands.items()}
    # the marginals alone, current catalog
    out["current_sqrts_marginal"] = {f"{f:.3g}": gap(c.sum(1), pp.sum(1), f) for f in F}
    out["current_logm_marginal"] = {f"{f:.3g}": gap(c.sum(0), pp.sum(0), f) for f in F}
    out["outside_support_2d"] = float(pp[(pp > 0) & (c == 0)].sum())
    return out


def main():
    d = json.load(open(os.path.join(DATA, "pp_support_reads.json")))
    rows = [r for r in d["rows"] if "error" not in r]
    per = {}
    for r in rows:
        o = one(r)
        if o is not None:
            n_fin = len(r["default"]["pdg"]) - 2 if isinstance(r["default"].get("pdg"), list) else None
            o["n_final"] = n_fin
            o["n_pp"] = r["default"]["n"]
            o["logm_above"] = r["default"]["logm_above"]
            per[r["probe"]] = o
    names = list(next(iter(per.values())).keys())
    o2 = np.array([p["outside_support_2d"] for p in per.values()]); n2 = np.array([p["n_pp"] for p in per.values()], float)
    la = np.array([p["logm_above"] for p in per.values()])
    supp = {"outside_support_2d": {"pooled": float((o2 * n2).sum() / n2.sum()), "median": float(np.median(o2)),
                                   "max": float(o2.max()), "n_classes_any": int((o2 > 0).sum()),
                                   "n_classes_over_1pct": int((o2 > 0.01).sum())},
            "logm_above_catalog_max": {"pooled": float((la * n2).sum() / n2.sum()), "n_classes_over_1pct": int((la > 0.01).sum())}}
    summ = {}
    for k in names:
        if not isinstance(per[next(iter(per))][k], dict):
            continue
        summ[k] = {}
        for f in per[next(iter(per))][k]:
            x = np.array([p[k][f] for p in per.values()])
            n = np.array([p["n_pp"] for p in per.values()], float)
            summ[k][f] = {"pooled": float((x * n).sum() / n.sum()), "median": float(np.median(x)),
                          "p90": float(np.quantile(x, 0.9)), "max": float(x.max()),
                          "n_classes_over_1pct": int((x > 0.01).sum()), "n_classes_over_10pct": int((x > 0.1).sum())}
    by_n = {}
    for nf in sorted({p["n_final"] for p in per.values()}):
        sel = [p for p in per.values() if p["n_final"] == nf]
        by_n[str(nf)] = {"n_classes": len(sel), **{k: float(np.median([p[k]["0.5"] for p in sel]))
                                                   for k in ("current", "train_mix25", "sqrts_pp_0.5", "sqrts_pp_unw_0.5",
                                                             "unw_0.6_s0.05", "unw_0.7_s0.1")}}
    out = {"source": "data/pp_support_reads.json (pp_support.py pull @841c5e29), envelope_design.py",
           "grid": "log10 sqrt(s_hat) 0.1 dex x ln|M|^2 2 nat, densities normalised per sample",
           "n_classes": len(per), "support": supp, "summary": summ, "median_gap_at_f0.5_by_n_final": by_n, "per_class": per}
    json.dump(out, open(os.path.join(DATA, "envelope_design.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, {f: (round(s["pooled"], 4), round(s["median"], 4), s["n_classes_over_10pct"]) for f, s in v.items()})
    print(json.dumps(by_n, indent=0)); print(supp)


if __name__ == "__main__":
    main()
