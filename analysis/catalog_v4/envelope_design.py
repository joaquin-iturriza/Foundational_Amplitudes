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
import json, os, sys
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


if __name__ == "__main__" and sys.argv[1:2] not in (["sampler"], ["own"]):
    main()


# ---------------------------------------------------------------------------------------------------------------
# sampler mode (step 49 redone, coordinator 2026-11-04 relaying Joaquin's "I'm not asking you to recreate the pp
# sampling"): catalog-sampler-only candidates, scored on support and on training events per bin inside the pp range.
#     python3 analysis/catalog_v4/envelope_design.py sampler   -> data/envelope_sampler.json
# Counts are expected training events per 2D bin (0.1 dex x 2 nat) at N_TRAIN events of the class (1e4: the short
# arms' in-memory load; x10 for the 1e5 pool). pp range = the bins holding pp events; "outside support" = pp events in
# bins whose expected catalog count at the 1e5 pool is below 0.5 event.
# Dose-response reference: the share arms' training mixture (1-s) catalog + s default-sampled (the pp sample stands in),
# s = 0, 0.25, 0.5, 1, on the same metric, beside their measured R on PP4 trained (tab:v4_bound_shares).
# Candidates (each a 2D density built from the class's catalog and pp occupancies, no MadEvent component):
#   current          catalog_v4 as generated
#   sq_w{w}          sqrt(s_hat): w x pp's sqrt(s_hat) marginal widened by a 0.1-dex Gaussian (the class's pp pool),
#                    1-w log-uniform on the class's window to 6 TeV; ln|M|^2 at each sqrt(s_hat) the catalog's own
#                    conditional (nearest filled row)
#   sq_w{w}_flat{g}  as sq_w{w}, with the ln|M|^2 conditional per sqrt(s_hat) row (1-g) catalog's + g flat in ln|M|^2
#                    over [pp min - 2 nat, pp max + 4 nat] of that row (the class's range where the row has no pp event):
#                    a flat-in-log|M|^2 part selected per sqrt(s_hat) slice; above the catalog's current max it assumes
#                    candidates that reach there (the initial-state collinear branch): an upper bound there
N_TRAIN = 1e4
NMIN = (1, 3, 10, 30)
SHARES = {0.0: 1.0, 0.25: 0.650, 0.5: 0.549, 1.0: 0.388}   # tab:v4_bound_shares, PP4 trained, MSE R


def log_uniform_rows(lo, hi):
    e = np.array(SQ2)
    a, b = np.log10(max(lo, 1.0)), np.log10(hi)
    w = np.clip(np.minimum(e[1:], b) - np.maximum(e[:-1], a), 0, None)
    return w / w.sum()


def flat_rows(pp, ncols):
    """per row the uniform ln|M|^2 density over [pp min - 1 bin, pp max + 2 bins] of the row (class range if empty)"""
    out = np.zeros_like(pp)
    cols = np.where(pp.sum(0) > 0)[0]
    glo, ghi = cols.min(), cols.max()
    for i in range(pp.shape[0]):
        cc = np.where(pp[i] > 0)[0]
        lo, hi = (cc.min(), cc.max()) if len(cc) else (glo, ghi)
        lo, hi = max(lo - 1, 0), min(hi + 2, ncols - 1)
        out[i, lo:hi + 1] = 1.0 / (hi - lo + 1)
    return out


def metrics(dens, pp, n_cat=1e5):
    m = pp > 0
    cnt = dens * N_TRAIN
    o = {"outside_support": float(pp[m & (dens * n_cat < 0.5)].sum())}
    for k in NMIN:
        o[f"below_{k}"] = float(pp[m & (cnt < k)].sum())
    o["pp_weighted_median_count"] = float(np.interp(0.5, np.cumsum(pp[m][np.argsort(cnt[m])]), np.sort(cnt[m])))
    o["catalog_mass_in_pp_range"] = float(dens[m].sum())
    o["n_pp_bins"] = int(m.sum())
    return o


def sampler_one(r):
    pp = np.array(r["default"]["hist2d"], float); c = np.array(r["catalog"]["hist2d"], float)
    if pp.sum() == 0 or c.sum() == 0:
        return None
    pp /= pp.sum(); c /= c.sum()
    cond = catalog_conditional(c)
    ppm = widen_rows(pp.sum(1, keepdims=True))[:, 0]
    lo, hi = r["window"] if r.get("window") else (r["catalog"]["sqrts_min"], 6000.0)
    lu = log_uniform_rows(lo, hi)
    flat = flat_rows(pp, pp.shape[1])
    out = {"current": metrics(c, pp)}
    for s in SHARES:
        out[f"share_{s}"] = metrics((1 - s) * c + s * pp, pp)
    for w in (0.5, 0.7):
        marg = w * ppm + (1 - w) * lu
        out[f"sq_w{w}"] = metrics(marg[:, None] * cond, pp)
        for g in (0.35, 0.5):
            out[f"sq_w{w}_flat{g}"] = metrics(marg[:, None] * ((1 - g) * cond + g * flat), pp)
    out["n_final"] = len(r["default"]["pdg"]) - 2
    out["n_pp"] = r["default"]["n"]
    return out


def sampler():
    global SQ2
    d = json.load(open(os.path.join(DATA, "pp_support_reads.json")))
    SQ2 = d["hist2d_edges"][0]
    per = {r["probe"]: o for r in d["rows"] if "error" not in r for o in [sampler_one(r)] if o is not None}
    cands = [k for k in next(iter(per.values())) if isinstance(next(iter(per.values()))[k], dict)]
    n = np.array([p["n_pp"] for p in per.values()], float)
    summ = {}
    for k in cands:
        summ[k] = {}
        for m in next(iter(per.values()))[k]:
            x = np.array([p[k][m] for p in per.values()], float)
            summ[k][m] = {"pooled": float((x * n).sum() / n.sum()), "median": float(np.median(x)),
                          "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9)), "max": float(x.max()),
                          "n_classes_over_1pct": int((x > 0.01).sum())}
    by_n = {str(nf): {k: {"below_10": float(np.median([p[k]["below_10"] for p in per.values() if p["n_final"] == nf])),
                          "outside": float(np.median([p[k]["outside_support"] for p in per.values() if p["n_final"] == nf]))}
                      for k in cands} for nf in sorted({p["n_final"] for p in per.values()})}
    out = {"source": "data/pp_support_reads.json (pp_support.py pull @841c5e29), envelope_design.py sampler",
           "n_train": N_TRAIN, "share_R_pp4_trained": SHARES, "n_classes": len(per), "summary": summ,
           "median_by_n_final": by_n, "per_class": per}
    json.dump(out, open(os.path.join(DATA, "envelope_sampler.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, {m: (round(s["pooled"], 4), round(s["median"], 4)) for m, s in v.items() if m != "n_pp_bins"})


if __name__ == "__main__" and sys.argv[1:2] == ["sampler"]:
    sampler()


# ---------------------------------------------------------------------------------------------------------------
# own mode (step 49, the main session's correction of 2026-11-04): the criterion of main() (gap at f, outside support)
# on the catalog's OWN partonic sampler, no events taken from the MadEvent pools (the default share is added on top
# in training, separately). Weights sum to 1:
#     python3 analysis/catalog_v4/envelope_design.py own   -> data/envelope_own.json
#   own_u{u}_s{S}_keep  u: our own |M|^2-unweighted partonic events (accept-reject / VEGAS on our candidates, the
#                       partonic cuts) at sqrt(s_hat) drawn from the class's pp sqrt(s_hat) marginal widened by a
#                       Gaussian of S dex; its ln|M|^2 at each sqrt(s_hat) is stood in for by pp's conditional (equal up
#                       to the lab vs partonic cut, 0.07% of pp events, and the PDF's boost, which does not change
#                       |M|^2 at fixed sqrt(s_hat)); 1-u: the current catalog_v4 rows (subsampled, no IS branch)
#   own_u{u}_s{S}_ext   1-u: the current sampling with the flat-in-log|M|^2 part (g = 0.35 of it) spanning
#                       [pp min - 2 nat, pp max + 4 nat] per sqrt(s_hat) slice: what the initial-state collinear branch
#                       must reach (upper bound above the current candidates' max); sqrt(s_hat) log-uniform to 6 TeV
# The unweighted part is built from the same pp sample it is scored against, so the gap is optimistic in sparse bins;
# the gate rescores built pools against the pp pools.
def own_one(r):
    pp = np.array(r["default"]["hist2d"], float); c = np.array(r["catalog"]["hist2d"], float)
    if pp.sum() == 0 or c.sum() == 0:
        return None
    pp /= pp.sum(); c /= c.sum()
    filled = np.where(pp.sum(1) > 0)[0]
    ppc0 = pp / np.where(pp.sum(1, keepdims=True) > 0, pp.sum(1, keepdims=True), 1)
    for i in range(ppc0.shape[0]):
        if ppc0[i].sum() == 0:
            ppc0[i] = ppc0[filled[np.argmin(np.abs(filled - i))]]
    lo, hi = r["window"] if r.get("window") else (r["catalog"]["sqrts_min"], 6000.0)
    lu = log_uniform_rows(lo, hi)
    rest_ext = lu[:, None] * (0.65 * catalog_conditional(c) + 0.35 * flat_rows(pp, pp.shape[1]))
    cands = {"current": c}
    for S in (0.05, 0.1):
        m_s = widen_rows(pp.sum(1, keepdims=True), S / 0.1)[:, 0]
        comp = m_s[:, None] * ppc0; comp /= comp.sum()
        for u in (0.5, 0.6, 0.7):
            cands[f"own_u{u}_s{S}_keep"] = u * comp + (1 - u) * c
            cands[f"own_u{u}_s{S}_ext"] = u * comp + (1 - u) * rest_ext
    m = pp > 0
    out = {k: {**{f"gap_{f:.3g}": gap(v, pp, f) for f in F}, "outside_support": float(pp[m & (v * 1e5 < 0.5)].sum()),
               "mass_outside_pp_bins": float(v[~m].sum())} for k, v in cands.items()}
    out["n_final"] = len(r["default"]["pdg"]) - 2
    out["n_pp"] = r["default"]["n"]
    return out


def own():
    global SQ2
    d = json.load(open(os.path.join(DATA, "pp_support_reads.json")))
    SQ2 = d["hist2d_edges"][0]
    per = {r["probe"]: o for r in d["rows"] if "error" not in r for o in [own_one(r)] if o is not None}
    p0 = next(iter(per.values()))
    n = np.array([p["n_pp"] for p in per.values()], float)
    summ = {k: {m: {"pooled": float((x * n).sum() / n.sum()), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
                    "max": float(x.max()), "n_classes_over_1pct": int((x > 0.01).sum())}
                for m in p0[k] for x in [np.array([p[k][m] for p in per.values()], float)]}
            for k in p0 if isinstance(p0[k], dict)}
    json.dump({"source": "data/pp_support_reads.json (pp_support.py pull @841c5e29), envelope_design.py own",
               "n_classes": len(per), "summary": summ, "per_class": per},
              open(os.path.join(DATA, "envelope_own.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, {m: (round(s["pooled"], 4), round(s["median"], 4), s["n_classes_over_1pct"]) for m, s in v.items()
                  if m in ("gap_0.5", "gap_0.333", "outside_support", "mass_outside_pp_bins")})


if __name__ == "__main__" and sys.argv[1:2] == ["own"]:
    own()
