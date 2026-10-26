"""catalog_v4's training density against the 13 TeV pp events' (catalog_coverage, CPU only, no new label).

The coverage check before catalog_v4 was generated (analysis/catalog_v3/coverage_check.py) measured containment: which
share of a class's pp events lies inside the new cuts and window. It did not measure density: how many of catalog_v4's
training events fall where the pp events sit. Its sqrt(s_hat) "effective sample size" was (sum w)^2 / (n sum w^2) over
the pp events with w = p_pp / q, which is 1 for any q proportional to p_pp on pp's support, whatever the magnitude of
q there: it measured the shape of the ratio, not how thin q is. This script measures the density.

Per hadronic tree class (the 1396 classes with a 13 TeV pp pool, recipes/catalog_v3_pp_test.yaml), four samples, every
event in the partonic rest frame (the pp events boosted along z, as pp_compare.support_masks):
  v4     catalog_v4's training pool of the class (its own, or the catalog_v4 class holding its tree after the merges and
         splits of data/class_changes.json), the first N_POOL rows (a pool is a concatenation of chunks, each a full
         mixture shuffled, so the first rows are the mixture)
  v3     catalog_v3's training pool of the class, the first N_POOL rows (the control: the pp-trained arm's gain over the
         window-trained arm is large, so a density measure that means anything must follow it there)
  pp     the class's 13 TeV pp training pool (MadEvent's unweighted events, 1e4): pp's density
  ppval  the first 2000 rows of its pp validation pool: the events every pp read scored
Three variables, binned jointly:
  s      log10 sqrt(s_hat) / GeV
  u      log(1/(1 - |cos theta|)) of the most forward final particle (coverage_check.u_forward)
  y      log10 of the smallest 2|p_i.p_j| / s_hat over every pair of external legs but the two beams: a final pair
         collinear, or a final particle collinear to a beam (small |t|), whichever is nearest
and one alone:
  pt     log10 of the smallest final-particle pT / GeV
    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/density_check.py run \
        | gzip > analysis/catalog_v4/data/density_hists.json.gz                     (login node, ~1 min)
    python3 analysis/catalog_v4/density_check.py      -> data/density_check.json, figs/density_check.{png,pdf}
(the site part prints its command line before the JSON; load() skips it). The last line printed is the summary JSON.
The summary's method and its own choices are in summary()'s docstring.
"""
import argparse, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

N_POOL = 20000
# fine bins written by `run`; the summary coarsens them
S_EDGES = np.round(np.arange(1.0, 3.80001, 0.1), 3)          # 10 GeV .. 6.3 TeV, 28 bins
U_EDGES = np.arange(0.0, 10.0001, 1.0)                        # |eta| < 5 is u = 9.31, 10 bins
Y_EDGES = np.arange(-8.0, 1.0001, 1.0)                        # 9 bins
PT_EDGES = np.round(np.arange(0.0, 3.80001, 0.1), 3)          # 1 GeV .. 6.3 TeV, 38 bins
SHAPE = (len(S_EDGES) - 1, len(U_EDGES) - 1, len(Y_EDGES) - 1)


def pools():
    """pp class -> (catalog_v4 training class or None, catalog_v3 training class or None)"""
    names = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]]
    tr4 = {e["name"]: e for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    tr3 = {e["name"]: e for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]}
    cc = json.load(open(os.path.join(HERE, "data", "class_changes.json")))
    holder = {}                                       # catalog_v4 member -> the class whose pool holds it
    for g in cc["merges"]:
        for x in g["members"]:
            holder.setdefault(x, g["v4_class"])
    for sp in cc["splits"]:
        for k, v in sp["v4_classes"].items():
            for x in v["members"]:
                holder.setdefault(x, k)
    out = {}
    for n in names:
        x = n[len("pp13_"):]
        c4 = "v4_" + x
        c4 = c4 if c4 in tr4 else holder.get(c4)
        out[n] = {"v4": (c4, tr4[c4]["sqrts"]) if c4 in tr4 else None,
                  "v3": ("v3_" + x, tr3["v3_" + x]["sqrts"]) if "v3_" + x in tr3 else None}
    return out


def features(a, k, boost):
    """rows -> (s, u, y, pt) per event"""
    P = np.array(a[:, :4 * k], dtype=np.float64).reshape(-1, k, 4)
    tot = P[:, 0] + P[:, 1]
    if boost:
        bz = tot[:, 3] / tot[:, 0]
        g = 1.0 / np.sqrt(1.0 - bz ** 2)
        E, pz = P[..., 0].copy(), P[..., 3].copy()
        P[..., 0] = g[:, None] * (E - bz[:, None] * pz)
        P[..., 3] = g[:, None] * (pz - bz[:, None] * E)
    sh = np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 1e-300, None)
    p = P[:, 2:, 1:]
    c = np.abs(p[..., 2]) / np.clip(np.linalg.norm(p, axis=-1), 1e-300, None)
    u = np.log(1.0 / np.clip(1.0 - c.max(1), 1e-300, None))
    dot = lambda x, z: x[..., 0] * z[..., 0] - (x[..., 1:] * z[..., 1:]).sum(-1)
    ys = [2.0 * np.abs(dot(P[:, i], P[:, j])) / sh for i in range(k) for j in range(i + 1, k) if (i, j) != (0, 1)]
    y = np.log10(np.clip(np.min(ys, axis=0), 1e-300, None))
    pt = np.log10(np.clip(np.hypot(P[:, 2:, 1], P[:, 2:, 2]).min(1), 1e-300, None))
    return 0.5 * np.log10(sh), u, y, pt


def bin_of(x, e):
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)


def hists(a, k, boost):
    s, u, y, pt = features(a, k, boost)
    j = np.ravel_multi_index((bin_of(s, S_EDGES), bin_of(u, U_EDGES), bin_of(y, Y_EDGES)), SHAPE)
    jc = np.bincount(j, minlength=int(np.prod(SHAPE)))
    nz = np.nonzero(jc)[0]
    return {"n": int(len(s)), "joint": [[int(i), int(jc[i])] for i in nz],
            "pt": [int(x) for x in np.bincount(bin_of(pt, PT_EDGES), minlength=len(PT_EDGES) - 1)]}


def one(task):
    n, P = task
    import mg5_pipeline_final as mg
    D, C = os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")
    row = {"name": n}
    try:
        k = len(mg.PROCESSES[n]["pdg_ids"])
        row["nfinal"] = k - 2
        row["pp"] = hists(np.load(os.path.join(C, f"{n}_13000-13000GeV_train_lhe_amplitudes.npy"), mmap_mode="r"), k, True)
        row["ppval"] = hists(np.load(os.path.join(D, f"{n}_13000-13000GeV_val_lhe_amplitudes.npy"),
                                     mmap_mode="r")[:2000], k, True)
        for cat in ("v4", "v3"):
            if P[cat] is None:
                continue
            c, (lo, hi) = P[cat]
            a = np.load(os.path.join(C, f"{c}_{int(lo)}-{int(hi)}GeV_train_smix_amplitudes.npy"), mmap_mode="r")
            kc = len(mg.PROCESSES[c]["pdg_ids"])
            assert kc == k, (c, kc, k)
            row[cat] = dict(hists(a[:N_POOL], k, False), pool=c, sqrts=[float(lo), float(hi)], n_pool_rows=int(a.shape[0]))
    except Exception as e:                   # a class that fails is listed with its error, never dropped silently
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run(workers, only=None):
    from multiprocessing import Pool
    P = pools()
    if only:
        P = {n: P[n] for n in sorted(P)[:only]}
    rows = {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, sorted(P.items()), chunksize=4)):
            rows[r["name"]] = r
    err = {n: r["error"] for n, r in rows.items() if "error" in r}
    json.dump({"errors": err, "edges": {"s": S_EDGES.tolist(), "u": U_EDGES.tolist(), "y": Y_EDGES.tolist(), "pt": PT_EDGES.tolist()},
               "n_pool": N_POOL, "rows": rows}, sys.stdout)

# ---------------------------------------------------------------------------------------------------------------------
# the summary, on the laptop
COARSE = (2, 2, 1)          # fine bins merged per coarse bin along s, u, y: 0.2 dex in sqrt(s_hat), 2 in u, 1 dex in y
GAP = 1.0 / 3               # a bin is thin when the sample's density there is below a third of pp's
FS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
PARTONS = {1, 2, 3, 4, 5, 21}


def load():
    import gzip
    t = gzip.open(os.path.join(HERE, "data", "density_hists.json.gz"), "rt").read()
    return json.loads(t[t.index("{"):])           # site run prints its command line first


def dense(h, coarse=COARSE):
    """sparse fine joint counts -> coarse dense array"""
    a = np.zeros(int(np.prod(SHAPE)))
    for i, c in h["joint"]:
        a[i] = c
    a = a.reshape(SHAPE)
    for ax, k in enumerate(coarse):
        if k > 1:
            n = a.shape[ax] // k * k
            a = np.take(a, range(n), axis=ax)
            a = a.reshape(a.shape[:ax] + (n // k, k) + a.shape[ax + 1:]).sum(ax + 1)
    return a


def metrics(q, p, v, gap_=None):
    """q the sample's density, p pp's (both normalised over the same bins), v the scored pp events' counts:
    gap   the share of the scored pp events in bins where q < GAP * p
    ratio the median over the scored pp events of q / p in their bin
    kl    KL(p || q) in nats (q floored at half an event of the sample, see summary)"""
    m = p > 0
    r = np.where(m, q / np.where(m, p, 1.0), np.inf)
    lq = np.log(np.where(q[m] > 0, q[m], 1e-300))
    vv = v.ravel(); rr = r.ravel()
    gap = float(vv[rr < (GAP if gap_ is None else gap_)].sum() / vv.sum())
    o = np.argsort(rr[vv > 0]); w = np.cumsum(vv[vv > 0][o]); med = float(rr[vv > 0][o][np.searchsorted(w, w[-1] / 2)])
    kl = float((p[m] * (np.log(p[m]) - lq)).sum())
    return {"gap": gap, "ratio": med, "kl": kl}


def norm_sample(c, n, floor=True):
    """counts -> density per bin; floor: an empty bin gets half an event (the sample's resolution), renormalised"""
    q = c.astype(float)
    if floor:
        q = q + 0.5 * (q == 0)
    return q / q.sum()


def marg(a, keep):
    return a.sum(tuple(i for i in range(a.ndim) if i not in keep))


def beams_kind(pdg):
    b = sorted(abs(int(x)) for x in pdg[:2])
    if b == [21, 21]:
        return "gg"
    if 21 in b:
        return "qg"
    return "qqbar" if int(pdg[0]) * int(pdg[1]) < 0 else "qq"


def mse_rows():
    """per class: catalog_v4 (Muon, geometric mean of seeds 42 and 43), catalog_v3 (Muon, same), the pp-trained arm and
    catalog_v3 (AdamW, one seed each): MSE of log|M|^2 on the 2000 scored pp events (compare_pp_trained.rows)"""
    import compare_pp_trained as cpt
    R, p, _ = cpt.rows()
    k = "mse"
    out = {}
    for n in p["_role"]:
        if all(n in R[k][r] for r in cpt.V4 + cpt.V3M) and all(n in R[k][a] for a in ("dimensionless_pp", "dimensionless")):
            out[n] = {"v4": cpt.gm(R, k, cpt.V4, n), "v3m": cpt.gm(R, k, cpt.V3M, n),
                      "pp": R[k]["dimensionless_pp"][n], "v3a": R[k]["dimensionless"][n]}
    return out, p["_role"], p["_b"]


def spear(x, y, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i])[0])
    return {"rho": rho, "ci95": [float(q) for q in np.nanquantile(bs, [0.025, 0.975])], "n": int(len(x))}


def summary():
    """Per class, on coarse joint bins (0.2 dex in sqrt(s_hat), 2 in u, 1 dex in y: 14 x 5 x 9), pp's density p from
    its training and scored events together (12000), each catalog's q from its N_POOL training events with an empty bin
    floored at half an event; the metrics of metrics() for catalog_v4 and catalog_v3, jointly and per variable alone.
    The check's own choices: the coarse bins, the third (GAP), the half-event floor, N_POOL; the gap share and the
    median ratio do not use the floor (an empty bin is thin whatever the floor), the KL does.
    Against the reads (compare_pp_trained, no b; the pp pools carry catalog_v3's 4.7 GeV b, catalog_v4's b is
    massless): Spearman rank correlation, with a 95% bootstrap interval over classes, of each density measure with
      direct      log(catalog_v4 MSE / pp-trained MSE) on the classes trained in both catalogs
      shortfall   log(catalog_v4 / catalog_v3 Muon) - log(pp-trained / catalog_v3 AdamW): how much less catalog_v4 gains
                  over its catalog_v3 reference than pp training over its own (the optimizer cancels if its factor is
                  the same on every class)
      control     catalog_v3's own density gap against log(pp-trained / catalog_v3 AdamW) on catalog_v3's training classes:
                  whether the density measure follows the error where pp training is known to buy a factor seven
    and within each multiplicity.
    The mixture (design): q_f = (1 - f) q_v4 + f c for a component c, per class, the metrics as f grows:
      pp      c = p: MadEvent's pp events themselves (the bound)
      s_own   catalog_v4's events with sqrt(s_hat) redrawn from the class's own pp marginal: c = q_v4 p(s) / q_v4(s)
      s_tmpl  the same with the marginal from a template shared by the classes of one beam pair (gg, qg, q qbar, qq) and
              multiplicity, in log10(sqrt(s_hat) / the class's catalog_v4 threshold) (what a class with no pp pool gets)
      su_own  sqrt(s_hat) and the forward angle redrawn from the class's pp (s, u) marginal: c = q_v4 p(s,u) / q_v4(s,u)
    A component's mass where catalog_v4 has no event at that s (or s, u) is lost (counted, `lost`): redrawing cannot put
    events where the mixture's conditional is empty. Breadth: q_f >= (1 - f) q_v4 in every bin, so no region of
    catalog_v4 thins by more than 1 - f; the share of catalog_v4's events in bins pp never reaches is reported."""
    import plot_style as ps
    import mg5_pipeline_final as mg
    D = load()
    rows = D["rows"]
    assert not D.get("errors"), D["errors"]
    M, role, isb = mse_rows()
    out = {"n_classes": len(rows), "coarse": COARSE, "gap": GAP}
    per = {}
    tmpl = {}
    for n, r in rows.items():
        P = dense(r["pp"]) + dense(r["ppval"])
        p = P / P.sum()
        v = dense(r["ppval"])
        e = {"nfinal": r["nfinal"], "beams": beams_kind(mg.PROCESSES[n]["pdg_ids"]), "b": bool(isb[n]), "role": role[n]}
        for cat in ("v4", "v3"):
            if cat not in r:
                continue
            c = dense(r[cat])
            q = norm_sample(c, r[cat]["n"])
            e[cat] = metrics(c / c.sum(), p, v)
            e[cat]["kl"] = metrics(q, p, v)["kl"]
            for nm, keep in (("s", (0,)), ("u", (1,)), ("y", (2,))):
                cm, pm, vm = marg(c, keep), marg(p, keep), marg(v, keep)
                e[cat][nm] = metrics(cm / cm.sum(), pm, vm)
            # pT alone, on its own fine bins merged by two
            cp = np.add.reduceat(np.asarray(r[cat]["pt"], float), np.arange(0, len(r[cat]["pt"]), 2))
            pp_ = np.add.reduceat(np.asarray(r["pp"]["pt"], float) + np.asarray(r["ppval"]["pt"], float),
                                  np.arange(0, len(r["pp"]["pt"]), 2))
            vp = np.add.reduceat(np.asarray(r["ppval"]["pt"], float), np.arange(0, len(r["pp"]["pt"]), 2))
            e[cat]["pt"] = metrics(cp / cp.sum(), pp_ / pp_.sum(), vp)
            e[cat]["outside_pp"] = float(c[P == 0].sum() / c.sum())
            if cat == "v3":
                e["_c3"] = c
            if cat == "v4":
                e["_q"], e["_p"], e["_v"], e["_c"] = q, p, v, c
                e["lo"] = float(r["v4"]["sqrts"][0])
                if not e["b"]:
                    # the template's input: pp's fine sqrt(s_hat) marginal shifted to the class's threshold
                    fs = dense(r["pp"], (1, 1, 1)).sum((1, 2)) + dense(r["ppval"], (1, 1, 1)).sum((1, 2))
                    i0 = int(np.clip(np.floor((np.log10(e["lo"]) - S_EDGES[0]) / 0.1 + 1e-9), 0, len(fs) - 1))
                    sh = np.zeros(len(fs)); sh[:len(fs) - i0] = fs[i0:]
                    e["_i0"] = i0
                    tmpl.setdefault((e["beams"], e["nfinal"]), []).append(sh / sh.sum())
        per[n] = e
    T = {k: np.mean(v, 0) for k, v in tmpl.items()}

    # --- correlations with the reads
    tr = [n for n, e in per.items() if not e["b"] and e["role"]["v3"] == "train" and e["role"]["v4"] == "train"
          and n in M and "v4" in e]
    lg = lambda n, a, b: np.log(M[n][a] / M[n][b])
    Y = {"direct": {n: lg(n, "v4", "pp") for n in tr},
         "shortfall": {n: lg(n, "v4", "v3m") - lg(n, "pp", "v3a") for n in tr}}
    out["n_train_both_no_b"] = len(tr)
    C = {}
    for y, yv in Y.items():
        for x in ("gap", "kl", "ratio"):
            sgn = -1.0 if x == "ratio" else 1.0          # a lower ratio is a larger gap: signed so + means "follows"
            C[f"{y}.{x}"] = spear([sgn * per[n]["v4"][x] for n in tr], [yv[n] for n in tr])
            for nf in (2, 3, 4):
                sel = [n for n in tr if per[n]["nfinal"] == nf]
                if len(sel) > 20:
                    C[f"{y}.{x}.2to{nf}"] = spear([sgn * per[n]["v4"][x] for n in sel], [yv[n] for n in sel])
        for var in ("s", "u", "y", "pt"):
            C[f"{y}.gap_{var}"] = spear([per[n]["v4"][var]["gap"] for n in tr], [yv[n] for n in tr])
            C[f"{y}.kl_{var}"] = spear([per[n]["v4"][var]["kl"] for n in tr], [yv[n] for n in tr])
    ctl = [n for n, e in per.items() if not e["b"] and e["role"]["v3"] == "train" and n in M and "v3" in e]
    for x in ("gap", "kl"):
        C[f"control.{x}"] = spear([per[n]["v3"][x] for n in ctl], [-np.log(M[n]["pp"] / M[n]["v3a"]) for n in ctl])
        C[f"v4_minus_v3.{x}"] = spear([per[n]["v4"][x] - per[n]["v3"][x] for n in tr], [lg(n, "v4", "v3m") for n in tr])
    out["corr"] = C
    out["n_control"] = len(ctl)

    # --- the densities, medians over the trained-in-both classes
    md = lambda xs: float(np.median([x for x in xs if x is not None and np.isfinite(x)]))
    out["density"] = {}
    for cat, sel in (("v4", tr), ("v3", tr), ("v3_all_train", ctl)):
        cc = "v3" if cat.startswith("v3") else cat
        out["density"][cat] = {"n": len(sel), **{x: md(per[n][cc][x] for n in sel) for x in ("gap", "ratio", "kl", "outside_pp")},
                               **{f"{x}_{var}": md(per[n][cc][var][x] for n in sel) for var in ("s", "u", "y", "pt")
                                  for x in ("gap", "ratio")}}
        for nf in (2, 3, 4):
            s2 = [n for n in sel if per[n]["nfinal"] == nf]
            out["density"][cat][f"2to{nf}"] = {"n": len(s2), "gap": md(per[n][cc]["gap"] for n in s2),
                                               "ratio": md(per[n][cc]["ratio"] for n in s2)}
    # the error by tercile of catalog_v4's density gap
    g = np.array([per[n]["v4"]["gap"] for n in tr]); d = np.array([Y["direct"][n] for n in tr])
    sh_ = np.array([Y["shortfall"][n] for n in tr])
    ed = np.quantile(g, [0, 1 / 3, 2 / 3, 1])
    out["terciles"] = [{"gap_range": [float(ed[i]), float(ed[i + 1])],
                        "direct": float(np.exp(np.median(d[(g >= ed[i]) & (g <= ed[i + 1])]))),
                        "shortfall": float(np.exp(np.median(sh_[(g >= ed[i]) & (g <= ed[i + 1])])))} for i in range(3)]

    # --- the mixture
    def comp(e, kind):
        q, p, c = e["_q"], e["_p"], e["_c"] / e["_c"].sum()
        if kind == "pp":
            return p, 0.0
        if kind in ("s_own", "s_tmpl"):
            ps_ = marg(p, (0,))
            if kind == "s_tmpl":
                t = T[(e["beams"], e["nfinal"])]
                fine = np.zeros(SHAPE[0]); fine[e["_i0"]:] = t[:SHAPE[0] - e["_i0"]]
                ps_ = fine.reshape(-1, COARSE[0]).sum(1) if SHAPE[0] % COARSE[0] == 0 else None
                ps_ = ps_ / ps_.sum()
            qs = marg(c, (0,))
            w = np.where(qs > 0, ps_ / np.where(qs > 0, qs, 1), 0.0)
            x = c * w[:, None, None]
            return x / max(x.sum(), 1e-300), float(ps_[qs == 0].sum())
        if kind == "su_own":
            pu, qu = marg(p, (0, 1)), marg(c, (0, 1))
            w = np.where(qu > 0, pu / np.where(qu > 0, qu, 1), 0.0)
            x = c * w[:, :, None]
            return x / max(x.sum(), 1e-300), float(pu[qu == 0].sum())
    KINDS = ("pp", "s_own", "s_tmpl", "su_own")
    mix = {k: {f: {"gap": [], "ratio": [], "kl": []} for f in FS} for k in KINDS}
    lost = {k: [] for k in KINDS}
    for n in tr:
        e = per[n]
        for k in KINDS:
            c, l = comp(e, k)
            lost[k].append(l)
            for f in FS:
                cf = (1 - f) * (e["_c"] / e["_c"].sum()) + f * c
                qf = (1 - f) * e["_q"] + f * c
                m = metrics(cf, e["_p"], e["_v"])
                m["kl"] = metrics(qf, e["_p"], e["_v"])["kl"]
                for x in m:
                    mix[k][f][x].append(m[x])
    out["mixture"] = {k: {str(f): {x: md(v) for x, v in mix[k][f].items()} for f in FS} for k in KINDS}
    out["mixture_lost"] = {k: md(v) for k, v in lost.items()}
    out["mixture_q90_gap"] = {k: {str(f): float(np.quantile(mix[k][f]["gap"], 0.9)) for f in FS} for k in KINDS}
    # the measures' dependence on the check's own choices: the coarse bins and the third
    rob = {}
    yd = [Y["direct"][n] for n in tr]
    for co in ((1, 1, 1), (2, 2, 1), (4, 5, 3), (2, 10, 9), (28, 2, 1)):
        for gp in (0.2, 1.0 / 3, 0.5):
            g_, k_ = [], []
            for n in tr:
                r = rows[n]
                P_ = dense(r["pp"], co) + dense(r["ppval"], co)
                c_ = dense(r["v4"], co)
                m_ = metrics(c_ / c_.sum(), P_ / P_.sum(), dense(r["ppval"], co), gp)
                g_.append(m_["gap"])
                k_.append(metrics(norm_sample(c_, 0), P_ / P_.sum(), dense(r["ppval"], co), gp)["kl"])
            rob[f"{co[0]}{co[1]}{co[2]}_{gp:.2f}"] = {"gap": md(g_), "rho_gap": spear(g_, yd, 300)["rho"],
                                                     "kl": md(k_), "rho_kl": spear(k_, yd, 300)["rho"]}
    out["robustness"] = rob
    out["template_groups"] = {f"{a}_2to{b}": len(v) for (a, b), v in tmpl.items()}
    json.dump(out, open(os.path.join(HERE, "data", "density_check.json"), "w"), indent=1)
    plot(per, tr, Y, mix, out)
    print(json.dumps(out))


def plot(per, tr, Y, mix, out):
    """(a), (b) the median over the trained-in-both classes of each catalog's density over pp's, per bin of sqrt(s_hat)
    and of the forward angle (bins holding at least 1% of a class's pp events, at least 10 classes), with pp's mean
    marginal scaled to its maximum; (c) per class the share of pp events in thin bins against catalog_v4's MSE over the
    pp-trained arm's; (d) the mixture: the median share of pp events in thin bins against f, per component"""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a, b), (c, d) = axs
    for ax, keep, edges, lab in ((a, (0,), S_EDGES[::COARSE[0]], r"$\log_{10}\sqrt{\hat s}$ / GeV"),
                                 (b, (1,), U_EDGES[::COARSE[1]], r"$\log\,1/(1-|\cos\theta|)$, most forward")):
        x = 0.5 * (edges[1:] + edges[:-1])
        wt = np.array([marg(per[n]["_p"], keep) for n in tr])
        for key, col, mk, nm in (("_c3", ps.C.blue, "o", "catalog v3 / pp"), ("_c", ps.C.vermillion, "s", "catalog v4 / pp")):
            cm = np.array([marg(per[n][key], keep) / per[n][key].sum() for n in tr])
            rat = np.where((wt > 0) & (cm > 0), cm / np.where(wt > 0, wt, 1), np.nan)   # no event: not drawn
            ok = wt > 0.01
            med = [np.nanmedian(rat[ok[:, i], i]) if ok[:, i].sum() >= 10 else np.nan for i in range(len(x))]
            ax.plot(x, med, mk + "-", color=col, label=nm)
        m = wt.mean(0)
        ax.plot(x, m / m.max(), "--", color=ps.C.grey, label="pp events, scaled")
        ax.axhline(GAP, color=ps.C.green, ls=":", label="a third")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("density ratio")
    ps.legend(a, "upper left", ncol=2)
    for nf, col, mk in ((2, ps.C.blue, "o"), (3, ps.C.orange, "^"), (4, ps.C.vermillion, "s")):
        sel = [n for n in tr if per[n]["nfinal"] == nf]
        c.plot([per[n]["v4"]["gap"] for n in sel], [np.exp(Y["direct"][n]) for n in sel], mk, color=col, mfc="none",
               alpha=0.6, label=rf"$2\to{nf}$")
    r_ = out["corr"]["direct.gap"]
    c.plot([], [], " ", label=rf"Spearman ${r_['rho']:.2f}$")
    c.set_yscale("log")
    c.set_xlabel("pp events in thin bins, catalog v4")
    c.set_ylabel("MSE, catalog v4 / pp-trained")
    ps.legend(c, "upper left", ncol=2)
    for k, col, mk, nm in (("pp", ps.C.green, "D", "MadEvent pp events"),
                           ("su_own", ps.C.purple, "v", r"$\sqrt{\hat s}$ and angle as pp"),
                           ("s_own", ps.C.orange, "^", r"$\sqrt{\hat s}$ as pp"),
                           ("s_tmpl", ps.C.sky, "o", r"$\sqrt{\hat s}$ template")):
        d.plot(FS, [np.median(mix[k][f]["gap"]) for f in FS], mk + "-", color=col, label=nm)
    d.set_xlabel(r"fraction $f$ of the added component")
    d.set_ylabel("pp events in thin bins")
    ps.legend(d, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "density_check"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="summary", choices=("run", "summary"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", type=int, default=None, help="the first N classes only (a test)")
    a = ap.parse_args()
    if a.mode == "run":
        run(a.workers, a.only)
    else:
        summary()
