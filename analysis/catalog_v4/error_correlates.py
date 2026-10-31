#!/usr/bin/env python3
"""Plan entry error_correlates (Joaquin 2026-10-31T20:42): beyond the largest |eta| and the collinearity, which event
and process quantities correlate most strongly with the per-event error and with the het sigma of catalog_v4's short
networks? CPU only, on the saved step-15/20 zero-shot reads; no new run.

    site run --item error_correlates ccin2p3 FA -- python analysis/catalog_v4/error_correlates.py probe
    site run --item error_correlates ccin2p3 FA -- python analysis/catalog_v4/error_correlates.py extract --part I --nparts N
    site run ... aggregate --run <arm> > A_<arm>;  ... gbt --run <arm> --target T > G_<arm>_T;  ... excluded > X
                                                                       (login node, CPU; each under site run's limit)
    python3 analysis/catalog_v4/error_correlates.py combine X A_* G_*   -> data/error_correlates_reads.json.gz
    python3 analysis/catalog_v4/error_correlates.py summary   -> data/error_correlates.json, figs/error_correlates.{png,pdf}

The reads: per probe <probe>.npz of tools/zero_shot_parent.py --save-dir in $SCRATCH/zz_v4_short_<arm>/<axis>/, with
per event t (truth log|M|^2), q (the prediction, log|M|^2; het: mu) and, het only, s = sd_c * sigma (log|M|^2), the
pool and the scored events' pool rows. Each run's read draws its own events of the probe's pool, so the features are
computed per run (once per distinct pool and rows); a read saved before the pool_row field (the MSE control's PP4 and
catalog-sampled held-out reads) has no momenta and is listed as excluded. The twelve runs are the het-loss and MSE twins of mixture_ab.py (control,
default50, default25; seeds 42 and 43), one job folder per arm. A probe is kept as het_uncertainty.read_probe keeps it
(finite t, q, s; physics status ok or allowed; the MSE control's own held-out reads, made before the physics check, as
mixture_ab.extract's allow_unchecked); the held-out axes H1..H7c read their v4_* probes only. Read groups: PP4 and EE4
(default-sampled test events of trained and held-out classes), held (the catalog-sampled held-out axes H1..H7c),
held_default (HD4 + HD5, the default-sampled held-out pools), all.

Per event, from the pool row's momenta (columns 4k momenta E, px, py, pz; k ids; 1 amplitude; beams first), the
partonic frame being the beams' c.o.m. (a boost along z of the lab frame):
    eta_max      largest final |eta|, lab frame                                 (het_uncertainty's eta)
    y_min        log10 min over pairs but the beam pair of 2|p_i.p_j|/s_hat    (het_uncertainty's y)
    y_ff, y_if   the same over final-final pairs, over beam-final pairs
    u_fwd        log10 1/(1 - max final |cos theta|), partonic frame
    pt_min, pt_max   log10 min, max final pT / sqrt(s_hat)
    e_min        log10 min final energy / sqrt(s_hat), partonic frame
    dr_min       min Delta R (eta, phi) between finals (nan with one final)
    sqrt_shat    log10 sqrt(s_hat) / GeV
    thr          log10 (sqrt(s_hat) - sum of final masses) / sqrt(s_hat)   (0 for massless finals; distance to threshold)
    y_boost      |rapidity| of the beams' system in the lab (0 on e+e-)
    res_min      log10 min over pairs and triples of finals and over V in {Z, W, H, t} of |m^2 - M_V^2| / (M_V Gamma_V)
                 (near an s-channel resonance; any combination, charges not matched)
    t_level      t;  t_z  (t - probe mean) / probe std
    loc_var      std of t over the 10 nearest events of the probe in the partonic-frame final momenta / sqrt(s_hat),
                 over the probe's std of t (the target's local variation)
Per probe (constant inside a probe): n_final, n_massive, n_gluon, n_quark (b included), n_lepton (charged), n_nu,
n_photon, n_heavy (W, Z, H, t), as_pow and aew_pow (the alpha_s and alpha_ew powers of mg5_pipeline_final.order_vector of the
probe's catalog entry, v4_<process>; nan when the table lacks it),
loop (one-loop target), pp (hadronic beams), t_std and t_mean (the probe's spread and level of log|M|^2), trained.

Targets per event: ae = |q - t|; s (het). Within a probe: ae_n = ae / probe RMSE, s_n = s / probe median s.
Measures per run and read group (every event read, each probe's 2000):
    within   median over probes of Spearman(feature, ae) and (feature, s), event features only, with the quartiles and
             the share of probes with a positive coefficient; mi_med the median over probes of the binned mutual
             information (8 x 8 equal-frequency bins on ranks, minus the mean of 3 shuffles), in nats
    pooled   Spearman over the group's events (SUB per probe) of the feature against ae_n and s_n (the class level
             taken out) and against ae and s (raw, the class level in); mi 20 x 20 bins, same correction
    classes  Spearman over classes (probes of a class pooled) of the class median of the feature against the class
             RMSE and the class median s
    profile  per decile of the feature (edges from the group's pooled events of the run), RMS of ae_n and median s_n
GBT (the four control runs, sklearn HistGradientBoostingRegressor, SUB events per probe): targets log ae, log ae_n,
log s, log s_n on all event and process features; classes split in two halves by a fixed hash, fit on one, R^2 and
permutation importance (3 repeats) and grouped importance (kinematic, process, target features permuted together)
on the other, then the halves swapped; the mean of the two reported.
"""
import argparse, gzip, hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
sys.path.insert(0, HERE)
sys.path.insert(1, ROOT)

HELD = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")    # mixture_ab.HELD
AXES = ("PP4", "EE4", "HD4", "HD5") + HELD
# arm folder -> (loss, share arm, seed)
RUNS = {"v4_het_control": ("het", "control", 42), "v4_het_control_seed2": ("het", "control", 43),
        "v4_het_default50": ("het", "default50", 42), "v4_het_default50_seed2": ("het", "default50", 43),
        "v4_het_default25": ("het", "default25", 42), "v4_het_default25_seed2": ("het", "default25", 43),
        "v4_dimensionless_muon": ("mse", "control", 42), "v4_dimensionless_muon_seed2": ("mse", "control", 43),
        "v4_dimensionless_muon_default50": ("mse", "default50", 42),
        "v4_dimensionless_muon_default50_seed2": ("mse", "default50", 43),
        "v4_dimensionless_muon_default25": ("mse", "default25", 42),
        "v4_dimensionless_muon_default25_seed2": ("mse", "default25", 43)}
UNCHECKED_OK = {"v4_dimensionless_muon", "v4_dimensionless_muon_seed2"}   # the MSE control's own held-out reads
GBT_RUNS = ("v4_het_control", "v4_het_control_seed2", "v4_dimensionless_muon", "v4_dimensionless_muon_seed2")
SUB = 250            # events per probe kept for the pooled measures, the profiles and the GBT
GBT_FIT, GBT_TEST = 150000, 30000    # GBT events fitted (of the training half) and scored (of the other half)
EV_FEATS = ("eta_max", "y_min", "y_ff", "y_if", "u_fwd", "pt_min", "pt_max", "e_min", "dr_min", "sqrt_shat", "thr",
            "y_boost", "res_min", "t_level", "t_z", "loc_var")
PR_FEATS = ("n_final", "n_massive", "n_gluon", "n_quark", "n_lepton", "n_nu", "n_photon", "n_heavy", "as_pow",
            "aew_pow", "loop", "pp", "t_std", "t_mean", "trained")
GROUPS = {"kinematic": EV_FEATS[:13], "process": PR_FEATS, "target": ("t_level", "t_z", "loc_var")}
READ_GROUPS = ("PP4", "EE4", "held", "held_default", "all")
# resonances: the catalog's sm card (G_mu scheme) and WT 1.4915 (fa-catalog-reference)
RES = ((91.188, 2.441404), (80.419, 2.0476), (125.0, 6.382e-3), (172.5, 1.4915))
QUARKS = {1, 2, 3, 4, 5, 5000000}


def group_of(axis):
    return axis if axis in ("PP4", "EE4") else "held_default" if axis in ("HD4", "HD5") else "held"


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    a, b = a[m], b[m]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if math.isfinite(c) else None


def _qbin(x, B):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (r * B // len(x)).astype(np.int64)


def mutual_info(a, b, B, rng):
    """binned MI in nats on ranks (equal-frequency bins), minus the mean of 3 shuffled baselines"""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 * B * B or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return None
    x, y = _qbin(a[m], B), _qbin(b[m], B)

    def mi(x, y):
        h = np.bincount(x * B + y, minlength=B * B).reshape(B, B).astype(float)
        p = h / h.sum()
        pxy = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / pxy[nz])).sum())
    return mi(x, y) - float(np.mean([mi(x, rng.permutation(y)) for _ in range(3)]))


def rank(x):
    """ranks with ties broken by order (as spearman and mutual_info)"""
    r = np.empty(len(x), np.int64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def rank_corr(rx, ry):
    if np.ptp(rx) == 0:
        return None
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c) if math.isfinite(c) else None


def rank_mi(rx, ry, B, rng):
    """mutual_info on precomputed ranks of finite arrays, one shuffle as the baseline (the pooled reads are large)"""
    n = len(rx)
    x, y = rx * B // n, ry * B // n

    def mi(x, y):
        h = np.bincount(x * B + y, minlength=B * B).reshape(B, B).astype(float)
        p = h / h.sum()
        pxy = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / pxy[nz])).sum())
    return mi(x, y) - mi(x, rng.permutation(y))


def mdot(x, z):
    return x[..., 0] * z[..., 0] - (x[..., 1:] * z[..., 1:]).sum(-1)


def lg(x):
    return np.log10(np.clip(x, 1e-300, None))


def event_features(P, t):
    """P (n, k, 4) lab momenta, beams first; t (n,) log|M|^2 -> {feature: (n,)}"""
    n, k, _ = P.shape
    nf = k - 2
    tot = P[:, 0] + P[:, 1]
    sh = np.clip(mdot(tot, tot), 1e-300, None)
    rs = np.sqrt(sh)
    f = {}
    p3 = P[:, 2:, 1:]
    pn = np.clip(np.linalg.norm(p3, axis=-1), 1e-300, None)
    eta = np.arctanh(np.clip(p3[..., 2] / pn, -1 + 1e-15, 1 - 1e-15))
    f["eta_max"] = np.abs(eta).max(1)
    inv = lambda i, j: 2.0 * np.abs(mdot(P[:, i], P[:, j])) / sh
    ff = [inv(i, j) for i in range(2, k) for j in range(i + 1, k)]
    fi = [inv(i, j) for i in (0, 1) for j in range(2, k)]
    f["y_ff"] = lg(np.min(ff, 0)) if ff else np.full(n, np.nan)
    f["y_if"] = lg(np.min(fi, 0))
    f["y_min"] = np.fmin(f["y_ff"], f["y_if"])
    # partonic frame: boost along z with beta = pz / E of the beams' system
    beta = np.clip(tot[:, 3] / tot[:, 0], -1 + 1e-15, 1 - 1e-15)
    g = 1.0 / np.sqrt(1 - beta ** 2)
    E, pz = P[:, 2:, 0], P[:, 2:, 3]
    Ep, pzp = g[:, None] * (E - beta[:, None] * pz), g[:, None] * (pz - beta[:, None] * E)
    pt = np.sqrt(P[:, 2:, 1] ** 2 + P[:, 2:, 2] ** 2)
    cth = np.abs(pzp / np.clip(np.sqrt(pzp ** 2 + pt ** 2), 1e-300, None)).max(1)
    f["u_fwd"] = lg(1.0 / np.clip(1 - cth, 1e-15, None))
    f["pt_min"], f["pt_max"] = lg(pt.min(1) / rs), lg(pt.max(1) / rs)
    f["e_min"] = lg(Ep.min(1) / rs)
    phi = np.arctan2(P[:, 2:, 2], P[:, 2:, 1])
    if nf >= 2:
        dr = [np.sqrt((eta[:, i] - eta[:, j]) ** 2 + ((phi[:, i] - phi[:, j] + np.pi) % (2 * np.pi) - np.pi) ** 2)
              for i in range(nf) for j in range(i + 1, nf)]
        f["dr_min"] = np.min(dr, 0)
    else:
        f["dr_min"] = np.full(n, np.nan)
    f["sqrt_shat"] = lg(rs)
    msum = np.sqrt(np.clip(mdot(P[:, 2:], P[:, 2:]), 0, None)).sum(1)
    f["thr"] = lg(np.clip(rs - msum, 0, None) / rs)
    f["y_boost"] = np.abs(0.5 * np.log(np.clip((tot[:, 0] + tot[:, 3]) / np.clip(tot[:, 0] - tot[:, 3], 1e-300, None),
                                               1e-300, None)))
    combos = [(i, j) for i in range(2, k) for j in range(i + 1, k)] + \
             [(i, j, l) for i in range(2, k) for j in range(i + 1, k) for l in range(j + 1, k)]
    if combos:
        best = np.full(n, np.inf)
        for c in combos:
            q = P[:, list(c)].sum(1)
            mm = mdot(q, q)
            for M, G in RES:
                best = np.minimum(best, np.abs(mm - M * M) / (M * G))
        f["res_min"] = lg(best)
    else:
        f["res_min"] = np.full(n, np.nan)
    f["t_level"] = t
    sd = float(np.std(t)) or 1.0
    f["t_z"] = (t - t.mean()) / sd
    X = np.stack([Ep, P[:, 2:, 1], P[:, 2:, 2], pzp], -1).reshape(n, -1) / rs[:, None]
    try:
        from scipy.spatial import cKDTree
        _, nb = cKDTree(X).query(X, k=min(11, n))
        f["loc_var"] = np.std(t[nb[:, 1:]], axis=1) / sd
    except Exception:
        f["loc_var"] = np.full(n, np.nan)
    return f


def order_of(probe):
    """mg5_pipeline_final.order_vector of the probe's catalog_v4 entry (the name mixture_ab.class_of reads)"""
    try:
        import mg5_pipeline_final as mg
        c = mg.PROCESSES.get("v4_" + probe.split("_", 1)[1])
        return [float(x) for x in mg.order_vector(c)] if c is not None else None
    except Exception:
        return None


def process_features(ids, masses, ov, axis, trained, t):
    a = np.abs(np.asarray(ids)[2:]).astype(int)
    b = np.abs(np.asarray(ids)[:2]).astype(int)
    ov = list(ov or []) + [np.nan] * (4 - len(ov or []))
    loop = float(axis in ("H5", "HD5") or (np.isfinite(ov[0]) and ov[0] + ov[1] > 0))
    return {"n_final": float(len(a)), "n_massive": float((np.asarray(masses) > 1.0).sum()),
            "n_gluon": float((a == 21).sum()), "n_quark": float(sum(int(x) in QUARKS for x in a)),
            "n_lepton": float(np.isin(a, (11, 13, 15)).sum()), "n_nu": float(np.isin(a, (12, 14, 16)).sum()),
            "n_photon": float((a == 22).sum()), "n_heavy": float(np.isin(a, (6, 23, 24, 25)).sum()),
            "as_pow": float(ov[2]), "aew_pow": float(ov[3]), "loop": loop,
            "pp": float(all(int(x) in QUARKS or x == 21 for x in b)), "t_std": float(np.std(t)),
            "t_mean": float(np.mean(t)), "trained": float(trained)}


def read_task(task):
    """one (axis, probe): the event features once per distinct event set (pool and rows) of the runs kept, then each
    run's targets and per-probe measures"""
    axis, probe, files, trained = task
    rng = np.random.default_rng(int(hashlib.md5(probe.encode()).hexdigest()[:8], 16))
    excluded, runs = {}, {}
    for arm, f in files:
        try:
            d = np.load(f)
            t, q = d["t"].astype(np.float64), d["q"].astype(np.float64)
            s = d["s"].astype(np.float64) if RUNS[arm][0] == "het" else None
            chk = (json.loads(str(d["physics_check"])) or {}) if "physics_check" in d.files else {}
            rows = np.asarray(d["pool_row"]) if "pool_row" in d.files else np.zeros(0, np.int64)
            pool = str(d["pool"]) if "pool" in d.files else ""
        except Exception as e:
            excluded[arm] = f"unreadable: {type(e).__name__}: {e}"
            continue
        st = chk.get("status", "unchecked")
        ok = st in ("ok", "allowed") or (st == "unchecked" and arm in UNCHECKED_OK and axis in HELD)
        if not len(t) or len(q) != len(t) or (s is not None and len(s) != len(t)):
            excluded[arm] = f"no q or s ({len(t)} {len(q)} {0 if s is None else len(s)})"
        elif not (np.all(np.isfinite(t)) and np.all(np.isfinite(q)) and (s is None or np.all(np.isfinite(s)))):
            excluded[arm] = "non-finite t, q or s"
        elif not ok:
            excluded[arm] = f"physics {st}"
        elif len(rows) != len(t):
            excluded[arm] = "no pool rows saved (a read older than the pool_row field)"
        else:
            runs[arm] = dict(t=t, q=q, s=s, pool=pool, rows=rows, chk=chk)
    if not runs:
        return axis, probe, None, excluded
    feats, sets = {}, {}       # one feature set per distinct (pool, rows): the runs' reads draw their own events
    for arm, r in runs.items():
        key = (r["pool"], hashlib.md5(r["rows"].tobytes()).hexdigest())
        if key not in sets:
            try:
                a = np.load(r["pool"], mmap_mode="r")
                k = (a.shape[1] - 1) // 5
                assert 5 * k + 1 == a.shape[1], a.shape
                rows = np.asarray(a[r["rows"]])
                P = rows[:, :4 * k].astype(np.float64).reshape(-1, k, 4)
                assert np.allclose(rows[:, -1] * 0 + 1, 1)
                masses = np.sqrt(np.clip(np.median(mdot(P, P), axis=0), 0, None))[2:]
                sets[key] = (event_features(P, r["t"]),
                             process_features(rows[0, 4 * k:5 * k], masses, order_of(probe),
                                              axis, trained, r["t"]))
            except Exception as e:
                excluded[arm] = f"features: {type(e).__name__}: {e}"
                continue
        feats[arm] = sets[key]
    if not feats:
        return axis, probe, None, excluded
    out = {"pr": next(iter(feats.values()))[1], "runs": {}, "n_event_sets": len(sets)}
    for arm, (ev, pr) in feats.items():
        r = runs[arm]
        t, q, s = r["t"], r["q"], r["s"]
        n = len(t)
        sub = np.sort(rng.choice(n, size=min(SUB, n), replace=False))
        ae = np.abs(q - t)
        rmse = float(np.sqrt(np.mean((q - t) ** 2)))
        x = {"n": n, "pr": pr, "rmse": rmse, "bias": float(np.mean(q - t)),
             "ev_sub": {k_: v[sub].astype(np.float32) for k_, v in ev.items()},
             "ev_med": {k_: float(np.nanmedian(v)) if np.isfinite(v).any() else None for k_, v in ev.items()},
             "ae_sub": (ae / max(rmse, 1e-300))[sub].astype(np.float32),
             "ae_raw_sub": ae[sub].astype(np.float32), "rho_ae": {}, "mi_ae": {}}
        if s is not None:
            ms = float(np.median(s))
            x.update(med_s=ms, s_sub=(s / ms)[sub].astype(np.float32), s_raw_sub=s[sub].astype(np.float32),
                     rho_s={}, mi_s={}, rho_s_ae=spearman(s, ae))
        for fn, v in ev.items():
            x["rho_ae"][fn] = spearman(v, ae)
            x["mi_ae"][fn] = mutual_info(v, ae, 8, rng)
            if s is not None:
                x["rho_s"][fn] = spearman(v, s)
                x["mi_s"][fn] = mutual_info(v, s, 8, rng)
        out["runs"][arm] = x
    return axis, probe, out, excluded


def tasks_of(S):
    import yaml
    from mixture_ab import class_of
    cls = class_of()
    trained = {cls(p["name"]) for p in
               yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    by = {}
    for arm in RUNS:
        for ax in AXES:
            dd = os.path.join(S, f"zz_v4_short_{arm}", ax)
            if not os.path.isdir(dd):
                continue
            for f in sorted(os.listdir(dd)):
                if f.endswith(".npz") and (ax not in HELD or f.startswith("v4_")):
                    by.setdefault((ax, f[:-4]), []).append((arm, os.path.join(dd, f)))
    return [(ax, p, fs, cls(p) in trained) for (ax, p), fs in sorted(by.items())], cls


def probe_cmd():
    import time
    tasks, _ = tasks_of(os.environ["SCRATCH"])
    cnt = {}
    for ax, p, fs, _ in tasks:
        for arm, _ in fs:
            cnt.setdefault(arm, {}).setdefault(ax, 0)
            cnt[arm][ax] += 1
    print(json.dumps(cnt))
    for ax in ("PP4", "EE4", "H2", "H5", "HD5"):
        x = next((t for t in tasks if t[0] == ax), None)
        if not x:
            continue
        d = np.load(x[2][0][1])
        print(ax, x[1], sorted(d.files))
        t0 = time.time()
        o = read_task(x)
        print("  one probe", round(time.time() - t0, 2), "s; excluded", o[3])
        if o[2]:
            r0 = next(iter(o[2]["runs"].values()))
            print("  event sets", o[2]["n_event_sets"], "ev_med",
                  {k: round(v, 3) for k, v in r0["ev_med"].items() if v is not None}, "\n  pr", o[2]["pr"],
                  "\n  rmse", {a: round(r["rmse"], 3) for a, r in o[2]["runs"].items()})


# The site part in stages, each one site run under the tool's two-minute limit: extract writes each run's per-probe
# results to $SCRATCH/zz_error_correlates/<arm>/part_<i>.pkl; aggregate (per run) and gbt (per run and target) read
# them and print JSON; the laptop's combine joins the printed outputs into data/error_correlates_reads.json.gz.
def out_dir():
    return os.path.join(os.environ["SCRATCH"], "zz_error_correlates")


def extract_cmd(part, nparts, workers):
    import pickle
    from multiprocessing import Pool
    tasks, cls = tasks_of(os.environ["SCRATCH"])
    mine = tasks[part::nparts]
    per_arm, excluded = {arm: [] for arm in RUNS}, {}
    with Pool(workers) as pool:
        for ax, p, o, exc in pool.imap_unordered(read_task, mine, chunksize=2):
            for arm, why in exc.items():
                excluded.setdefault(arm, {})[f"{ax} {p}"] = why
            if o is not None:
                for arm, r in o["runs"].items():
                    per_arm[arm].append((ax, p, cls(p), {"runs": {arm: r}}))
    for arm, ps in per_arm.items():
        os.makedirs(os.path.join(out_dir(), arm), exist_ok=True)
        pickle.dump(ps, open(os.path.join(out_dir(), arm, f"part_{part}.pkl"), "wb"))
    json.dump({"part": part, "nparts": nparts, "n_tasks": len(mine), "n_tasks_all": len(tasks),
               "kept": {a: len(v) for a, v in per_arm.items()}, "excluded": excluded}, open(
        os.path.join(out_dir(), f"excluded_{part}.json"), "w"))
    print(json.dumps({"part": part, "n_tasks": len(mine), "kept": {a: len(v) for a, v in per_arm.items()}}))


def load_arm(arm):
    import glob, pickle
    ps = []
    for f in sorted(glob.glob(os.path.join(out_dir(), arm, "part_*.pkl"))):
        ps += pickle.load(open(f, "rb"))
    return sorted(ps, key=lambda x: (x[0], x[1]))


def aggregate_cmd(arm):
    loss = RUNS[arm][0]
    probes = load_arm(arm)
    rng = np.random.default_rng(0)
    res = {"arm": arm, "n_probes": {}, "stats": {}}
    for g in READ_GROUPS:
        ps = [x for x in probes if g == "all" or group_of(x[0]) == g]
        res["n_probes"][g] = len(ps)
        if ps:
            res["stats"][g] = group_stats(arm, loss, ps, rng)
    print(json.dumps(res).replace("NaN", "null").replace("Infinity", "null"))


def gbt_cmd(arm, target):
    print(json.dumps({"arm": arm, "target": target, "gbt": gbt(arm, RUNS[arm][0], load_arm(arm), target)}))


def excluded_cmd():
    import glob
    out = {"excluded": {}, "n_tasks": None, "parts": []}
    for f in sorted(glob.glob(os.path.join(out_dir(), "excluded_*.json"))):
        d = json.load(open(f))
        out["parts"].append(d["part"])
        out["n_tasks"] = d["n_tasks_all"]
        for arm, v in d["excluded"].items():
            out["excluded"].setdefault(arm, {}).update(v)
    print(json.dumps(out))


def combine_cmd(paths):
    """join the printed outputs of excluded, aggregate and gbt (one JSON object per file, after site's command line)"""
    res = {"runs": RUNS, "per_run": {}, "n_probes": {}, "gbt": {}}
    for f in paths:
        t = open(f).read()
        d = json.loads(t[t.index("{"):])
        if "excluded" in d:
            res.update(excluded=d["excluded"], n_tasks=d["n_tasks"], parts=d["parts"])
        elif "stats" in d:
            res["per_run"][d["arm"]], res["n_probes"][d["arm"]] = d["stats"], d["n_probes"]
        else:
            res["gbt"].setdefault(d["arm"], {}).update(d["gbt"])
    with gzip.open(os.path.join(DATA, "error_correlates_reads.json.gz"), "wt") as fh:
        json.dump(res, fh)
    print({k: len(v) for k, v in res.items() if isinstance(v, dict)})


def stack(arm, ps, key):
    return np.concatenate([x[3]["runs"][arm][key] for x in ps]).astype(np.float64)


def feat_matrix(arm, ps):
    rr = [x[3]["runs"][arm] for x in ps]
    ev = {f: np.concatenate([r["ev_sub"][f] for r in rr]).astype(np.float64) for f in EV_FEATS}
    pr = {f: np.concatenate([np.full(len(r["ev_sub"]["t_level"]), r["pr"][f]) for r in rr]) for f in PR_FEATS}
    return {**ev, **pr}


def r4(x):
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def group_stats(arm, loss, ps, rng):
    het = loss == "het"
    tgts = ("ae", "s") if het else ("ae",)
    out = {"n_probes": len(ps), "n_classes": len({x[2] for x in ps}), "within": {}, "pooled": {}, "classes": {},
           "profile": {}}
    rr = [x[3]["runs"][arm] for x in ps]
    if het:
        v = [x["rho_s_ae"] for x in rr if x["rho_s_ae"] is not None]
        out["rho_s_ae_within"] = r4(np.median(v)) if v else None
    for fn in EV_FEATS:
        w = {}
        for tgt in tgts:
            v = np.array([x[f"rho_{tgt}"][fn] for x in rr if x[f"rho_{tgt}"].get(fn) is not None])
            mi = np.array([x[f"mi_{tgt}"][fn] for x in rr if x[f"mi_{tgt}"].get(fn) is not None])
            w[tgt] = {"rho_med": r4(np.median(v)) if len(v) else None,
                      "rho_q25": r4(np.quantile(v, 0.25)) if len(v) else None,
                      "rho_q75": r4(np.quantile(v, 0.75)) if len(v) else None,
                      "frac_pos": r4(np.mean(v > 0)) if len(v) else None, "n": int(len(v)),
                      "mi_med": r4(np.median(mi)) if len(mi) else None}
        out["within"][fn] = w
    F = feat_matrix(arm, ps)
    T = {"ae_n": stack(arm, ps, "ae_sub"), "ae": stack(arm, ps, "ae_raw_sub")}
    if het:
        T.update(s_n=stack(arm, ps, "s_sub"), s=stack(arm, ps, "s_raw_sub"))
        out["pooled_s_ae"] = {"rho_raw": r4(spearman(T["s"], T["ae"])), "rho_n": r4(spearman(T["s_n"], T["ae_n"]))}
    TR = {tg: rank(y) for tg, y in T.items()}          # every target is finite: its ranks once
    for fn, x in F.items():
        if np.isfinite(x).all():
            rx = rank(x)
            out["pooled"][fn] = {tg: {"rho": r4(rank_corr(rx, ry)), "mi": r4(rank_mi(rx, ry, 20, rng))}
                                 for tg, ry in TR.items()}
        else:
            out["pooled"][fn] = {tg: {"rho": r4(spearman(x, y)), "mi": r4(mutual_info(x, y, 20, rng))}
                                 for tg, y in T.items()}
        if fn not in EV_FEATS:
            continue
        m = np.isfinite(x)
        if m.sum() < 100 or np.ptp(x[m]) == 0:
            continue
        e = np.quantile(x[m], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, 9)
        prof = {"edges": [r4(z) for z in e], "rms_ae_n": [], "med_s_n": [], "n": []}
        for i in range(10):
            mm = m & (b == i)
            prof["n"].append(int(mm.sum()))
            prof["rms_ae_n"].append(r4(np.sqrt(np.mean(T["ae_n"][mm] ** 2))) if mm.any() else None)
            if het:
                prof["med_s_n"].append(r4(np.median(T["s_n"][mm])) if mm.any() else None)
        out["profile"][fn] = prof
    # classes: probes of a class pooled; RMSE over the class's events, median s as the median of its probes' medians
    by = {}
    for x in ps:
        by.setdefault(x[2], []).append(x[3])
    crm = np.array([math.sqrt(np.average([o["runs"][arm]["rmse"] ** 2 for o in os_],
                                         weights=[o["runs"][arm]["n"] for o in os_]))
                    for os_ in by.values()])
    cms = np.array([np.median([o["runs"][arm]["med_s"] for o in os_]) for os_ in by.values()]) if het else None
    val = lambda o, fn: (o["runs"][arm]["ev_med"].get(fn) if fn in EV_FEATS else o["runs"][arm]["pr"][fn])
    for fn in EV_FEATS + PR_FEATS:
        cf = np.array([np.nanmedian([np.nan if val(o, fn) is None else val(o, fn) for o in os_])
                       if any(val(o, fn) is not None and np.isfinite(val(o, fn)) for o in os_) else np.nan
                       for os_ in by.values()])
        out["classes"][fn] = {"rmse": r4(spearman(cf, crm)), **({"med_s": r4(spearman(cf, cms))} if het else {})}
    if het:
        out["classes_s_rmse"] = r4(spearman(cms, crm))
    return out


def gbt(arm, loss, ps, target):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
    except Exception as e:
        return {"error": f"no sklearn: {e}"}
    het = loss == "het"
    F = feat_matrix(arm, ps)
    names = [f for f in F if np.isfinite(F[f]).any() and np.nanmax(F[f]) > np.nanmin(F[f])]   # all-nan or constant out
    X = np.stack([F[f] for f in names], 1)
    half = np.concatenate([np.full(len(x[3]["runs"][arm]["ev_sub"]["t_level"]), int(hashlib.md5(x[2].encode()).hexdigest()[:8], 16) % 2)
                           for x in ps])
    T = {"log_ae": np.log10(np.clip(stack(arm, ps, "ae_raw_sub"), 1e-12, None)),
         "log_ae_n": np.log10(np.clip(stack(arm, ps, "ae_sub"), 1e-12, None))}
    if het:
        T.update(log_s=np.log10(stack(arm, ps, "s_raw_sub")), log_s_n=np.log10(stack(arm, ps, "s_sub")))
    T = {target: T[target]}
    out = {"n_events": int(len(X)), "dropped": [f for f in F if f not in names], "n_classes_half": [int(len({x[2] for x in ps if
                                                                 int(hashlib.md5(x[2].encode()).hexdigest()[:8], 16) % 2 == h}))
                                                        for h in (0, 1)]}
    rng = np.random.default_rng(1)
    for tg, y in T.items():
        r2, imp, gimp = [], [], {g: [] for g in GROUPS}
        for h in (0, 1):
            tr, te = half == h, half != h
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
            itr = np.flatnonzero(tr)
            if len(itr) > GBT_FIT:
                itr = rng.choice(itr, GBT_FIT, replace=False)
            m.fit(X[itr], y[itr])
            Xt, yt = X[te], y[te]
            if len(yt) > GBT_TEST:
                sel = rng.choice(len(yt), GBT_TEST, replace=False)
                Xt, yt = Xt[sel], yt[sel]
            base = m.score(Xt, yt)
            r2.append(base)
            pi = permutation_importance(m, Xt, yt, n_repeats=3, random_state=0)
            imp.append(pi.importances_mean)
            for g, fs in GROUPS.items():
                Xp = Xt.copy()
                perm = rng.permutation(len(Xp))
                for f in (f for f in fs if f in names):
                    Xp[:, names.index(f)] = Xt[perm, names.index(f)]
                gimp[g].append(base - m.score(Xp, yt))
        out[tg] = {"r2_heldout_classes": [r4(x) for x in r2],
                   "perm_importance": {n: r4(v) for n, v in zip(names, np.mean(imp, 0))},
                   "group_importance": {g: r4(np.mean(v)) for g, v in gimp.items()}}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the summary, on the laptop
LABEL = {"eta_max": r"largest final $|\eta|$", "y_min": r"$\log_{10}\min 2|p_i\cdot p_j|/\hat s$",
         "y_ff": r"$\log_{10}\min_{\rm ff} 2p_i\cdot p_j/\hat s$", "y_if": r"$\log_{10}\min_{\rm if} 2|p_i\cdot p_j|/\hat s$",
         "u_fwd": r"$\log_{10}1/(1-|\cos\theta|_{\max})$", "pt_min": r"$\log_{10}p_{T,\min}/\sqrt{\hat s}$",
         "pt_max": r"$\log_{10}p_{T,\max}/\sqrt{\hat s}$", "e_min": r"$\log_{10}E_{\min}/\sqrt{\hat s}$",
         "dr_min": r"$\Delta R_{\min}$", "sqrt_shat": r"$\log_{10}\sqrt{\hat s}$",
         "thr": r"$\log_{10}(1-\sum m/\sqrt{\hat s})$", "y_boost": r"$|y_{\rm boost}|$",
         "res_min": r"$\log_{10}\min|m^2-M_V^2|/M_V\Gamma_V$", "t_level": r"$\log|\mathcal{M}|^2$",
         "t_z": r"$\log|\mathcal{M}|^2$, probe $z$", "loc_var": r"local spread of $\log|\mathcal{M}|^2$"}


def load(path=None):
    t = gzip.open(path or os.path.join(DATA, "error_correlates_reads.json.gz"), "rt").read()
    return json.loads(t[t.index("{"):])           # site run prints its command line first


def summary(path=None):
    raw = load(path)
    out = {"design": "plan error_correlates step 1 (CPU, existing reads)", "n_tasks": raw["n_tasks"],
           "n_probes": raw["n_probes"], "excluded_counts": {a: len(v) for a, v in raw["excluded"].items()},
           "excluded": raw["excluded"], "tables": {}, "gbt": raw["gbt"]}
    # per read group and target: each feature's within-probe median Spearman, its MI, the pooled Spearman against the
    # normalised target and the class-level Spearman, per run (share_seed)
    for g in READ_GROUPS:
        tab = {}
        for loss, tgts in (("het", ("ae", "s")), ("mse", ("ae",))):
            for tgt in tgts:
                rows = {}
                for fn in EV_FEATS + PR_FEATS:
                    vals = {}
                    for arm, (l, share, seed) in raw["runs"].items():
                        x = raw["per_run"].get(arm, {}).get(g)
                        if l != loss or not x:
                            continue
                        w = x["within"].get(fn, {}).get(tgt, {})
                        vals[f"{share}_{seed}"] = {"within_rho": w.get("rho_med"), "within_mi": w.get("mi_med"),
                                                   "frac_pos": w.get("frac_pos"),
                                                   "pooled_rho_n": x["pooled"][fn][tgt + "_n"]["rho"],
                                                   "pooled_rho_raw": x["pooled"][fn][tgt]["rho"],
                                                   "pooled_mi_raw": x["pooled"][fn][tgt]["mi"],
                                                   "class_rho": x["classes"][fn]["rmse" if tgt == "ae" else "med_s"]}
                    rows[fn] = vals
                tab[f"{loss}_{tgt}"] = rows
        out["tables"][g] = tab
    json.dump(out, open(os.path.join(DATA, "error_correlates.json"), "w"), indent=1)
    figures(raw)
    return out


def figures(raw):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    g = "all"
    het = [raw["per_run"]["v4_het_control"][g], raw["per_run"]["v4_het_control_seed2"][g]]
    mse = [raw["per_run"]["v4_dimensionless_muon"][g], raw["per_run"]["v4_dimensionless_muon_seed2"][g]]
    mean_abs = lambda f: -np.nanmean([abs(x["within"][f]["ae"]["rho_med"]) if x["within"][f]["ae"]["rho_med"] is not None
                                      else np.nan for x in het + mse])
    order = sorted(EV_FEATS, key=mean_abs)
    fig, (axa, axb) = ps.figure(ncols=2)
    yy = np.arange(len(order))[::-1]
    for runs, tgt, col, mk, lab, dy in ((mse, "ae", ps.C.blue, "s", r"MSE net, $|q-t|$", 0.2),
                                        (het, "ae", ps.C.green, "D", r"het net, $|\mu-t|$", 0.0),
                                        (het, "s", ps.C.vermillion, "o", r"het net, $\sigma$", -0.2)):
        for x, fill, sd in zip(runs, ("full", "none"), ("42", "43")):
            v = [x["within"][f][tgt]["rho_med"] for f in order]
            axa.plot([np.nan if z is None else z for z in v], yy + dy, mk, color=col, fillstyle=fill, ls="none",
                     label=f"{lab}, seed {sd}")
    axa.axvline(0, color="0.6", lw=0.8)
    axa.set_yticks(yy)
    axa.set_yticklabels([LABEL[f] for f in order], fontsize=7)
    axa.set_xlabel("median over probes of Spearman with the feature")
    ps.legend(axa, "lower right")
    x = np.arange(1, 11)
    for f, c in zip(order[:3], (ps.C.blue, ps.C.vermillion, ps.C.green)):
        p = het[0]["profile"][f]
        axb.plot(x, p["rms_ae_n"], "-", color=c, marker="s", label=LABEL[f] + ": error")
        axb.plot(x, p["med_s_n"], "--", color=c, marker="o", label=LABEL[f] + r": $\sigma$")
    axb.set_xlabel("decile of the feature, all reads pooled")
    axb.set_ylabel("bin / whole probe, het control seed 42")
    ps.legend(axb, "upper left")
    ps.save(fig, os.path.join(FIGS, "error_correlates"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("probe")
    e = sub.add_parser("extract")
    e.add_argument("--part", type=int, required=True)
    e.add_argument("--nparts", type=int, required=True)
    e.add_argument("--workers", type=int, default=8)
    g = sub.add_parser("aggregate")
    g.add_argument("--run", required=True, choices=list(RUNS))
    b = sub.add_parser("gbt")
    b.add_argument("--run", required=True, choices=GBT_RUNS)
    b.add_argument("--target", required=True, choices=("log_ae", "log_ae_n", "log_s", "log_s_n"))
    sub.add_parser("excluded")
    c = sub.add_parser("combine")
    c.add_argument("paths", nargs="+")
    sm = sub.add_parser("summary")
    sm.add_argument("--reads", default=None)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe_cmd()
    elif a.cmd == "extract":
        extract_cmd(a.part, a.nparts, a.workers)
    elif a.cmd == "aggregate":
        aggregate_cmd(a.run)
    elif a.cmd == "gbt":
        gbt_cmd(a.run, a.target)
    elif a.cmd == "excluded":
        excluded_cmd()
    elif a.cmd == "combine":
        combine_cmd(a.paths)
    else:
        out = summary(getattr(a, "reads", None))
        print(json.dumps({k: out[k] for k in ("n_tasks", "excluded_counts", "gbt")})[:6000])


if __name__ == "__main__":
    main()
