#!/usr/bin/env python3
"""Does catalog_v4's sampling envelop the 13 TeV pp events? Support, not density (Joaquin 2026-11-04, plan
catalog_coverage step 46: "The whole point was to envelop the pp beam range, but neither for the target, or the
energy, the catalog sampling does that").

    python3 analysis/catalog_v4/pp_support.py pull    one read-only `site run` on CC-IN2P3 (CPU) -> data/pp_support_reads.json
    python3 analysis/catalog_v4/pp_support.py         the summary -> data/pp_support.json, figs/pp_support.{png,pdf}

Classes: every probe of recipes/catalog_v4_pp_test.yaml (the PP4 read: 1301 hadronic catalog_v4 classes, pp13v4_<x>
with catalog_v4's physics or pp13_<x> reused), its catalog_v4 counterpart mixture_ab.class_of's class, whose pool is
the catalog entry's (recipes/catalog_v4_base_train.yaml, else the held-out recipe holding it): the train-role pool, or
the val-role pool where the class has no train pool here. Every row of both pools is read:
  default  the pp pool's train and test roles (1e4 + 2e3 MadEvent unweighted events, Daniel's 13 TeV run card:
           pT > 10 GeV and |eta| < 5 in the lab frame on jets, b, photons, charged leptons and neutrinos; MadGraph's
           run-card defaults for every cut the card does not write)
  catalog  the catalog pool (its window [lo, hi] from the recipe; catalog sampling: V4_CUTS in the partonic rest frame,
           sqrt(s_hat) log-uniform on [v4_sqrts_lo, 6000 GeV], the mixture of mg5_pipeline_final.V4_SAMPLING)
Per class, on the default events (sqrt(s_hat) = the beams' invariant mass, a Lorentz invariant; log|M|^2 = the
natural log of the label):
  sqrt(s_hat)  share below the window's lo, above its hi; the catalog pool's own min and max
  log|M|^2     share below the catalog pool's min, above its max; the same on the default events inside the window and
               passing the catalog's cuts (what the support misses beyond energy and cuts)
  cuts         share failing the catalog's cuts (mg5_pipeline_final.fiducial_pass_mask, the class's own fiducial_cuts) in
               the partonic frame, the whole set and each cut alone: pT, |cos theta| (partonic |eta| < 5), Delta R
               (pairs with a b or a tau apart from the rest); and the lab-frame |eta| of those failing the angle.
"""
import json, os, subprocess, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
HELD = ("H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")


def classes():
    """[(probe, catalog class, (lo, hi), where)] for every PP4 probe"""
    from mixture_ab import class_of
    cls = class_of()
    win = {e["name"]: (tuple(e["sqrts"]), "train") for e in
           yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    for h in HELD:
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{h}.yaml")))["processes"]:
            win.setdefault(e["name"], (tuple(e["sqrts"]), h))
    out = []
    for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_pp_test.yaml")))["processes"]:
        p = e["name"]
        c = cls(p)
        out.append([p, c] + ([list(win[c][0]), win[c][1]] if c in win else [None, None]))
    return out


REMOTE = r'''
import glob, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import mg5_pipeline_final as mg
D, C = os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")
NU = {12, 14, 16}

def find(stem):
    for d in (C, D):
        g = sorted(glob.glob(os.path.join(d, stem)))
        if g:
            return g[0]

def sqrts(P):
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))

def ymin(P):
    """log10 of the smallest 2|p_i.p_j| / s_hat over the pairs of external legs but the two beams, and whether that
    pair holds a beam (initial-state collinear, small |t|) or two finals"""
    k = P.shape[1]
    tot = P[:, 0] + P[:, 1]
    sh = np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 1e-300, None)
    dot = lambda x, z: x[..., 0] * z[..., 0] - (x[..., 1:] * z[..., 1:]).sum(-1)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k) if (i, j) != (0, 1)]
    ys = np.stack([2.0 * np.abs(dot(P[:, i], P[:, j])) / sh for i, j in pairs], 1)
    a = np.argmin(ys, 1)
    beam = np.array([i < 2 for i, _ in pairs])[a]
    ymin.n_small = (ys < 1e-2).sum(1)          # how many invariants are small at once (multiply singular corners)
    ymin.lp = np.log10(np.clip(ys, 1e-300, None)).sum(1)   # log10 of the product of every invariant over s_hat
    return np.log10(np.clip(ys[np.arange(len(a)), a], 1e-300, None)), beam

def one(task):
    probe, c, win, where = task
    r = {"probe": probe, "class": c, "window": win, "where": where}
    try:
        if win is None:
            raise RuntimeError("no catalog entry")
        lo, hi = win
        cp = find(f"{c}_{lo}-{hi}GeV_train_*amplitudes.npy") or find(f"{c}_{lo}-{hi}GeV_val_*amplitudes.npy")
        cfg = mg.PROCESSES[c]
        k = len(cfg["pdg_ids"])
        a = np.load(cp, mmap_mode="r")
        assert a.shape[1] == 5 * k + 1, (cp, a.shape, k)
        Pc = np.asarray(a[:, :8], dtype=np.float64).reshape(-1, 2, 4)
        mc = np.asarray(a[:, -1], dtype=np.float64)
        sc = sqrts(Pc)
        with np.errstate(divide="ignore"):
            lc = np.log(np.abs(mc))
        Pfull = np.asarray(a[:, :4 * k], dtype=np.float64).reshape(-1, k, 4)
        yc, _ = ymin(Pfull)
        nsc, lpc = ymin.n_small, ymin.lp
        lc = lc[np.isfinite(lc)]
        r["catalog"] = {"pool": os.path.basename(cp), "n": int(len(mc)), "sqrts_min": float(sc.min()),
                        "sqrts_max": float(sc.max()), "logm_min": float(lc.min()), "logm_max": float(lc.max()),
                        "logm_q001": float(np.quantile(lc, 0.001)), "logm_q999": float(np.quantile(lc, 0.999)),
                        "ymin_min": float(yc.min()), "ymin_q001": float(np.quantile(yc, 0.001)),
                        "n_logm_above_default_q99": None}
        rows = []
        for role in ("train", "test"):
            f = find(f"{probe}_13000-13000GeV_{role}_lhe_amplitudes.npy")
            if f:
                rows.append(np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64))
        if not rows:
            raise RuntimeError("no pp pool")
        x = np.concatenate(rows)
        assert x.shape[1] == 5 * k + 1, (probe, x.shape, k)
        P = x[:, :4 * k].reshape(-1, k, 4)
        pdg = x[0, 4 * k:5 * k].astype(int)
        me = x[:, -1]
        mass = dict(zip((abs(int(p)) for p in cfg["pdg_ids"][2:]), (float(m) for m in cfg["m_finals"])))
        mf = np.array([mass[abs(int(p))] for p in pdg[2:]])
        cuts = cfg["fiducial_cuts"]
        s = sqrts(P)
        with np.errstate(divide="ignore"):
            lm = np.log(np.abs(me))
        Q = mg.to_partonic_frame(P)
        ok = mg.fiducial_pass_mask(Q, mf, pdg, cuts)
        extra = {abs(int(p)) for p in (cuts.get("cut_pdgs") or ())}
        slots = [2 + i for i in range(len(mf)) if abs(int(pdg[2 + i])) not in NU and (mf[i] == 0.0 or abs(int(pdg[2 + i])) in extra)]
        pt = lambda p: np.hypot(p[..., 1], p[..., 2])
        cth = lambda p: np.abs(p[..., 3]) / np.clip(np.linalg.norm(p[..., 1:], axis=-1), 1e-12, None)
        eta = lambda p: np.arcsinh(p[..., 3] / np.clip(pt(p), 1e-9, None))
        phi = lambda p: np.arctan2(p[..., 2], p[..., 1])
        n = len(s)
        f_pt, f_ang, f_dr_b, f_dr = (np.zeros(n, bool) for _ in range(4))
        lab_eta_ang = []
        for i in slots:
            f_pt |= pt(Q[:, i]) <= cuts["pt_min"]
            bad = cth(Q[:, i]) >= cuts["cos_max"]
            f_ang |= bad
        hv = {5, 15}
        for i, j in itertools.combinations(slots, 2):
            dphi = np.abs(phi(Q[:, i]) - phi(Q[:, j])); dphi = np.minimum(dphi, 2 * np.pi - dphi)
            dr = np.sqrt((eta(Q[:, i]) - eta(Q[:, j])) ** 2 + dphi ** 2) <= cuts["dr_min"]
            if abs(int(pdg[i])) in hv or abs(int(pdg[j])) in hv:
                f_dr_b |= dr
            else:
                f_dr |= dr
        lab_max_eta = np.max(np.abs(eta(P[:, slots])), axis=1) if slots else np.zeros(n)
        part_max_eta = np.max(np.abs(eta(Q[:, slots])), axis=1) if slots else np.zeros(n)
        ybst = 0.5 * np.log(np.clip((P[:, 0, 0] + P[:, 1, 0] + P[:, 0, 3] + P[:, 1, 3]) /
                                    np.clip(P[:, 0, 0] + P[:, 1, 0] - P[:, 0, 3] - P[:, 1, 3], 1e-300, None), 1e-300, None))
        inw = (s >= lo) & (s <= hi)
        fin = np.isfinite(lm)
        sup = inw & ok
        yd, bd = ymin(P)
        nsd, lpd = ymin.n_small, ymin.lp
        hi_ev = fin & sup & (lm > r["catalog"]["logm_max"])
        q99 = float(np.quantile(lm[fin], 0.99))
        r["catalog"]["n_logm_above_default_q99"] = int(np.sum(np.log(np.abs(mc[np.abs(mc) > 0])) > q99))
        r["default_q99_logm"] = q99
        r["above_max_events"] = {"n": int(hi_ev.sum()),
            "ymin_median": float(np.median(yd[hi_ev])) if hi_ev.any() else None,
            "below_catalog_ymin_min": float(np.mean(yd[hi_ev] < r["catalog"]["ymin_min"])) if hi_ev.any() else None,
            "beam_pair_share": float(np.mean(bd[hi_ev])) if hi_ev.any() else None,
            "sqrts_median": float(np.median(s[hi_ev])) if hi_ev.any() else None,
            "part_eta_median": float(np.median(part_max_eta[hi_ev])) if hi_ev.any() else None}
        # a resonance: the invariant mass of some 2 or 3 finals within 3 widths of a W, Z, H or top pole (charge-blind,
        # an indicator); the share of the above-max events and of all default and catalog events near one
        def near_pole(PP):
            fins = range(2, PP.shape[1])
            hit = np.zeros(len(PP), bool)
            for m in (2, 3):
                for sub in itertools.combinations(fins, m):
                    q = PP[:, list(sub)].sum(1)
                    mm = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))
                    for M, G in ((80.419, 2.0476), (91.188, 2.441404), (125.0, 6.382e-3), (172.5, 1.4915)):
                        hit |= np.abs(mm - M) < 3 * G
            return hit
        npd = near_pole(P)
        npc = near_pole(Pfull[:20000])
        r["above_max_events"]["near_pole_share"] = float(np.mean(npd[hi_ev])) if hi_ev.any() else None
        r["above_max_events"]["n_small_median"] = float(np.median(nsd[hi_ev])) if hi_ev.any() else None
        r["above_max_events"]["lp_below_catalog_min"] = float(np.mean(lpd[hi_ev] < lpc.min())) if hi_ev.any() else None
        r["n_small_catalog_q999"] = float(np.quantile(nsc, 0.999)); r["n_small_catalog_max"] = int(nsc.max())
        r["n_small_default_median"] = float(np.median(nsd)); r["lp_default_below_catalog_min"] = float(np.mean(lpd < lpc.min()))
        r["near_pole_default"] = float(np.mean(npd)); r["near_pole_catalog"] = float(np.mean(npc))
        r["default_ymin_below_catalog_min"] = float(np.mean(yd < r["catalog"]["ymin_min"]))
        r["default"] = {
            "n": int(n), "n_nonfinite_logm": int((~fin).sum()), "slots": len(slots),
            "pdg": [int(p) for p in pdg],
            "sqrts_min": float(s.min()), "sqrts_max": float(s.max()),
            "below_lo": float(np.mean(s < lo)), "above_hi": float(np.mean(s > hi)),
            "below_pool_min": float(np.mean(s < r["catalog"]["sqrts_min"])),
            "above_pool_max": float(np.mean(s > r["catalog"]["sqrts_max"])),
            "fail_cuts": float(np.mean(~ok)), "fail_pt": float(np.mean(f_pt)), "fail_angle": float(np.mean(f_ang)),
            "fail_dr_b": float(np.mean(f_dr_b)), "fail_dr_other": float(np.mean(f_dr)),
            "fail_angle_only": float(np.mean(f_ang & ~f_pt & ~f_dr & ~f_dr_b)),
            "fail_dr_b_only": float(np.mean(f_dr_b & ~f_pt & ~f_ang & ~f_dr)),
            "angle_fail_lab_eta_max": float(lab_max_eta[f_ang].max()) if f_ang.any() else None,
            "angle_fail_ybst_median": float(np.median(np.abs(ybst[f_ang]))) if f_ang.any() else None,
            "lab_eta_max": float(lab_max_eta.max()) if slots else None,
            "part_eta_max": float(part_max_eta.max()) if slots else None,
            "outside_any": float(np.mean(~sup)),
            "logm_min": float(lm[fin].min()), "logm_max": float(lm[fin].max()),
            "logm_below": float(np.mean(lm[fin] < r["catalog"]["logm_min"])),
            "logm_above": float(np.mean(lm[fin] > r["catalog"]["logm_max"])),
            "logm_below_in_support": float(np.mean(lm[fin & sup] < r["catalog"]["logm_min"])) if (fin & sup).any() else None,
            "logm_above_in_support": float(np.mean(lm[fin & sup] > r["catalog"]["logm_max"])) if (fin & sup).any() else None,
            "logm_above_outside_cuts": float(np.mean(lm[fin & ~ok] > r["catalog"]["logm_max"])) if (fin & ~ok).any() else None,
            "logm_below_below_lo": float(np.mean(lm[fin & (s < lo)] < r["catalog"]["logm_min"])) if (fin & (s < lo)).any() else None,
            "hist_sqrts": np.histogram(np.log10(np.clip(s, 1, None)), np.arange(0.5, 4.15, 0.05))[0].tolist(),
        }
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    return r

if __name__ == "__main__":
    from multiprocessing import Pool
    tasks = json.load(open(sys.argv[1]))
    with Pool(int(sys.argv[2])) as pool:
        res = list(pool.imap_unordered(one, tasks, chunksize=4))
    print("PPSUP " + json.dumps(res))
'''


def pull(workers=8, limit=None):
    tasks = classes()[:limit] if limit else classes()
    cmd = (f"F=$(mktemp /tmp/zz_probe_ppsup_XXXXXX.py); J=$(mktemp /tmp/zz_probe_ppsup_XXXXXX.json); cat > $F <<'PYEOF'\n{REMOTE}PYEOF\n"
           f"cat > $J <<'JEOF'\n{json.dumps(tasks)}\nJEOF\n"
           f"python3 $F $J {workers}; rm -f $F $J")
    res = subprocess.run(["site", "--timeout", "3000", "run", "--item", "catalog_coverage", "ccin2p3", "FA", "--", cmd],
                         capture_output=True, text=True, timeout=3100)
    line = [l for l in res.stdout.splitlines() if l.startswith("PPSUP ")]
    if not line:
        sys.exit("pull: no PPSUP line\n" + res.stdout[-3000:] + res.stderr[-2000:])
    rows = json.loads(line[-1][len("PPSUP "):])
    json.dump({"rows": sorted(rows, key=lambda r: r["probe"]), "sqrts_hist_edges_log10": np.arange(0.5, 4.15, 0.05).tolist()},
              open(os.path.join(DATA, "pp_support_reads.json"), "w"))
    print(len(rows), sum("error" in r for r in rows))


if __name__ == "__main__":
    if sys.argv[1:2] == ["pull"]:
        pull(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)


def summary():
    d = json.load(open(os.path.join(DATA, "pp_support_reads.json")))
    rows = [r for r in d["rows"] if "error" not in r]
    err = {r["probe"]: r["error"] for r in d["rows"] if "error" in r}
    D = lambda k: np.array([r["default"][k] if r["default"][k] is not None else np.nan for r in rows], float)
    n_ev = D("n")
    pooled = lambda k: float(np.nansum(D(k) * n_ev) / n_ev[~np.isnan(D(k))].sum())
    q = lambda x: [float(v) for v in np.nanquantile(x, [0.5, 0.9, 0.99])] + [float(np.nanmax(x))]
    keys = ("below_lo", "above_hi", "fail_cuts", "fail_pt", "fail_angle", "fail_dr_b", "fail_dr_other",
            "outside_any", "logm_below", "logm_above", "logm_above_in_support")
    out = {"source": "data/pp_support_reads.json (pp_support.py pull)", "n_classes": len(rows), "errors": err,
           "n_default_events": int(n_ev.sum()),
           "pooled_share": {k: pooled(k) for k in keys},
           "per_class_median_p90_p99_max": {k: q(D(k)) for k in keys},
           "n_classes_any": {k: int(np.nansum(D(k) > 0)) for k in keys},
           "n_classes_over_1pct": {k: int(np.nansum(D(k) > 0.01)) for k in keys}}
    a = [r["above_max_events"] for r in rows if r["above_max_events"]["n"]]
    out["above_max_events"] = {"n_classes": len(a), "n_events": int(sum(x["n"] for x in a)),
                               "sqrts_median_median": float(np.median([x["sqrts_median"] for x in a])),
                               "ymin_median_median": float(np.median([x["ymin_median"] for x in a])),
                               "share_below_catalog_ymin_min_max": float(max(x["below_catalog_ymin_min"] for x in a)),
                               "beam_pair_share_median": float(np.median([x["beam_pair_share"] for x in a]))}
    gap = np.array([r["default"]["logm_max"] - r["catalog"]["logm_max"] for r in rows])
    gapq = np.array([r["default"]["logm_max"] - r["catalog"]["logm_q999"] for r in rows])
    low = np.array([r["catalog"]["logm_min"] - r["default"]["logm_min"] for r in rows])
    out["logm_edges"] = {"default_max_minus_catalog_max_median_p90_max": q(gap),
                         "n_default_max_above_catalog_max": int((gap > 0).sum()),
                         "default_max_minus_catalog_q999_median": float(np.median(gapq)),
                         "catalog_min_minus_default_min_median": float(np.median(low)),
                         "n_catalog_min_below_default_min": int((low > 0).sum())}
    out["sqrts_edges"] = {"catalog_pool_min_over_lo_max": float(max(r["catalog"]["sqrts_min"] / r["window"][0] for r in rows)),
                          "default_sqrts_max_median": float(np.median([r["default"]["sqrts_max"] for r in rows])),
                          "default_sqrts_min_over_lo_median": float(np.median([r["default"]["sqrts_min"] / r["window"][0] for r in rows])),
                          "n_default_below_lo": int(sum(r["default"]["below_lo"] > 0 for r in rows)),
                          "n_default_above_6TeV": int(sum(r["default"]["above_hi"] > 0 for r in rows))}
    ang = [r for r in rows if r["default"]["fail_angle"] > 0]
    out["angle_cut"] = {"n_classes": len(ang),
                        "lab_eta_max_of_failing": float(max(r["default"]["angle_fail_lab_eta_max"] for r in ang)) if ang else None,
                        "boost_rapidity_median_of_failing": float(np.median([r["default"]["angle_fail_ybst_median"] for r in ang])) if ang else None}
    out["worst_outside_any"] = sorted(([r["probe"], round(r["default"]["outside_any"], 4), round(r["default"]["fail_angle"], 4),
                                        round(r["default"]["fail_dr_b"], 4), round(r["default"]["fail_pt"], 4),
                                        round(r["default"]["below_lo"], 4)] for r in rows), key=lambda x: -x[1])[:15]
    out["worst_logm_above"] = sorted(([r["probe"], round(r["default"]["logm_above"], 4), round(r["default"]["logm_max"] - r["catalog"]["logm_max"], 2)]
                                      for r in rows), key=lambda x: -x[1])[:15]
    json.dump(out, open(os.path.join(DATA, "pp_support.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return rows, d


def figure(rows):
    """(a) per class, sorted: the share of its pp events outside the catalog's cuts or window, and the share above the
    catalog pool's largest log|M|^2; (b) per class, the largest log|M|^2 of the pp events against the catalog pool's"""
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    out = np.sort([r["default"]["outside_any"] for r in rows])[::-1]
    up = np.sort([r["default"]["logm_above"] for r in rows])[::-1]
    x = np.arange(1, len(rows) + 1)
    a.plot(x, np.clip(out, 1e-5, None), color=ps.C.blue, label="outside cuts or window")
    a.plot(x, np.clip(up, 1e-5, None), color=ps.C.vermillion, label=r"above catalog max $\log|M|^2$")
    a.set(yscale="log", ylim=(1e-5, 1.5), xlabel="classes, sorted", ylabel="share of pp events")
    ps.legend(a, "upper right")
    cm = np.array([r["catalog"]["logm_max"] for r in rows]); dm = np.array([r["default"]["logm_max"] for r in rows])
    lim = [min(cm.min(), dm.min()) - 2, max(cm.max(), dm.max()) + 2]
    b.plot(cm, dm, "o", color=ps.C.sky, mfc="none", linestyle="none", label="class")
    b.plot(lim, lim, "k--", label="equal")
    b.set(xlim=lim, ylim=lim, xlabel=r"catalog: max $\log|M|^2$", ylabel=r"pp: max $\log|M|^2$")
    ps.legend(b, "upper left")
    ps.save(fig, os.path.join(FIGS, "pp_support"))


if __name__ == "__main__" and sys.argv[1:2] != ["pull"]:
    figure(summary()[0])
