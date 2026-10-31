#!/usr/bin/env python3
"""Step 15 of plan entry catalog_coverage, question 2 of the design passed 2026-10-31T04:56 (D220): where is a
catalog_v4 network uncertain (the het loss's per-event sigma), on default-sampled events of its trained classes and on
the held-out classes, and does sigma rank the error there?

    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/het_uncertainty.py run \
        | gzip > analysis/catalog_v4/data/het_reads.json.gz                       (login node, CPU)
    python3 analysis/catalog_v4/het_uncertainty.py        -> data/het_uncertainty.json, figs/het_where.{png,pdf}

The site part reads each het run's zero-shot reads (tools/zero_shot_parent.py --save-dir, the job's
$SCRATCH/zz_v4_short_<arm>/<axis>/<probe>.npz, read from the run's sigma_stage parent: mu frozen there, so mu is the
best checkpoint by val_loss_no_reg) with per event t (truth log|M|^2), q (prediction) and s = sd_c * sigma (sigma in
log|M|^2). Every run's training curve is checked for a blow-up (mixture_ab.py blowups: the het NLL rule, an NLL above its
step-500 value + 0.5 ln 10); an arm with a run flagged keeps its numbers, its verdicts withheld.
A probe is used only as mixture_ab.py extract keeps it (finite t, q, s; physics status ok or allowed);
held-out axes catalog_v4 probes (v4_*) only, as mixture_ab.read_of. A probe's class is mixture_ab.class_of's.
Per (run, read, class), events of the class's probes pooled:
    med_s       median s
    rmse        sqrt(mean (q - t)^2)
    pull_std    std of (q - t) / s;  cover68  share of events with |q - t| < s
    rho_within  Spearman over the class's events of s against |q - t|
The where-map (PP4 and EE4 only: default-sampled test events whose momenta are the pool rows the read scored,
npz pool / pool_row): per event
    eta   the largest final-state |eta| in the lab frame (pp: the LHE frame, ee: the c.o.m.)
    y     log10 of the smallest 2|p_i.p_j| / s_hat over every pair of external legs but the two beams
          (as density_check.features)
binned in deciles of the read (edges from control seed 42's events of the same read group), per run and bin: median
s, RMSE, and the same on per-probe normalised values (s over its probe's median s, |q - t| over its probe's RMSE;
information: it takes the class mix out of the bins).
Read groups: PP4 and EE4 each split into trained classes (recipes/catalog_v4_train.yaml) and not trained; held-out
H1..H7c pooled and per axis (catalog-sampled); held_default, the default-sampled held-out pools HD4 (114 tree classes)
and HD5 (28 one-loop classes) of the step-20 REREADs, pooled and per pool, rule B on it beside (D227).

The summary (laptop), measures and rules of the design, fixed before the read, per run; a verdict on the control pair
(v4_het_control, both seeds), the share arms reported beside (information; their reads in hand only, the missing ones
listed):
    rank_across  Spearman over classes of med_s against rmse;  rank_within  the median over classes of rho_within
    (A) trained classes, per read (PP4 trained, EE4 trained): YES rank_across >= 0.5 and rank_within >= 0.3;
        NO rank_across < 0.3 or rank_within < 0.15; otherwise weak. A verdict needs both seeds (split otherwise).
    (B) held-out pooled: calibrated when the median over classes of pull_std <= 3 and of cover68 >= 0.4; otherwise
        rank_across >= 0.5 orders the worst held-out cells, < 0.3 does not, between weak.
    bin contrast: last over first decile of median s and of RMSE, on each binning.
"""
import argparse, gzip, json, math, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from mixture_ab import HELD, MIX_HET, blowups, class_of   # noqa: E402

# run id -> the job's arm (scripts/job_catalog_v3_short.sh ARM, the prediction dir's name), seed
RUNS = {"FA-ccin2p3-87ea24d0": ("v4_het_control", 42), "FA-ccin2p3-321a3de2": ("v4_het_control_seed2", 43),
        "FA-ccin2p3-466656b2": ("v4_het_default50", 42), "FA-ccin2p3-9897833d": ("v4_het_default50_seed2", 43),
        "FA-ccin2p3-f0156d88": ("v4_het_default25", 42), "FA-ccin2p3-df858d33": ("v4_het_default25_seed2", 43)}
# an arm's reads are its job folder's (the step-20 eval-only REREADs wrote PP4, EE4, HD4 and HD5 there from the same
# parent), keyed by the training run
ARMS = {a: MIX_HET[a]["held"] for a in ("control", "default50", "default25")}
DEFAULT_HELD = ("HD4", "HD5")       # the default-sampled held-out pools (mixture_ab.DEFAULT_EXTRA), read held_default
AXES = ("PP4", "EE4") + HELD + DEFAULT_HELD
REF = "FA-ccin2p3-87ea24d0"          # the decile edges: control seed 42's events of each read group
YES_A, YES_W, NO_A, NO_W = 0.5, 0.3, 0.3, 0.15
PULL_MAX, COVER_MIN = 3.0, 0.4


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if math.isfinite(c) else None


def kinematics(pool, rows):
    """(eta, y) per scored event from its pool rows: columns 4k momenta (E, px, py, pz), k ids, 1 amplitude"""
    a = np.load(pool, mmap_mode="r")
    k = (a.shape[1] - 1) // 5
    assert 5 * k + 1 == a.shape[1], (pool, a.shape)
    P = np.array(a[np.asarray(rows)][:, :4 * k], dtype=np.float64).reshape(-1, k, 4)
    p = P[:, 2:, 1:]
    pn = np.clip(np.linalg.norm(p, axis=-1), 1e-300, None)
    eta = np.abs(np.arctanh(np.clip(p[..., 2] / pn, -1 + 1e-15, 1 - 1e-15))).max(1)
    tot = P[:, 0] + P[:, 1]
    sh = np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 1e-300, None)
    dot = lambda x, z: x[..., 0] * z[..., 0] - (x[..., 1:] * z[..., 1:]).sum(-1)
    ys = [2.0 * np.abs(dot(P[:, i], P[:, j])) / sh for i in range(k) for j in range(i + 1, k) if (i, j) != (0, 1)]
    return eta, np.log10(np.clip(np.min(ys, axis=0), 1e-300, None))


def read_probe(task):
    run, axis, f = task
    d = np.load(f)
    probe = os.path.basename(f)[:-4]
    t, q, s = d["t"].astype(np.float64), d["q"].astype(np.float64), d["s"].astype(np.float64)
    st = (json.loads(str(d["physics_check"])) or {}).get("status", "unchecked")
    if len(q) != len(t) or len(s) != len(t) or not len(t):
        return run, axis, probe, {"excluded": f"no q or s ({len(t)} {len(q)} {len(s)})"}
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q)) and np.all(np.isfinite(s))):
        return run, axis, probe, {"excluded": "non-finite t, q or s"}
    if st not in ("ok", "allowed"):
        return run, axis, probe, {"excluded": f"physics {st}"}
    out = {"t": t, "q": q, "s": s}
    if axis in ("PP4", "EE4"):
        try:
            out["eta"], out["y"] = kinematics(str(d["pool"]), d["pool_row"])
        except Exception as e:                 # listed, never dropped silently
            out["kin_error"] = f"{type(e).__name__}: {e}"
    return run, axis, probe, out


def class_stats(t, q, s):
    e = q - t
    pull = e / s
    return {"n": int(len(t)), "med_s": float(np.median(s)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "pull_std": float(np.std(pull)), "cover68": float(np.mean(np.abs(pull) < 1)),
            "rho_within": spearman(s, np.abs(e))}


def bins_of(x, edges, s, e, sn, en):
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    out = []
    for i in range(len(edges) - 1):
        m = b == i
        out.append({"n": int(m.sum()),
                    "med_s": float(np.median(s[m])) if m.any() else None,
                    "rmse": float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None,
                    "med_s_norm": float(np.median(sn[m])) if m.any() else None,
                    "rmse_norm": float(np.sqrt(np.mean(en[m] ** 2))) if m.any() else None})
    return out


def run(workers, runs=None):
    from multiprocessing import Pool
    cls = class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    S = os.environ["SCRATCH"]
    tasks = []
    for run_id, (arm, _) in RUNS.items():
        if runs and run_id not in runs:
            continue
        for ax in AXES:
            dd = os.path.join(S, f"zz_v4_short_{arm}", ax)
            if os.path.isdir(dd):
                tasks += [(run_id, ax, os.path.join(dd, f)) for f in sorted(os.listdir(dd))
                          if f.endswith(".npz") and (ax in ("PP4", "EE4") + DEFAULT_HELD or f.startswith("v4_"))]
    ev, per_class, excluded = {}, {}, {}
    with Pool(workers) as pool:
        for run_id, ax, probe, o in pool.imap_unordered(read_probe, tasks, chunksize=8):
            if "excluded" in o:
                excluded.setdefault(run_id, {})[f"{ax} {probe}"] = o["excluded"]
                continue
            c = cls(probe)
            read = ax if ax in ("PP4", "EE4") else "held_default" if ax in DEFAULT_HELD else "held"
            per_class.setdefault((run_id, read), {}).setdefault(c, []).append(dict(o, axis=ax, probe=probe))
            if ax in ("PP4", "EE4"):
                if "eta" in o:
                    ev.setdefault((run_id, f"{ax}_{'trained' if c in trained else 'not_trained'}"), []).append(o)
                else:
                    excluded.setdefault(run_id, {})[f"{ax} {probe} (where-map only)"] = o["kin_error"]
    res = {"runs": {r: {"arm": a, "seed": s} for r, (a, s) in RUNS.items()}, "excluded": excluded,
           "classes": {}, "n_probes": {}, "where": {}}
    for (run_id, read), by in per_class.items():
        rows = {}
        for c, ps in by.items():
            cat = lambda k: np.concatenate([p[k] for p in ps])
            st = class_stats(cat("t"), cat("q"), cat("s"))
            st.update(axis=ps[0]["axis"], trained=c in trained, probes=sorted(p["probe"] for p in ps))
            rows[c] = st
        res["classes"].setdefault(run_id, {})[read] = rows
        res["n_probes"].setdefault(run_id, {})[read] = sum(len(v) for v in by.values())
    edges = {}
    for g in sorted({g for _, g in ev}):
        ref = ev.get((REF, g))
        if ref:
            edges[g] = {v: np.quantile(np.concatenate([p[v] for p in ref]), np.linspace(0, 1, 11)).tolist()
                        for v in ("eta", "y")}
    res["edges"] = edges
    for (run_id, g), ps in ev.items():
        if g not in edges:
            continue
        s = np.concatenate([p["s"] for p in ps])
        e = np.concatenate([p["q"] - p["t"] for p in ps])
        sn = np.concatenate([p["s"] / np.median(p["s"]) for p in ps])
        en = np.concatenate([(p["q"] - p["t"]) / np.sqrt(np.mean((p["q"] - p["t"]) ** 2)) for p in ps])
        out = {"n_probes": len(ps), "n_events": int(len(s))}
        for v in ("eta", "y"):
            out[v] = bins_of(np.concatenate([p[v] for p in ps]), edges[g][v], s, e, sn, en)
        res["where"].setdefault(run_id, {})[g] = out
    json.dump(res, sys.stdout)


# ---------------------------------------------------------------------------------------------------------------------
# the summary, on the laptop
def load(path=None):
    t = gzip.open(path or os.path.join(DATA, "het_reads.json.gz"), "rt").read()
    return json.loads(t[t.index("{"):])           # site run prints its command line first


def ranks(rows):
    rows = [r for r in rows if r["n"] >= 3]
    ra = spearman(np.array([r["med_s"] for r in rows]), np.array([r["rmse"] for r in rows]))
    w = [r["rho_within"] for r in rows if r["rho_within"] is not None]
    return {"n_classes": len(rows), "rank_across": None if ra is None else round(ra, 4),
            "rank_within": round(float(np.median(w)), 4) if w else None,
            "pull_std_median": round(float(np.median([r["pull_std"] for r in rows])), 4) if rows else None,
            "cover68_median": round(float(np.median([r["cover68"] for r in rows])), 4) if rows else None}


def rule_a(x):
    if x["rank_across"] is None or x["rank_within"] is None:
        return None
    if x["rank_across"] >= YES_A and x["rank_within"] >= YES_W:
        return "YES"
    if x["rank_across"] < NO_A or x["rank_within"] < NO_W:
        return "NO"
    return "weak"


def rule_b(x):
    if x["pull_std_median"] is None:
        return None
    if x["pull_std_median"] <= PULL_MAX and x["cover68_median"] >= COVER_MIN:
        return "calibrated"
    ra = x["rank_across"]
    return "not calibrated; " + ("orders the worst held-out cells" if ra is not None and ra >= YES_A else
                                 "does not order them" if ra is not None and ra < NO_A else "weak ordering")


def both(vs):
    if any(v is None for v in vs):
        return None
    return vs[0] if len(set(vs)) == 1 else "split: " + " / ".join(vs)


def contrast(bins, k):
    lo, hi = bins[0][k], bins[-1][k]
    return round(hi / lo, 4) if lo and hi else None


def summary(path=None, out_dir=None):
    raw = load(path)
    cls_reads = raw["classes"]
    expected = dict(raw["n_probes"][REF])          # the control's probe count per read: what a full read has
    out = {"design": "plan catalog_coverage step 15, passed 2026-10-31T04:56, question 2 (sigma)",
           "rules": {"A": f"trained classes per read: YES rank_across >= {YES_A} and rank_within >= {YES_W}; NO "
                          f"rank_across < {NO_A} or rank_within < {NO_W}; else weak; both seeds",
                     "B": f"held-out pooled: calibrated median pull_std <= {PULL_MAX} and median cover68 >= "
                          f"{COVER_MIN}; else rank_across >= {YES_A} orders the worst cells, < {NO_A} does not"},
           "excluded": raw["excluded"], "edges": raw["edges"], "arms": {},
           "blowups": blowups(RUNS)}      # each run's training curve (mixture_ab.py blowup_of; the het NLL rule)
    flagged = {f.split(":")[0] for f in out["blowups"]["flagged"]}
    for arm, pair in ARMS.items():
        a = {"runs": list(pair), "per_seed": {}, "missing": {}}
        for run_id in pair:
            seed = RUNS[run_id][1]
            cr = cls_reads.get(run_id, {})
            ps = {}
            for read in ("PP4", "EE4", "held", "held_default"):
                n = raw["n_probes"].get(run_id, {}).get(read, 0)
                if n < expected.get(read, 0):
                    a["missing"][f"seed{seed} {read}"] = f"{expected[read] - n} of {expected[read]} probes not read"
                rows = cr.get(read)
                if not rows:
                    continue
                if read == "held_default":       # HD4 and HD5 by held-out axis (the class's), pooled
                    ps["held_default_pooled"] = ranks(list(rows.values()))
                    for ax in DEFAULT_HELD:
                        r = [x for x in rows.values() if x["axis"] == ax]
                        if r:
                            ps[f"held_default_{ax}"] = ranks(r)
                elif read == "held":
                    ps["held_pooled"] = ranks(list(rows.values()))
                    for ax in HELD:
                        r = [x for x in rows.values() if x["axis"] == ax]
                        if r:
                            ps[f"held_{ax}"] = ranks(r)
                else:
                    for part, sel in (("trained", True), ("not_trained", False)):
                        ps[f"{read}_{part}"] = ranks([x for x in rows.values() if x["trained"] == sel])
            where = {}
            for g, w in raw["where"].get(run_id, {}).items():
                where[g] = {"n_probes": w["n_probes"], "n_events": w["n_events"],
                            **{f"{v}_contrast": {k: contrast(w[v], k) for k in ("med_s", "rmse", "med_s_norm", "rmse_norm")}
                               for v in ("eta", "y")},
                            "bins": {v: w[v] for v in ("eta", "y")}}
            ps["where"] = where
            a["per_seed"][f"seed{seed}"] = ps
        s = a["per_seed"]
        verd = {}
        part = lambda read: any(k.split()[1] == read.split("_")[0] for k in a["missing"])   # a seed's read is partial
        for read in ("PP4_trained", "EE4_trained"):
            verd[f"A {read}"] = "withheld: read incomplete" if part(read) else \
                both([rule_a(s[k][read]) if read in s.get(k, {}) else None for k in ("seed42", "seed43")])
        verd["B held_pooled"] = "withheld: read incomplete" if part("held_pooled") else \
            both([rule_b(s[k]["held_pooled"]) if "held_pooled" in s.get(k, {}) else None for k in ("seed42", "seed43")])
        # the same rule on the default-sampled held-out pools HD4 + HD5 (D227: the held-out check on default events)
        verd["B held_default_pooled"] = "withheld: read incomplete" if any(
            k.split()[1] == "held_default" for k in a["missing"]) else \
            both([rule_b(s[k]["held_default_pooled"]) if "held_default_pooled" in s.get(k, {}) else None
                  for k in ("seed42", "seed43")])
        bu = sorted(r for r in pair if r in flagged)
        if bu:                    # a blown-up or unchecked run: its numbers stay above, listed apart in blowups
            verd = {k: f"withheld: {', '.join(bu)} blew up or unchecked (blowups); on the reads: {v}"
                    for k, v in verd.items()}
        a["verdicts"] = verd
        a["complete"] = not a["missing"] and len(s) == 2
        out["arms"][arm] = a
    json.dump(out, open(os.path.join(out_dir or DATA, "het_uncertainty.json"), "w"), indent=1)
    figure(out, out_dir)
    print(json.dumps({arm: {"verdicts": a["verdicts"], "missing": a["missing"],
                            **{k: {r: {m: x for m, x in v.items()} for r, v in ps.items() if r != "where"}
                               for k, ps in a["per_seed"].items()},
                            "where_contrast": {k: {g: {c: w[c] for c in ("eta_contrast", "y_contrast")}
                                                   for g, w in ps["where"].items()} for k, ps in a["per_seed"].items()}}
                      for arm, a in out["arms"].items()}))


def figure(out, out_dir=None):
    """Per-probe normalised median s and RMSE per decile of the largest final |eta| (a) and of log10 min pair
    invariant / s_hat (b), PP4 trained classes, the het control's two seeds."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    fig, (axa, axb) = ps.figure(ncols=2)
    ctl = out["arms"]["control"]["per_seed"]
    x = np.arange(1, 11)
    for ax, v, xl in ((axa, "eta", r"decile of largest final $|\eta|$ (10 = forward)"),
                      (axb, "y", r"decile of $\log_{10}\min 2p_i\cdot p_j/\hat s$ (1 = collinear)")):
        for seed, ls in (("seed42", "-"), ("seed43", "--")):
            b = ctl[seed]["where"]["PP4_trained"]["bins"][v]
            ax.plot(x, [z["med_s_norm"] for z in b], ls, color=ps.C.vermillion, marker="o",
                    label=rf"predicted $\sigma$, median (seed {seed[-2:]})")
            ax.plot(x, [z["rmse_norm"] for z in b], ls, color=ps.C.blue, marker="s",
                    label=rf"actual error, RMS (seed {seed[-2:]})")
        ax.set_xlabel(xl)
        ax.set_ylabel("bin / whole probe")
    ps.legend(axa, "upper left")
    ps.save(fig, os.path.join(out_dir or os.path.join(HERE, "figs"), "het_where"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--runs", nargs="*", help="only these run ids (default: the six)")
    sm = sub.add_parser("summary")
    sm.add_argument("--reads", default=None, help="the run part's output (default data/het_reads.json.gz)")
    sm.add_argument("--out-dir", default=None, help="where the JSON and figure go (default data/, figs/)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.workers, a.runs)
    else:
        summary(getattr(a, "reads", None), getattr(a, "out_dir", None))


if __name__ == "__main__":
    main()
