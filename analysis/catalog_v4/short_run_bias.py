#!/usr/bin/env python3
"""Does a MadEvent run that stops short of its target ("fail to reach target") bias the events it returns?
(catalog_coverage, the catalog_v4 generation wave, 2026-10-30.)

    measure [--out F.json] [--min-events N] [--max-events M]     on the site (reads $SCRATCH/lhe_parts and the pools)
    fig     [--json F.json]                                       locally: the figure from the JSON

Samples. Every MadEvent part file under $SCRATCH/lhe_parts/<process>/ (one run: its events, the events it asked, in its
file name, and its banner), and the train pool of every serially built lhe_pp pool (pp13v4_*, no parts directory) of a
class that has parts, its runs read from its recipe sidecar (lhe_run.events_per_run against the events each asked).
A run is complete when it returned every event it asked, short otherwise, deep-short under 0.3 of its ask. A serial
pool counts as short when any of its runs is.

Like for like. Samples are grouped by the class (the process name without its pool prefix: pp13v4_, dsh_, ee4_, pp13_)
and by the physics of the run read from its banner: the generate lines and model (process card without its output
line), the run card without nevents, iseed, run_tag and the hidden second_refine_treshold, and the whole parameter
card. Two samples are compared only inside one such group, so a run size or a stop is the one thing that differs.

Measures, per group with enough events on both sides: two-sample Kolmogorov-Smirnov of short against complete on
sqrt(s_hat) (pp; the e+e- beams fix it) and on cos(theta*) of the first final-state particle in the partonic rest frame,
and the same for deep-short against complete; twice: over events (pooled, as if independent) and with the run as the
unit (run_perm: a run's events are not independent draws, its channels' shares come from its own integration, so
complete runs differ from each other by more than the event count allows; p from relabelling runs). Two controls in the same group: complete runs split in two by run
(alternate runs by seed), which calibrates the statistic under no bias, and complete runs of the group's smallest ask
against those of its largest when 3x apart (whether a run that completes depends on its size). Per-channel (subprocess)
shares: not measured; each pool is one partonic subprocess (fixed beam flavours) and neither the part files nor the
pools keep MadEvent's integration channel of an event.
"""
import argparse, glob, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "data", "short_run_bias.json")
FIG = os.path.join(HERE, "figs", "short_run_bias")
PART_RE = re.compile(r"^(?P<proc>.+)_(?P<eid>[0-9a-f]{16})_(?P<run>run_\d+)_seed(?P<seed>\d+)_ev(?P<ev>\d+)\.npz$")
PREFIXES = ("pp13v4_", "dsh_", "ee4_", "pp13_")
DEEP = 0.3                  # deep-short: under this fraction of the ask (the hand-off's threshold, open)

CAP = 200000                # events per side at most (a seeded subsample beyond)
NBIN = 20                   # quantile bins of the complete side, for the figure


def cls(name):
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _section(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    return m.group(1) if m else ""


def physics_key(banner):
    """sha256 of what fixes the events' density: process card less its output line, run card less nevents, iseed,
    run_tag and second_refine_treshold, the parameter card."""
    if not banner:
        return None
    proc = [l.split("#")[0].strip() for l in _section(banner, "MGProcCard").splitlines()]
    proc = [l for l in proc if l and not l.startswith("output")]
    run = []
    for l in _section(banner, "MGRunCard").splitlines():
        l = l.split("!")[0].strip()
        if not l or l.startswith("#"):
            continue
        if re.search(r"=\s*(nevents|iseed|run_tag|second_refine_treshold)\b", l):
            continue
        run.append(" ".join(l.split()))
    param = [" ".join(l.split("#")[0].split()) for l in _section(banner, "slha").splitlines()]
    param = [l for l in param if l]
    if not proc or not run or not param:
        return None
    return hashlib.sha256("\n".join(proc + ["--"] + run + ["--"] + param).encode()).hexdigest()[:16]


def _npz_len(path):
    """Rows of the part's mom array, read from its .npy header (no data loaded)."""
    import zipfile
    import numpy as np
    with zipfile.ZipFile(path) as z, z.open("mom.npy") as f:
        v = np.lib.format.read_magic(f)
        shape = (np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f))[0]
    return int(shape[0])


def features(P):
    """sqrt(s_hat) and cos(theta*) of the first final in the partonic rest frame (boost along z)."""
    import numpy as np
    q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0.0, None))
    beta = q[:, 3] / q[:, 0]
    g = 1.0 / np.sqrt(np.clip(1.0 - beta ** 2, 1e-300, None))
    f = P[:, 2]
    pz = g * (f[:, 3] - beta * f[:, 0])
    pabs = np.sqrt(f[:, 1] ** 2 + f[:, 2] ** 2 + pz ** 2)
    return {"sqrt_shat": sq, "cos_theta_star": pz / np.clip(pabs, 1e-300, None)}


def _ks(a, b):
    from scipy.stats import ks_2samp
    t = ks_2samp(a, b)
    n, m = len(a), len(b)
    crit = 1.358 * ((n + m) / (n * m)) ** 0.5            # the 5% critical value
    return {"ks": round(float(t.statistic), 5), "p": float(t.pvalue), "ks_over_crit5": round(float(t.statistic) / crit, 3),
            "n": n, "m": m}


def _ratio(short, comp):
    """short / complete density in NBIN quantile bins of the complete side, with the Poisson error of the short count."""
    import numpy as np
    edges = np.unique(np.quantile(comp, np.linspace(0, 1, NBIN + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = min(edges[0], short.min()), max(edges[-1], short.max())
    hs, _ = np.histogram(short, edges)
    hc, _ = np.histogram(comp, edges)
    exp = hc * (len(short) / len(comp))
    ok = exp > 0
    return {"edges": [float(x) for x in edges], "ratio": [float(r) if o else None for r, o in zip(hs / np.where(ok, exp, 1), ok)],
            "err": [float(e) if o else None for e, o in zip(np.sqrt(np.maximum(hs, 1)) / np.where(ok, exp, 1), ok)]}


def _samples():
    """[{process, class, key, asked, got, status, path, kind}] of every part file and serial pool."""
    import numpy as np
    sys.path.insert(0, ROOT)
    out = []
    pdir = os.path.join(os.environ["SCRATCH"], "lhe_parts")
    classes = set()
    for d in sorted(glob.glob(os.path.join(pdir, "*"))):
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            m = PART_RE.match(os.path.basename(f))
            if not m:
                continue
            try:
                got = _npz_len(f)
                with np.load(f) as z:
                    key = physics_key(str(z["banner"]))
            except Exception:
                continue
            asked = int(m["ev"])
            out.append({"process": m["proc"], "class": cls(m["proc"]), "key": key, "asked": asked, "got": got,
                        "seed": int(m["seed"]), "path": f, "kind": "part"})
            classes.add(cls(m["proc"]))
    # serial pp13v4 pools of the classes that have parts (their events are in the train pool, shuffled over runs)
    import mg5_pipeline_final as mg
    import datagen
    have_parts = {os.path.basename(d) for d in glob.glob(os.path.join(pdir, "*"))}
    for name, cfg in sorted(mg.PROCESSES.items()):
        if not name.startswith("pp13v4_") or name in have_parts or cls(name) not in classes or not mg.is_lhe_pp(name):
            continue
        try:
            lo, hi = mg.lhe_layout(cfg)["train"]
            rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role="train", seed=42)
            path = mg.recipe_output_path(rec, datagen.dest_for_role("train"))
            side = json.load(open(path + ".recipe.json"))
        except Exception:
            continue
        lr = side.get("lhe_run") or {}
        per = lr.get("events_per_run") or []
        runs = mg.lhe_runs(name, mg.lhe_iseed(name, 42), mg.lhe_n_requested(cfg))
        asked = [int(r[1]) for r in runs]
        if not per or len(per) != len(asked):
            continue
        b = lr.get("banner")
        key = physics_key(open(b).read()) if b and os.path.exists(b) else None
        out.append({"process": name, "class": cls(name), "key": key, "asked": sum(asked), "got": sum(per),
                    "runs_asked": asked, "runs_got": per, "seed": 0, "path": path, "kind": "pool",
                    "short_any": any(g < a for g, a in zip(per, asked)),
                    "deep_any": any(g < DEEP * a for g, a in zip(per, asked))})
    for s in out:
        if s["kind"] == "part":
            s["status"] = "complete" if s["got"] >= s["asked"] else ("deep" if s["got"] < DEEP * s["asked"] else "short")
        else:
            s["status"] = "deep" if s["deep_any"] else ("short" if s["short_any"] else "complete")
    return out


def _load_one(s, cap):
    """Momenta of one sample (a run's part, or a serial pool), a seeded subsample of at most `cap` events."""
    import numpy as np
    sys.path.insert(0, ROOT)
    import mg5_pipeline_final as mg
    if s["kind"] == "part":
        with np.load(s["path"]) as z:
            P = np.asarray(z["mom"], dtype=np.float64)
    else:
        x = np.load(s["path"], mmap_mode="r")
        k = int(mg.PROCESSES[s["process"]]["nfinal"]) + 2
        P = np.asarray(x[:, :4 * k], dtype=np.float64).reshape(len(x), k, 4)
    if len(P) > cap:
        P = P[np.sort(np.random.default_rng(s["seed"]).choice(len(P), cap, replace=False))]
    return P


def _binned_ks(H, a, b):
    """KS distance of the pooled bin counts of units `a` against units `b` (H: units x bins)."""
    import numpy as np
    ca, cb = H[a].sum(0), H[b].sum(0)
    return float(np.abs(np.cumsum(ca) / ca.sum() - np.cumsum(cb) / cb.sum()).max())


def run_perm(units_x, units_y, nperm=2000, nbin=50):
    """Run-level permutation test: a run's events are not independent draws (MadEvent fixes each channel's share of a
    run's events from that run's own integration), so the unit is the run. KS distance of the pooled binned
    distributions (nbin quantile bins of all events) of the x runs against the y runs; p from relabelling runs
    (every relabelling when there are fewer than nperm). None when either side has fewer than 2 runs."""
    import itertools, math
    import numpy as np
    nx, ny = len(units_x), len(units_y)
    if nx < 2 or ny < 2:
        return None
    allv = np.concatenate(units_x + units_y)
    edges = np.unique(np.quantile(allv, np.linspace(0, 1, nbin + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    H = np.array([np.histogram(u, edges)[0] for u in units_x + units_y], dtype=np.float64)
    idx = np.arange(nx + ny)
    obs = _binned_ks(H, idx[:nx], idx[nx:])
    total = math.comb(nx + ny, nx)
    if total <= nperm:
        sets = [np.array(c) for c in itertools.combinations(idx, nx)]
    else:
        rng = np.random.default_rng(0)
        sets = [rng.permutation(idx)[:nx] for _ in range(nperm)]
    ge = 0
    for a in sets:
        m = np.zeros(nx + ny, bool); m[a] = True
        ge += _binned_ks(H, idx[m], idx[~m]) >= obs - 1e-12
    return {"ks_binned": round(obs, 5), "p_runs": ge / len(sets), "runs_x": nx, "runs_y": ny,
            "relabellings": len(sets), "p_min": 1.0 / len(sets)}


def _feats_of(samples, per):
    return [features(_load_one(s, per)) for s in samples]


def _compare(X, Y, feats):
    """Pooled event KS and the run-level permutation test of feature arrays per run X against Y."""
    import numpy as np
    out = {}
    for f in feats:
        r = _ks(np.concatenate([x[f] for x in X]), np.concatenate([y[f] for y in Y]))
        rp = run_perm([x[f] for x in X], [y[f] for y in Y])
        if rp:
            r.update(rp)
        out[f] = r
    return out


def cmd_measure(a):
    import numpy as np
    S = _samples()
    groups = {}
    for s in S:
        if s["key"]:
            groups.setdefault((s["class"], s["key"]), []).append(s)
    inv = {"parts": sum(s["kind"] == "part" for s in S), "pools": sum(s["kind"] == "pool" for s in S),
           "no_banner_key": sum(not s["key"] for s in S),
           "parts_short": sum(s["kind"] == "part" and s["status"] != "complete" for s in S),
           "parts_deep": sum(s["kind"] == "part" and s["status"] == "deep" for s in S),
           "pools_short": sum(s["kind"] == "pool" and s["status"] != "complete" for s in S)}
    res = {}
    for (c, key), ss in sorted(groups.items()):
        comp = [s for s in ss if s["status"] == "complete"]
        short = [s for s in ss if s["status"] != "complete"]
        deep = [s for s in ss if s["status"] == "deep"]
        n_c, n_s = sum(s["got"] for s in comp), sum(s["got"] for s in short)
        if n_c < a.min_events or n_s < a.min_events:
            continue
        r = {"key": key, "processes": sorted({s["process"] for s in ss}),
             "complete": {"runs": len(comp), "events": n_c, "asked": sorted({s["asked"] for s in comp})},
             "short": {"runs": len(short), "events": n_s, "asked": sum(s["asked"] for s in short),
                       "yield": round(n_s / max(1, sum(s["asked"] for s in short)), 4),
                       "from_pools": sorted(s["process"] for s in short if s["kind"] == "pool")},
             "deep": {"runs": len(deep), "events": sum(s["got"] for s in deep)}}
        per = max(500, CAP // max(1, len(comp) + len(short)))
        Fc, Fs = _feats_of(comp, per), _feats_of(short, per)
        sq = np.concatenate([x["sqrt_shat"] for x in Fc])
        pp = float(np.std(sq)) > 1e-6 * float(np.mean(sq))
        feats = [f for f in ("sqrt_shat", "cos_theta_star") if f != "sqrt_shat" or pp]
        r["beam"] = "pp" if pp else "e+e-"
        r["short_vs_complete"] = _compare(Fs, Fc, feats)
        r["hist"] = {f: _ratio(np.concatenate([x[f] for x in Fs]), np.concatenate([x[f] for x in Fc])) for f in feats}
        if sum(s["got"] for s in deep) >= a.min_events // 2:
            r["deep_vs_complete"] = _compare([F for F, s in zip(Fs, short) if s["status"] == "deep"], Fc, feats)
        # control 1: complete runs split in two by run (alternate runs by seed)
        order = sorted(range(len(comp)), key=lambda i: comp[i]["seed"])
        A, B = [Fc[i] for i in order[0::2]], [Fc[i] for i in order[1::2]]
        if A and B:
            r["control_complete_halves"] = _compare(A, B, feats)
        # control 2: complete runs of the smallest ask against those of the largest (when 3x apart)
        asks = sorted({s["asked"] for s in comp})
        if len(asks) > 1 and asks[-1] >= 3 * asks[0]:
            sm = [F for F, s in zip(Fc, comp) if s["asked"] == asks[0]]
            lg = [F for F, s in zip(Fc, comp) if s["asked"] == asks[-1]]
            r["control_complete_small_vs_large"] = dict(_compare(lg, sm, feats), asks=[asks[0], asks[-1]])
        res[c] = r
        print(f"[short_run_bias] {c}: short {n_s} ({len(short)} runs) vs complete {n_c} ({len(comp)} runs): "
              + ", ".join(f"{f} KS {v['ks']} p {v['p']:.2g} p_runs {v.get('p_runs')}"
                          for f, v in r["short_vs_complete"].items()), flush=True)
    # every short source, compared or not (step 2's list): process -> runs short of their ask
    short_by_proc = {}
    for s in S:
        if s["status"] != "complete":
            short_by_proc.setdefault(s["process"], []).append([s["asked"], s["got"]])
    out = {"inventory": inv, "groups": res, "short_sources": short_by_proc,
           "settings": {"deep": DEEP, "cap": CAP, "min_events": a.min_events}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print("SHORTBIAS_JSON " + json.dumps({"inventory": inv, "groups": len(res), "out": a.out}), flush=True)


def summary(d, alpha=0.05):
    """Per comparison kind, over the groups and features where the run-level test can be made (2+ runs a side): the
    groups whose run-level p is under alpha after a Bonferroni correction over that kind's tests, and the event-level
    (pooled KS, events taken as independent) count for contrast."""
    out = {"groups": len(d["groups"])}
    for kind in ("short_vs_complete", "deep_vs_complete", "control_complete_halves", "control_complete_small_vs_large"):
        T = [(c, f, v) for c, r in d["groups"].items() for f, v in (r.get(kind) or {}).items() if isinstance(v, dict)]
        R = [(c, f, v) for c, f, v in T if "p_runs" in v]
        out[kind] = {"tests_events": len(T),
                     "event_level_p_below_bonf": sorted(f"{c}:{f}" for c, f, v in T if v["p"] < alpha / max(1, len(T))),
                     "tests_runs": len(R),
                     "run_level_p_below_bonf": sorted(f"{c}:{f}" for c, f, v in R if v["p_runs"] < alpha / max(1, len(R))),
                     "run_level_p_below_alpha": sorted(f"{c}:{f}" for c, f, v in R if v["p_runs"] < alpha)}
    return out


def cmd_fig(a):
    import numpy as np
    sys.path.insert(0, ROOT)
    import plot_style as ps
    import matplotlib.pyplot as plt
    d = json.load(open(a.json))
    ps.use()
    G = d["groups"]
    fig, axes = ps.figure(ncols=2, nrows=1)
    axes = np.atleast_1d(axes).ravel()
    # (a) KS of short against complete over the 5% critical value, per group and feature, against the control
    ax = axes[0]
    names = sorted(G, key=lambda c: -max(v["ks_over_crit5"] for v in G[c]["short_vs_complete"].values()))
    y = np.arange(len(names))
    for j, (f, mk) in enumerate((("sqrt_shat", "o"), ("cos_theta_star", "s"))):
        xs = [G[c]["short_vs_complete"].get(f, {}).get("ks_over_crit5", np.nan) for c in names]
        ax.plot(xs, y + 0.15 * (j - 0.5), mk, color=ps.CYCLE[j], ms=3.5, ls="none",
                label=r"$\sqrt{\hat s}$, stopped short" if f == "sqrt_shat" else r"$\cos\theta^*$, stopped short")
        xc = [(G[c].get("control_complete_halves") or {}).get(f, {}).get("ks_over_crit5", np.nan) for c in names]
        ax.plot(xc, y + 0.15 * (j - 0.5), mk, mfc="none", color=ps.C.grey, ms=3.5, ls="none",
                label="complete halves" if j == 0 else None)
    ax.axvline(1.0, color="k", lw=0.6, ls="--")
    ax.set_yticks(y, [c.replace("_", " ") for c in names], fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{\rm KS} / D_{5\%}$")
    ax.legend(fontsize=6, loc="lower right")
    # (b) density ratio short / complete in sqrt(s_hat) for the groups with the largest effect
    ax = axes[1]
    top = [c for c in names if G[c]["hist"].get("sqrt_shat")][:4]
    for i, c in enumerate(top):
        h = G[c]["hist"]["sqrt_shat"]
        e = np.asarray(h["edges"]); mid = np.sqrt(e[:-1] * e[1:]) if e[0] > 0 else 0.5 * (e[:-1] + e[1:])
        r = np.array([np.nan if v is None else v for v in h["ratio"]])
        er = np.array([np.nan if v is None else v for v in h["err"]])
        ax.errorbar(mid, r, er, color=ps.CYCLE[i], lw=0.8, ms=2.5, marker="o", label=c.replace("_", " "))
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sqrt{\hat s}$ [GeV]")
    ax.set_ylabel("stopped short / complete")
    ax.legend(fontsize=6)
    ps.layout(fig)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG + ".png", dpi=200); fig.savefig(FIG + ".pdf")
    print(json.dumps({"fig": FIG + ".png", **summary(d)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("--out", default=OUT)
    m.add_argument("--min-events", type=int, default=2000)
    f = sub.add_parser("fig"); f.add_argument("--json", default=OUT)
    s = sub.add_parser("summary"); s.add_argument("--json", default=OUT)
    a = ap.parse_args()
    if a.cmd == "summary":
        print(json.dumps(summary(json.load(open(a.json))), indent=1))
        return
    {"measure": cmd_measure, "fig": cmd_fig}[a.cmd](a)


if __name__ == "__main__":
    main()
