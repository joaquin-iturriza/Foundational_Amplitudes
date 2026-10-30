#!/usr/bin/env python3
"""Does a MadEvent run that stops short of its target ("fail to reach target") bias the events it returns?
(catalog_coverage, the catalog_v4 generation wave, 2026-10-30.)

    measure [--out F.json] [--min-events N] [--max-events M]     on the site (reads $SCRATCH/lhe_parts and the pools)
    fig     [--json F.json]                                       locally: the figure from the JSON
    sizesummary --json F.json                                     locally: a sizecheck's outcome per run size

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
import argparse, glob, hashlib, json, math, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "data", "short_run_bias.json")
COMPLETE = os.path.join(HERE, "data", "complete_runs.json")
PENDING = os.path.join(HERE, "data", "complete_runs_pending.json")
FIG = os.path.join(HERE, "figs", "short_run_bias")
PART_RE = re.compile(r"^(?P<proc>.+)_(?P<eid>[0-9a-f]{16})_(?P<run>run_\d+)_seed(?P<seed>\d+)_ev(?P<ev>\d+)\.npz$")
PROBE_RE = re.compile(r"^(?P<proc>.+)_probe_seed(?P<seed>\d+)_ev(?P<ev>\d+)\.npz$")
GPACK_RE = re.compile(r"^(?P<proc>.+)_gpack(?P<grid>\d+)_seed(?P<seed>\d+)_ev(?P<ev>\d+)\.npz$")
PREFIXES = ("pp13v4_", "dsh_", "ee4_", "pp13_")
DEEP = 0.3                  # deep-short: under this fraction of the ask (the hand-off's threshold, open)

SIZE_SMALL, SIZE_LARGE = 1500, 5000   # run-size test: complete runs asking at most / at least this many (open)
CAP = 200000                # events per side at most (a seeded subsample beyond)
NPERM = 2000                # run-level relabellings (--nperm)
SCALED = True               # run-level p from the scaled KS distance (cdcae2cd); --unscaled re-reads the old test
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
        if re.search(r"=\s*(nevents|iseed|run_tag|second_refine_treshold|gridpack)\b", l):
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


def _samples(before=None):
    """[{process, class, key, asked, got, status, path, kind, mtime}] of every part file and serial pool; with `before`
    (epoch seconds) the files as they were then (_select), so an earlier measurement is re-read on the files it saw."""
    S = _samples_all()
    if before is None:                                     # today's files: the pools at their paths
        return _select([s for s in S if s.get("current", True)], float("inf"),
                       {os.path.basename(d) for d in glob.glob(os.path.join(os.environ["SCRATCH"], "lhe_parts", "*"))})
    return _select(S, before)


def _select(S, t, have_parts=None):
    """The samples on disk at time t: parts written by then; per serial pool the version that sat at its path then (the
    latest written by t, the current file or one set aside later by lhe_set_aside), and none for a process whose parts
    had started by then (it was then built from parts, not in series) or whose class had no part yet (parts of
    lhe_parts only, not the probes' kept runs). have_parts (today's files): the process dirs under lhe_parts, the rule
    the measurements were made with. A set-aside version left its path at its set_aside stamp ("until")."""
    parts = [s for s in S if s["kind"] == "part" and s["mtime"] <= t]
    own = [s for s in parts if not s.get("probe") and not s.get("gridpack")]
    procs = have_parts if have_parts is not None else {s["process"] for s in own}
    classes = {s["class"] for s in own}
    best = {}
    for s in S:
        if s["kind"] == "pool" and s["mtime"] <= t < s.get("until", float("inf")) and s["process"] not in procs \
                and s["class"] in classes:
            if s["process"] not in best or s["mtime"] > best[s["process"]]["mtime"]:
                best[s["process"]] = s
    return parts + sorted(best.values(), key=lambda s: s["process"])


def _samples_all():
    """Every part file and every version of every serial pool (the file at its path, "current", and the copies
    lhe_set_aside moved to <pool dir>/set_aside/), each with its mtime (shutil.move keeps it)."""
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
                        "seed": int(m["seed"]), "path": f, "kind": "part", "mtime": os.path.getmtime(f)})
            classes.add(cls(m["proc"]))
    # complete runs kept by the run-size probe (tools/lhe_probe.py run --keep: <process>_probe_seed<iseed>_ev<n>.npz)
    # and runs drawn from one gridpack (tools/lhe_probe.py gridpack --keep: <process>_gpack<grid iseed>_seed<s>_ev<n>.npz)
    for f in sorted(glob.glob(os.path.join(os.environ["SCRATCH"], "lhe_probe_keep", "*", "*.npz"))):
        m, g = PROBE_RE.match(os.path.basename(f)), GPACK_RE.match(os.path.basename(f))
        if not (m or g):
            continue
        try:
            got = _npz_len(f)
            with np.load(f) as z:
                key = physics_key(str(z["banner"]))
        except Exception:
            continue
        x = m or g
        s = {"process": x["proc"], "class": cls(x["proc"]), "key": key, "asked": int(x["ev"]), "got": got,
             "seed": int(x["seed"]), "path": f, "kind": "part", "mtime": os.path.getmtime(f)}
        s.update({"probe": True} if m else {"gridpack": int(g["grid"])})
        out.append(s)
    # serial pp13v4 pools of the classes that have parts (their events are in the train pool, shuffled over runs);
    # a version's runs asked from its own sidecar's lhe section (run_events, else n_requested: pp runs all ask one size)
    import mg5_pipeline_final as mg
    import datagen
    for name, cfg in sorted(mg.PROCESSES.items()):
        if not name.startswith("pp13v4_") or not mg.is_lhe_pp(name):
            continue
        try:
            lo, hi = mg.lhe_layout(cfg)["train"]
            rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role="train", seed=42)
            path = mg.recipe_output_path(rec, datagen.dest_for_role("train"))
            runs = mg.lhe_runs(name, mg.lhe_iseed(name, 42), mg.lhe_n_requested(cfg))
        except Exception:
            continue
        ask_of = {int(r[0]): int(r[1]) for r in runs}
        aside = glob.glob(os.path.join(os.path.dirname(path), "set_aside", f"{name}_*", os.path.basename(path)))
        for v in [path] + sorted(aside):
            try:
                side = json.load(open(v + ".recipe.json"))
            except Exception:
                continue
            lr, lv = side.get("lhe_run") or {}, side.get("lhe") or {}
            per, seeds = lr.get("events_per_run") or [], lr.get("iseeds") or []
            own_ask = lv.get("run_events") or lv.get("n_requested")
            if not per or len(per) != len(seeds) or (not own_ask and (v != path or any(int(i) not in ask_of
                                                                                       for i in seeds))):
                continue
            asked = [int(own_ask) if own_ask else ask_of[int(i)] for i in seeds]
            b = lr.get("banner")
            if b and v != path:                            # a set-aside version: its own banner copy, moved with it
                b = os.path.join(os.path.dirname(v), os.path.basename(b))
            key = physics_key(open(b).read()) if b and os.path.exists(b) else None
            until = float("inf")
            if v != path:                                  # <process>_<%Y%m%dT%H%M%S, the cluster's local time>
                try:
                    until = time.mktime(time.strptime(os.path.basename(os.path.dirname(v)).rsplit("_", 1)[1],
                                                      "%Y%m%dT%H%M%S"))
                except (ValueError, IndexError):
                    continue
            out.append({"process": name, "class": cls(name), "key": key, "asked": sum(asked), "got": sum(per),
                        "runs_asked": asked, "runs_got": per, "seed": 0, "path": v, "kind": "pool",
                        "mtime": os.path.getmtime(v), "current": v == path, "until": until,
                        "short_any": any(g < a for g, a in zip(per, asked)),
                        "deep_any": any(g < DEEP * a for g, a in zip(per, asked))})
    for s in out:
        if s["kind"] == "part":
            s["status"] = "complete" if s["got"] >= s["asked"] else ("deep" if s["got"] < DEEP * s["asked"] else "short")
        else:
            s["status"] = "deep" if s["deep_any"] else ("short" if s["short_any"] else "complete")
    return out


def _side_counts(S):
    """Per (class, physics key): the counts a measured group records (complete, short, deep runs and events; the
    processes), from the samples alone, so a time at which the files match a measurement can be found (cutoff)."""
    g = {}
    for s in S:
        if not s["key"] or s.get("gridpack"):
            continue
        e = g.setdefault((s["class"], s["key"]), {"complete": [0, 0], "short": [0, 0], "deep": [0, 0], "processes": set()})
        e["processes"].add(s["process"])
        e["complete" if s["status"] == "complete" else "short"][0] += 1
        e["complete" if s["status"] == "complete" else "short"][1] += s["got"]
        if s["status"] == "deep":
            e["deep"][0] += 1; e["deep"][1] += s["got"]
    return g


def cmd_cutoff(a):
    """The time at which the files on disk matched an earlier measurement (--old): every distinct file time in the
    --hours before --before is tried, each measured group's complete, short and deep runs and events and its processes
    compared with what the files at that time give; the earliest time matching the most groups, and the groups
    matching at none. One JSON line (CUTOFF_JSON)."""
    S = _samples_all()
    for old, before in zip(a.old, a.before):
        _cutoff_one(S, old, before, a.hours)


def _cutoff_one(S, old, before, hours):
    import datetime
    o = json.load(open(old))["groups"]
    want = {(n.split("#")[0], r["key"]): (n, r) for n, r in o.items()}
    hi = _epoch(before); lo = hi - 3600 * hours
    ts = sorted({s["mtime"] for s in S if lo <= s["mtime"] <= hi} | {lo, hi})
    def score(t):
        g = _side_counts(_select(S, t))
        ok = []
        for (c, key), (n, r) in want.items():
            e = g.get((c, key))
            if e and all(e[k] == [r[k]["runs"], r[k]["events"]] for k in ("complete", "short", "deep")) \
                    and sorted(e["processes"]) == r["processes"]:
                ok.append(n)
        return ok
    best, bt, ever = [], None, set()
    for t in ts:
        ok = score(t)
        ever |= set(ok)
        if len(ok) > len(best):
            best, bt = ok, t
    iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()
    print("CUTOFF_JSON " + json.dumps({"old": old, "groups": len(want), "times_tried": len(ts),
                                       "best_time": iso(bt) if bt else None, "matched": len(best),
                                       "unmatched_at_best": sorted(set(n for n, _ in want.values()) - set(best)),
                                       "never_matched": sorted(set(n for n, _ in want.values()) - ever),
                                       "set_aside_versions": sum(s["kind"] == "pool" and not s.get("current", True)
                                                                 for s in S)}), flush=True)

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


def _detectable(X, Y):
    """The KS distance an event-level two-sample test detects at 5 percent for the pooled sizes (a floor: the run-level
    test, with runs as units, needs at least this)."""
    n, m = sum(len(next(iter(x.values()))) for x in X), sum(len(next(iter(y.values()))) for y in Y)
    return round(1.358 * ((n + m) / (n * m)) ** 0.5, 4)


def _binned_ks(H, a, b, scaled=False):
    """KS distance of the pooled bin counts of units `a` against units `b` (H: units x bins); scaled: times
    sqrt(n_a n_b / (n_a + n_b)), the event counts of the two sides, so that relabellings that move events between
    sides (units of unequal size) are compared on the scale of their own sampling noise."""
    import numpy as np
    ca, cb = H[a].sum(0), H[b].sum(0)
    d = float(np.abs(np.cumsum(ca) / ca.sum() - np.cumsum(cb) / cb.sum()).max())
    if scaled:
        na, nb = float(ca.sum()), float(cb.sum())
        d *= (na * nb / (na + nb)) ** 0.5
    return d


def run_perm(units_x, units_y, nperm=None, nbin=50):
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
    # the p compares scaled distances: with the raw distance, a relabelling that hands the side of fewer events more
    # events (units of unequal size) has a narrower null, and p comes out too small (catalog_coverage 2026-10-30)
    nperm = nperm or NPERM
    obs_s = _binned_ks(H, idx[:nx], idx[nx:], scaled=SCALED)
    total = math.comb(nx + ny, nx)
    if total <= nperm:
        sets = [np.array(c) for c in itertools.combinations(idx, nx)]
    else:
        rng = np.random.default_rng(0)
        sets = [rng.permutation(idx)[:nx] for _ in range(nperm)]
    ge = 0
    for a in sets:
        m = np.zeros(nx + ny, bool); m[a] = True
        ge += _binned_ks(H, idx[m], idx[~m], scaled=SCALED) >= obs_s - 1e-9
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


def _epoch(t):
    """ISO time (with its offset) -> epoch seconds; None stays None."""
    import datetime
    return None if not t else datetime.datetime.fromisoformat(t).timestamp()


def cmd_measure(a):
    import numpy as np
    S = _samples(_epoch(a.before))
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
        gp = [s for s in ss if s.get("gridpack") and s["status"] == "complete"]    # compared only in the ladder below
        ss = [s for s in ss if not s.get("gridpack")]
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
        # run size: complete runs of each small ask (at most SIZE_SMALL events) against every complete run of at least
        # SIZE_LARGE, and the complete-halves control of the small runs themselves
        lg = [F for F, s in zip(Fc, comp) if s["asked"] >= SIZE_LARGE]
        for ask in sorted({s["asked"] for s in comp if s["asked"] <= SIZE_SMALL}):
            sm = [F for F, s in zip(Fc, comp) if s["asked"] == ask]
            if lg and len(sm) >= 2:
                r[f"size_{ask}"] = dict(_compare(sm, lg, feats), runs_small=len(sm), runs_large=len(lg),
                                        detectable_ks_5pct=_detectable(sm, lg))
        # size ladder: complete runs of each ask against those of the group's largest ask that has 2+ runs, so a size
        # is tested against the largest runs alone rather than a mix of large sizes; keyed "<ask>/<feature>"
        by = {}
        for F, s in zip(Fc, comp):
            by.setdefault(s["asked"], []).append(F)
        top = max((a for a in by if len(by[a]) >= 2), default=None)   # the largest ask with 2+ runs
        if top:
            lad = {}
            for ask in sorted(by):
                if ask < top and len(by[ask]) >= 2:
                    for f, v in _compare(by[ask], by[top], feats).items():
                        lad[f"{ask}/{f}"] = dict(v, ask=ask, ask_large=top, runs_small=len(by[ask]),
                                                 runs_large=len(by[top]),
                                                 detectable_ks_5pct=_detectable(by[ask], by[top]))
            if lad:
                r["ladder"] = lad
            # the reference itself: the largest runs split in two by seed (do independent large runs agree?)
            ts = [F for _, F in sorted(((s["seed"], F) for F, s in zip(Fc, comp) if s["asked"] == top),
                                       key=lambda t: t[0])]
            if len(ts) >= 4:
                r["control_top_halves"] = _compare(ts[0::2], ts[1::2], feats)
            # runs drawn from gridpacks: per ask, all the group's gridpacks pooled against the same largest runs ("all")
            # and against the per-run-integrated complete runs of the same ask (the change itself), each gridpack alone
            # against the largest runs ("g<iseed>"), the two first gridpacks against each other, and each split in two
            gl, gv, gpair, gh = {}, {}, {}, {}
            for ask in sorted({s["asked"] for s in gp}):
                grids = sorted({s["gridpack"] for s in gp if s["asked"] == ask})
                FG = {g: _feats_of(sorted([s for s in gp if s["gridpack"] == g and s["asked"] == ask],
                                          key=lambda s: s["seed"]), per) for g in grids}
                allg = [F for g in grids for F in FG[g]]
                meta = lambda X, Y: dict(ask=ask, runs_small=len(X), runs_large=len(Y), detectable_ks_5pct=_detectable(X, Y))
                for tag, X in [("all", allg)] + [(f"g{g}", FG[g]) for g in grids]:
                    if len(X) < 2 or (tag != "all" and len(grids) < 2):
                        continue
                    for f, v in _compare(X, by[top], feats).items():
                        gl[f"{tag}/{ask}/{f}"] = dict(v, ask_large=top, **meta(X, by[top]))
                own = by.get(ask, [])
                if len(allg) >= 2 and len(own) >= 2:
                    for f, v in _compare(allg, own, feats).items():
                        gv[f"{ask}/{f}"] = dict(v, **meta(allg, own))
                if len(grids) >= 2 and min(len(FG[g]) for g in grids[:2]) >= 2:
                    for f, v in _compare(FG[grids[0]], FG[grids[1]], feats).items():
                        gpair[f"{ask}/{f}"] = dict(v, **meta(FG[grids[0]], FG[grids[1]]))
                for g in grids:
                    if len(FG[g]) >= 4:
                        for f, v in _compare(FG[g][0::2], FG[g][1::2], feats).items():
                            gh[f"g{g}/{ask}/{f}"] = v
            for k, v in (("gridpack", gl), ("gridpack_vs_own_runs", gv), ("gridpack_pair", gpair),
                         ("control_gridpack_halves", gh)):
                if v:
                    r[k] = v
        res[c if c not in res else f"{c}#{key[:6]}"] = r
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


def _cp(v, conf=0.95):
    """Clopper-Pearson interval of a relabelling p (p_runs over its relabellings); None without one."""
    from scipy.stats import beta
    if not isinstance(v, dict) or v.get("p_runs") is None or not v.get("relabellings"):
        return None
    n = int(v["relabellings"]); k = int(round(v["p_runs"] * n)); a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [round(lo, 5), round(hi, 5)]


def cmd_reread(a):
    """An earlier measurement (--old) against its re-read (--new, same files, a changed test): every test present in
    both (a dict with an event-level "ks"), keyed by its path; whether the event-level KS reproduces (the check that the
    re-read saw the same events) and the run-level p before and after. One JSON line."""
    o, n = json.load(open(a.old)), json.load(open(a.new))
    rows, same, diff = {}, 0, 0

    def walk(x, y, path):
        nonlocal same, diff
        if isinstance(x, dict) and isinstance(y, dict):
            if "ks" in x and "ks" in y and not isinstance(x["ks"], dict):
                ok = abs(float(x["ks"]) - float(y["ks"])) < 1e-9
                same += ok; diff += not ok
                if "p_runs" in x or "p_runs" in y:
                    rows[path] = {"ks_event_same": ok, "p_runs_old": x.get("p_runs"), "p_runs_new": y.get("p_runs"),
                                  "ci_old": _cp(x), "ci_new": _cp(y)}
                return
            for k in x:
                if k in y and k != "hist":
                    walk(x[k], y[k], f"{path}/{k}" if path else k)
    walk(o.get("groups", o), n.get("groups", n), "")
    # why a test did not reproduce: per group of the old measurement, its sides' runs and events against the re-read's
    # (a group gone from the re-read, or a side of other runs or events, means the files it saw are not those on disk)
    sides = {}
    og, ng = o.get("groups") or {}, n.get("groups") or {}
    bad = {p.split("/")[0] for p, r in rows.items() if not r["ks_event_same"]}
    for g, r in og.items():
        if g not in ng:
            sides[g] = "missing in re-read"
            continue
        if g not in bad:
            continue
        sides[g] = {s: {k: [r[s].get(k), ng[g].get(s, {}).get(k)] for k in ("runs", "events", "asked")
                        if r[s].get(k) != ng[g].get(s, {}).get(k)}
                    for s in ("complete", "short", "deep") if isinstance(r.get(s), dict)}
        sides[g]["processes"] = [r.get("processes"), ng[g].get("processes")]
    print("REREAD_JSON " + json.dumps({"event_ks_same": same, "event_ks_differ": diff, "tests": rows,
                                       "groups_new_only": sorted(set(ng) - set(og)), "sides": sides}), flush=True)


def summary(d, alpha=0.05):
    """Per comparison kind, over the groups and features where the run-level test can be made (2+ runs a side): the
    groups whose run-level p is under alpha after a Bonferroni correction over that kind's tests, and the event-level
    (pooled KS, events taken as independent) count for contrast."""
    out = {"groups": len(d["groups"])}
    sizes = sorted({k for r in d["groups"].values() for k in r if k.startswith("size_")})
    for kind in ("short_vs_complete", "deep_vs_complete", "control_complete_halves",
                 "control_complete_small_vs_large", *sizes, "ladder", "control_top_halves", "gridpack",
                 "gridpack_vs_own_runs", "gridpack_pair", "control_gridpack_halves"):
        T = [(c, f, v) for c, r in d["groups"].items() for f, v in (r.get(kind) or {}).items() if isinstance(v, dict)]
        R = [(c, f, v) for c, f, v in T if "p_runs" in v]
        out[kind] = {"tests_events": len(T),
                     "event_level_p_below_bonf": sorted(f"{c}:{f}" for c, f, v in T if v["p"] < alpha / max(1, len(T))),
                     "tests_runs": len(R),
                     "run_level_p_below_bonf": sorted(f"{c}:{f}" for c, f, v in R if v["p_runs"] < alpha / max(1, len(R))),
                     "run_level_p_below_alpha": sorted(f"{c}:{f}" for c, f, v in R if v["p_runs"] < alpha)}
        o = out[kind]
        o.update(n_bonf=len(o["run_level_p_below_bonf"]), n_alpha=len(o["run_level_p_below_alpha"]),
                 n_event_bonf=len(o["event_level_p_below_bonf"]))
    lad = {}
    for c, r in d["groups"].items():
        for k, v in (r.get("ladder") or {}).items():
            if "p_runs" in v:
                lad.setdefault(c, {})[k] = {"ks": round(v["ks_binned"], 3), "p": v["p_runs"], "runs": v["runs_small"],
                                            "ref": v["ask_large"], "ref_runs": v["runs_large"]}
    # the gridpack probe (catalog_coverage step 7): per group, the runs of its gridpacks pooled against the largest
    # complete runs (primary, Bonferroni over the primary tests only: GPACK_PRIMARY), the direct test against the
    # per-run-integrated runs of the same ask, the two gridpacks against each other
    gp = {}
    for c, r in d["groups"].items():
        for k, v in (r.get("gridpack") or {}).items():
            if k.startswith("all/") and "p_runs" in v:
                f = k.split("/")[-1]
                e = gp.setdefault(c, {})
                e[f] = {"ks": round(v["ks_binned"], 4), "p": v["p_runs"], "detectable": v["detectable_ks_5pct"],
                        "ask": v["ask"],
                        "runs": v["runs_small"], "ref": v["ask_large"], "ref_runs": v["runs_large"]}
                for kind, key in (("direct", "gridpack_vs_own_runs"), ("pair", "gridpack_pair")):
                    w = (r.get(key) or {}).get(k[len("all/"):])
                    if w and "p_runs" in w:
                        e[f][kind] = {"ks": round(w["ks_binned"], 4), "p": w["p_runs"]}
                w = (r.get("ladder") or {}).get(k[len("all/"):])
                if w and "p_runs" in w:
                    e[f]["per_run"] = {"ks": round(w["ks_binned"], 4), "p": w["p_runs"]}
    if gp:
        n = sum(len(v) for v in gp.values())
        out["gridpack_detail"] = gp
        out["gridpack_primary_tests"] = n
        out["gridpack_level"] = round(alpha / max(1, n), 5)
        out["gridpack_below_level"] = sorted(f"{c}:{f}" for c, v in gp.items() for f, x in v.items()
                                             if x["p"] < alpha / max(1, n))
        pp = [x for c, v in gp.items() if not c.startswith("epem_") for x in v.values()]
        th = [w["ks_binned"] for c in gp if not c.startswith("epem_")
              for w in (d["groups"][c].get("control_top_halves") or {}).values() if isinstance(w, dict)]
        out["gridpack_n_below"] = sum(x["p"] < alpha / max(1, n) for v in gp.values() for x in v.values())
        out["gridpack_pair_n_below"] = sum(x.get("pair", {}).get("p", 1) < alpha / max(1, n)
                                           for v in gp.values() for x in v.values())
        # pp groups: the gridpack distances, the large runs' own halves, and the tests under alpha uncorrected
        out["gridpack_pp_ks_min"], out["gridpack_pp_ks_max"] = (round(min(x["ks"] for x in pp), 3),
                                                                round(max(x["ks"] for x in pp), 3))
        if th:
            out["top_halves_pp_ks_min"], out["top_halves_pp_ks_max"] = round(min(th), 3), round(max(th), 3)
        out["gridpack_pp_tests"] = len(pp)
        out["gridpack_pp_below_alpha"] = sum(x["p"] < alpha for x in pp)
        out["gridpack_pair_below_level"] = sorted(f"{c}:{f}" for c, v in gp.items() for f, x in v.items()
                                                  if x.get("pair", {}).get("p", 1) < alpha / max(1, n))
    if lad:
        out["ladder_detail"] = lad
        out["ladder_bonf"] = round(alpha / max(1, out["ladder"]["tests_runs"]), 5)
    out["inventory"] = d["inventory"]
    g = d["groups"].get("uu_aguu")
    if g:
        h = g["hist"]["sqrt_shat"]
        out["uu_aguu"] = {"ks": round(g["short_vs_complete"]["sqrt_shat"]["ks"], 3),
                          "short_runs": g["short"]["runs"], "complete_runs": g["complete"]["runs"],
                          "ratio_top2": round(sum(h["ratio"][-2:]) / 2, 2), "edge_top2": round(h["edges"][-3])}
    if os.path.exists(COMPLETE):
        c = json.load(open(COMPLETE))
        out["rebuild"] = {"pools": len(c), "short_built": sum("stopped short" in v["reason"] for v in c.values()),
                          "gridpack": sum(bool(v.get("gridpack")) for v in c.values()),
                          "after_w3_short_built": sum("after catalog_v4_w3" in v["reason"] for v in c.values()),
                          "w3_unassembled": sum(v["reason"].startswith("not assembled in catalog_v4_w3")
                                                for v in c.values())}
        out["rebuild"]["design"] = (out["rebuild"]["pools"] - out["rebuild"]["after_w3_short_built"]
                                    - out["rebuild"]["w3_unassembled"])
        # pools found after catalog_v4_w3 with the same problem, waiting on the user's call (complete_runs_pending.json)
        pend = json.load(open(PENDING)) if os.path.exists(PENDING) else {}
        out["rebuild"]["after_w3_short_built"] += sum("after catalog_v4_w3" in v["reason"] for v in pend.values())
        out["rebuild"]["w3_unassembled"] += sum(v["reason"].startswith("not assembled in catalog_v4_w3")
                                                for v in pend.values())
        out["rebuild"]["added_after_w3"] = len(pend)
        out["rebuild"]["with_pending"] = len(c) + len(pend)
    return out


_TOK = [("wp", "W^+"), ("wm", "W^-"), ("tap", r"\tau^+"), ("tam", r"\tau^-"), ("mup", r"\mu^+"), ("mum", r"\mu^-"),
        ("ep", "e^+"), ("em", "e^-"), ("ux", r"\bar u"), ("dx", r"\bar d"), ("bx", r"\bar b"), ("cx", r"\bar c"),
        ("sx", r"\bar s"), ("tx", r"\bar t"), ("vt", r"\nu_\tau"), ("vm", r"\nu_\mu"), ("ve", r"\nu_e"),
        ("a", r"\gamma"), ("z", "Z"), ("h", "H")] + [(q, q) for q in "udscbtg"]


def proc_label(c):
    """A class name (uu_aguu) as its process in math ($uu\\to\\gamma guu$)."""
    def toks(x):
        out = []
        while x:
            t = next(((k, v) for k, v in _TOK if x.startswith(k)), (x[0], x[0]))
            out.append(t[1]); x = x[len(t[0]):]
        return " ".join(out)
    if c.startswith("epem_"):
        return "$e^+e^-\\to " + toks(c[5:]) + "$"
    b, f = c.split("_", 1)
    return "$" + toks(b) + "\\to " + toks(f) + "$"


def _pvals(G, kind):
    import numpy as np
    return np.sort([v["p_runs"] for r in G.values() for v in (r.get(kind) or {}).values()
                    if isinstance(v, dict) and "p_runs" in v])


def cmd_fig(a):
    """(a) run-level p of every test against its rank's expectation under no difference (uniform p): stopped-short
    runs against complete runs, deep-short ones, and the control (complete runs split in two). (b) The density of
    stopped-short over complete runs' events in sqrt(s_hat), in quantile bins of the complete ones, for the groups
    whose run-level p is lowest. Both png and pdf (plot_style.save)."""
    import numpy as np
    sys.path.insert(0, ROOT)
    import plot_style as ps
    d = json.load(open(a.json))
    ps.use()
    G = d["groups"]
    (fa, ax), (fb, bx) = ps.panels(2)
    for kind, lab, col, mk in (("short_vs_complete", "stopped short", ps.C.vermillion, "o"),
                               ("deep_vs_complete", r"returned $<0.3$ of ask", ps.C.orange, "^"),
                               ("control_complete_halves", "complete halves", ps.C.grey, "s")):
        p = np.maximum(_pvals(G, kind), 2.5e-4)
        if not len(p):
            continue
        q = (np.arange(1, len(p) + 1) - 0.5) / len(p)
        ax.plot(-np.log10(q), -np.log10(p), mk, color=col, ms=3, ls="none", label=lab)
    lim = 3.8
    ax.plot([0, lim], [0, lim], color="k", lw=0.6, ls="--")
    ax.set_xlim(0, 2.2); ax.set_ylim(0, lim)
    ax.set_xlabel(r"$-\log_{10} p$ expected")
    ax.set_ylabel(r"$-\log_{10} p$, run level")
    ax.legend(loc="upper left")
    best = sorted((v["p_runs"], c) for c, r in G.items() if r["beam"] == "pp"
                  for f, v in r["short_vs_complete"].items() if f == "sqrt_shat" and "p_runs" in v)[:3]
    for i, (_, c) in enumerate(best):
        h = G[c]["hist"]["sqrt_shat"]
        e = np.asarray(h["edges"]); mid = np.sqrt(e[:-1] * e[1:])
        r = np.array([np.nan if v is None else v for v in h["ratio"]])
        er = np.array([np.nan if v is None else v for v in h["err"]])
        bx.errorbar(mid, r, er, color=ps.CYCLE[i], lw=0.8, ms=2.5, marker="o", capsize=0,
                    label=proc_label(c))
    bx.axhline(1.0, color="k", lw=0.6, ls="--")
    bx.set_xscale("log")
    bx.set_xlabel(r"$\sqrt{\hat s}$ [GeV]")
    bx.set_ylabel("stopped short / complete")
    bx.legend(loc="upper left")
    ps.save_panels([fa, fb], FIG)
    print(json.dumps({"fig": [FIG + "_a.png", FIG + "_b.png"], "panel_b": [c for _, c in best], **summary(d)}))


PROBED = (150, 200, 1000)   # the run-size probe's asks


def cmd_figsize(a):
    """Run size: the run-level binned KS distance of complete runs of each probed size against the complete runs of the
    group's largest ask with two or more runs (the ladder), per group and variable; filled where the run-level p is under
    the Bonferroni level over every ladder test. png and pdf (plot_style.save)."""
    import numpy as np
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    d = json.load(open(a.json))
    sm = summary(d)
    bonf = sm["ladder_bonf"]
    ps.use()
    fig, ax = ps.figure()
    groups = [c for c, r in d["groups"].items() if any(v["ask"] in PROBED for v in (r.get("ladder") or {}).values())]
    handles = []
    for i, c in enumerate(sorted(groups)):
        col = ps.CYCLE[i]
        for k, v in d["groups"][c]["ladder"].items():
            if "p_runs" not in v:
                continue
            mk = "o" if k.endswith("sqrt_shat") else "^"
            ax.plot([v["ask"]], [v["ks_binned"]], mk, color=col, mfc=col if v["p_runs"] < bonf else "none", ls="none")
        # runs of the same ask drawn from one gridpack (pooled over the group's two), just right of their ask
        for f, v in (sm.get("gridpack_detail") or {}).get(c, {}).items():
            mk = "s" if f == "sqrt_shat" else "D"
            ax.plot([v["ask"] * 1.12], [v["ks"]], mk, color=col, mfc=col if v["p"] < sm["gridpack_level"] else "none",
                    ls="none")
        handles.append(Patch(color=col, label=proc_label(c)))
    nt = sm["ladder"]["tests_runs"]
    handles += [Line2D([], [], color="k", marker="o", mfc="none", ls="none", label=r"$\sqrt{\hat s}$"),
                Line2D([], [], color="k", marker="^", mfc="none", ls="none", label=r"$\cos\theta^*$"),
                Line2D([], [], color=ps.C.grey, marker="o", ls="none", label=rf"$p_{{\rm runs}}<0.05/{nt}$")]
    if sm.get("gridpack_detail"):
        handles += [Line2D([], [], color="k", marker="s", mfc="none", ls="none", label=r"$\sqrt{\hat s}$, gridpack"),
                    Line2D([], [], color="k", marker="D", mfc="none", ls="none", label=r"$\cos\theta^*$, gridpack")]
    ax.set_xscale("log")
    ax.set_xticks(list(PROBED)); ax.set_xticklabels([str(x) for x in PROBED])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("events per run")
    ax.set_ylabel(r"KS distance to the largest runs")
    ax.set_ylim(0, None)
    ps.legend(ax, "upper right", handles=handles, ncol=2)
    ps.make_room(ax)
    ps.save(fig, FIG + "_size")
    print(json.dumps({"fig": FIG + "_size.png", "groups": groups, **sm}))


def _ebeam(banner):
    """ebeam1 of a banner's run card (None when absent)."""
    m = re.search(r"^\s*([0-9.eE+-]+)\s*=\s*ebeam1\b", _section(banner, "MGRunCard"), re.M)
    return float(m.group(1)) if m else None


def _gpack_rows(keep, process, energy):
    """The runs `lhe_probe.py gridpool --keep` saved under KEEP/<process>/ at collider energy `energy`: path, ask, seed,
    gridpack, events returned and physics key (one per run)."""
    import numpy as np
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.expandvars(keep), process, "*.npz"))):
        m = GPACK_RE.match(os.path.basename(f))
        if not (m and m["proc"] == process):
            continue
        with np.load(f) as z:
            b = str(z["banner"])
        e = _ebeam(b)
        if e is None or abs(2.0 * e - energy) > 1e-6:
            continue
        rows.append({"path": f, "asked": int(m["ev"]), "seed": int(m["seed"]), "src": f"g{m['grid']}", "kind": "part",
                     "got": _npz_len(f), "key": physics_key(b)})
    return rows


def _armcheck(a):
    """sizecheck --ref-keep: the runs under --keep (one arm of a probe, e.g. other gridpack generation limits) against
    the runs of --ref-ask under --ref-keep from the same gridpack and cards: the arm's complete share per ask from its
    logs, its runs (all of them, and its complete ones) against the reference's complete runs (vs_ref) and against the
    reference's short runs (vs_ref_short), and its runs split in two by seed (control_arm_halves). Refused when the two
    directories hold other physics or other gridpacks."""
    A, R = _gpack_rows(a.keep, a.process, a.energy), _gpack_rows(a.ref_keep, a.process, a.energy)
    keys, srcs = {s["key"] for s in A + R}, {s["src"] for s in A + R}
    if len(keys) != 1 or len(srcs) != 1:
        raise SystemExit(f"{a.process} at {a.energy:g} GeV: {len(keys)} physics keys, gridpacks {sorted(srcs)}: "
                         "the arm and the reference must share both")
    ref = [s for s in R if s["asked"] == a.ref_ask]
    rc = sorted([s for s in ref if s["got"] >= s["asked"]], key=lambda s: s["seed"])
    rs = sorted([s for s in ref if 0 < s["got"] < s["asked"]], key=lambda s: s["seed"])
    if {s["seed"] for s in A} & {s["seed"] for s in ref}:
        raise SystemExit("the arm and the reference share seeds: the same events would be compared with themselves")
    feats = ["cos_theta_star"]
    Fc, Fs = _feats_of(rc, CAP), _feats_of(rs, CAP)
    logged = _logged(a.logs, a.process, a.energy)
    out = {"process": a.process, "energy": a.energy, "ref_ask": a.ref_ask, "physics_keys": sorted(map(str, keys)),
           "gridpack": sorted(srcs)[0], "ref": {"complete_runs": len(rc), "short_runs": len(rs),
                                                "events_complete": sum(s["got"] for s in rc)}, "asks": {}}
    for ask in sorted({s["asked"] for s in A}):
        ss = sorted([s for s in A if s["asked"] == ask], key=lambda s: s["seed"])
        comp = [s for s in ss if s["got"] >= ask]
        lg = logged.get(ask)
        r = {"runs_kept": len(ss), "complete": len(comp), "events": sum(s["got"] for s in ss),
             **({"logged": lg, "complete_share": round(lg["complete"] / lg["runs"], 4)} if lg else {}),
             "short_returned": sorted(s["got"] for s in ss if s["got"] < ask)}
        Fa = _feats_of([s for s in ss if s["got"] > 0], CAP)
        if len(Fa) >= 2:
            for k, Fy in (("vs_ref", Fc), ("vs_ref_short", Fs)):
                if len(Fy) >= 2:
                    r[k] = dict(_compare(Fa, Fy, feats)["cos_theta_star"], detectable_ks_5pct=_detectable(Fa, Fy))
            if len(Fa) >= 4:
                r["control_arm_halves"] = _compare(Fa[0::2], Fa[1::2], feats)["cos_theta_star"]
        out["asks"][str(ask)] = r
        print(f"[armcheck] ask {ask}: {len(comp)} complete of {len(ss)} kept, share {r.get('complete_share')}, "
              f"vs ref KS {r.get('vs_ref', {}).get('ks_binned')} p {r.get('vs_ref', {}).get('p_runs')}, "
              f"vs short KS {r.get('vs_ref_short', {}).get('ks_binned')}", flush=True)
    if len(Fc) >= 4:
        out["control_ref_halves"] = _compare(Fc[0::2], Fc[1::2], feats)["cos_theta_star"]
    if len(Fc) >= 2 and len(Fs) >= 2:
        out["ref_short_vs_complete"] = _compare(Fs, Fc, feats)["cos_theta_star"]
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    print("ARMCHECK_JSON " + json.dumps(out), flush=True)


def _logged(logs, process, energy):
    """Per ask, runs, complete runs and errors from the probe logs' PROBE_JSON lines at that energy (an errored run
    counts as a run that returned nothing)."""
    logged = {}
    for lp in logs or []:
        for line in open(lp, errors="replace"):
            if not line.startswith("PROBE_JSON "):
                continue
            r = json.loads(line[len("PROBE_JSON "):])
            e = 2.0 * float((r.get("patches") or {}).get("ebeam1", 0.0))
            if r.get("process") != process or abs(e - energy) > 1e-6:
                continue
            ask = int(r["events_asked"])
            d = logged.setdefault(ask, {"runs": 0, "complete": 0, "errors": 0})
            d["runs"] += 1
            d["complete"] += int((r.get("events_returned") or 0) >= ask)
            d["errors"] += int("error" in r)
    return logged


def cmd_sizecheck(a):
    """Run size on one gridpack (the run-size probe of a pool whose runs stop short at one energy; catalog_coverage,
    ee4_epem_cgsxwm at 3000 GeV). Runs of --process kept by `lhe_probe.py gridpool --keep` (KEEP/<process>/
    <process>_gpack<grid>_seed<s>_ev<n>.npz) at collider energy --energy, and with --parts the queue's part files of
    that energy whose name holds the events' identity --eid ($SCRATCH/lhe_parts/<process>/, tagged "parts"): per ask,
    the complete share read from the probe logs --logs (PROBE_JSON lines: a run that errored keeps no file but counts as
    a run that returned nothing, as `lhe_probe.py kept` counts it; without logs no share is given), and the run-level test
    (run_perm, as `measure`) of its complete runs against the complete runs of --ref-ask from the probe; the control is
    those reference runs split in two by seed. One physics key required (same cards and energy): refused otherwise.
    --ref-ask defaults to the pool's run_events in complete_runs.json."""
    import numpy as np
    sys.path.insert(0, ROOT)
    if a.ref_ask is None:
        c = json.load(open(COMPLETE)).get(a.process, {})
        if not c.get("run_events"):
            raise SystemExit(f"{a.process}: no run_events in {COMPLETE}; give --ref-ask")
        a.ref_ask = int(c["run_events"])
    if a.ref_keep:
        if a.parts:
            raise SystemExit("--ref-keep compares a probe arm with a probe reference; --parts is the queue's check")
        a.keep = a.keep or os.path.join(os.environ["SCRATCH"], "lhe_probe_keep")
        return _armcheck(a)
    if a.parts and not a.eid:
        raise SystemExit("--parts needs --eid: the queue's parts of the current layout only")
    keep = os.path.expandvars(a.keep) if a.keep else os.path.join(os.environ["SCRATCH"], "lhe_probe_keep")
    S = []
    for f in sorted(glob.glob(os.path.join(keep, a.process, "*.npz"))):
        m = GPACK_RE.match(os.path.basename(f))
        if m and m["proc"] == a.process:
            S.append({"path": f, "asked": int(m["ev"]), "seed": int(m["seed"]), "src": f"g{m['grid']}", "kind": "part"})
    if a.parts:
        for f in sorted(glob.glob(os.path.join(os.environ["SCRATCH"], "lhe_parts", a.process, "*.npz"))):
            m = PART_RE.match(os.path.basename(f))
            if m and m["proc"] == a.process and m["eid"] == a.eid:
                S.append({"path": f, "asked": int(m["ev"]), "seed": int(m["seed"]), "src": "parts", "kind": "part"})
    rows, keys = [], set()
    for s in S:
        with np.load(s["path"]) as z:
            b = str(z["banner"])
        e = _ebeam(b)
        if e is None or abs(2.0 * e - a.energy) > 1e-6:
            continue
        s["got"], s["key"] = _npz_len(s["path"]), physics_key(b)
        keys.add(s["key"])
        rows.append(s)
    if len(keys) != 1:
        raise SystemExit(f"{a.process} at {a.energy:g} GeV: {len(keys)} physics keys {sorted(map(str, keys))}: "
                         "runs of other cards mixed in")
    # complete share per ask from the probe logs, errored runs included (asked, returned 0)
    logged = _logged(a.logs, a.process, a.energy)
    ref = sorted([s for s in rows if s["asked"] == a.ref_ask and s["got"] >= s["asked"] and s["src"] != "parts"],
                 key=lambda s: s["seed"])
    # a queue run that shares a reference run's seed draws the same events from the same gridpack: that reference run
    # is left out, so the test never compares a run with itself
    ref = [s for s in ref if s["seed"] not in {t["seed"] for t in rows if t["src"] == "parts"}]
    feats = ["cos_theta_star"]
    Fr = _feats_of(ref, CAP)
    out = {"process": a.process, "energy": a.energy, "ref_ask": a.ref_ask, "physics_keys": sorted(map(str, keys)),
           "ref": {"runs": len(ref), "events": sum(s["got"] for s in ref)}, "asks": {}}
    if len(Fr) >= 4:
        out["control_ref_halves"] = _compare(Fr[0::2], Fr[1::2], feats)
    for src in sorted({s["src"] for s in rows}):
        for ask in sorted({s["asked"] for s in rows if s["src"] == src}):
            ss = [s for s in rows if s["src"] == src and s["asked"] == ask]
            comp = [s for s in ss if s["got"] >= ask]
            lg = logged.get(ask) if src != "parts" else None
            r = {"runs_kept": len(ss), "complete": len(comp), "events_complete": sum(s["got"] for s in comp),
                 **({"logged": lg, "complete_share": round(lg["complete"] / lg["runs"], 4)} if lg else {}),
                 "short_returned": sorted(s["got"] for s in ss if s["got"] < ask)}
            if ask != a.ref_ask or src == "parts":
                Fc = _feats_of(comp, CAP)
                if len(Fc) >= 2 and len(Fr) >= 2:
                    r["vs_ref"] = dict(_compare(Fc, Fr, feats)["cos_theta_star"], detectable_ks_5pct=_detectable(Fc, Fr))
            # the short runs of this ask against the same reference (is the complete-run reference itself a selected
            # sample? short and complete runs of one gridpack should then differ)
            short = [s for s in ss if 0 < s["got"] < ask]
            Fs = _feats_of(short, CAP) if len(short) >= 2 and len(Fr) >= 2 else []
            if Fs:
                r["short_vs_ref"] = dict(_compare(Fs, Fr, feats)["cos_theta_star"], runs_short=len(Fs),
                                         detectable_ks_5pct=_detectable(Fs, Fr))
            out["asks"][f"{src}/{ask}"] = r
            print(f"[sizecheck] {src} ask {ask}: {len(comp)} complete of {len(ss)} kept, share {r.get('complete_share')}"
                  + (f", vs ref {a.ref_ask}: KS {r['vs_ref']['ks_binned']} p_runs {r['vs_ref']['p_runs']}"
                     if r.get("vs_ref", {}).get("ks_binned") is not None else ""), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    print("SIZECHECK_JSON " + json.dumps(out), flush=True)


def sizesummary(d, alpha=0.05, share_min=0.9, floor_max=0.013):
    """The run-size probe's outcome per size (catalog_coverage, design of 2026-10-30 11:15): PASS when the complete share
    is at least share_min, the run-level p is not under alpha / (the sizes tested) and the event-level detectable
    distance at that level is at most floor_max; DIFFERS when the p is under it; UNDERPOWERED when share and p pass but the floor
    does not; LOW_SHARE when the share alone fails. The largest PASS size is the layout's (none: nothing laid out). Also
    the reference halves control and the short runs against the reference (whether the reference is a selected sample)."""
    srcs = {k.split("/")[0] for k in d["asks"] if not k.startswith("parts")}
    if len(srcs) != 1:
        raise SystemExit(f"sizesummary: probe runs from {len(srcs)} gridpacks {sorted(srcs)}; one gridpack per check")
    A = {k.split("/")[-1]: v for k, v in d["asks"].items() if not k.startswith("parts")}
    tested = [k for k, v in A.items() if "vs_ref" in v]
    lvl = alpha / max(1, len(tested))
    # the event-level detectable distance at the Bonferroni level (the design's floor), from the one at 5 percent that
    # sizecheck stores: the KS critical coefficient sqrt(-ln(level / 2) / 2) over its value at 5 percent, 1.358
    at_lvl = math.sqrt(-math.log(lvl / 2.0) / 2.0) / 1.358
    out = {"process": d["process"], "energy": d["energy"], "ref_ask": d["ref_ask"], "ref_runs": d["ref"]["runs"],
           "ref_events": d["ref"]["events"], "level": round(lvl, 5), "sizes": {}}
    c = (d.get("control_ref_halves") or {}).get("cos_theta_star")
    if c:
        out["control_halves"] = {"ks": round(c["ks_binned"], 4), "p": c["p_runs"]}
    for k in sorted(A, key=int):
        v = A[k]
        e = {"runs": v.get("logged", {}).get("runs", v["runs_kept"]), "complete": v["complete"],
             "share": v.get("complete_share")}
        if "vs_ref" in v:
            w = v["vs_ref"]
            e.update(ks=round(w["ks_binned"], 4), p=w["p_runs"], floor=round(w["detectable_ks_5pct"] * at_lvl, 4))
            ok_share = e["share"] is not None and e["share"] >= share_min
            e["outcome"] = ("DIFFERS" if w["p_runs"] < lvl else "LOW_SHARE" if not ok_share else
                            "UNDERPOWERED" if e["floor"] > floor_max else "PASS")
        if "short_vs_ref" in v:
            w = v["short_vs_ref"]
            e["short"] = {"runs": w["runs_short"], "ks": round(w["ks_binned"], 4), "p": w["p_runs"]}
        out["sizes"][k] = e
    ps = [int(k) for k, e in out["sizes"].items() if e.get("outcome") == "PASS"]
    out["layout_size"] = max(ps) if ps else None
    out["n_pass"] = len(ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("--out", default=OUT)
    m.add_argument("--nperm", type=int, default=2000); m.add_argument("--unscaled", action="store_true")
    m.add_argument("--before", default=None, help="ISO time: only files last written before it (re-read an earlier "
                   "measurement on the files it saw; checked by its event-level numbers reproducing)")
    m.add_argument("--min-events", type=int, default=2000)
    f = sub.add_parser("fig"); f.add_argument("--json", default=OUT)
    s = sub.add_parser("summary"); s.add_argument("--json", default=OUT)
    z = sub.add_parser("figsize"); z.add_argument("--json", default=OUT)
    k = sub.add_parser("sizecheck"); k.add_argument("--process", required=True)
    k.add_argument("--nperm", type=int, default=2000); k.add_argument("--unscaled", action="store_true")
    k.add_argument("--energy", type=float, required=True); k.add_argument("--ref-ask", type=int, default=None)
    k.add_argument("--keep", default=None); k.add_argument("--parts", action="store_true")
    k.add_argument("--eid", default=None, help="with --parts: the events' identity of the layout whose parts to test")
    k.add_argument("--logs", nargs="*", default=[], help="the probe jobs' logs (PROBE_JSON lines) for the shares")
    k.add_argument("--out", default=None)
    k.add_argument("--ref-keep", default=None, help="the reference runs' keep dir: --keep is then one probe arm "
                   "compared with them (_armcheck)")
    r = sub.add_parser("reread"); r.add_argument("--old", required=True); r.add_argument("--new", required=True)
    c = sub.add_parser("cutoff"); c.add_argument("--old", nargs="+", required=True)
    c.add_argument("--before", nargs="+", required=True, help="per --old, the latest time it can have been measured")
    c.add_argument("--hours", type=float, default=12.0)
    q = sub.add_parser("sizesummary"); q.add_argument("--json", required=True)
    a = ap.parse_args()
    global NPERM, SCALED
    NPERM = getattr(a, "nperm", None) or NPERM
    SCALED = not getattr(a, "unscaled", False)
    if a.cmd == "sizesummary":
        print(json.dumps(sizesummary(json.load(open(a.json)))))
        return
    if a.cmd == "sizecheck":
        cmd_sizecheck(a)
        return
    if a.cmd == "summary":
        print(json.dumps(summary(json.load(open(a.json)))))
        return
    {"measure": cmd_measure, "reread": cmd_reread, "cutoff": cmd_cutoff, "fig": cmd_fig, "figsize": cmd_figsize}[a.cmd](a)


if __name__ == "__main__":
    main()
