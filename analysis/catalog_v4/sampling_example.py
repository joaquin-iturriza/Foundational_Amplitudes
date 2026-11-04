#!/usr/bin/env python3
"""Default against catalog sampling on one catalog_v4 tree class, for a deck slide (plan catalog_coverage).

The class is u u~ > e+ e- g (v4_uux_epemg, Drell-Yan plus a jet), train role, seed 42, recipe catalog_v4_train.yaml:
  catalog    its catalog_v4 pool (the entry without the default share): RAMBO bulk, 0.15 forward (rotate_forward),
             0.35 flat in log|M|^2 (6x oversampled candidates, half splitter half RAMBO, adaptive sqrt(s_hat));
             each row's component from mix_compose.provenance (the bulk and forward draws replayed and matched to
             the pool's rows by their momenta; the rest is the flat part)
  default    every row of the class's default-sampled MadEvent pools (pp at 13 TeV: pp13v4_uux_epemg reused plus
             dsh_uux_epemg) in the partonic frame inside the pool's sqrt(s_hat) window and cuts
             (mix_compose._default_accepted), i.e. the generator's own sampling with pp beams
  trained    the default-share 0.25 mixture as the loader composes it (mix_compose.compose): what training reads
The flat part's candidates are half democratic splitter, half RAMBO (pol["democratic"]; bulk and forward draw RAMBO
only). The pool keeps no tag, so each chunk is regenerated here exactly (datagen.gen_chunk at the chunk's seed, the
compiled standalone labelling the candidates as in the build, so the adaptive sqrt(s_hat) pass replays too) with
mg.democratic_draw and mg.rotate_forward recorded; the replayed chunk must equal the pool's rows bit for bit, and a
kept flat row is the splitter's when its final momenta are a recorded splitter draw (or that draw turned forward).

    python analysis/catalog_v4/sampling_example.py pull     # the hist step on CC-IN2P3 (site run) -> data/sampling_example.json
    python analysis/catalog_v4/sampling_example.py figs     # local, from analysis/catalog_v4/data/sampling_example.json
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT_JSON = os.path.join(ROOT, "analysis", "catalog_v4", "data", "sampling_example.json")
FIGS = os.path.join(ROOT, "analysis", "catalog_v4", "figs", "sampling_example")
RECIPE = os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")
NAME, ROLE, SEED = "v4_uux_epemg", "train", 42
LOGM_BINS = np.linspace(-14.0, 4.0, 73)              # log10 |M|^2, 0.25 per bin; under/overflow counted apart
LOGM_XLIM = (-9.5, 1.0)                              # the figures' range: every event of the class lies inside
SQ_BINS = np.logspace(np.log10(30.0), np.log10(6000.0), 61)   # sqrt(s_hat) in GeV, the pool's window


def _sqrts(P):
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))


def _h(me, sq):
    with np.errstate(divide="ignore"):
        lg = np.log10(np.abs(me))
    fin = np.isfinite(lg)
    return {"n": int(len(me)), "n_nonfinite": int((~fin).sum()),
            "under": int((lg[fin] < LOGM_BINS[0]).sum()), "over": int((lg[fin] > LOGM_BINS[-1]).sum()),
            "logm": np.histogram(lg[fin], LOGM_BINS)[0].tolist(), "sqrts": np.histogram(sq, SQ_BINS)[0].tolist(),
            "logm_mean": float(lg[fin].mean()), "logm_min": float(lg[fin].min()), "logm_max": float(lg[fin].max()),
            "sqrts_median": float(np.median(sq))}


def hist():
    import yaml
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
    spec = yaml.safe_load(open(RECIPE))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in spec["processes"]],
                                 default_sampling=spec.get("sampling"))
    p = {q["name"]: q for q in spec["processes"]}[NAME]
    lo, hi = map(float, p["sqrts"]); n = int(p["n_" + ROLE])
    cfg = mg.PROCESSES[NAME]; k = int(cfg["nfinal"]) + 2
    with mc._Swap(NAME, mc.control_config(NAME)):
        rec_c = mg.variable_energy_recipe(NAME, lo, hi, n, role=ROLE, seed=SEED)
        ctrl = mg.recipe_output_path(rec_c, datagen.dest_for_role(ROLE))
    pool = np.asarray(np.load(ctrl, mmap_mode="r"), np.float64)
    prov = mc.provenance(NAME, lo, hi, n, ROLE, SEED, pool)
    Pc = pool[:, :4 * k].reshape(len(pool), k, 4); sq_c = _sqrts(Pc); me_c = pool[:, -1]
    m_list, cuts, pol, _, _ = mc._sampler_args(cfg, ROLE)
    Pd, sq_d, me_d = mc._default_accepted(pol["default_source"], ROLE, SEED, (lo, hi, m_list, cfg["pdg_ids"], cuts))
    src_pools = [(nm, path) for nm, _, path in mg.default_share_pools(pol["default_source"], ROLE, SEED)]
    n_src = sum(int(np.load(path, mmap_mode="r").shape[0]) for _, path in src_pools)
    rows, kinds = mc.compose(NAME, lo, hi, n, ROLE, SEED, ctrl)
    Pt = rows[:, :4 * k].reshape(len(rows), k, 4); sq_t = _sqrts(Pt); me_t = rows[:, -1]
    out = {"class": NAME, "role": ROLE, "seed": SEED, "recipe": os.path.relpath(RECIPE, ROOT), "window": [lo, hi],
           "catalog_pool": ctrl, "default_source": pol["default_source"], "f_default": float(pol["f_default"]),
           "default_pools": [[nm, path] for nm, path in src_pools], "default_rows_total": n_src,
           "logm_bins": LOGM_BINS.tolist(), "sqrts_bins": SQ_BINS.tolist(), "h": {}}
    dem = flat_tags(NAME, lo, hi, n, ROLE, SEED, pool, k)
    out["flat_tags"] = {"n_flat": int((prov == mc.KIND["flat"]).sum()), "n_flat_splitter": int(dem.sum()),
                        "splitter_share_of_kept_flat": float(dem.sum() / max(1, (prov == mc.KIND["flat"]).sum()))}
    out["h"]["catalog"] = _h(me_c, sq_c)
    for kind in ("bulk", "fwd", "flat"):
        s = prov == mc.KIND[kind]
        out["h"]["catalog_" + kind] = _h(me_c[s], sq_c[s])
    for key, s in (("catalog_flat_splitter", dem), ("catalog_flat_rambo", (prov == mc.KIND["flat"]) & ~dem)):
        out["h"][key] = _h(me_c[s], sq_c[s])
    out["h"]["default"] = _h(me_d, sq_d)
    out["h"]["trained"] = _h(me_t, sq_t)
    keys = {Pc[i, 2:].tobytes(): i for i in range(len(Pc))}
    t_idx = np.array([keys.get(Pt[i, 2:].tobytes(), -1) for i in range(len(Pt))])
    for kind in ("bulk", "fwd", "flat", "default"):
        s = kinds == mc.KIND[kind]
        out["h"]["trained_" + kind] = _h(me_t[s], sq_t[s])
    fl = kinds == mc.KIND["flat"]
    if (t_idx[fl] < 0).any():
        raise RuntimeError(f"{int((t_idx[fl] < 0).sum())} trained flat rows not in the catalog pool")
    t_dem = np.zeros(len(Pt), bool); t_dem[fl] = dem[t_idx[fl]]
    out["h"]["trained_flat_splitter"] = _h(me_t[t_dem], sq_t[t_dem])
    out["h"]["trained_flat_rambo"] = _h(me_t[fl & ~t_dem], sq_t[fl & ~t_dem])
    # to a file, not stdout: site run merges stderr into stdout, and a long stdout line (block-buffered in a pipe) can
    # be cut by stderr output written between its flushes (E3584: 'pull: no SAMPEX line'); pull cats the file
    with open(_site_json(), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("hist: wrote", _site_json())


def _site_json():
    """where hist writes its histograms on the site (outside the checkout, so the checkout stays clean)"""
    return os.path.join(os.environ.get("SCRATCH") or "/tmp", "sampling_example.json")


def pull(step="hist", out_file="sampling_example.json"):
    """a step (hist, shape) on CC-IN2P3 through site run; the file it writes in $SCRATCH, printed after a marker line
    once the step has exited (nothing else writes then) -> analysis/catalog_v4/data/<out_file>"""
    import subprocess
    src = "sampling_example" if step == "hist" else "sampling_shape"
    res = subprocess.run(["site", "--timeout", "3000", "run", "--item", "catalog_coverage", "ccin2p3", "FA", "--",
                          f"python analysis/catalog_v4/sampling_example.py {step} && echo && echo SAMPEX_BEGIN && "
                          f"cat \"${{SCRATCH:-/tmp}}/{src}.json\" && echo"],
                         capture_output=True, text=True, timeout=3100)
    lines = res.stdout.splitlines()
    k = [i for i, l in enumerate(lines) if l.strip() == "SAMPEX_BEGIN"]
    if res.returncode or not k or k[-1] + 1 >= len(lines):
        sys.exit("pull: rc %d, no SAMPEX_BEGIN block\n" % res.returncode + res.stdout[-3000:] + res.stderr[-2000:])
    d = json.loads(lines[k[-1] + 1])
    open(os.path.join(ROOT, "analysis", "catalog_v4", "data", out_file), "w").write(lines[k[-1] + 1])
    print(d.get("flat_tags") or {x: d.get(x) for x in ("n_candidates", "n_kept", "n_default")})


def flat_tags(name, lo, hi, n, role, seed, pool, k):
    """bool per pool row: a kept flat-part row drawn by the democratic splitter (see the module note)"""
    import tempfile
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
    out = np.zeros(len(pool), bool)
    rec = {"dem": [], "rot": []}
    dd, rf = mg.democratic_draw, mg.rotate_forward
    def dem_rec(*a, **kw):
        P = dd(*a, **kw); rec["dem"].append(np.array(P)); return P
    def rot_rec(P, *a, **kw):
        Q = rf(P, *a, **kw); rec["rot"].append((np.array(P), np.array(Q))); return Q
    with mc._Swap(name, mc.control_config(name)):
        datagen.ensure_backend(name)
        tasks = mc.chunk_plan(name, lo, hi, n, role, seed)
        mg.democratic_draw, mg.rotate_forward = dem_rec, rot_rec
        try:
            start = 0
            for t in tasks:
                rec["dem"].clear(); rec["rot"].clear()
                with tempfile.TemporaryDirectory(prefix="zz_probe_sampling_") as td:
                    t = dict(t, out_dir=td)
                    _, path = datagen.gen_chunk(t)
                    ch = np.load(path)
                m = int(t["count"])
                # row by row the same events in the same order, to rounding: the pool's rows differ from a replay on
                # another node in the last digits (E3587: max |dp| 2.2e-9 GeV, max rel |M|^2 1.6e-11; 6695 of 16667
                # rows bit-equal), so equality is to rtol 1e-9; a different selection or order differs at O(1).
                # The tags come from the replay's own records, compared bit for bit against the replayed rows.
                b = pool[start:start + m]
                if ch.shape != b.shape or not (np.allclose(ch, b, rtol=1e-9, atol=0)
                                               and np.array_equal(ch[:, 4 * k:-1], b[:, 4 * k:-1])):
                    raise RuntimeError(f"chunk {t['idx']}: the replay differs from the pool's rows: "
                                       + _mismatch(ch, pool[start:start + m], k))
                dem_f = {r.tobytes() for D in rec["dem"] for r in np.asarray(D, np.float64)}
                for Pin, Q in rec["rot"]:
                    for a, b in zip(np.asarray(Pin, np.float64)[:, 2:], np.asarray(Q, np.float64)[:, 2:]):
                        if a.tobytes() in dem_f:
                            dem_f.add(b.tobytes())
                fin = ch[:, :4 * k].reshape(m, k, 4)[:, 2:]
                out[start:start + m] = [fin[i].tobytes() in dem_f for i in range(m)]
                start += m
        finally:
            mg.democratic_draw, mg.rotate_forward = dd, rf
    return out


def _mismatch(a, b, k):
    """what differs between a replayed chunk a and the pool's rows b: shapes, rows equal in momenta and in |M|^2,
    rows of a found anywhere in b (an order change), the largest relative |M|^2 and momentum differences"""
    if a.shape != b.shape:
        return f"shapes {a.shape} vs {b.shape}"
    pa, pb = a[:, :4 * k], b[:, :4 * k]
    mom = (pa == pb).all(1); me = a[:, -1] == b[:, -1]
    inb = {r.tobytes() for r in pb}
    anyw = sum(r.tobytes() in inb for r in pa)
    dme = float(np.max(np.abs(a[:, -1] - b[:, -1]) / np.maximum(np.abs(b[:, -1]), 1e-300)))
    dp = float(np.max(np.abs(pa - pb)))
    first = int(np.argmin(mom)) if not mom.all() else -1
    return (f"{len(a)} rows, momenta equal {int(mom.sum())}, |M|^2 equal {int(me.sum())}, replayed momenta found "
            f"anywhere in the pool {anyw}, max rel |M|^2 diff {dme:.3g}, max |dp| {dp:.3g}, first momentum mismatch "
            f"row {first}, extra columns {a.shape[1] - 4 * k - 1}")


# phase-space variables of the shape study (Joaquin 2026-11-04, on figure c): name -> bin edges
SHAPE_VARS = {"sqrts": np.logspace(np.log10(30.0), np.log10(6000.0), 31),
              "mll_over_mz": np.logspace(np.log10(0.1), np.log10(60.0), 41),
              "ptg": np.logspace(np.log10(10.0), np.log10(3000.0), 31),
              "absyg": np.linspace(0.0, 5.0, 26),
              "drgl": np.logspace(np.log10(0.4), np.log10(8.0), 26)}
SHAPE_LAB = {"sqrts": r"$\sqrt{\hat s}$ [GeV]", "mll_over_mz": r"$m_{\ell\ell}/m_Z$", "ptg": r"gluon $p_T$ [GeV]",
             "absyg": r"gluon $|y|$", "drgl": r"gluon $\min\Delta R_{g\ell}$"}
# "the top events": log10|M|^2 above TOP, TOP derived in shape_figs from the kept flat part itself; rule and value
# written to the report
TOP_RULE = "lower edge of the first 0.25-dex bin above the kept part's median-count bin where kept < 0.5 x median"
MZ = 91.1876


def _shape_vars(P, pdg):
    """per event (P: (N, k, 4) partonic frame, beams first) the shape variables: sqrt(s_hat), the lepton pair's mass
    over m_Z, the gluon's pT and |rapidity| to the beam axis, and its smallest Delta R to a lepton"""
    pdg = list(int(x) for x in pdg)
    ig = pdg.index(21, 2); il = [i for i in range(2, len(pdg)) if abs(pdg[i]) in (11, 13)]
    q = P[:, il[0]] + P[:, il[1]]
    mll = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))
    g = P[:, ig]
    pt = np.hypot(g[:, 1], g[:, 2])
    y = 0.5 * np.log((g[:, 0] + g[:, 3]) / np.clip(g[:, 0] - g[:, 3], 1e-300, None))
    def eta_phi(v):
        ptv = np.hypot(v[:, 1], v[:, 2]); return np.arcsinh(v[:, 3] / ptv), np.arctan2(v[:, 2], v[:, 1])
    eg, pg = eta_phi(g)
    dr = np.full(len(P), np.inf)
    for i in il:
        el, pl = eta_phi(P[:, i]); dphi = np.angle(np.exp(1j * (pg - pl)))
        dr = np.minimum(dr, np.hypot(eg - el, dphi))
    return {"sqrts": _sqrts(P), "mll_over_mz": mll / MZ, "ptg": pt, "absyg": np.abs(y), "drgl": dr}


def _h2(lm, v):
    """{var: 2D counts [log10|M|^2 bin][var bin]} (out-of-range values clipped into the edge bins)"""
    out = {}
    for k, e in SHAPE_VARS.items():
        x = np.clip(v[k], e[0], e[-1] * (1 - 1e-12))
        out[k] = np.histogram2d(np.clip(lm, LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), x, bins=[LOGM_BINS, e])[0].astype(int).tolist()
    return out


def shape():
    """on CC-IN2P3 (the pools and the standalone): the flat part's candidates of every chunk (replayed as flat_tags
    does, _candidates and _flat_logm_select recorded), which of them were kept, the kept flat events and the default
    (MadEvent) events, each as 2D histograms of log10|M|^2 against the SHAPE_VARS -> $SCRATCH/sampling_shape.json"""
    import tempfile, yaml
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
    spec = yaml.safe_load(open(RECIPE))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in spec["processes"]],
                                 default_sampling=spec.get("sampling"))
    p = {q["name"]: q for q in spec["processes"]}[NAME]
    lo, hi = map(float, p["sqrts"]); n = int(p["n_" + ROLE])
    cfg = mg.PROCESSES[NAME]; k = int(cfg["nfinal"]) + 2; pdg = cfg["pdg_ids"]
    with mc._Swap(NAME, mc.control_config(NAME)):
        ctrl = mg.recipe_output_path(mg.variable_energy_recipe(NAME, lo, hi, n, role=ROLE, seed=SEED),
                                     datagen.dest_for_role(ROLE))
    pool = np.asarray(np.load(ctrl, mmap_mode="r"), np.float64)
    calls, sel = [], []
    cd, fs = mg._candidates, mg._flat_logm_select
    def cd_rec(nb, *a, **kw):
        ev, sq = cd(nb, *a, **kw); calls.append(np.stack([np.asarray(e[0], float) for e in ev])); return ev, sq
    def fs_rec(logm, n_keep, rng, n_bins=60):
        keep = fs(logm, n_keep, rng, n_bins=n_bins)
        sel.append((np.asarray(logm, float).copy(), np.asarray(keep).copy(), int(n_bins))); return keep
    cand_P, cand_lm, kept_mask, nbins = [], [], [], []
    with mc._Swap(NAME, mc.control_config(NAME)):
        datagen.ensure_backend(NAME)
        tasks = mc.chunk_plan(NAME, lo, hi, n, ROLE, SEED)
        mg._candidates, mg._flat_logm_select = cd_rec, fs_rec
        try:
            start = 0
            for t in tasks:
                calls.clear(); sel.clear()
                with tempfile.TemporaryDirectory(prefix="zz_probe_sampling_") as td:
                    _, path = datagen.gen_chunk(dict(t, out_dir=td))
                    ch = np.load(path)
                m = int(t["count"]); b = pool[start:start + m]
                if ch.shape != b.shape or not np.allclose(ch, b, rtol=1e-9, atol=0):
                    raise RuntimeError(f"chunk {t['idx']}: the replay differs from the pool's rows: " + _mismatch(ch, b, k))
                if len(sel) != 1:
                    raise RuntimeError(f"chunk {t['idx']}: {len(sel)} flat selections")
                lm, keep, nb = sel[0]
                Pc = np.concatenate(calls[-2:]) if sum(len(c) for c in calls[-2:]) == len(lm) else calls[-1]
                if len(Pc) != len(lm):
                    raise RuntimeError(f"chunk {t['idx']}: {len(Pc)} recorded candidates against {len(lm)} labels")
                km = np.zeros(len(lm), bool); km[keep] = True
                cand_P.append(Pc); cand_lm.append(lm / np.log(10.0)); kept_mask.append(km); nbins.append(nb)
                start += m
        finally:
            mg._candidates, mg._flat_logm_select = cd, fs
    Pc = np.concatenate(cand_P); lmc = np.concatenate(cand_lm); km = np.concatenate(kept_mask)
    fin = np.isfinite(lmc)
    vc = _shape_vars(Pc[fin], pdg)
    out = {"class": NAME, "role": ROLE, "seed": SEED, "logm_bins": LOGM_BINS.tolist(),
           "vars": {kk: e.tolist() for kk, e in SHAPE_VARS.items()}, "n_chunks": len(tasks), "select_bins": nbins,
           "n_candidates": int(len(lmc)), "n_candidates_finite": int(fin.sum()), "n_kept": int(km.sum())}
    out["candidates"] = _h2(lmc[fin], vc)
    out["kept"] = _h2(lmc[fin][km[fin]], {kk: v[km[fin]] for kk, v in vc.items()})
    _, _, pol, _, _ = mc._sampler_args(cfg, ROLE)
    m_list, cuts, _, _, _ = mc._sampler_args(cfg, ROLE)
    Pd, sq_d, me_d = mc._default_accepted(pol["default_source"], ROLE, SEED, (lo, hi, m_list, pdg, cuts))
    Pd = np.asarray(Pd, float).reshape(len(Pd), k, 4)
    out["default"] = _h2(np.log10(np.asarray(me_d, float)), _shape_vars(Pd, pdg))
    out["n_default"] = int(len(Pd))
    with open(os.path.join(os.environ.get("SCRATCH") or "/tmp", "sampling_shape.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("shape: wrote", out["n_candidates"], "candidates,", out["n_kept"], "kept,", out["n_default"], "default")


def pull_shape():
    """shape on CC-IN2P3 -> data/sampling_shape.json"""
    pull("shape", "sampling_shape.json")


def shape_figs():
    """from data/sampling_shape.json (values outside a variable's range are counted in its edge bins): (a) the flat part's candidates and kept events per 0.25-dex bin of log10|M|^2
    (kept counts against their mean over the flat range), printed per bin; (b-f) per variable, the share of events
    with log10|M|^2 > TOP among the default events, the candidates and the kept flat events, per variable bin, and
    each set's distribution of that variable among its own top events -> figs/sampling_shape_*.{png,pdf}"""
    import plot_style as ps
    d = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "data", "sampling_shape.json")))
    lb = np.array(d["logm_bins"]); mid = 0.5 * (lb[1:] + lb[:-1])
    rep = {}
    c1 = {s: np.array(d[s]["sqrts"]).sum(1) for s in ("candidates", "kept", "default")}
    nz = np.nonzero(c1["kept"])[0]
    rep["kept_per_bin"] = {f"{lb[i]:.2f}": int(c1["kept"][i]) for i in range(nz[0], nz[-1] + 1)}
    rep["candidates_per_bin"] = {f"{lb[i]:.2f}": int(c1["candidates"][i]) for i in range(nz[0], nz[-1] + 1)}
    rep["kept_over_candidates"] = {f"{lb[i]:.2f}": round(float(c1["kept"][i] / c1["candidates"][i]), 4)
                                   for i in range(nz[0], nz[-1] + 1) if c1["candidates"][i]}
    figs = ps.panels(1 + len(SHAPE_VARS))
    fig, ax = figs[0]
    ax.stairs(c1["candidates"], lb, color=ps.C.grey, label="candidates")
    ax.stairs(c1["kept"], lb, color=ps.C.blue, label="kept flat")
    ax.stairs(c1["default"] * c1["kept"].sum() / c1["default"].sum(), lb, color=ps.C.vermillion, ls="--",
              label="default, same total")
    ax.set_yscale("log"); ax.set_xlim(*LOGM_XLIM); ax.set_ylim(1, None)
    ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); ax.set_ylabel("events per 0.25")
    ps.legend(ax, "upper left")
    occ = c1["kept"] > 0
    plateau = float(np.median(c1["kept"][occ]))
    i_pl = int(np.nonzero(occ)[0][np.argmin(np.abs(c1["kept"][occ] - plateau))])
    i_top = next(i for i in range(i_pl, len(mid)) if c1["kept"][i] < 0.5 * plateau)
    TOP = float(lb[i_top])
    rep["top_rule"] = {"rule": TOP_RULE, "plateau": plateau, "TOP": TOP}
    top = mid > TOP
    rep["top_share"] = {s: round(float(c1[s][top].sum() / c1[s].sum()), 4) for s in c1}
    # the Z bin: the m_ll/m_Z bin holding 1 (edges a factor 1.17 apart); per set its share of all events and of the
    # top events, and the share of its events above TOP
    em = np.array(d["vars"]["mll_over_mz"]); iz = int(np.searchsorted(em, 1.0) - 1)
    rep["z_bin"] = {"edges": [round(float(em[iz]), 4), round(float(em[iz + 1]), 4)]}
    for s in c1:
        H = np.array(d[s]["mll_over_mz"], float)
        rep["z_bin"][s] = {"share_of_all": round(float(H[:, iz].sum() / H.sum()), 4),
                           "share_of_top": round(float(H[top, iz].sum() / max(H[top].sum(), 1)), 4),
                           "top_share_in_bin": round(float(H[top, iz].sum() / max(H[:, iz].sum(), 1)), 4),
                           "n_in_bin": int(H[:, iz].sum())}
    for (fig, ax), v in zip(figs[1:], SHAPE_VARS):
        e = np.array(d["vars"][v]); xm = np.sqrt(e[1:] * e[:-1]) if v != "absyg" else 0.5 * (e[1:] + e[:-1])
        rep[v] = {}
        for s, col, lab in (("default", ps.C.vermillion, "default"), ("candidates", ps.C.grey, "candidates"),
                            ("kept", ps.C.blue, "kept flat")):
            H = np.array(d[s][v], float)
            t = H[top].sum(0)
            ax.stairs(t / max(t.sum(), 1), e, color=col, label=lab)
            a = H.sum(0)
            rep[v][s] = {"top_median": float(xm[np.searchsorted(np.cumsum(t), t.sum() / 2)]) if t.sum() else None,
                         "all_median": float(xm[np.searchsorted(np.cumsum(a), a.sum() / 2)]),
                         "n_top": int(t.sum())}
        if v != "absyg":
            ax.set_xscale("log")
        if v == "drgl":
            from matplotlib.ticker import NullFormatter
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(SHAPE_LAB[v]); ax.set_ylabel("fraction of the top events")
        ps.legend(ax, "upper left")
    ps.save_panels(figs, FIGS.replace("sampling_example", "sampling_shape"))
    json.dump(rep, open(os.path.join(ROOT, "analysis", "catalog_v4", "data", "sampling_shape_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=0))


def figs():
    import plot_style as ps
    d = json.load(open(OUT_JSON)); h = d["h"]
    lb, sb = np.array(d["logm_bins"]), np.array(d["sqrts_bins"])

    def dens(key, var):
        c = np.asarray(h[key][var], float); e = lb if var == "logm" else np.log10(sb)
        return c / (h[key.split("_")[0]]["n"] * np.diff(e))   # per unit log10, per event of its whole pool

    def step(ax, key, var, **kw):
        x = lb if var == "logm" else sb
        y = dens(key, var)
        ax.stairs(y, x, **kw)

    proc = r"$u\bar u\to e^+e^-g$"
    # (a) the catalog mixture by component, stacked shares of the one pool
    fig, (axL, axR) = ps.figure(ncols=2)
    comps = [("trained_bulk", "bulk (RAMBO)", ps.C.sky), ("trained_fwd", "forward (RAMBO)", ps.C.green),
             ("trained_flat_rambo", r"flat in $\log|\mathcal{M}|^2$ (RAMBO)", ps.C.orange),
             ("trained_flat_splitter", r"flat in $\log|\mathcal{M}|^2$ (splitter)", ps.C.purple),
             ("trained_default", "default share (MadEvent)", ps.C.vermillion)]
    for ax, var in ((axL, "logm"), (axR, "sqrts")):
        x = lb if var == "logm" else sb
        base = np.zeros(len(x) - 1)
        for key, lab, col in comps:
            y = dens(key, var)
            ax.stairs(base + y, x, baseline=base, fill=True, color=col, alpha=0.6, label=lab)
            base = base + y
        step(ax, "trained", var, color=ps.C.blue, label="trained mixture (sum)")
        if var == "logm":
            ax.set_ylim(0, 2.0 * base.max())         # headroom for the six-entry legend above the stack
    axL.set_xlim(*LOGM_XLIM); axL.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); axL.set_ylabel("density")
    axR.set_xscale("log"); axR.set_xlabel(r"$\sqrt{\hat s}$ [GeV]"); axR.set_ylabel("density")
    ps.legend(axL, "upper left")
    ps.process_label(axR, proc)
    ps.save(fig, FIGS + "_a")
    # (b) generator default (pp 13 TeV MadEvent), catalog, and the trained 0.25 mixture
    fig, (axL, axR) = ps.figure(ncols=2)
    nd = h["default"]["n"]; nc = h["catalog"]["n"]; nt = h["trained"]["n"]
    for ax, var in ((axL, "logm"), (axR, "sqrts")):
        x = lb if var == "logm" else sb
        e = lb if var == "logm" else np.log10(sb)
        for key, lab, col, nn, ls in (("default", "default (MadEvent, pp 13 TeV)", ps.C.vermillion, nd, "-"),
                                      ("catalog", "catalog", ps.C.blue, nc, "-"),
                                      ("trained", r"trained: $0.75\,$catalog$\,+\,0.25\,$default", ps.C.grey, nt, "--")):
            y = np.asarray(h[key][var], float) / (nn * np.diff(e))
            ax.stairs(y, x, color=col, linestyle=ls, label=lab)
    axL.set_xlim(*LOGM_XLIM); axL.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); axL.set_ylabel("density")
    axR.set_xscale("log"); axR.set_xlabel(r"$\sqrt{\hat s}$ [GeV]"); axR.set_ylabel("density")
    ps.legend(axL, "upper left")
    ps.process_label(axR, proc)
    ps.save(fig, FIGS + "_b")
    # (c) the flat-in-log|M|^2 part alone (Joaquin 2026-11-04: 'to see that they are indeed flat'), its two generators
    # stacked, density per unit log10|M|^2 per kept flat event of the catalog pool
    fig, ax = ps.figure()
    nf = h["catalog_flat"]["n"]; base = np.zeros(len(lb) - 1)
    for key, lab, col in (("catalog_flat_rambo", "RAMBO candidates", ps.C.orange),
                          ("catalog_flat_splitter", "splitter candidates", ps.C.purple)):
        y = np.asarray(h[key]["logm"], float) / (nf * np.diff(lb))
        ax.stairs(base + y, lb, baseline=base, fill=True, color=col, alpha=0.6, label=lab)
        base = base + y
    ax.stairs(base, lb, color=ps.C.blue, label=r"flat in $\log|\mathcal{M}|^2$ (sum)")
    ax.set_xlim(*LOGM_XLIM)
    ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); ax.set_ylabel("density")
    ps.legend(ax, "upper left", title=proc)
    ps.save(fig, FIGS + "_c")


if __name__ == "__main__":
    {"hist": hist, "pull": pull, "figs": figs, "shape": shape, "pull_shape": pull_shape,
     "shape_figs": shape_figs}[sys.argv[1]]()
