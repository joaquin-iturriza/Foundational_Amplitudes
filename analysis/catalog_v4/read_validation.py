"""catalog_v4's validation and census read (runs FA-ccin2p3-44e6a15b and, with the sampling fix 63bc5de, the rerun
FA-ccin2p3-2e1a0894; scripts/job_catalog_v4_validate.sh).

Inputs (analysis/catalog_v4/data/, fetched from CC): validation.json (the rerun; analysis/catalog_v4/validate_v4.py: per
sampled entry the diagram counts, the backend's cards, the pool checks), validation_first.json (the first run, whose
pools all stopped on the missing sampling block; its backend build times, the rerun reused the backends),
v4_disk_rerun.txt (the rerun's du of the standalones and the free space), checked.json (recipes/gen_catalog_v4.py
--check: every catalog_v3 tree on its v4 string through MadGraph), v4_disk.txt (the job's du of the v4 backends it
built, the free space, and the count of catalog_v3's MadGraph events dirs).
Output: figs/census_diagrams (v4 against v3 diagram count per string, by layer), figs/pool_sqrts_forward (three
entries' sqrt(s_hat) per decade in the pool against log-uniform and log-uniform after the cuts, and the forward
component's u = log(1/(1-|cos|)) against uniform) and, last line, the JSON of the numbers
docs/results.tex cites. The regeneration's pool size is rows x (5 k + 1) float64 columns over every entry of
recipes/catalog_v4_gen_stage{1..4}.yaml (k legs; mg5_pipeline_final's registry), its backends the count of distinct
standalone names times the measured mean size per backend.
    python3 analysis/catalog_v4/read_validation.py
"""
import json, os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
D = os.path.join(HERE, "data")
C = json.load(open(os.path.join(D, "checked.json")))
V = json.load(open(os.path.join(D, "validation.json")))
V0 = {r["name"]: r for r in json.load(open(os.path.join(D, "validation_first.json")))["rows"]}
FIG_ENTRIES = ["v4_uux_ddx", "v4_ddx_ddxgg", "v4_epem_ccxgvevex"]
NLOC = 20000          # local draws per entry for the after-cuts expectation and the forward replay


def census():
    out = {}
    for L in sorted({r["layer"] for r in C}):
        rs = [r for r in C if r["layer"] == L]; ex = [r for r in rs if r["exists"]]
        out[f"l{L}"] = {"n": len(rs), "exists": len(ex), "d3": sum(r["diagrams_v3"] for r in ex),
                        "d4": sum(r["diagrams"] for r in ex), "grew": sum(r["diagrams"] > r["diagrams_v3"] for r in ex),
                        "shrank": sum(r["diagrams"] < r["diagrams_v3"] for r in ex)}
    tot = {k: sum(v[k] for v in out.values()) for k in ("n", "exists", "d3", "d4", "grew", "shrank")}
    tot["ratio"] = round(tot["d4"] / tot["d3"], 2)
    van = [r for r in C if not r["exists"]]
    # every vanished string has a Higgs and only b quarks: made by the b Yukawa alone, zero in sm-no_b_mass
    q = lambda g: [t for t in g.split(">")[0].split() + g.split(">")[1].split() if t.rstrip("~") in "udscb" and t]
    tot["vanished"] = len(van)
    tot["vanished_all_bh"] = all(" h" in r["generate"] and set(x.rstrip("~") for x in q(r["generate"][9:].split("QED")[0]))
                                 <= {"b"} for r in van)
    tot["vanished_nodiagram"] = sum("NoDiagram" in (r.get("error") or "") for r in van)
    return out, tot


def validation():
    rows = {r["name"]: r for r in V["rows"]}
    out = {"n_rows": V["n_rows"], "n_ok": V["n_ok"],
           "n_pool_error": sum("mixture path only" in (r.get("error") or "") for r in rows.values()),
           # the builds ran in the first run; the rerun found them built (0 s)
           "t_build_min": round(min(r["t_build"] for r in V0.values()), 0),
           "t_build_max": round(max(r["t_build"] for r in V0.values()), 0),
           "t_build_mean": round(float(np.mean([r["t_build"] for r in V0.values()])), 0),
           "tool_matches_v3": sum(bool(r["diagrams"].get("tool_matches_v3")) for r in rows.values()),
           "bound_attained": sum(r["diagrams"].get("bound_attained") is True for r in rows.values()),
           "bound_tested": sum(r["diagrams"].get("bound_attained") is not None for r in rows.values()),
           "mt_172": sum(r["card"]["MT"] == 172.5 and r["card"]["ymt"] == 172.5 for r in rows.values()),
           "wt": rows["v4_uux_ddx"]["card"]["WT"], "wt_mt50": rows["v4_bb_abtwm__mt50"]["card"]["WT"],
           "wt_mt300": rows["v4_bb_abtwm__mt300"]["card"]["WT"],
           "no_mb": sum(r["card"]["MB"] in (None, 0.0) and r["card"]["ymb"] in (None, 0.0) for r in rows.values()),
           "model_read": sum(r["card"]["model"] is not None for r in rows.values())}
    for n in ("v4_uux_ddx", "v4_uu_uuz", "v4_ud_dhu", "v4_bbx_wpwm", "v4_bbx_ggtaptamz"):
        d = rows[n]["diagrams"]
        out[n] = {"v3": d["diagrams_v3_recorded"], "v4": d["v4"], "fivefs": d["v3_string_5fs"]}
    out["card_ok"] = sum(bool(r.get("card_ok")) for r in rows.values())
    L = V["loops"]
    out["loops"] = {"entries": L["entries"], "bad": len(L["bad"]),
                    "loop_qcd_qed_import": L["models"]["loop_qcd_qed_sm-no_b_mass"],
                    "generate_ok": sum(g["ok"] for g in L.get("generate", {}).values()),
                    "generate_n": len(L.get("generate", {}))}
    from tools import nlo_virtual_pipeline as NV
    out["loops"]["loop_qcd_qed_entries"] = sum(v.get("_v4", False) and v["model"] == "loop_qcd_qed_sm-no_b_mass"
                                               for v in NV.VIRT_PROCESSES.values())
    return out


def disk():
    lines = [l.split() for l in open(os.path.join(D, "v4_disk.txt")) if l.startswith("V")]
    ev = [int(l[1]) for l in lines if l[0] == "V4SIZE_KB" and l[2].endswith("_events")]
    sa = [int(l[1]) for l in lines if l[0] == "V4SIZE_KB" and l[2].endswith("_standalone")]
    free = next(int(l[1]) for l in lines if l[0] == "V4SIZE_FREE_KB")
    n3 = next(int(l[1]) for l in lines if l[0] == "V3_EVENTS_DIRS")
    import mg5_pipeline_final as mg
    gb, rows, back, nent = 0.0, 0, set(), 0
    for s in (1, 2, 3, 4):
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_gen_stage{s}.yaml")))["processes"]:
            b = e.get("base", e["name"]); k = len(mg.PROCESSES[b]["pdg_ids"])
            r = sum(e.get(x, 0) for x in ("n_train", "n_val", "n_test"))
            gb += r * (5 * k + 1) * 8 / 1e9; rows += r; nent += 1
            if not b.endswith("_nlo"):
                back.add(mg.standalone_name(b))
    mev, msa = np.mean(ev) * 1.024e-3, np.mean(sa) * 1.024e-3   # MB (1e6 bytes) per backend; du -k counts KiB
    return {"events_mb": round(mev, 0), "standalone_mb": round(msa, 1), "n_built": len(ev),
            "free_gb": round(free * 1.024e-6, 1), "pool_entries": nent, "pool_rows_m": round(rows / 1e6, 0),
            "pool_gb": round(gb, 0), "tree_backends": len(back),
            "backends_gb": round(len(back) * (mev + msa) / 1e3, 0),
            "backends_sa_gb": round(len(back) * msa / 1e3, 1),
            "total_gb": round(gb + len(back) * (mev + msa) / 1e3, 0),
            "total_sa_gb": round(gb + len(back) * msa / 1e3, 0),
            "v3_events_dirs": n3, "v3_events_gb": round(n3 * mev / 1e3, 0)}


def local_draws(name, spec):
    """the entry registered as the regeneration does; NLOC draws of its log-uniform measure through the pipeline's
    candidate path with the cuts (bulk: no turn, forward: every draw turned), and the validation's forward replay
    (raw draws, no cuts, turned by rotate_forward with the job's seeds) giving the turned particle's u"""
    import mg5_pipeline_final as mg
    sys.path.insert(0, HERE)
    import validate_v4 as vv
    vv.register(spec)
    c = mg.PROCESSES[name]; lo, hi = spec["sqrts"]
    rng = np.random.default_rng(1)
    sm = mg._sqrts_sampler(lo, hi, c.get("sqrts_measure"), rng)
    cut = [mg._candidates(NLOC, sm, c["m_finals"], c["pdg_ids"], rng, c["fiducial_cuts"], 0.0, 1e-6, ff,
                          c.get("beam_masses"))[1] for ff in (0.0, 1.0)]
    seed = 7000 + sorted(vv.sample()).index(name)
    k = len(c["pdg_ids"]); r = np.random.default_rng(seed + 1)
    mg._sqrts_sampler(lo, hi, "log", r)(vv.NKS)
    ev, _ = mg._candidates(vv.NKS, mg._sqrts_sampler(lo, hi, "log", r), [float(m) for m in c["m_finals"]],
                           list(c["pdg_ids"]), r, None, 0.5, 1e-6, 0.0, {})
    cap = float(c["fiducial_cuts"]["cos_max"])
    Q = mg.rotate_forward(np.stack([e[0] for e in ev]), cap, np.random.default_rng(seed + 2))
    j = np.random.default_rng(seed + 2).integers(2, k, vv.NKS)
    return mg.sampling_policy(c), cut, vv.u_of(Q[np.arange(vv.NKS), j]), float(np.log(1.0 / (1.0 - cap)))


def pools():
    """the rerun's pool checks over the 35 entries, and the pool's sqrt(s_hat) per decade against log-uniform and
    against log-uniform after the cuts (the bulk and forward parts drawn locally; the flat-|M|^2 part needs labels)"""
    sys.path.insert(0, HERE)
    import mg5_pipeline_final as mg
    import validate_v4 as vv
    S = vv.sample()
    for spec in S.values():
        vv.register(spec)
    rows = [r for r in V["rows"] if "error" not in r]
    ks = [r["ks_log_sqrts"] for r in rows]; fk = [r["forward"]["ks_u"] for r in rows]
    top = [r for r in rows if 6 in [abs(p) for p in mg.PROCESSES[r["name"]]["pdg_ids"][2:]]]
    out = {"n": len(rows), "errors": len(V["rows"]) - len(rows),
           "cuts_pass": sum(r["pass_v4"] == 1.0 for r in rows),
           "outside_v3_min": round(min(r["outside_v3_cuts"] for r in rows if "outside_v3_cuts" in r), 2),
           "outside_v3_max": round(max(r["outside_v3_cuts"] for r in rows if "outside_v3_cuts" in r), 2),
           "in_window": sum(r["in_window"] for r in rows),
           "ks_sqrts_pass": sum(d < c for d, c in ks), "ks_sqrts_max": round(max(d for d, _ in ks), 4),
           "ks_crit": round(ks[0][1], 4),
           "ks_u_pass": sum(d < c for d, c in fk), "ks_u_max": round(max(d for d, _ in fk), 4),
           "u_replay_maxdev": float(f"{max(r['forward']['u_replay_maxdev'] for r in rows):.0e}"),
           "fwd_conservation_max": float(f"{max(r['forward']['conservation_rel'] for r in rows):.0e}"),
           "fwd_mass_dev_max": float(f"{max(r['forward']['mass_dev_rel'] for r in rows):.0e}"),
           "conservation_max": float(f"{max(r['conservation_rel'] for r in rows):.0e}"),
           "mass_dev_max": float(f"{max(r['mass_dev_rel'] for r in rows):.0e}"),
           "beams_massless": sum(r["beams_massless"] for r in rows), "b_massless": sum(r["b_masses_zero"] for r in rows),
           "top_entries": len(top), "labels_positive_min": round(min(r["labels_positive"] for r in rows), 3),
           "fwd_share_v3_edge_min": round(min(r["forward_share_v3_edge"] for r in rows), 2),
           "fwd_share_v3_edge_max": round(max(r["forward_share_v3_edge"] for r in rows), 2),
           "ms_per_event_min": round(1e3 * min(r["sec_per_event"] for r in rows), 2),
           "ms_per_event_max": round(1e3 * max(r["sec_per_event"] for r in rows), 2)}
    # the pool's decades: binned CDF distance (a lower bound on the KS distance) to log-uniform and to log-uniform
    # after the cuts; the lowest decade's share in the pool, after the cuts and under log-uniform
    dec, loc = {}, {}
    for r in rows:
        pol, cut, u, ue = local_draws(r["name"], S[r["name"]])
        e = np.log10(r["decades"]["edges"]); fw = float(pol.get("f_fwd", 0)); fb = 1 - float(pol["f_flat"]) - fw
        exp = (fb * np.histogram(np.log10(cut[0]), e)[0] + fw * np.histogram(np.log10(cut[1]), e)[0]) / (NLOC * (fb + fw))
        sh, lu = np.array(r["decades"]["share"]), np.array(r["decades"]["log_uniform"])
        cdf = lambda x: np.cumsum(x)[:-1]
        dec[r["name"]] = {"d_log_uniform": float(np.abs(cdf(sh) - cdf(lu)).max()),
                          "d_after_cuts": float(np.abs(cdf(sh) - cdf(exp)).max()),
                          "low": [float(sh[0]), float(exp[0]), float(lu[0])]}
        loc[r["name"]] = (exp, u, ue)
    dl = [v["d_log_uniform"] for v in dec.values()]; dc = [v["d_after_cuts"] for v in dec.values()]
    out["pool_d_log_uniform_median"] = round(float(np.median(dl)), 3)
    out["pool_d_log_uniform_max"] = round(float(max(dl)), 2)
    out["pool_d_after_cuts_median"] = round(float(np.median(dc)), 3)
    out["pool_d_after_cuts_max"] = round(float(max(dc)), 2)
    for n in FIG_ENTRIES:
        out[n] = {"low_pool": round(dec[n]["low"][0], 3), "low_cuts": round(dec[n]["low"][1], 3),
                  "low_log_uniform": round(dec[n]["low"][2], 3)}
    return out, loc


def regeneration(disk_res):
    """the full regeneration's CPU from the measured seconds per kept event (per number of finals, mean over the
    sample; 2000-event chunks with 16 entries at once on one node, so a chunk's startup is in it) and its disk"""
    import mg5_pipeline_final as mg
    rows = [r for r in V["rows"] if "error" not in r]
    nf = lambda n: len(mg.PROCESSES[n]["pdg_ids"]) - 2
    spe = {}
    for r in rows:
        spe.setdefault(nf(r["name"]), []).append(r["sec_per_event"])
    m = {k: float(np.mean(v)) for k, v in spe.items()}
    cpu, ev, nl, nt = 0.0, 0, 0, 0
    for s in (1, 2, 3, 4):
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_gen_stage{s}.yaml")))["processes"]:
            b = e.get("base", e["name"]); n = sum(e.get(x, 0) for x in ("n_train", "n_val", "n_test"))
            if b.endswith("_nlo"):
                nl += 1; continue
            k = nf(b); cpu += n * m[min(k, max(m))] / 3600; ev += n; nt += 1
    inv = json.load(open(os.path.join(ROOT, "analysis", "storage", "inventory.json")))["mg5amcnlo"]["catalog_v4_projection"]
    lines = [l.split() for l in open(os.path.join(D, "v4_disk_rerun.txt")) if l.startswith("V")]
    free = next(int(l[1]) for l in lines if l[0] == "V4SIZE_FREE_KB") * 1.024e-6
    return {"ms_per_event_by_nfinal": {str(k): round(1e3 * v, 2) for k, v in sorted(m.items())},
            "tree_entries": nt, "loop_entries": nl, "tree_events_m": round(ev / 1e6, 0),
            "tree_cpu_h": round(cpu, 0), "backends_no_events_gb": inv["without_events_dirs_GB"],
            "pool_gb": disk_res["pool_gb"], "total_gb": round(disk_res["pool_gb"] + inv["without_events_dirs_GB"], 0),
            "free_gb": round(free, 1)}


def figure_pools(loc):
    import plot_style as ps
    fig, (axL, axR) = ps.figure(ncols=2)
    rows = {r["name"]: r for r in V["rows"]}
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green]
    lab = {"v4_uux_ddx": r"$u\bar u\to d\bar d$", "v4_ddx_ddxgg": r"$d\bar d\to d\bar dgg$",
           "v4_epem_ccxgvevex": r"$e^+e^-\to c\bar cg\nu_e\bar\nu_e$"}
    for c, n in zip(cols, FIG_ENTRIES):
        d = rows[n]["decades"]; e = np.log10(d["edges"]); w = np.diff(e)
        exp, u, ue = loc[n]
        axL.stairs(np.array(d["share"]) / w, e, color=c, label=lab[n], baseline=None)
        axL.stairs(exp / w, e, color=c, linestyle="--", baseline=None)
        axL.stairs(np.array(d["log_uniform"]) / w, e, color=c, linestyle=":", baseline=None)
        h, b = np.histogram(u, np.linspace(0, ue, 21), density=True)
        axR.stairs(h, b, color=c, baseline=None)
        axR.plot([0, ue], [1 / ue] * 2, color=c, linestyle=":")
    axL.plot([], [], color="k", label="sample"); axL.plot([], [], color="k", linestyle="--", label="after the cuts")
    axL.plot([], [], color="k", linestyle=":", label="target")
    axL.set_xlabel(r"$\log_{10}\sqrt{\hat s}$ [GeV]"); axL.set_ylabel(r"$dP/d\log_{10}\sqrt{\hat s}$")
    axR.set_xlabel(r"$u=\log(1/(1-|\cos\theta|))$"); axR.set_ylabel(r"$dP/du$")
    axR.set_ylim(0, 0.62)
    ps.legend(axR, "upper right", handles=axL.get_legend_handles_labels()[0], ncol=2)
    ps.save(fig, os.path.join(HERE, "figs", "pool_sqrts_forward"))


def figure():
    import plot_style as ps
    fig, ax = ps.figure()
    rng = np.random.default_rng(0)
    cols = [ps.C.blue, ps.C.orange, ps.C.green, ps.C.purple]
    for i, L in enumerate(sorted({r["layer"] for r in C})):
        ex = [r for r in C if r["layer"] == L and r["exists"]]
        x = np.array([r["diagrams_v3"] for r in ex], float); y = np.array([r["diagrams"] for r in ex], float)
        j = lambda a: a * np.exp(rng.uniform(-0.06, 0.06, len(a)))        # jitter on the integer lattice
        ax.scatter(j(x), j(y), s=4, alpha=0.5, color=cols[i], label=rf"$2\to{L}$")
    lim = [0.7, 2e3]
    ax.plot(lim, lim, color="k", linestyle="--", label="equal")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("diagrams, catalog v3"); ax.set_ylabel("diagrams, catalog v4")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "census_diagrams"))


def dropped():
    """what the drop of the vanished trees (D136, recipes/gen_catalog_v4.py) took out of the generated v4 files: trees,
    one-loop entries (a virtual on a vanished Born) and recipe entries named after or based on them, per role"""
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    import gen_catalog_v4 as g
    van = {r["name"] for r in C if not r["exists"]}
    P = g.load(os.path.join(ROOT, "recipes", "catalog_v4_processes.yaml"))["processes"]
    L3, L4 = (g.load(os.path.join(ROOT, "recipes", f"catalog_{v}_loops.yaml")) for v in ("v3", "v4"))
    out = {"trees": len(van), "trees_left_in_v4": len(van & set(P)),
           "loop_entries": len(L3["processes"]) - len(L4["processes"]), "loop_strings": len(L3["virt"]) - len(L4["virt"])}
    per = {}
    for r in g.RECIPES:
        n3 = sum(not e["name"].endswith("__E10") for e in g.load(os.path.join(ROOT, "recipes", f"catalog_v3_{r}.yaml"))["processes"])
        n4 = len(g.load(os.path.join(ROOT, "recipes", f"catalog_v4_{r}.yaml"))["processes"])
        if n3 != n4:
            per[r] = n3 - n4
    out["recipe_entries"] = sum(per.values()); out["by_recipe"] = per
    return out


def main():
    by, tot = census()
    res = {"census": by, "total": tot, "val": validation(), "disk": disk(), "drop": dropped()}
    res["pools"], loc = pools()
    res["regen"] = regeneration(res["disk"])
    # the tree backends' build, one core each (35 builds on 16 workers took 64 s of wall): measured mean x count
    res["disk"]["build_core_h"] = round(res["disk"]["tree_backends"] * res["val"]["t_build_mean"] / 3600, 0)
    if "--no-fig" not in sys.argv:
        figure(); figure_pools(loc)
    print(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
