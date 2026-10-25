"""catalog_v4's validation and census read (run FA-ccin2p3-44e6a15b, scripts/job_catalog_v4_validate.sh).

Inputs (analysis/catalog_v4/data/, fetched from CC): validation.json (analysis/catalog_v4/validate_v4.py: per sampled
entry the diagram counts, the backend build and its cards, the pool checks), checked.json (recipes/gen_catalog_v4.py
--check: every catalog_v3 tree on its v4 string through MadGraph), v4_disk.txt (the job's du of the v4 backends it
built, the free space, and the count of catalog_v3's MadGraph events dirs).
Output: figs/census_diagrams (v4 against v3 diagram count per string, by layer) and, last line, the JSON of the numbers
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
           "t_build_min": round(min(r["t_build"] for r in rows.values()), 0),
           "t_build_max": round(max(r["t_build"] for r in rows.values()), 0),
           "t_build_mean": round(float(np.mean([r["t_build"] for r in rows.values()])), 0),
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
    L = V["loops"]
    out["loops"] = {"entries": L["entries"], "bad": len(L["bad"]),
                    "loop_qcd_qed_import": L["models"]["loop_qcd_qed_sm-no_b_mass"]}
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


def main():
    by, tot = census()
    res = {"census": by, "total": tot, "val": validation(), "disk": disk()}
    # the tree backends' build, one core each (35 builds on 16 workers took 64 s of wall): measured mean x count
    res["disk"]["build_core_h"] = round(res["disk"]["tree_backends"] * res["val"]["t_build_mean"] / 3600, 0)
    if "--no-fig" not in sys.argv:
        figure()
    print(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
