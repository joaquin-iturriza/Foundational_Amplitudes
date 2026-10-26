"""catalog_v4 against catalog_v3 as pretraining catalogs (catalog_coverage): the dimensionless arm with Muon at the Muon
search's best, two seeds per catalog (42, 43), 4264 steps each (scripts/job_catalog_v3_short.sh; catalog_v3:
dimensionless_muon_best, dimensionless_muon_best_seed2; catalog_v4: v4_dimensionless_muon, v4_dimensionless_muon_seed2).
Every read is zero-shot at the run's best checkpoint (tools/zero_shot_parent.py, 2000 validation events per pool;
predictions copied home into analysis/catalog_v3/data/short_reads/<run>/<axis>/<probe>.npz). Per probe, from the saved
truth t and prediction q (the run's target, log|M|^2 up to the target factor, which cancels in q - t):
  mse  mean (q - t)^2;  level  (mean (q - t))^2;  shape  var(q - t)       (analysis/catalog_v3/short_reads.py)
Three comparisons:
  shared  the 555 processes held out in both catalogs (recipes/make_catalog_v3v4_shared.py), each run read on its own
          catalog's pool of the process and on the other catalog's (XV3/XV4): on catalog_v3's pools and on catalog_v4's,
          per axis (catalog_v3's axis of the process), processes with a b quark apart (catalog_v3's b is 4.7 GeV,
          catalog_v4's massless: the two pools of such a process are different |M|^2 functions, and each run reads
          the other catalog's on a b it never saw)
  pp      the 13 TeV MadEvent pools of the 1396 hadronic tree classes (recipes/catalog_v3_pp_test.yaml), which neither
          catalog trains on, by the class's role in each catalog (training or held out in both; the role changes
          apart) and split into the events inside what catalog_v3's window pools cover (sqrt(s_hat) window and
          catalog_v3's fiducial cuts in the partonic frame, analysis/catalog_v3/pp_compare.py) and the rest; b
          classes apart (the pp pools have catalog_v3's 4.7 GeV b, so they are off catalog_v4's b)
  own     each catalog's own held-out axes, per run: different pools, cuts and windows, not comparable across catalogs
Per group the median MSE over probes for each run; the catalog ratio is the median over probes of the per-probe ratio of
the two catalogs' seed-geometric-mean MSEs (catalog_v4 over catalog_v3), with the share of probes lower in both seeds;
the gap (median |log10 ratio|) is read against each catalog's seed spread (median |log10 seed 43 / seed 42|), the
convention of analysis/catalog_v3/muon_compare.py, and a difference exceeds the seed spread when the gap is larger
than both. A pp class's catalog_v4 role is its catalog_v4 class's (data/class_changes.json: a catalog_v3 class merged
into or split across catalog_v4 classes takes the role of the one holding its tree); the 33 b-Higgs classes catalog_v4
dropped have none and are counted apart.
    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/compare_v3.py collect > analysis/catalog_v4/data/compare_v3_site.json
    python3 analysis/catalog_v4/compare_v3.py   -> data/compare_v3.json, figs/compare_v3_{shared,pp}.{png,pdf}
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import glob, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v3"))
SR = os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads")
RUNS = {"v3": ("dimensionless_muon_best", "dimensionless_muon_best_seed2"),
        "v4": ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2")}
RUN_DIRS = {"dimensionless_muon_best": "runs/catalog_v3_short/dimensionless_muon_best",
            "dimensionless_muon_best_seed2": "runs/catalog_v3_short/dimensionless_muon_best_seed2",
            "v4_dimensionless_muon": "runs/catalog_v4_short/v4_dimensionless_muon",
            "v4_dimensionless_muon_seed2": "runs/catalog_v4_short/v4_dimensionless_muon_seed2"}
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
NAMES = {"H1": "new combinations", "H2": r"$2\to4$ families", "H2p5": r"$2\to5$", "H3": "the top", "H5": "one loop",
         "H6": r"$\alpha_s=0.13$", "H7E": "energy", "H7c": r"$p_T\geq30$ GeV"}
CHECK = 40                                    # every 50th of the 2000 sorted log|M|^2, to check the event match


def collect():
    """on the site: per pp probe, which scored events lie inside catalog_v3's window coverage, ordered by the event's
    log|M|^2 (the reads' loader does not keep the pool's row order; matched at home by sorting, as pp_compare.collect
    does), with a sample of the sorted log|M|^2 to check that match; and each run's result.json step counts"""
    import pp_compare
    sup = pp_compare.support_masks()
    out = {"pp": {}, "runs": {}}
    for p, m in sup.items():
        o = np.argsort(m["log_amp"], kind="stable")
        out["pp"][p] = {"inside": "".join("1" if x else "0" for x in (m["in_cuts"] & m["in_sqrts"])[o]),
                        "check": [round(float(x), 6) for x in m["log_amp"][o][::2000 // CHECK]]}
    for r, d in RUN_DIRS.items():
        res = json.load(open(os.path.join(ROOT, d, "result.json")))
        out["runs"][r] = {k: res.get(k) for k in ("best_step", "iterations", "n_steps", "steps", "best_val_loss",
                                                  "val_loss_no_reg", "diverged")}
        cfg = yaml.safe_load(open(os.path.join(ROOT, d, "config.yaml")))
        out["runs"][r].update({"cfg_iterations": cfg["training"].get("iterations"), "seed": cfg.get("seed"),
                               "optimizer": cfg["training"].get("optimizer"), "lr": cfg["training"].get("lr"),
                               "processes_file": cfg["data"].get("processes_file")})
    json.dump(out, sys.stdout)


def red(f):
    z = np.load(f)
    if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
        return None
    d = z["q"].astype(float) - z["t"].astype(float)
    return {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2), "shape": float(np.var(d))}


def axis_of(cat):
    """name without prefix -> axis, from the catalog's held-out axis recipes"""
    out = {}
    for a in AXES:
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_{cat}_heldout_{a}.yaml")))["processes"]:
            out[e["name"][3:]] = a
    return out


def has_b(name):
    import mg5_pipeline_final as mg
    return any(abs(int(x)) == 5 for x in mg.PROCESSES[name]["pdg_ids"])


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def compare(rows, probes):
    """rows[run][probe] -> mse, on the probes every run has: each run's median; per probe the ratio of the two catalogs'
    seed-geometric-mean MSEs (catalog_v4 over catalog_v3), its median and 10%/90% quantiles; the share of probes lower
    with catalog_v4 in both seeds (and with catalog_v3 in both); the gap, the median over probes of |log10 ratio|,
    read against each catalog's seed spread, the median over probes of |log10 (seed 43 / seed 42)|, as
    analysis/catalog_v3/muon_compare.py compare() does; the difference exceeds the seed spread when the gap is larger
    than both catalogs' spreads"""
    ps_ = [p for p in probes if all(p in rows[r] for c in RUNS for r in RUNS[c])]
    if not ps_:
        return None
    e = {"n": len(ps_), "n_expected": len(probes), "missing": sorted(set(probes) - set(ps_))}
    for c in RUNS:
        e[c] = [float(np.median([rows[r][p] for p in ps_])) for r in RUNS[c]]
    A = {c: [np.array([rows[r][p] for p in ps_]) for r in RUNS[c]] for c in RUNS}
    ratio = np.sqrt(A["v4"][0] * A["v4"][1]) / np.sqrt(A["v3"][0] * A["v3"][1])
    lr = np.log10(ratio)
    e["ratio"] = float(np.median(ratio))
    e["ratio_q10_q90"] = [float(x) for x in np.quantile(ratio, [0.1, 0.9])]
    e["v4_lower_both_pct"] = float(100 * np.mean((A["v4"][0] < A["v3"][0]) & (A["v4"][1] < A["v3"][1])))
    e["v3_lower_both_pct"] = float(100 * np.mean((A["v3"][0] < A["v4"][0]) & (A["v3"][1] < A["v4"][1])))
    e["gap"] = float(np.median(np.abs(lr)))
    e["seed_spread"] = {c: float(np.median(np.abs(np.log10(A[c][1] / A[c][0])))) for c in RUNS}
    e["exceeds_seed_spread"] = bool(e["gap"] > max(e["seed_spread"].values()))
    e["v3_med"], e["v4_med"] = gmean(e["v3"]), gmean(e["v4"])
    return e


def shared():
    s3 = [e["name"][3:] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_shared_heldout.yaml")))["processes"]]
    a3, a4 = axis_of("v3"), axis_of("v4")
    reads = {}                                         # pool set -> run -> name -> row
    for pool in ("v3", "v4"):
        for c in RUNS:
            for r in RUNS[c]:
                d = {}
                for n in s3:
                    f = (os.path.join(SR, r, (a3 if pool == "v3" else a4)[n], f"{pool}_{n}.npz") if pool == c else
                         os.path.join(SR, r, "XV4" if pool == "v4" else "XV3", f"{pool}_{n}.npz"))
                    if os.path.exists(f):
                        x = red(f)
                        if x:
                            d[n] = x
                reads.setdefault(pool, {})[r] = d
    out = {"n_shared": len(s3), "axis_pairs": {}}
    for n in s3:
        k = f"{a3[n]}/{a4[n]}"
        out["axis_pairs"][k] = out["axis_pairs"].get(k, 0) + 1
    base = {e["name"][3:]: e.get("base", e["name"]) for e in
            yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_shared_heldout.yaml")))["processes"]}
    bq = {n: has_b(base[n]) for n in s3}
    for pool in ("v3", "v4"):
        mse = {r: {n: v["mse"] for n, v in reads[pool][r].items()} for r in reads[pool]}
        lev = {r: {n: v["level"] for n, v in reads[pool][r].items()} for r in reads[pool]}
        shp = {r: {n: v["shape"] for n, v in reads[pool][r].items()} for r in reads[pool]}
        for grp, sel in (("all_no_b", lambda n: not bq[n]), ("b", lambda n: bq[n])):
            out.setdefault(pool, {})[grp] = {"all": compare(mse, [n for n in s3 if sel(n)]),
                                             "level": compare(lev, [n for n in s3 if sel(n)]),
                                             "shape": compare(shp, [n for n in s3 if sel(n)])}
            for a in AXES:
                pr = [n for n in s3 if sel(n) and a3[n] == a]
                if pr:
                    out[pool][grp][a] = compare(mse, pr)
    out["n_b"] = sum(bq.values())
    out["_rows"] = {pool: {r: {n: v["mse"] for n, v in reads[pool][r].items()} for r in reads[pool]} for pool in reads}
    out["_axis"] = a3
    out["_b"] = bq
    return out


def pp(site):
    import pp_compare
    R3 = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"]
    R4 = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_roles.json")))["trees"]
    names = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]]
    cc = json.load(open(os.path.join(HERE, "data", "class_changes.json")))
    m4 = {k: v["role"] for k, v in R4.items()}
    for g in cc["merges"]:
        for x in g["members"]:
            m4.setdefault(x, g["role"][0])
    for sp in cc["splits"]:
        for v in sp["v4_classes"].values():
            for x in v["members"]:
                m4.setdefault(x, v["role"][0])
    role = {}
    for n in names:
        x = n[len("pp13_"):]
        r3 = R3["v3_" + x]["role"]
        r4 = m4.get("v4_" + x, "absent")
        role[n] = {"v3": r3, "v4": r4, "axis3": R3["v3_" + x].get("axis")}
    rows = {k: {} for k in ("mse", "mse_inside", "mse_outside")}
    bad, frac, srt = {}, {}, {}
    for c in RUNS:
        for r in RUNS[c]:
            for k in rows:
                rows[k][r] = {}
            bad[r] = []
            for n in names:
                f = os.path.join(SR, r, "PP", n + ".npz")
                if not os.path.exists(f):
                    bad[r].append(n)
                    continue
                z = np.load(f)
                if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
                    bad[r].append(n)
                    continue
                d = z["q"].astype(float) - z["t"].astype(float)
                rows["mse"][r][n] = float(np.mean(d ** 2))
                s = site["pp"].get(n)
                if s is None:
                    continue
                lf = z["logF"].astype(float) if ("logF" in z.files and len(z["logF"]) == len(d)) else 0.0
                la = z["t"].astype(float) - lf
                srt.setdefault(n, []).append(np.sort(la))
                o = np.argsort(la, kind="stable")
                chk = np.max(np.abs(la[o][::2000 // CHECK] - np.array(s["check"])))
                if chk > 1e-3:                 # not the same events: no split for this probe, counted
                    bad[r].append(n + ":order")
                    continue
                ins = np.empty(len(d), bool)
                ins[o] = np.frombuffer(s["inside"].encode(), dtype=np.uint8) == ord("1")
                frac[n] = float(ins.mean())
                if ins.any():
                    rows["mse_inside"][r][n] = float(np.mean(d[ins] ** 2))
                if (~ins).any():
                    rows["mse_outside"][r][n] = float(np.mean(d[~ins] ** 2))
    # the 40 sorted quantiles checked each read against the pool; the four runs' reads of a probe are checked against
    # each other on all 2000 sorted events (the same events, whatever their order)
    xmax = max(float(np.max(np.abs(a - v[0]))) for v in srt.values() for a in v[1:])
    assert xmax < 1e-3, f"the runs' reads of a pp probe are not the same events (max {xmax})"
    b = {n: has_b(n) for n in names}
    groups = {"train": lambda n: role[n]["v3"] == "train" and role[n]["v4"] == "train",
              "heldout": lambda n: role[n]["v3"] == "heldout" and role[n]["v4"] == "heldout",
              "role_changed": lambda n: role[n]["v3"] != role[n]["v4"] and role[n]["v4"] != "absent",
              "dropped_in_v4": lambda n: role[n]["v4"] == "absent"}
    out = {"_rows": rows, "_role": role, "_b": b, "_frac_inside": frac, "n": len(names), "max_event_mismatch": xmax, "no_prediction": {r: v for r, v in bad.items() if v}, "n_b": sum(b.values()),
           "roles": {}, "frac_inside": float(np.median(list(frac.values()))) if frac else None}
    for n in names:
        k = f"{role[n]['v3']}/{role[n]['v4']}" + ("/b" if b[n] else "")
        out["roles"][k] = out["roles"].get(k, 0) + 1
    for bk, bsel in (("no_b", lambda n: not b[n]), ("b", lambda n: b[n])):
        for g, gsel in groups.items():
            pr = [n for n in names if bsel(n) and gsel(n)]
            if not pr:
                continue
            e = out.setdefault(bk, {}).setdefault(g, {})
            for k in rows:
                e[k] = compare(rows[k], pr)
            e["frac_inside"] = float(np.median([frac[n] for n in pr if n in frac])) if any(n in frac for n in pr) else None
            if g == "heldout":                 # per held-out axis of catalog_v3
                for a in ("H1", "H2", "H3"):
                    pa = [n for n in pr if role[n]["axis3"] == a]
                    if pa:
                        e[a] = compare(rows["mse"], pa)
    return out


def own():
    out = {}
    for c in RUNS:
        for a in AXES:
            exp = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_{c}_heldout_{a}.yaml")))["processes"]]
            for r in RUNS[c]:
                v = [red(os.path.join(SR, r, a, n + ".npz")) for n in exp if os.path.exists(os.path.join(SR, r, a, n + ".npz"))]
                v = [x["mse"] for x in v if x]
                out.setdefault(c, {}).setdefault(a, {})[r] = {"n": len(v), "n_expected": len(exp),
                                                              "mse": float(np.median(v)) if v else None}
    return out


def plot(sh, p):
    import plot_style as ps
    # shared: per axis, the four runs' median MSE on each pool set (no b), one panel per pool set
    fig, axs = ps.figure(ncols=2)
    for ax, pool, lab in zip(axs, ("v3", "v4"), ("catalog v3's pools", "catalog v4's pools")):
        ax_ = [a for a in AXES if sh[pool]["all_no_b"].get(a)]
        x = np.arange(len(ax_))
        for c, col, mk, dx in (("v3", ps.C.blue, "o", -0.12), ("v4", ps.C.vermillion, "s", 0.12)):
            for j in range(2):
                y = [sh[pool]["all_no_b"][a][c][j] for a in ax_]
                ax.plot(x + dx, y, mk, color=col, mfc=col if j == 0 else "none",
                        label=f"catalog {c}, seed {42 + j}")
        ax.set_xticks(x)
        ax.set_xticklabels([a for a in ax_])
        ax.set_yscale("log")
        ax.set_ylabel("median MSE")
        ax.set_ylim(0.008, 60)                           # one range for both panels, room for the legend
        ax.set_xlabel("held-out axis, on " + lab)
    ps.legend(axs[0], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "compare_v3_shared"))
    # pp: per group, the four runs' median MSE on the whole pool, inside and outside catalog_v3's coverage
    fig, axs = ps.figure(ncols=2)
    for ax, g, lab in zip(axs, ("train", "heldout"), ("trained in both", "held out in both")):
        ks = [("mse", "pp pool"), ("mse_inside", "inside v3 coverage"), ("mse_outside", "outside v3 coverage")]
        x = np.arange(len(ks))
        for c, col, mk, dx in (("v3", ps.C.blue, "o", -0.12), ("v4", ps.C.vermillion, "s", 0.12)):
            for j in range(2):
                y = [p["no_b"][g][k][c][j] for k, _ in ks]
                ax.plot(x + dx, y, mk, color=col, mfc=col if j == 0 else "none", label=f"catalog {c}, seed {42 + j}")
        ax.set_xticks(x)
        ax.set_xticklabels(["whole pool", "inside", "outside"])
        ax.set_yscale("log")
        ax.set_ylabel("median MSE, 13 TeV pp")
        ax.set_ylim(0.08, 30)                            # one range for both panels, room for the legend
        ax.set_xlabel(f"classes {lab}")
    ps.legend(axs[0], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "compare_v3_pp"))


def main():
    site = json.load(open(os.path.join(HERE, "data", "compare_v3_site.json")))
    sh, p, o = shared(), pp(site), own()
    res = {"shared": sh, "pp": {k: v for k, v in p.items() if not k.startswith("_")}, "own": o, "runs": site["runs"]}
    json.dump(res, open(os.path.join(HERE, "data", "compare_v3.json"), "w"), indent=1)
    plot(sh, p)
    fmt = lambda e: (f"n={e['n']:3d} v3 {e['v3'][0]:.3g}/{e['v3'][1]:.3g} v4 {e['v4'][0]:.3g}/{e['v4'][1]:.3g} "
                     f"ratio {e['ratio']:.3g} [{e['ratio_q10_q90'][0]:.2g},{e['ratio_q10_q90'][1]:.2g}] "
                     f"v4 lower both {e['v4_lower_both_pct']:.0f}% v3 lower both {e['v3_lower_both_pct']:.0f}% gap {e['gap']:.3g} "
                     f"spread v3 {e['seed_spread']['v3']:.3g} v4 {e['seed_spread']['v4']:.3g}"
                     f"{' EXCEEDS' if e['exceeds_seed_spread'] else ''}") if e else "none"
    print("runs", json.dumps(site["runs"]))
    print("shared axis pairs", sh["axis_pairs"], "b", sh["n_b"])
    for pool in ("v3", "v4"):
        for grp in ("all_no_b", "b"):
            for k, e in sh[pool][grp].items():
                print(f"shared on {pool} pools {grp:8s} {k:6s}", fmt(e))
    print("pp roles", p["roles"], "no prediction", p["no_prediction"], "frac inside", p["frac_inside"])
    for bk in ("no_b", "b"):
        for g, e in p.get(bk, {}).items():
            for k, v in e.items():
                if isinstance(v, dict) or v is None:
                    print(f"pp {bk:4s} {g:12s} {k:12s}", fmt(v))
    for c in o:
        for a, e in o[c].items():
            print(f"own {c} {a:5s}", {r: (v["n"], v["n_expected"], v["mse"] and round(v["mse"], 3)) for r, v in e.items()})
    claims = {"shared": {pool: {grp: {k: {kk: e[kk] for kk in ("n", "ratio", "v4_lower_both_pct", "v3_lower_both_pct",
                                                                "gap", "v3_med", "v4_med", "exceeds_seed_spread")} | {"spread_v3": e["seed_spread"]["v3"], "spread_v4": e["seed_spread"]["v4"], "v3_lo": min(e["v3"]),
                                                                "v3_hi": max(e["v3"]), "v4_lo": min(e["v4"]), "v4_hi": max(e["v4"])}
                                     for k, e in sh[pool][grp].items() if e} for grp in sh[pool] if not grp.startswith("_")}
                         for pool in ("v3", "v4")},
              "pp": {bk: {g: {k: {kk: v[kk] for kk in ("n", "ratio", "v4_lower_both_pct", "v3_lower_both_pct", "gap", "v3_med", "v4_med", "exceeds_seed_spread")}
                              | {"spread_v3": v["seed_spread"]["v3"], "spread_v4": v["seed_spread"]["v4"], "v3_lo": min(v["v3"]), "v3_hi": max(v["v3"]), "v4_lo": min(v["v4"]), "v4_hi": max(v["v4"])}
                              for k, v in e.items() if isinstance(v, dict)} for g, e in p[bk].items()} for bk in ("no_b", "b")},
              }
    claims["own"] = {c: {a: {"seed42": o[c][a][RUNS[c][0]]["mse"], "seed43": o[c][a][RUNS[c][1]]["mse"],
                             "n": o[c][a][RUNS[c][0]]["n"]} for a in o[c]} for c in o}
    claims["n_shared"] = sh["n_shared"]
    claims["n_shared_b"] = sh["n_b"]
    claims["n_pp"] = p["n"]
    claims["n_pp_b"] = p["n_b"]
    claims["pp_frac_inside"] = p["frac_inside"]
    print(json.dumps(claims))


if __name__ == "__main__":
    {"collect": collect}.get((sys.argv[1:] or [""])[0], main)()
