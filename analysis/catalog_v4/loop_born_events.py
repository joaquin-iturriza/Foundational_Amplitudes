"""The Born-relative one-loop target per event against its pool-level proxy (v3_prep_explore, the user's first priority
D178, the zero-shot bias; par:v4_loop_born). par:v4_loop_born read the candidate t_virt - t_Born = log|c0 + shift| at
the pool level, as the one-loop pool's level minus its Born tree pool's level (loop_levels.py --loop-born); a proxy,
the tree pools being mixture-shaped and the one-loop pools never shaped. Here the Born is re-evaluated on the one-loop
pools' own train events (tools/nlo_virtual_pipeline.py --born-of: the generator's evaluator, the stored target
recomputed and checked against the stored column), so per event t_virt - t_Born = log|stored| - log(Born) =
log|c0 + shift|, with no F and no tree pool. (For the signed pools the trained target is abslog:s, log max(|x|, s):
below its floor, the 1% quantile, the per-event difference is not log|c0 + shift|; the share of such events is not
read here.)

The pools, fixed before any number was read: five matched held-out pools whose whole matched class (the one rule,
match_key: multiplicity, initial state, born orders) is re-evaluated with them, so their Born-relative departure is
recomputed end to end: v4_bb_bb (class bb as2 a2, 4 particles), v4_bbx_hz and v4_bbx_zz (bb as0 a2, 4), v4_ddx_ag (qq
as1 a1, 4), v4_bbx_ccxh (bb as2 a3, 5); two more held-out pools in large classes (v4_uux_wpwm, v4_ucx_cxhu, step only)
and one training pool with no massive final and a large step (v4_udx_csx). 2000 train events per pool (seed 0).

The check's rule, fixed before reading: the pool-level proxy holds if (a) the median over the checked pools of
|per-event mean - pool-level step| is below 0.2 (a quarter of the median Born-relative departure, 0.851) and (b) no
fully re-evaluated held-out pool changes its verdict (narrower with the Born removed, or not) when its departure is
recomputed from the per-event means of it and its class. Otherwise the read is redone on per-event values (the plan's
second branch).

    on CC, as a CPU job (scripts/job_v4_loop_born.sh):  python analysis/catalog_v4/loop_born_events.py --job
        -> one LOOP_BORN_EVENTS JSON line, saved home as data/loop_born_events_cc.json
    laptop:  python3 analysis/catalog_v4/loop_born_events.py   -> data/loop_born_events.json
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

FULL_HELDOUT = ("v4_bb_bb_nlo", "v4_bbx_hz_nlo", "v4_bbx_zz_nlo", "v4_ddx_ag_nlo", "v4_bbx_ccxh_nlo")
CLASSES = ("v4_bbx_bbx_nlo",                                                    # bb as2 a2, 4
           "v4_bbx_aa_nlo", "v4_bbx_az_nlo", "v4_bbx_taptam_nlo", "v4_bbx_vevex_nlo", "v4_bbx_wpwm_nlo",  # bb as0 a2, 4
           "v4_udx_gwp_nlo",                                                    # qq as1 a1, 4
           "v4_bb_bbz_nlo", "v4_bbx_addx_nlo")                                  # bb as2 a3, 5
EXTRA = ("v4_uux_wpwm_nlo", "v4_ucx_cxhu_nlo", "v4_udx_csx_nlo")
POOLS = FULL_HELDOUT + CLASSES + EXTRA
N_EVENTS, TOL_STEP = 2000, 0.2


def recipe_entries():
    out = {}
    for f in ("catalog_v4_train.yaml", "catalog_v4_heldout_H5.yaml"):
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]:
            out[p["name"]] = p
    return out


def one_pool(name, outdir, n):
    import mg5_pipeline_final as mg
    p = recipe_entries()[name]
    lo, hi = p["sqrts"]
    fs = glob.glob(os.path.join(os.environ["SCRATCH"], "amp_data_cache", f"{name}_{lo}-{hi}GeV_train_*amplitudes.npy"))
    if len(fs) != 1:
        return name, {"error": f"train pool not found uniquely: {fs}"}
    base = mg.PROCESSES[name]["virt_base"]
    out = os.path.join(outdir, f"{name}.npz")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "nlo_virtual_pipeline.py"), base,
                        "--born-of", fs[0], "--n", str(n), "--seed", "0", "--out", out],
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("BORN_OF_POOL ")]
    if r.returncode or not line:
        return name, {"error": f"rc {r.returncode}: {(r.stderr or r.stdout)[-600:]}"}
    s = json.loads(line[-1][len("BORN_OF_POOL "):])
    z = np.load(out)
    ok = z["rc_ok"] & (z["born"] > 0)
    rel = np.log(np.abs(z["stored"][ok])) - np.log(z["born"][ok])          # log|c0 + shift| per event
    nos = np.log(np.abs(z["c0"][ok]))                                       # log|c0|, the shift left out
    s.update({"n_used": int(ok.sum()), "mean_rel": float(rel.mean()), "sd_rel": float(rel.std()),
              "se_rel": float(rel.std() / np.sqrt(ok.sum())), "median_rel": float(np.median(rel)),
              "mean_log_c0": float(nos.mean()), "mean_shift": float(z["shift"][ok].mean()),
              "neg_share": float(np.mean((z["c0"][ok] + z["shift"][ok]) < 0)),
              "mean_log_born": float(np.log(z["born"][ok]).mean())})
    return name, s


def job(n, workers):
    outdir = os.path.join(os.environ["SCRATCH"], f"zz_v4_loop_born_{os.environ.get('SLURM_JOB_ID', 'x')}")
    os.makedirs(outdir, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(lambda nm: one_pool(nm, outdir, n), POOLS))
    for k, v in res.items():
        print(f"  {k:24s} " + (v["error"] if "error" in v else
              f"n {v['n_used']} max_rel {v['max_rel']:.2e} mean log|c0+shift| {v['mean_rel']:+.3f} "
              f"(log|c0| {v['mean_log_c0']:+.3f}, shift {v['mean_shift']:+.2f})"), flush=True)
    print("LOOP_BORN_EVENTS " + json.dumps(res), flush=True)


def read():
    from loop_levels import match_key  # noqa: E402  (the one match rule)
    ev = json.load(open(os.path.join(HERE, "data", "loop_born_events_cc.json")))
    lb = json.load(open(os.path.join(HERE, "data", "loop_born.json")))
    R, dep = lb["rows"], lb["departure"]
    bad = {k: v["error"] for k, v in ev.items() if "error" in v}
    E = {k: v for k, v in ev.items() if "error" not in v}
    rows = {}
    for k, v in E.items():
        r = R[k]
        rows[k] = {"heldout": r["heldout"], "step": r["step"], "per_event": v["mean_rel"], "se": v["se_rel"],
                   "diff": v["mean_rel"] - r["step"], "log_c0": v["mean_log_c0"], "shift": v["mean_shift"],
                   "max_rel": v["max_rel"], "n": v["n_used"], "neg_share": v["neg_share"],
                   "born_gap": v["mean_log_born"]}
    med = float(np.median([abs(r["diff"]) for r in rows.values()]))
    # (b) the fully re-evaluated held-out pools: departure from the per-event means of their class
    P = {k: R[k] for k in R}
    T = [k for k, r in R.items() if not r["heldout"] and r["pole"] is not None]   # as loop_born_read
    flips, full = [], {}
    for h in FULL_HELDOUT:
        if h not in E:
            continue
        cls = [m for m in T if match_key(P, m) == match_key(P, h)]
        if not all(m in E for m in cls):
            continue
        d_ev = E[h]["mean_rel"] - float(np.mean([E[m]["mean_rel"] for m in cls]))
        d0, dB = dep["none"][h], dep["born_rel"][h]
        narrow_proxy, narrow_ev = abs(dB) < abs(d0), abs(d_ev) < abs(d0)
        full[h] = {"class": cls, "dep_none": d0, "dep_born_proxy": dB, "dep_born_event": d_ev,
                   "narrow_proxy": bool(narrow_proxy), "narrow_event": bool(narrow_ev)}
        if narrow_proxy != narrow_ev:
            flips.append(h)
    holds = bool(med < TOL_STEP and not flips)
    out = {"rows": rows, "errors": bad, "full": full, "median_abs_diff": med, "flips": flips, "proxy_holds": holds}
    json.dump(out, open(os.path.join(HERE, "data", "loop_born_events.json"), "w"), indent=1)
    for k, r in sorted(rows.items(), key=lambda x: -abs(x[1]["diff"])):
        print(f"  {k:24s} {'H' if r['heldout'] else 'T'} step {r['step']:+.2f} per-event {r['per_event']:+.2f} "
              f"(+-{r['se']:.2f}) diff {r['diff']:+.2f}  log|c0| {r['log_c0']:+.2f} shift {r['shift']:+.2f} "
              f"neg {r['neg_share']:.2f} max_rel {r['max_rel']:.1e}")
    for h, f in full.items():
        print(f"  {h:24s} departure none {f['dep_none']:+.2f}  Born removed: proxy {f['dep_born_proxy']:+.2f}, "
              f"per event {f['dep_born_event']:+.2f}  narrower: {f['narrow_proxy']} / {f['narrow_event']}")
    flat = {"lbe.n_pools": len(rows), "lbe.n_errors": len(bad), "lbe.median_abs_diff": round(med, 3),
            "lbe.n_full": len(full), "lbe.n_flips": len(flips), "lbe.proxy_holds": holds,
            "lbe.n_shifted": len(sh := [r for r in rows.values() if r["shift"] > 0]),
            "lbe.median_shift": round(float(np.median([r["shift"] for r in sh])), 1) if sh else None,
            "lbe.median_shift_gain": round(float(np.median([r["per_event"] - r["log_c0"] for r in sh])), 2) if sh else None,
            "lbe.max_rel": float(f"{max(r['max_rel'] for r in rows.values()):.2g}") if rows else None}
    plot(rows)
    cen = shift_census()
    flat.update({f"lbe.census.{k}": v for k, v in cen.items() if k != "by_mass"})
    flat.update({f"lbe.census.m{k}": v for k, v in cen["by_mass"].items()})
    print(json.dumps(flat))


def shift_census():
    """catalog_v4's one-loop entries (the _nlo entries of its training and held-out recipes) by the mass-scheme shift
    generate_virt_dataset gives them: the rule of virt_evaluator (no _v3 flag, not loop-induced or electroweak: shift
    with m the first massive final), counted by role and by that final's mass."""
    import mg5_pipeline_final as mg
    import nlo_virtual_pipeline as nv
    out = {"shifted_train": 0, "shifted_heldout": 0, "none_train": 0, "none_heldout": 0, "by_mass": {}}
    for f, role in (("catalog_v4_train.yaml", "train"), ("catalog_v4_heldout_H5.yaml", "heldout")):
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]:
            if not p["name"].endswith("_nlo"):
                continue
            cfg = nv.VIRT_PROCESSES[mg.PROCESSES[p["name"]]["virt_base"]]
            m = 0.0 if (cfg.get("_v3") or cfg.get("loopind") or cfg.get("ew")) else \
                next((x for x in cfg["m_finals"] if x and x > 0), 0.0)
            out[("shifted_" if m else "none_") + role] += 1
            if m:
                out["by_mass"][f"{m:g}"] = out["by_mass"].get(f"{m:g}", 0) + 1
    return out


def plot(rows):
    """Fig. v4_loop_born_events: left, per checked pool its per-event mean of log|c0+shift| against the pool-level
    step of par:v4_loop_born; right, per pool the per-event mean of log|c0| (no shift) against that of log|c0+shift|,
    pools with a massive final (the shift applied) apart."""
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    for held, c, lab in ((False, ps.C.blue, "training"), (True, ps.C.vermillion, "held out")):
        rs = [r for r in rows.values() if r["heldout"] == held]
        a.plot([r["step"] for r in rs], [r["per_event"] for r in rs], "o", color=c, label=lab)
    for shifted, c, lab in ((False, ps.C.grey, r"$\delta=0$"), (True, ps.C.green, r"$\delta>0$")):
        rs = [r for r in rows.values() if (r["shift"] > 0) == shifted]
        b.plot([r["log_c0"] for r in rs], [r["per_event"] for r in rs], "o", color=c, label=lab)
    for ax in (a, b):
        lo, hi = -3, 5
        ax.plot([lo, hi], [lo, hi], "-", color="k", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    a.set_xlabel(r"$\overline{t}-\overline{t}_{\mathrm{Born}}$"); a.set_ylabel(r"$\langle\log|c_0+\delta|\rangle$")
    b.set_xlabel(r"$\langle\log|c_0|\rangle$"); b.set_ylabel(r"$\langle\log|c_0+\delta|\rangle$")
    ps.legend(a, "lower right"); ps.legend(b, "lower right")
    ps.save(fig, os.path.join(HERE, "figs", "loop_born_events"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="store_true")
    ap.add_argument("--n", type=int, default=N_EVENTS)
    ap.add_argument("--workers", type=int, default=len(POOLS))
    A = ap.parse_args()
    job(A.n, A.workers) if A.job else read()
