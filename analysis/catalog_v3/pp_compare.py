"""The pp study (D117, scope D119): does training on catalog_v3's shaped sqrt(s_hat) windows serve pp-sampled use, maybe
more than training on pp? Every catalog_v3 short arm read zero-shot on the 13 TeV MadEvent pools of the 1396 hadronic
tree classes (scripts/job_catalog_v3_pp_score.sh; recipes/catalog_v3_pp_test.yaml: 410 held-out classes of H1, H2, H3
and the 986 training classes; each pp13_<class> is its class's |M|^2 function on pp-distributed events, 2000 validation
events). The predictions stay on the site ($SCRATCH/zz_v3_pp_<arm>/<probe>.npz); `collect`, run there, reduces them to
one row per probe and adds each training pool's loss in the arm's own run at its best checkpoint (result.json
proc_val_losses_no_reg, 200 validation events of the pool it trained on, times the shared standardization's variance:
log|M|^2 units, EMA weights as the pp read). For a window-trained arm that pool is the class's shaped window (v3_<x>);
for the pp-trained arm a hadronic tree class trains as pp13_<x>, so its in-run loss is a loss on pp events, kept apart
from the window comparison (in_run_pp). It prints the rows as JSON:
    site run --item pp_sampling ccin2p3 FA -- python analysis/catalog_v3/pp_compare.py collect > analysis/catalog_v3/data/pp_rows.json
  window-trained   baseline, dimensionless, dimensionless_context, dimensionless_twin (the four short pretrainings,
                   every hadronic class on its shaped windows)
  pp-trained       dimensionless_pp: the dimensionless arm with every hadronic training tree class on 13 TeV pp events
                   (the other pools on their windows, D119)
Per probe, from the saved truth t and the arm's prediction q (log|M|^2), as analysis/catalog_v3/short_reads.py:
  mse    mean (q - t)^2;  level  (mean (q - t))^2;  shape  var(q - t)
Per arm and group (H1, H2, H3, and the training classes, in distribution for the pp-trained arm only on pp events):
the medians over probes, the 10% and 90% MSE quantiles, the count against the recipe, every missing probe listed by name.
For the held-out classes, the same arm's read of the same class on its window pool (data/short_reads.json) beside it.
Each probe's pp events are also split into those inside what its class's window pools cover (collect) and the rest.
Figure pp_compare: (a) the window-trained dimensionless arm's MSE per class on the pp pool against its window pool;
(b) the same on the pp events inside the windows' support. Figure pp_trained (once its reads exist): the pp-trained
arm's MSE per class against the window-trained dimensionless arm's, both on the pp pools. The last line printed is
the numbers docs/results.tex quotes, as one JSON object.
    python3 analysis/catalog_v3/pp_compare.py   -> data/pp_compare.json, figs/pp_compare.{png,pdf}
"""
import glob, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
ARMS = ("baseline", "dimensionless", "dimensionless_context", "dimensionless_twin", "dimensionless_pp")
GROUPS = ("H1", "H2", "H3", "train")
NAMES = {"H1": "new combinations", "H2": r"$2\to4$ families", "H3": "the top", "train": "training classes"}


def groups():
    """pp13_<x> -> its class's role: the held-out axis, or 'train'; every probe of the pp test recipe"""
    T = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"]
    names = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]]
    out = {}
    for n in names:
        r = T["v3_" + n[len("pp13_"):]]
        out[n] = "train" if r["role"] == "train" else r["axis"]
    return out


def support_masks():
    """per pp probe, which of its 2000 scored events lie inside what its class's window pools cover: sqrt(s_hat) in the
    class's window and the catalog's fiducial cuts (mg5_pipeline_final.FIDUCIAL_CUTS: pT > 10 GeV, |cos theta| < 0.9,
    Delta R > 0.4, m_ij > 10 GeV on the massless visible finals), both read in the partonic rest frame, where the window
    sampler draws. MadEvent's run card cuts in the lab (pT > 10 GeV, |eta| < 5), so a pp event can be forward, collinear
    or above the window's top, none of which a window pool holds. The first 2000 rows of the val pool are the events the
    read scored (tools/zero_shot_parent.py, data.eval_subsample: an mmap slice), in another order (matched in collect)."""
    import mg5_pipeline_final as mg
    win = {}
    for f in ["catalog_v3_train.yaml"] + [f"catalog_v3_heldout_{a}.yaml" for a in ("H1", "H2", "H3")]:
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]:
            win.setdefault(e["name"], tuple(float(x) for x in e["sqrts"]))
    out = {}
    for n in groups():
        c = mg.PROCESSES[n]
        k = len(c["pdg_ids"])
        a = np.load(os.path.join(os.environ["DATA_DIR"], f"{n}_13000-13000GeV_val_lhe_amplitudes.npy"), mmap_mode="r")
        a = np.array(a[:2000], dtype=np.float64)          # a copy: the mmap is read-only
        P = a[:, :4 * k].reshape(-1, k, 4)
        tot = P[:, 0] + P[:, 1]
        bz = tot[:, 3] / tot[:, 0]
        g = 1.0 / np.sqrt(1.0 - bz ** 2)
        E, pz = P[..., 0].copy(), P[..., 3].copy()
        P[..., 0] = g[:, None] * (E - bz[:, None] * pz)
        P[..., 3] = g[:, None] * (pz - bz[:, None] * E)
        rs = np.sqrt(np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 0, None))
        lo, hi = win["v3_" + n[len("pp13_"):]]
        cut = mg.fiducial_pass_mask(P, c["m_finals"], c["pdg_ids"], mg.FIDUCIAL_CUTS)
        inw = (rs >= lo) & (rs <= hi)
        out[n] = {"in_sqrts": inw, "in_cuts": cut, "log_amp": np.log(np.abs(a[:, -1])), "sqrts": rs,
                  "fail": cut_fails(P, c, mg.FIDUCIAL_CUTS, cut)}
    return out


def cut_fails(P, c, C, combined):
    """the share of events each fiducial cut alone removes (as mg5_pipeline_final.fiducial_pass_mask, one cut at a
    time; their conjunction checked against it)"""
    import itertools
    k = len(c["pdg_ids"])
    sl = [i for j, i in enumerate(range(2, k)) if c["m_finals"][j] == 0.0 and abs(int(c["pdg_ids"][i])) not in (12, 14, 16)]
    f = {x: np.ones(len(P), bool) for x in ("pt", "cos", "dr", "m")}
    pt = lambda p: np.hypot(p[:, 1], p[:, 2])
    eta = lambda p: np.arcsinh(p[:, 3] / np.clip(pt(p), 1e-9, None))
    phi = lambda p: np.arctan2(p[:, 2], p[:, 1])
    for s_ in sl:
        p = P[:, s_]
        f["pt"] &= pt(p) > C["pt_min"]
        f["cos"] &= np.abs(p[:, 3]) / np.clip(np.linalg.norm(p[:, 1:], axis=1), 1e-12, None) < C["cos_max"]
    for i, j in itertools.combinations(sl, 2):
        dp = np.abs(phi(P[:, i]) - phi(P[:, j])); dp = np.minimum(dp, 2 * np.pi - dp)
        q = P[:, i] + P[:, j]
        f["dr"] &= np.hypot(eta(P[:, i]) - eta(P[:, j]), dp) > C["dr_min"]
        f["m"] &= np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None)) > C["m_min"]
    assert np.array_equal(f["pt"] & f["cos"] & f["dr"] & f["m"], combined), c["pdg_ids"]
    return {x: float(1 - v.mean()) for x, v in f.items()}


def collect():
    """on the site: every arm's pp predictions reduced to rows, and its training classes' window losses"""
    out = {}
    sup = support_masks()
    for arm in ARMS:
        pred = os.path.join(os.environ["SCRATCH"], f"zz_v3_pp_{arm}")
        files = sorted(glob.glob(os.path.join(pred, "*.npz")))
        if not files:
            continue
        rows, bad = {}, []
        for f in files:
            z = np.load(f)
            p = os.path.basename(f)[:-4]
            if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
                bad.append(p)                  # no prediction in the parent's units: listed, never dropped silently
                continue
            d = z["q"].astype(float) - z["t"].astype(float)
            rows[p] = {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2), "shape": float(np.var(d)),
                       "var_t": float(np.var(z["t"].astype(float))), "n": int(len(d))}
            m = sup.get(p)
            lf = z["logF"].astype(float) if ("logF" in z.files and len(z["logF"]) == len(d)) else 0.0
            if m is not None and len(m["in_cuts"]) == len(d):
                # the read's validation loader does not keep the pool's row order: each scored event is matched to its
                # pool row by its log|M|^2 (t minus the target factor), sorted against sorted, and the match checked
                la = z["t"].astype(float) - lf
                perm = np.empty(len(la), int)
                perm[np.argsort(la, kind="stable")] = np.argsort(m["log_amp"], kind="stable")
                chk = float(np.max(np.abs(la - m["log_amp"][perm])))
                rows[p]["order_check"] = chk
                if chk > 1e-3:                     # not the same events: no split for this probe, listed
                    continue
                inc, ins = m["in_cuts"][perm], m["in_sqrts"][perm]
                inside = inc & ins
                rows[p].update({"frac_in_cuts": float(inc.mean()), "frac_in_sqrts": float(ins.mean()),
                                "frac_inside": float(inside.mean()),
                                "mse_inside": float(np.mean(d[inside] ** 2)) if inside.any() else None,
                                "mse_outside": float(np.mean(d[~inside] ** 2)) if (~inside).any() else None,
                                "mse_out_cuts": float(np.mean(d[~inc] ** 2)) if (~inc).any() else None,
                                "mse_above_window": float(np.mean(d[~ins] ** 2)) if (~ins).any() else None,
                                "sqrts_median": float(np.median(m["sqrts"])), "cut_fail": m["fail"]})
        run = os.path.join(ROOT, "runs", "catalog_v3_short", arm)
        res = json.load(open(os.path.join(run, "result.json")))
        st = json.load(open(os.path.join(run, "data_stats.json")))
        sd = st["prepd_std"]
        assert st.get("shared_standardization") and max(sd) - min(sd) < 1e-9 * max(sd), f"{arm}: not one shared std"
        win = {k: float(v) * sd[0] ** 2 for k, v in res["proc_val_losses_no_reg"].items()}
        out[arm] = {"pp": rows, "no_prediction": bad, "window_train": win, "best_step": res.get("best_step"),
                    "shared_std": sd[0]}
    json.dump(out, sys.stdout)


def read(arm, rows_all):
    a = rows_all.get(arm)
    return (a["pp"], a["no_prediction"]) if a else (None, None)


def summ(rows):
    m = np.array([v["mse"] for v in rows.values()])
    return {"n": len(rows), "mse": float(np.median(m)), "mse_q10_q90": [float(x) for x in np.quantile(m, [0.1, 0.9])],
            "level": float(np.median([v["level"] for v in rows.values()])),
            "shape": float(np.median([v["shape"] for v in rows.values()]))}


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def main():
    g = groups()
    sr_path = os.path.join(HERE, "data", "short_reads.json")
    sr = json.load(open(sr_path)) if os.path.exists(sr_path) else {}
    rows_all = json.load(open(os.path.join(HERE, "data", "pp_rows.json")))
    res, claims = {}, {}
    for arm in ARMS:
        rows, bad = read(arm, rows_all)
        if rows is None:
            print(f"{arm}: no pp reads")
            continue
        res[arm] = {"no_prediction": bad}
        for grp in GROUPS:
            exp = sorted(p for p, a in g.items() if a == grp)
            r = {p: rows[p] for p in exp if p in rows}
            ent = {"n_expected": len(exp), "missing": sorted(set(exp) - set(r))}
            if not r:
                ent["n"] = 0
                res[arm][grp] = ent
                print(f"{arm:24s} {grp:5s} n=   0/{len(exp):4d} NO READS")
                continue
            ent.update(summ(r))
            ent["probes"] = r
            # the same arm on the same classes' window pools: a held-out class's zero-shot window read
            # (data/short_reads.json, 2000 events), a training class's validation loss in the arm's own run
            # (200 events, its best checkpoint); a class the arm never trained on (the twin's removed ones) has
            # neither and is counted apart
            if grp == "train":
                own = rows_all[arm]["window_train"]          # every pool of the arm's own run, by its training name
                win = {k: {"mse": v} for k, v in own.items() if k.startswith("v3_")}
                # the pp-trained arm trained the class on pp events (pp13_<x>): its in-run loss is on pp, not a window
                ent["in_run_pp"] = {p: own[p] for p in r if p in own}
                ent["not_in_arm_training"] = sorted(p for p in r if p not in own and "v3_" + p[len("pp13_"):] not in own)
            else:
                win = sr.get(arm, {}).get(grp, {}).get("probes", {})
            wk = lambda p: "v3_" + p[len("pp13_"):]
            both = [p for p in r if wk(p) in win and r[p].get("mse_inside") is not None]
            ent["window_pairs"] = {p: win[wk(p)]["mse"] for p in both}
            if both:
                lr = np.log10([r[p]["mse"] / win[wk(p)]["mse"] for p in both])
                li = np.log10([r[p]["mse_inside"] / win[wk(p)]["mse"] for p in both])
                ent["window_same_classes"] = {
                    "n": len(both), "mse_pp": med(r[p]["mse"] for p in both),
                    "mse_window": med(win[wk(p)]["mse"] for p in both),
                    "mse_pp_inside": med(r[p]["mse_inside"] for p in both),
                    "mse_pp_outside": med(r[p]["mse_outside"] for p in both),
                    "ratio": float(10 ** np.median(lr)), "ratio_q10_q90": [float(10 ** x) for x in np.quantile(lr, [0.1, 0.9])],
                    "n_pp_worse": int((lr > 0).sum()),
                    "ratio_inside": float(10 ** np.median(li)), "n_inside_worse": int((li > 0).sum())}
            ent["support"] = {"frac_inside": med(v["frac_inside"] for v in r.values()),
                              "frac_out_cuts": med(1 - v["frac_in_cuts"] for v in r.values()),
                              "frac_above_window": med(1 - v["frac_in_sqrts"] for v in r.values()),
                              "frac_fail_angle": med(v["cut_fail"]["cos"] for v in r.values()),
                              "frac_fail_pair_mass": med(v["cut_fail"]["m"] for v in r.values()),
                              "frac_fail_dr": med(v["cut_fail"]["dr"] for v in r.values()),
                              "frac_fail_pt": med(v["cut_fail"]["pt"] for v in r.values())}
            res[arm][grp] = ent
            w = ent.get("window_same_classes")
            c = claims.setdefault(arm, {})[grp] = {"n": ent["n"], "mse": ent["mse"], "level": ent["level"],
                                                   "shape": ent["shape"], **({k: w[k] for k in (
                                                       "mse_window", "mse_pp_inside", "mse_pp_outside", "ratio",
                                                       "n_pp_worse", "ratio_inside", "n_inside_worse")} if w else {})}
            if w:
                c["n_pairs"] = w["n"]
            claims.setdefault("support", {})[grp] = ent["support"]
            print(f"{arm:24s} {grp:5s} n={ent['n']:4d}/{len(exp):4d} mse {ent['mse']:.3g} level {ent['level']:.3g} "
                  f"shape {ent['shape']:.3g}" +
                  (f" | window {w['mse_window']:.3g} on the same {w['n']}, ratio {w['ratio']:.3g} ({w['n_pp_worse']} "
                   f"worse); pp inside the window's support {w['mse_pp_inside']:.3g} (ratio {w['ratio_inside']:.3g}, "
                   f"{w['n_inside_worse']} worse), outside {w['mse_pp_outside']:.3g}" if w else ""))
    for grp in GROUPS:
        sp = claims["support"][grp]
        sp.update({k.replace("frac_", "pct_"): 100 * v for k, v in list(sp.items())})      # the shares as percentages
        print(f"support {grp:5s}", {k: round(v, 3) for k, v in sp.items()})
    claims["n_classes"] = len(g)
    claims["n_no_prediction"] = sum(len(res[a]["no_prediction"]) for a in res)
    json.dump(res, open(os.path.join(HERE, "data", "pp_compare.json"), "w"), indent=1)
    plot(res)
    print(json.dumps(claims))


def plot(res):
    """(a) per class, the window-trained dimensionless arm's MSE on the pp pool against its MSE on the class's window
    pool; (b) the same pp MSE on the events inside what the window pools cover only. With the pp-trained arm's reads, a
    second figure: its pp MSE against the window-trained arm's, per class (pp_trained.{png,pdf})."""
    import plot_style as ps
    marks = {"H1": ("o", ps.C.blue), "H2": ("s", ps.C.green), "H3": ("^", ps.C.vermillion), "train": ("D", ps.C.grey)}
    fig, axs = ps.figure(ncols=2)
    a = res.get("dimensionless", {})
    lo, hi = np.inf, 0
    for ax, key in zip(axs, ("mse", "mse_inside")):
        for grp in ("train", "H1", "H2", "H3"):
            e = a.get(grp, {})
            wp, pr = e.get("window_pairs", {}), e.get("probes", {})
            ps_ = sorted(wp)
            if not ps_:
                continue
            x = [wp[p] for p in ps_]
            y = [pr[p][key] for p in ps_]
            ax.plot(x, y, marks[grp][0], mfc="none", ms=3, alpha=0.6, color=marks[grp][1], label=NAMES[grp])
            lo, hi = min(lo, min(x), min(y)), max(hi, max(x), max(y))
    for ax in axs:
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo / 2, hi * 2); ax.set_ylim(lo / 2, hi * 2)
        ax.set_xlabel("MSE, window pool")
    axs[0].set_ylabel("MSE, pp pool")
    axs[1].set_ylabel("MSE, pp inside the windows")
    ps.shared_legend(fig, axs[0], ncol=5)
    ps.save(fig, os.path.join(HERE, "figs", "pp_compare"))
    b = res.get("dimensionless_pp", {})
    if not any(b.get(grp, {}).get("probes") for grp in GROUPS):
        return
    fig, ax = ps.figure()
    lo, hi = np.inf, 0
    for grp in GROUPS:
        x_, y_ = a.get(grp, {}).get("probes", {}), b.get(grp, {}).get("probes", {})
        ps_ = sorted(set(x_) & set(y_))
        if ps_:
            x, y = [x_[p]["mse"] for p in ps_], [y_[p]["mse"] for p in ps_]
            ax.plot(x, y, marks[grp][0], mfc="none", ms=3, alpha=0.6, color=marks[grp][1], label=NAMES[grp])
            lo, hi = min(lo, min(x), min(y)), max(hi, max(x), max(y))
    ax.plot([lo, hi], [lo, hi], ls="--", color="k", label="equal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MSE on pp, trained on windows"); ax.set_ylabel("MSE on pp, trained on pp")
    ps.legend(ax, "upper left")
    ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "pp_trained"))


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else main()
