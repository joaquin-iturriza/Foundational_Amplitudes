"""Reads the signed-data check on catalog_v3 (prep_signed_pools.py, run on CC; its SIGNED_DATA lines copied home into
data/prep_signed_raw.txt) against the short pretrainings' signs on the same pools (data/short_reads/<arm>/H5_signed.txt,
tools/signed_ab_score.py: each arm's wrong-sign share on the held-out one-loop pools it resolves as sign-changing).
Per set (the one-loop pools of the training recipe, and of the one-loop held-out axis H5), over the sign-changing pools:
the medians of the negative share, the shares within a factor 10 and 100 of zero, the wrong-sign share of one sign per
pool (err_major), of a classifier on the invariants (err_mlp, and within a factor 10 of zero) and on log s_hat alone
(err_shat). The classifier's statistics (err_mlp, err_mlp_near, err_shat, their counts, and err_major, err_major_near
quoted against them) are over the pools it was fitted on only: a pool with fewer than 20 fit events of one sign (rare)
has no classifier, its stored err_mlp and err_shat are one sign per pool's, so it is listed apart (rare_pools) with
its counts and left out of them. Comparisons are strict (no margin); and the magnitude's shape per target (raw, + (N - 4) log s_hat, + the emission scales) with the per-pool
ratio of each to the raw one. On the held-out pools, per arm: the median wrong-sign share, how many pools it gets worse
than one sign per pool would, how many it gets more than half wrong, and the median of its wrong-sign share less the
classifier's (over the pools the classifier was fitted on); the held-out pools of one sign in every event that the scorer still reads as
sign-changing count there with no error for one sign per pool.
Figure: per sign-changing pool, (a) the classifier's wrong-sign share against one sign per pool's, (b) on the held-out
pools, each arm's wrong-sign share against one sign per pool's.
    python3 analysis/catalog_v3/prep_signed.py   -> data/prep_signed.json, figs/prep_signed.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
SETS = ("train", "H5")
TARGETS = ("raw", "dless", "qperp")
PCT = ("neg", "near10", "near100", "err_major", "err_mlp", "err_mlp_near", "err_major_near", "err_shat")
ARMS = {"baseline": "base", "dimensionless": "qperp", "dimensionless_shat": "shat", "dimensionless_hybrid": "hyb",
        "dimensionless_blend025": "b25", "dimensionless_blend050": "b50", "dimensionless_context": "ctx",
        "dimensionless_twin": "twin"}


def load_raw():
    rows = []
    for l in open(os.path.join(HERE, "data", "prep_signed_raw.txt")):
        if l.startswith("SIGNED_DATA "):
            rows.append(json.loads(l[len("SIGNED_DATA "):]))
    return rows


def key(r):
    f = r["files"][0]
    return f[: f.index("GeV_") + 3]                    # v3_<name>_<lo>-<hi>GeV, the scorer's pool name


def arm_signs(arm):
    f = os.path.join(HERE, "data", "short_reads", arm, "H5_signed.txt")
    out = {}
    for l in open(f):
        if "SIGNED_AB " in l:
            d = json.loads(l.split("SIGNED_AB ", 1)[1])
            if "skipped" not in d and "error" not in d:
                out[d["pool"]] = d["wrong_sign"]
    return out


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    rows = load_raw()
    out = {"errors": [(r["set"], r["name"], r["error"]) for r in rows if "error" in r]}
    flat = {"n_errors": len(out["errors"])}
    sig = {s: [r for r in rows if r["set"] == s and r.get("signed") and "error" not in r] for s in SETS}
    for s in SETS:
        allp = [r for r in rows if r["set"] == s and "error" not in r]
        d = sig[s]
        c = [r for r in d if not r.get("rare")]      # the pools a classifier was fitted on
        o = {"n_pools": len(allp), "n_signed": len(d), "n_clf": len(c)}
        for k in ("neg", "near10", "near100", "err_major", "err_major_near"):
            o[k + "_all"] = med([r[k] for r in d])     # the data alone, every sign-changing pool
        for k in ("neg", "near10", "near100"):
            o[k] = o[k + "_all"]
        for k in ("err_major", "err_mlp", "err_mlp_near", "err_major_near", "err_shat"):
            o[k] = med([r[k] for r in c])
        o["mlp_over_major"] = med([r["err_mlp"] / r["err_major"] for r in c if r["err_major"] > 0])
        o["n_mlp_below_5pct"] = int(sum(r["err_mlp"] < 0.05 for r in c))
        o["n_major_below_5pct"] = int(sum(r["err_major"] < 0.05 for r in c))
        o["n_rare"] = len(d) - len(c)
        o["n_shat_beats_major"] = int(sum(r["err_shat"] < r["err_major"] for r in c))
        o["rare_pools"] = {r["name"]: {"n_minor_fit": r["n_minor_fit"], "neg": r["neg"], "err_major": r["err_major"]}
                           for r in d if r.get("rare")}
        o["rare_max_err_major"] = max([r["err_major"] for r in d if r.get("rare")], default=None)
        o["rare_max_minor_fit"] = max([r["n_minor_fit"] for r in d if r.get("rare")], default=None)
        for t in TARGETS:
            o[f"shape_{t}"] = med([r[f"shape_{t}"] for r in d])
            o[f"frac_{t}"] = med([r[f"frac_{t}"] for r in d])
            if t != "raw":
                rr = np.array([r[f"shape_{t}"] / r["shape_raw"] for r in d])
                o[f"{t}_over_raw"] = float(np.median(rr))
                o[f"{t}_above_raw"] = int((rr > 1).sum())
        out[s] = o
        print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != "rare_pools"}))
        for n_, v_ in o["rare_pools"].items():
            print(f"  no classifier (rare) {n_:24s} {v_['n_minor_fit']:3d} of one sign in the fit events, "
                  f"negative {100 * v_['neg']:.2f}%, one sign per pool wrong {100 * v_['err_major']:.2f}%")
        sh = {"train": "tr", "H5": "H5"}[s]
        for k in ("n_pools", "n_signed", "n_clf", "neg", "near10", "near100", "err_major", "err_mlp", "err_mlp_near",
                  "err_major_near", "err_shat", "mlp_over_major", "n_mlp_below_5pct", "n_major_below_5pct", "n_rare", "n_shat_beats_major", "shape_raw", "shape_dless", "shape_qperp",
                  "dless_over_raw", "qperp_over_raw", "qperp_above_raw", "err_major_all", "err_major_near_all",
                  "rare_max_err_major", "rare_max_minor_fit"):
            v = o[k]
            if k in PCT or k in ("err_major_all", "err_major_near_all", "rare_max_err_major"):                              # shares of events, in percent as the notes write them
                v = round(100 * v, 1)
            flat[f"{sh}_{k}"] = round(v, 3) if isinstance(v, float) else v
        for n_, v_ in o["rare_pools"].items():            # the rare pools, by their minority count in the fit events
            flat[f"{sh}_rare_{n_[3:]}"] = v_["n_minor_fit"]
    # the arms' signs on the held-out pools, against the data's floors on the same pools
    h5 = {key(r): r for r in sig["H5"]}
    # a held-out pool of one sign in every validation and test event that the scorer still reads as sign-changing:
    # the transform is resolved on every role of the pool (experiment._resolve_amp_preprocessing), the train split
    # included, which this check does not read, and any non-positive value there makes it signed (an all-negative
    # pool too, which has no plain log). The truth's sign is the raw amplitude's, so the arms' shares there are real
    # errors; one sign per pool and the classifier make none. Per role, the negative share from the pool audit.
    one_sign = {key(r): {**r, "err_major": 0.0, "err_mlp": 0.0} for r in rows
                if r["set"] == "H5" and "error" not in r and not r.get("signed")}
    h5all = {**one_sign, **h5}
    out["one_sign_scored"] = {}
    out["arms"] = {}
    for arm, sh in ARMS.items():
        ws = arm_signs(arm)
        ps = sorted(set(ws) & set(h5all))
        w = np.array([ws[p] for p in ps]); mj = np.array([h5all[p]["err_major"] for p in ps])
        ml = np.array([h5all[p]["err_mlp"] for p in ps])
        clf = np.array([not h5all[p].get("rare") for p in ps])     # the classifier's pools (and the one-sign ones)
        for p in sorted(set(ws) & set(one_sign)):
            out["one_sign_scored"].setdefault(p, {"neg": one_sign[p]["neg"]})[arm] = ws[p]
        o = {"n": len(ps), "n_one_sign": len(set(ws) & set(one_sign)), "unmatched": sorted(set(ws) - set(h5all)),
             "wrong_sign": float(np.median(w)),
             "worse_than_major": int((w > mj).sum()), "minus_mlp": float(np.median((w - ml)[clf])),
             "per_pool": {p: ws[p] for p in ps}}
        out["arms"][arm] = o
        print(f"{arm:24s} n={o['n']:2d} wrong {o['wrong_sign']:.3f} worse-than-one-sign {o['worse_than_major']} "
              f"minus-classifier {o['minus_mlp']:.3f} unmatched {o['unmatched']}")
        flat[f"{sh}_n"] = o["n"]
        flat[f"{sh}_n_one_sign"] = o["n_one_sign"]
        flat[f"{sh}_inverted"] = int((w > 0.5).sum())
        flat[f"{sh}_wrong"] = round(100 * o["wrong_sign"])
        flat[f"{sh}_worse_major"] = o["worse_than_major"]
        flat[f"{sh}_minus_mlp"] = round(100 * o["minus_mlp"])
    out["pools"] = {s: {key(r): {k: r.get(k) for k in ("name", "neg", "near10", "err_major", "err_mlp", "err_shat",
                                                         "shape_raw", "shape_dless", "shape_qperp")} for r in sig[s]}
                    for s in SETS}
    json.dump(out, open(os.path.join(HERE, "data", "prep_signed.json"), "w"), indent=1)
    plot(sig, h5, out)
    # e+e- -> c cbar g at one loop: the pool every arm gets most wrong
    cg = "v3_epem_ccxg_nlo_25-1000GeV"
    wc = [out["arms"][a]["per_pool"][cg] for a in ARMS]
    flat.update({"ccg_neg": round(100 * h5[cg]["neg"]), "ccg_mlp": round(100 * h5[cg]["err_mlp"], 1),
                 "ccg_wrong_min": round(100 * min(wc)), "ccg_wrong_max": round(100 * max(wc))})
    flat["n_one_sign_scored"] = len(out["one_sign_scored"])
    audit = json.load(open(os.path.join(HERE, "pool_audit.json")))["pools"]
    n_allneg = 0
    for p, o in out["one_sign_scored"].items():
        au = audit[one_sign[p]["name"]]
        o["audit"] = {r: {"rows": au[r]["rows"], "negative": au[r]["negative"]} for r in ("train", "val", "test")}
        if all(au[r]["negative"] == 1.0 for r in ("train", "val", "test")):
            n_allneg += 1
            wa = [o[a] for a in ARMS]
            if max(wa) > 0.05:                        # the all-negative pool the arms get wrong
                flat["neg_one_sign_pool"] = one_sign[p]["name"]
                flat["neg_one_sign_wrong_min"] = round(100 * min(wa))
                flat["neg_one_sign_wrong_max"] = round(100 * max(wa))
        if one_sign[p]["neg"] == 0.0:                 # positive in val and test: the negatives are in train
            tr = au["train"]
            flat["pos_one_sign_pool"] = one_sign[p]["name"]
            flat["pos_one_sign_train_neg"] = int(round(tr["negative"] * tr["rows"]))
            flat["pos_one_sign_train_rows"] = tr["rows"]
            flat["pos_one_sign_base_wrong"] = round(100 * o["baseline"])
    flat["n_one_sign_allneg"] = n_allneg
    print("one sign in every event, scored as sign-changing:", json.dumps(out["one_sign_scored"]))
    print(json.dumps(flat))


def plot(sig, h5, out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    ax = axs[0]
    for s, col, mk, lab in (("train", ps.C.blue, "o", "training pools"), ("H5", ps.C.vermillion, "^", "held-out pools")):
        c = [r for r in sig[s] if not r.get("rare")]  # no classifier on the rare pools
        ax.plot([r["err_major"] for r in c], [r["err_mlp"] for r in c], mk, mfc="none", color=col, label=lab)
    ax.plot([0, 0.5], [0, 0.5], ls="--", color=ps.C.grey, label="equal")
    ax.set_xlabel("wrong sign, one sign per pool"); ax.set_ylabel("wrong sign, classifier")
    ps.legend(ax, "upper left")
    ax = axs[1]
    ps_ = sorted(h5)
    for arm, col, mk, lab in (("baseline", ps.C.grey, "s", "baseline"), ("dimensionless", ps.C.blue, "o", "emission scales"),
                              ("dimensionless_hybrid", ps.C.vermillion, "^", "hybrid")):
        pp = out["arms"][arm]["per_pool"]
        xs = [h5[p]["err_major"] for p in ps_ if p in pp]; ys = [pp[p] for p in ps_ if p in pp]
        # the pools of one sign in every event: at zero on one sign per pool's axis
        xs += [0.0 for p in sorted(out["one_sign_scored"]) if p in pp]
        ys += [pp[p] for p in sorted(out["one_sign_scored"]) if p in pp]
        ax.plot(xs, ys, mk, mfc="none", color=col, label=lab)
    ax.plot([0, 0.5], [0, 0.5], ls="--", color=ps.C.grey, label="one sign per pool")
    ax.set_xlabel("wrong sign, one sign per pool"); ax.set_ylabel("wrong sign, zero-shot")
    ps.legend(ax, "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_signed"))


if __name__ == "__main__":
    main()
