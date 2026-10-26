"""The sign head's search on catalog_v3 (sweep/sweep_config_ccin2p3_sign_head_search.yaml; plan entry v3_prep_explore;
docs/results.tex par:v3_prep_signed). The question: once its own knobs and the learning rate are searched with Muon, does
the sign head carry the one-loop sign to held-out processes? A single-fidelity DyHPO, 20 trials at the short
pretraining's 4264 steps, every trial the more-signed-pools arm (baseline target, one shared standardization,
recipes/catalog_v3_train_signext.yaml, 1492 pools, seed 42) with MuonAdamW at decoupled weight decay 0.0217; searched the
lr, the warmup fraction, the pair sign head (width 0 = off, hidden, tanh bound) and the near-zero weighting (weight 1 =
off, band). Objective val_total: the magnitude MSE plus the sign cross-entropy on training-process validation events,
geometric mean over processes, at the best val_loss_no_reg checkpoint.
  collect (on the site): every trial's result file (val_loss = best-checkpoint val_loss_no_reg, objective_value =
           val_total there, the curve), its run config (the HPs it ran), the failed trials' logs; and the two reference
           runs' result files (the Muon baseline baseline_more_signed_muon, the AdamW arm baseline_more_signed):
    site run --item v3_prep_explore ccin2p3 FA -- python analysis/catalog_v3/sign_head_search.py collect \
        > analysis/catalog_v3/data/sign_head_search_trials.json
  main: every trial's objective and magnitude loss with its settings; blow-ups (a validation over BLOWUP_FACTOR times its
        running minimum, pair_head_blowup.py's rule) listed with their step and kept in every number; the best trial
        with both heads off (pair width 0 and near weight 1) against the best with either on, inside the search; both
        against the Muon baseline's own val_total and val_loss_no_reg (its lr and weight decay come from the Muon search
        on the dimensionless target, so the in-search comparison is the fair one). If the best trial's zero-shot reads
        are home (data/short_reads/sign_search_best/, arm sign_search_best of scripts/job_catalog_v3_short.sh, mode
        eval), its held-out wrong-sign share and zero-shot MSE per probe against the Muon baseline's and the AdamW
        arm's, as prep_signed.py reads them.
    python3 analysis/catalog_v3/sign_head_search.py   -> data/sign_head_search.json, figs/sign_head_search.{png,pdf}
Figure (1x2): (a) every trial's objective against its lr, heads off and on marked apart, blow-ups hollow, the Muon
baseline starred; (b) the objective over the magnitude loss (how much the sign cross-entropy adds) against the lr.
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
SWEEP = "sign_head_search"
HPS = ("training.lr", "training.cosanneal_warmup_frac", "model.sign_pair_dim", "model.sign_pair_hidden",
       "model.sign_pair_bound", "training.sign_near_weight", "training.sign_near_factor", "training.optimizer",
       "training.muon_weight_decay", "seed")
REFS = {"muon_baseline": "runs/catalog_v3_short/baseline_more_signed_muon",
        "adamw_more_signed": "runs/catalog_v3_short/baseline_more_signed"}
DATA = os.path.join(HERE, "data", "sign_head_search_trials.json")


def _get(c, k):
    for p in k.split("."):
        if not isinstance(c, dict) or p not in c:
            return None
        c = c[p]
    return c


def collect():
    import yaml
    small = lambda r: {k: v for k, v in r.items() if not isinstance(v, dict) or k.startswith("val_total")}
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "results", "hp*_t*.json"))):
        hp = int(os.path.basename(f)[2:6])
        r = json.load(open(f))
        d = os.path.join(ROOT, "runs", SWEEP, f"trial_{hp:04d}")
        c = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        rj = os.path.join(d, "result.json")
        rows.append({"hp_idx": hp, "result_file": small(r), "hps": {k: _get(c, k) for k in HPS},
                     "run_result": small(json.load(open(rj))) if os.path.exists(rj) else None,
                     "has_best_model": any(os.path.exists(os.path.join(d, "models", "model_run0_best.pt" + z))
                                           for z in ("", ".gz"))})
    failed = [os.path.basename(e) for e in glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "error", "*.err"))
              if "Trial FAILED" in open(e, errors="replace").read()]
    refs = {}
    for k, d in REFS.items():
        rj = os.path.join(ROOT, d, "result.json")
        refs[k] = small(json.load(open(rj))) if os.path.exists(rj) else None
    json.dump({"sweep": SWEEP, "trials": rows, "failed_logs": failed, "refs": refs,
               "n_jobs": len(glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "jobs", "trial_*.sh")))}, sys.stdout)


def blowup(curve, every):
    """the step of the first validation over BLOWUP_FACTOR times its running minimum, None if none"""
    from pair_head_blowup import BLOWUP_FACTOR
    j = next((i for i in range(1, len(curve)) if curve[i] > BLOWUP_FACTOR * min(curve[:i])), None)
    return None if j is None else every * (j + 1)


def kind(h):
    """which of the two sign knobs a trial turns on (width 0 = no pair head, weight 1 = no near-zero weighting)"""
    pair, near = (h["model.sign_pair_dim"] or 0) > 0, (h["training.sign_near_weight"] or 1.0) != 1.0
    return {(False, False): "off", (True, False): "pair", (False, True): "near", (True, True): "both"}[(pair, near)]


ZS_ARMS = ("sign_search_best", "baseline_more_signed_muon", "baseline_more_signed")


def zero_shot(best_hp, flat, out):
    """the best trial's zero-shot reads against the Muon baseline's and the AdamW arm's: on the held-out one-loop pools
    the scorer reads as sign-changing in all three, the median wrong-sign share, how many pools the best trial gets
    less and more wrong than the Muon baseline; per held-out probe of every axis read by all three, the median of the
    zero-shot MSE ratio to the Muon baseline's, how many probes it is above, and the median per axis"""
    from prep_signed import arm_signs
    from short_reads import READ, read
    if not os.path.isdir(os.path.join(HERE, "data", "short_reads", ZS_ARMS[0])):
        print("sign_search_best: no zero-shot reads copied home yet")
        return
    w = {a: arm_signs(a) for a in ZS_ARMS}
    pools = sorted(set.intersection(*[set(x) for x in w.values()]))
    mse = {a: {ax: read(a, ax) for ax in READ} for a in ZS_ARMS}
    o = {"best_hp": best_hp, "n_pools": len(pools), "wrong": {a: med([w[a][p] for p in pools]) for a in ZS_ARMS}}
    ref = "baseline_more_signed_muon"
    for a in (ZS_ARMS[0], ZS_ARMS[2]):
        sh = {"sign_search_best": "best", "baseline_more_signed": "adamw"}[a]
        rat, ax_ = [], {}
        for ax in READ:
            rr = [mse[a][ax][p]["mse"] / mse[ref][ax][p]["mse"] for p in mse[a][ax]
                  if p in mse[ref][ax] and mse[ref][ax][p]["mse"] > 0]
            rat += rr
            ax_[ax] = med(rr)
        o[a] = {"better": int(sum(w[a][p] < w[ref][p] for p in pools)), "worse": int(sum(w[a][p] > w[ref][p] for p in pools)),
                "mse_ratio": med(rat), "n_probes": len(rat), "n_probes_worse": int(sum(r > 1 for r in rat)), "axes": ax_,
                "per_pool": {p: [w[ref][p], w[a][p]] for p in pools}}
        flat.update({f"zs_{sh}_better": o[a]["better"], f"zs_{sh}_worse": o[a]["worse"],
                     f"zs_{sh}_mse_ratio": round(o[a]["mse_ratio"], 2), f"zs_{sh}_n_probes": len(rat),
                     f"zs_{sh}_n_probes_worse": o[a]["n_probes_worse"]})
        flat.update({f"zs_{sh}_{ax}_ratio": round(r, 2) for ax, r in ax_.items() if r is not None})
    flat["zs_n_pools"] = len(pools)
    for a, sh in zip(ZS_ARMS, ("best", "muon", "adamw")):
        flat[f"zs_{sh}_wrong"] = round(100 * o["wrong"][a], 1)
    out["zero_shot"] = o
    print("zero-shot", json.dumps({k: v for k, v in o.items() if k not in ZS_ARMS}),
          {a: {k: v for k, v in o[a].items() if k != "per_pool"} for a in (ZS_ARMS[0], ZS_ARMS[2])})


def sig(x, n=4):
    """x to n significant figures, as the notes write it"""
    return float(f"{x:.{n - 1}e}")


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    d = json.load(open(DATA))
    rows = []
    for t in d["trials"]:
        r, h = t["result_file"], t["hps"]
        rows.append({"hp_idx": t["hp_idx"], "kind": kind(h), "objective": r["val_total"], "val_loss_no_reg": r["val_loss"],
                     "best_step": r["best_step"], "blowup_step": blowup(r["val_curve"], r["validate_every"]),
                     "has_best_model": t["has_best_model"],
                     **{k.split(".")[-1]: v for k, v in h.items() if k not in ("training.optimizer", "seed",
                                                                                "training.muon_weight_decay")}})
    rows.sort(key=lambda x: x["objective"])
    mb = d["refs"]["muon_baseline"]
    ref = {"objective": mb["val_total"], "val_loss_no_reg": mb["val_loss"], "lr": 5e-3, "cosanneal_warmup_frac": 0.1584336,
           "blowup_step": blowup(mb["val_curve"], mb["validate_every"])}
    out = {"n_trials": len(rows), "n_jobs": d["n_jobs"], "failed_logs": d["failed_logs"], "trials": rows,
           "muon_baseline": ref, "adamw_more_signed": {"val_loss_no_reg": d["refs"]["adamw_more_signed"]["val_loss"]}}
    flat = {"n_trials": len(rows), "n_failed": len(d["failed_logs"]),
            "n_blowup": sum(x["blowup_step"] is not None for x in rows),
            "mb_obj": sig(ref["objective"]), "mb_mag": sig(ref["val_loss_no_reg"])}
    blown = [x for x in rows if x["blowup_step"] is not None]
    flat["blowup_lr_min"] = sig(min(x["lr"] for x in blown)) if blown else None
    flat["calm_lr_max"] = sig(max(x["lr"] for x in rows if x["blowup_step"] is None))
    for k in ("off", "pair", "near", "both"):
        g = [x for x in rows if x["kind"] == k]
        flat[f"n_{k}"] = len(g)
        if g:
            b = g[0]
            out[f"best_{k}"] = b
            flat.update({f"{k}_hp": b["hp_idx"], f"{k}_obj": sig(b["objective"]),
                         f"{k}_mag": sig(b["val_loss_no_reg"]), f"{k}_lr": sig(b["lr"])})
    best = rows[0]
    on = [x for x in rows if x["kind"] != "off"]
    flat.update({"best_hp": best["hp_idx"], "best_kind": best["kind"], "best_obj": sig(best["objective"]),
                 "best_mag": sig(best["val_loss_no_reg"]), "best_lr": sig(best["lr"]),
                 "best_pair_dim": best["sign_pair_dim"], "best_pair_hidden": best["sign_pair_hidden"],
                 "best_warmup": round(best["cosanneal_warmup_frac"], 3),
                 "best_on_over_off": round(on[0]["objective"] / out["best_off"]["objective"], 2),
                 "best_over_mb_obj": round(best["objective"] / ref["objective"], 2),
                 "best_over_mb_mag": round(best["val_loss_no_reg"] / ref["val_loss_no_reg"], 2),
                 "off_lr_max": sig(max(x["lr"] for x in rows if x["kind"] == "off")),
                 "n_beat_mb": sum(x["objective"] < ref["objective"] for x in rows)})
    # the lr confound: the objective's rank correlation with lr over the trials that did not blow up
    calm = [x for x in rows if x["blowup_step"] is None]
    rk = lambda v: np.argsort(np.argsort(v))
    flat["calm_rho_lr"] = round(float(np.corrcoef(rk([x["lr"] for x in calm]), rk([x["objective"] for x in calm]))[0, 1]), 2)
    flat["n_calm"] = len(calm)
    q = [x["objective"] / x["val_loss_no_reg"] for x in rows]
    flat.update({"obj_over_mag_min": round(min(q), 2), "obj_over_mag_max": round(max(q), 2),
                 "mb_obj_over_mag": round(ref["objective"] / ref["val_loss_no_reg"], 2)})
    # the blow-ups against the pair head's bound: trials with an unbounded pair head at or above the lowest blown lr
    hi = [x for x in rows if x["lr"] >= min((y["lr"] for y in blown), default=np.inf)]
    flat["n_hi_lr"] = len(hi)
    flat["n_hi_lr_unbounded_pair"] = sum(x["kind"] in ("pair", "both") and not x["sign_pair_bound"] for x in hi)
    flat["n_blowup_unbounded_pair"] = sum(x["kind"] in ("pair", "both") and not x["sign_pair_bound"] for x in blown)
    for x in rows:
        print(f"hp{x['hp_idx']:04d} {x['kind']:5s} obj {x['objective']:.4g} mag {x['val_loss_no_reg']:.4g} "
              f"lr {x['lr']:.3g} wu {x['cosanneal_warmup_frac']:.3f} pair {x['sign_pair_dim']}/{x['sign_pair_hidden']}"
              f"/{x['sign_pair_bound']} near {x['sign_near_weight']}x{x['sign_near_factor']} best@{x['best_step']}"
              + (f" BLOWUP@{x['blowup_step']}" if x["blowup_step"] else ""))
    zero_shot(best["hp_idx"], flat, out)
    json.dump(out, open(os.path.join(HERE, "data", "sign_head_search.json"), "w"), indent=1)
    plot(rows, ref)
    print(json.dumps(flat))


def plot(rows, ref):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"off": (ps.C.grey, "s", "both off"), "pair": (ps.C.blue, "o", "pair head"),
           "near": (ps.C.vermillion, "^", "near-zero weight"), "both": (ps.C.purple, "D", "both on")}
    for k, (col, mk, lab) in sty.items():
        for blown, fill in ((False, col), (True, "none")):
            g = [x for x in rows if x["kind"] == k and (x["blowup_step"] is not None) == blown]
            if g:
                kw = dict(color=col, mfc=fill, label=lab + (", blew up" if blown else ""))
                axs[0].plot([x["lr"] for x in g], [x["objective"] for x in g], mk, **kw)
                axs[1].plot([x["lr"] for x in g], [x["objective"] / x["val_loss_no_reg"] for x in g], mk, **kw)
    axs[0].plot([ref["lr"]], [ref["objective"]], "*", color=ps.C.green, ms=10, label="Muon baseline")
    axs[1].plot([ref["lr"]], [ref["objective"] / ref["val_loss_no_reg"]], "*", color=ps.C.green, ms=10, label="Muon baseline")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("learning rate"); axs[0].set_ylabel("objective (magnitude + sign)")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("learning rate"); axs[1].set_ylabel("objective / magnitude loss")
    for ax in axs:
        ps.legend(ax, "upper left")
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "sign_head_search"))


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    else:
        sys.path.insert(0, ROOT)
        main()
