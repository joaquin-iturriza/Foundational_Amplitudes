"""Reads where and how the pair sign head's short pretraining on catalog_v3 blew up, from the training logs of the sign
arms (scripts/job_catalog_v3_short.sh; each run's out_0.log copied home into data/short_reads/<arm>/train.log).
Per arm: the validation curve (val_loss_no_reg every 85 steps), the median sign accuracy on the 60 sign-changing
training pools at each validation, the steps skipped for a non-finite gradient norm (placed between the validations
they fall between), and the run's pre-clip gradient-norm summary (median, 90% quantile, max, share of steps clip_grad_norm
5 bounds; the only gradient-norm record the run keeps). At the validation where the pair head's val_loss_no_reg first
jumps, the median over pools of each pool's val loss ratio to the validation before, for all pools and for the one-loop
pools, and the share of pools that rose more than threefold.
Figure: (a) val_loss_no_reg against step per arm, the skipped steps marked; (b) the median sign accuracy on the
sign-changing training pools against step.
The runs with the per-step record (train_steps.npz, base_experiment._write_step_log: per step the pre-clip gradient
norm, the training loss, sign and magnitude terms summed, and the first parameter group's lr), STEP_ARMS: the pair sign
head with tanh-bounded pair channels, and the more-signed-pools arm's seed 43 run again unchanged (its first run blew
up at steps 1360 to 1445). Per run: the validation curve and its jump as above (the bounded head also by pool group:
the pools with a negative value in some role per the pool audits, the other one-loop pools, the trees), the step where
warmup ends (the lr's maximum), and the shares of steps clipping bounds, before and after step 600; for a run that
jumps, its lead-up read step by step against the reference window, the validation interval before the last
validation under twice the running minimum: the first step from which the five-step running median of the loss stays
above LOSS_RISE times the window's median loss, the first from which that of the gradient norm stays above the clip
bound, the first single gradient norm over GRAD_JUMP times the window's median, the first non-finite one, and the first
loss over LOSS_BLOW times the window's median; for one that does not jump, the gradient norm's maximum and clipping
share over the first run's jump interval. Figure pair_head_steps: per step, (a) the pre-clip gradient norm, (b) the
training loss, both runs, the clip bound and the end of warmup marked.
    python3 analysis/catalog_v3/pair_head_blowup.py   -> data/pair_head_blowup.json, figs/pair_head_blowup.{png,pdf},
                                                          figs/pair_head_steps.{png,pdf}
"""
import json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ARMS = ["baseline", "baseline_sign_near", "baseline_more_signed", "baseline_sign_pairs", "baseline_sign_pairs_bounded"]
STEP_ARMS = ["baseline_sign_pairs_bounded", "baseline_more_signed_seed2_steplog"]
CLIP = 5.0                         # training.clip_grad_norm of every arm (CLAUDE.md canonical setup)
# the lead-up's read, all open (unsourced), chosen for this read with no decision of the user's behind them: the
# thresholds below, each far from the steady state's step-to-step spread (the loss's 5-step median moves under 1.5x,
# the norm under 10x, in the windows); the five-step running median (roll_med); the reference window (the validation
# interval ending at the last validation under twice the running minimum); the step-600 split of the clipping shares
LOSS_RISE, GRAD_JUMP, LOSS_BLOW = 3.0, 1e3, 100.0
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
# A run blows up at the first validation over BLOWUP_FACTOR times its running minimum (shared with muon_compare.py).
# open (unsourced): chosen by the main session, no decision of the user's behind it; 5x and 10x pick the same
# validations on every run so far (here 0.0208 -> 0.391, the Muon search's trials alike).
BLOWUP_FACTOR = 10


def read(arm):
    lines = open(os.path.join(HERE, "data", "short_reads", arm, "train.log")).read().splitlines()
    every = int(re.search(r"validating every (\d+) steps", "\n".join(lines)).group(1))
    iters = int(re.search(r"Starting to train for (\d+) iterations", "\n".join(lines)).group(1))
    vals, acc, skips, grad = [], [], [], None
    for l in lines:
        if "Val loss (combined)" in l:
            vals.append({"step": min(every * (len(vals) + 1), iters),     # the last one at the last step
                         "val_loss_no_reg": float(re.search(r"val_loss_no_reg: ([0-9.eE+-]+)", l).group(1)),
                         "per_pool": {k: float(v) for k, v in re.findall(r"(v3_\w+)=([0-9.eE+-]+)", l)}})
        elif "sign accuracy on the" in l:
            acc.append(float(re.search(r"median ([0-9.]+)", l).group(1)))
        elif "Skipping update, non-finite" in l:
            skips.append(len(vals))          # after this many validations
        elif "Grad-norm (pre-clip)" in l:
            m = re.search(r"median=([0-9.e+-]+) p90=([0-9.e+-]+) max=([0-9.e+-]+) \| clip=\S+ bound ([0-9.]+)%", l)
            grad = dict(zip(["median", "p90", "max", "clipped_pct"], map(float, m.groups())))
    return {"every": every, "vals": vals, "acc": acc, "skips": skips, "grad": grad}


def main():
    R = {a: read(a) for a in ARMS}
    out = {"arms": {}}
    for a, r in R.items():
        v = [x["val_loss_no_reg"] for x in r["vals"]]
        out["arms"][a] = {"grad_norm": r["grad"], "n_skipped": len(r["skips"]),
                          "best_val_loss_no_reg": min(v), "best_step": r["vals"][v.index(min(v))]["step"],
                          "final_val_loss_no_reg": v[-1]}
    p = R["baseline_sign_pairs"]; v = [x["val_loss_no_reg"] for x in p["vals"]]
    # the first validation BLOWUP_FACTOR times above the best so far
    j = next(i for i in range(1, len(v)) if v[i] > BLOWUP_FACTOR * min(v[:i]))
    a, b = p["vals"][j - 1]["per_pool"], p["vals"][j]["per_pool"]
    rat = {k: b[k] / a[k] for k in a if a[k] > 0 and k in b}
    loop = [x for k, x in rat.items() if k.endswith(("_nlo", "_loop"))]
    base_at = R["baseline"]["vals"][j]["val_loss_no_reg"]
    out["jump"] = {"validation": j + 1, "step_before": p["vals"][j - 1]["step"], "step": p["vals"][j]["step"],
                   "val_before": v[j - 1], "val_at": v[j], "baseline_at": base_at,
                   "median_ratio_all": statistics.median(rat.values()), "n_pools": len(rat),
                   "median_ratio_one_loop": statistics.median(loop), "n_one_loop": len(loop),
                   "share_above_3x": sum(x > 3 for x in rat.values()) / len(rat),
                   "sign_acc_before": p["acc"][j - 1], "sign_acc_at": p["acc"][j],
                   "baseline_sign_acc_at": R["baseline"]["acc"][j],
                   "first_skip_after_validation": min(p["skips"]) if p["skips"] else None,
                   "pairs_vs_baseline_before": v[j - 1] / R["baseline"]["vals"][j - 1]["val_loss_no_reg"],
                   "final_over_best": v[-1] / min(v), "peak_over_best": max(v[j:]) / min(v)}
    out["steps"] = {a: steps(a, R.get(a) or read(a)) for a in STEP_ARMS}
    out["bounded_groups"] = groups(R["baseline_sign_pairs_bounded"])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "pair_head_blowup.json"), "w"), indent=1)
    g, jj = out["arms"]["baseline_sign_pairs"]["grad_norm"], out["jump"]
    flat = {"pairs_grad_median": f"{g['median']:.3g}", "pairs_clip_pct": f"{g['clipped_pct']:.1f}",
            "base_grad_median": f"{out['arms']['baseline']['grad_norm']['median']:.3g}",
            "base_clip_pct": f"{out['arms']['baseline']['grad_norm']['clipped_pct']:.1f}",
            "pairs_skipped": out["arms"]["baseline_sign_pairs"]["n_skipped"],
            "jump_validation": jj["validation"], "jump_step": jj["step"], "jump_step_before": jj["step_before"],
            "jump_val_before": f"{jj['val_before']:.3g}", "jump_val_at": f"{jj['val_at']:.3g}",
            "jump_ratio_all": f"{jj['median_ratio_all']:.1f}", "jump_ratio_loop": f"{jj['median_ratio_one_loop']:.0f}",
            "jump_share_3x": round(100 * jj["share_above_3x"]),
            "sign_acc_before": jj["sign_acc_before"], "sign_acc_at": jj["sign_acc_at"],
            "first_skip_after_validation": jj["first_skip_after_validation"],
            "pairs_vs_base_before": f"{jj['pairs_vs_baseline_before']:.2f}",
            "peak_over_best": int(round(jj["peak_over_best"], -2)), "final_over_best": round(jj["final_over_best"])}
    flat = {k: (float(v) if isinstance(v, str) else v) for k, v in flat.items()}
    for a in ARMS[:3]:
        flat[f"{a}_skipped"] = out["arms"][a]["n_skipped"]
    sb, sr = out["steps"]["baseline_sign_pairs_bounded"], out["steps"]["baseline_more_signed_seed2_steplog"]
    for k, v in sb.items():
        if not isinstance(v, (dict, list)):
            flat[f"b_{k}"] = float(f"{v:.3g}") if isinstance(v, float) else v
    for k, v in sr.items():
        if not isinstance(v, (dict, list)):
            flat[f"r_{k}"] = float(f"{v:.3g}") if isinstance(v, float) else v
    for i, gr in enumerate(out["bounded_groups"]):
        for k, v in gr.items():
            flat[f"bg{i}_{k}"] = float(f"{v:.3g}") if isinstance(v, float) else v
    print(json.dumps(flat))
    plot(R)
    plot_steps(out)


def load_steps(arm):
    import numpy as np
    z = np.load(os.path.join(HERE, "data", "short_reads", arm, "train_steps.npz"))
    return {k: z[k] for k in z.files}


def roll_med(x, k=5):
    import numpy as np
    return np.array([np.nanmedian(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


def steps(arm, r):
    """The per-step record of one run (see the module docstring)."""
    import numpy as np
    z = load_steps(arm); g, L, lr = z["grad_norm"], z["loss"], z["lr"]
    v = [x["val_loss_no_reg"] for x in r["vals"]]
    o = {"n_steps": len(g), "warmup_end": int(np.argmax(lr)), "lr_peak": float(lr.max()),
         "best_val": min(v), "best_step": r["vals"][v.index(min(v))]["step"], "final_val": v[-1],
         "n_nonfinite": int((~np.isfinite(g)).sum()),
         "clip_pct_before600": 100 * float(np.mean(g[:600] > CLIP)), "clip_pct_after600": 100 * float(np.mean(g[600:] > CLIP))}
    j = next((i for i in range(1, len(v)) if v[i] > BLOWUP_FACTOR * min(v[:i])), None)
    if j is None:
        a, b = 1275, 1530              # the first seed-43 run's jump interval (1360 to 1445) and a validation each side
        o.update({"jumps": 0, "win_grad_max": float(np.nanmax(g[a:b])), "win_clip_pct": 100 * float(np.mean(g[a:b] > CLIP)),
                  "win_loss_med": float(np.nanmedian(L[a:b])), "grad_med_after600": float(np.nanmedian(g[600:]))})
        return o
    # the reference window: the validation interval ending at the last validation under twice the running minimum
    k = max(i for i in range(j) if v[i] <= 2 * min(v[:i + 1]))
    e = r["vals"][k]["step"]; s0 = e - r["every"]
    lm, gm = float(np.nanmedian(L[s0:e])), float(np.nanmedian(g[s0:e]))
    rl, rg = roll_med(L), roll_med(g)

    def first_stay(x, thr, start):
        for i in range(start, len(x)):
            if np.all(x[i:i + r["every"]] > thr):
                return i
        return None
    fin = np.isfinite(g)

    def first(cond, start):
        """The first step from start where cond holds, None if none does."""
        i = np.flatnonzero(cond[start:])
        return int(i[0]) + start if len(i) else None
    o.update({"jumps": 1, "jump_validation": j + 1, "jump_step": r["vals"][j]["step"],
              "jump_step_before": r["vals"][j - 1]["step"], "jump_val_before": v[j - 1], "jump_val_at": v[j],
              "ref_from": s0, "ref_to": e, "ref_loss": lm, "ref_grad": gm,
              "loss_rise_step": first_stay(rl, LOSS_RISE * lm, e), "grad_clip_step": first_stay(rg, CLIP, e),
              "grad_jump_step": first(g > GRAD_JUMP * gm, e), "first_nonfinite_step": first(~fin, 0),
              "loss_blow_step": first(L > LOSS_BLOW * lm, e)})
    gj, lrs = o["grad_jump_step"], o["loss_rise_step"]          # None where the event never happens
    o["loss_at_grad_jump"] = None if gj is None else float(L[gj])
    o["loss_at_grad_jump_over_ref"] = None if gj is None else o["loss_at_grad_jump"] / lm
    o["grad_at_loss_rise"] = None if lrs is None else float(rg[lrs])
    o["lr_at_loss_rise_over_peak"] = None if lrs is None else float(lr[lrs] / lr.max())
    return o


def groups(r):
    """The bounded head's validations around its jump, per pool group: the median over pools of each pool's val loss
    over the validation before, for the pools with a negative value in some role (pool audits), the other one-loop
    pools and the trees, with the median sign accuracy on the sign-changing training pools."""
    au = json.load(open(os.path.join(HERE, "pool_audit.json")))["pools"]
    au.update(json.load(open(os.path.join(HERE, "pool_audit_signext.json")))["pools"])
    signed = lambda n: any(isinstance(x, dict) and x.get("negative", 0) > 0 for x in au.get(n, {}).values())
    loop = lambda n: n.endswith(("_nlo", "_loop"))
    v = [x["val_loss_no_reg"] for x in r["vals"]]
    j = next(i for i in range(1, len(v)) if v[i] > BLOWUP_FACTOR * min(v[:i]))
    res = []
    for i in (j - 1, j):
        a, b = r["vals"][i - 1]["per_pool"], r["vals"][i]["per_pool"]
        rat = {k: b[k] / a[k] for k in a if k in b and a[k] > 0}
        res.append({"validation": i + 1, "step": r["vals"][i]["step"], "val_before": v[i - 1], "val_at": v[i],
                    "signed": statistics.median([x for k, x in rat.items() if signed(k)]),
                    "n_signed": sum(signed(k) for k in rat),
                    "loop_pos": statistics.median([x for k, x in rat.items() if loop(k) and not signed(k)]),
                    "trees": statistics.median([x for k, x in rat.items() if not loop(k)]),
                    "acc_before": r["acc"][i - 1], "acc_at": r["acc"][i]})
    return res


def plot_steps(out):
    """(a) the pre-clip gradient norm per step over the whole run, both STEP_ARMS, the clip bound and the end of warmup
    marked; (b) the bounded head's lead-up, steps 600 to 695: its gradient norm and loss per step, each over its
    median in the reference window, with the steps of the lead-up marked."""
    import numpy as np
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"baseline_sign_pairs_bounded": (ps.C.purple, "bounded pair sign head"),
           "baseline_more_signed_seed2_steplog": (ps.C.green, "more signed pools, seed 43 rerun")}
    for a, (col, lab) in sty.items():
        z = load_steps(a)
        axs[0].plot(z["step"], np.where(np.isfinite(z["grad_norm"]), z["grad_norm"], np.nan), color=col, lw=0.5, label=lab)
    sb = out["steps"]["baseline_sign_pairs_bounded"]
    axs[0].axvline(sb["warmup_end"], color=ps.C.grey, ls=":", label="end of warmup")
    axs[0].axhline(CLIP, color=ps.C.grey, ls="--", label="clip bound")
    axs[0].set_yscale("log"); axs[0].set_xlabel("step"); axs[0].set_ylabel("gradient norm before clipping")
    ps.legend(axs[0], "upper right")
    z = load_steps("baseline_sign_pairs_bounded"); w = slice(600, 696)
    axs[1].plot(z["step"][w], z["grad_norm"][w] / sb["ref_grad"], color=ps.C.purple, lw=0.8, label="gradient norm")
    axs[1].plot(z["step"][w], z["loss"][w] / sb["ref_loss"], color=ps.C.orange, lw=0.8, label="training loss")
    for k, ls_, lab in (("loss_rise_step", "-", "loss stays above 3x"), ("grad_clip_step", "--", "norm stays above the clip bound"),
                        ("grad_jump_step", "-.", "norm above 1000x"), ("first_nonfinite_step", ":", "first non-finite norm")):
        if sb[k] is not None:                       # a lead-up event that never happened draws no line
            axs[1].axvline(sb[k], color=ps.C.grey, ls=ls_, lw=0.8, label=lab)
    axs[1].set_yscale("log"); axs[1].set_xlabel("step"); axs[1].set_ylabel("over its reference median")
    ps.legend(axs[1], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "pair_head_steps"))


def plot(R):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"baseline": (ps.C.grey, "baseline"), "baseline_sign_near": (ps.C.vermillion, "near-zero weighting"),
           "baseline_more_signed": (ps.C.green, "more signed pools"), "baseline_sign_pairs": (ps.C.blue, "pair sign head"),
           "baseline_sign_pairs_bounded": (ps.C.purple, "bounded pair sign head")}
    for a, (col, lab) in sty.items():
        r = R[a]; st = [x["step"] for x in r["vals"]]
        axs[0].plot(st, [x["val_loss_no_reg"] for x in r["vals"]], color=col, label=lab)
        axs[1].plot(st[:len(r["acc"])], r["acc"], color=col, label=lab)
        if r["skips"]:
            sk = [r["every"] * (k + 0.5) for k in r["skips"]]
            axs[0].plot(sk, [r["vals"][min(k, len(r["vals"]) - 1)]["val_loss_no_reg"] for k in r["skips"]], "x",
                        color=col, label=f"{lab}, skipped step")
    axs[0].set_yscale("log"); axs[0].set_xlabel("step"); axs[0].set_ylabel(r"validation loss (no reg.)")
    axs[1].set_xlabel("step"); axs[1].set_ylabel("median sign accuracy")
    ps.legend(axs[0], "upper right")
    ps.make_room(axs[0], max_growth=4)
    axs[1].set_ylim(0, 1.05)
    ps.save(fig, os.path.join(HERE, "figs", "pair_head_blowup"))


if __name__ == "__main__":
    main()
