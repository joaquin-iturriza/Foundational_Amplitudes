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
    python3 analysis/catalog_v3/pair_head_blowup.py   -> data/pair_head_blowup.json, figs/pair_head_blowup.{png,pdf}
"""
import json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ARMS = ["baseline", "baseline_sign_near", "baseline_more_signed", "baseline_sign_pairs"]
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def read(arm):
    lines = open(os.path.join(HERE, "data", "short_reads", arm, "train.log")).read().splitlines()
    every = int(re.search(r"validating every (\d+) steps", "\n".join(lines)).group(1))
    vals, acc, skips, grad = [], [], [], None
    for l in lines:
        if "Val loss (combined)" in l:
            vals.append({"step": every * (len(vals) + 1),
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
    j = next(i for i in range(1, len(v)) if v[i] > 5 * min(v[:i]))     # the first validation 5x above the best so far
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
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "pair_head_blowup.json"), "w"), indent=1)
    g, jj = out["arms"]["baseline_sign_pairs"]["grad_norm"], out["jump"]
    flat = {"pairs_grad_median_e6": f"{g['median'] / 1e6:.3g}", "pairs_clip_pct": f"{g['clipped_pct']:.1f}",
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
    print(json.dumps(flat))
    plot(R)


def plot(R):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"baseline": (ps.C.grey, "baseline"), "baseline_sign_near": (ps.C.vermillion, "near-zero weighting"),
           "baseline_more_signed": (ps.C.green, "more signed pools"), "baseline_sign_pairs": (ps.C.blue, "pair sign head")}
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
