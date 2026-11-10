"""The current-defaults V/B test read out (plan virtual_over_born, D366): MSE of log|V| on the test pool, per process,
arm and seed.

Arms, per process (e+e- -> t t~ ee_ttbar_nlo, e+e- -> u u~ ee_uu_nlo): V (the stored one-loop virtual) at its DyHPO
sweep's best config (vob_tt_V hp_1333, vob_uu_V hp_0670; seed 42 = the sweep trial itself, seeds 43, 44 =
scripts/job_vob_seed.sh), and V/B (data.target_over_born, the virtual over MadLoop's exact Born) at that same config,
seeds 42, 43, 44. Each run's model is its final one, after the HETEROSC sigma stage (D276; the stage freezes the mean,
so its test predictions are the mean's), read from <run>/sigma_stage/preds_test.npz on CC:
    MSE = mean((pred - truth)^2) x prepd_std^2   (standardized log target back to log units, data_stats.json),
which for V is the MSE of log|V| and for V/B the MSE of log|V/B|, equal to that of the recombined log|V| (B exact).
June's from-scratch points beside them (the June 2026 V and V/B solo sweeps at t = 4000, scaling_solo_full, their best
validation loss x the transformed target's std^2: june_targets.py's conversion, from its cached site read and
analysis/hpo_optima/hpo_optima.json; a different setup: AdamW era, no sigma stage, other pools).
The same MSE straight from the raw float32 columns, mean((log raw_pred - log raw_truth)^2), is printed beside it as a
check.

    python3 analysis/virtual_over_born/seed_mse.py [--cached]   -> data/seed_mse_site.json, figs/vob_seeds.{png,pdf}
The last line printed is the JSON of the numbers quoted.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from analysis.common.readers import site_python

CACHE = os.path.join(HERE, "data", "seed_mse_site.json")
# (process, arm, seed) -> run directory under the CC checkout's runs/
RUNS = {}
for proc, sw, hp in (("ee_ttbar", "vob_tt", 1333), ("ee_uu", "vob_uu", 670)):
    RUNS[(proc, "V", 42)] = "%s_V/trial_%04d" % (sw, hp)
    for s in (43, 44):
        RUNS[(proc, "V", s)] = "%s_V_hp%04d_s%d" % (sw, hp, s)
    for s in (42, 43, 44):
        # the first e+e- -> u u~ V/B runs died in plots.py on the constant target (fixed in plots.py, E4605): reruns _r2
        RUNS[(proc, "V/B", s)] = "%s_VB_atVhp%04d_s%d%s" % (sw, hp, s, "_r2" if proc == "ee_uu" else "")

REMOTE = r'''
import json, os, sys, numpy as np
out = {}
for d in sys.argv[1:]:
    p = os.path.join("runs", d, "sigma_stage", "preds_test.npz")
    if not os.path.exists(p):
        out[d] = None
        continue
    z = np.load(p)
    std = json.load(open(os.path.join("runs", d, "sigma_stage", "data_stats.json")))["prepd_std"][0]
    pr, tr = z["pred"].astype(np.float64).ravel(), z["truth"].astype(np.float64).ravel()
    rp, rt = z["raw_pred"].astype(np.float64).ravel(), z["raw_truth"].astype(np.float64).ravel()
    out[d] = {"n": int(pr.size), "prepd_std": float(std), "mse": float(np.mean((pr - tr) ** 2) * std ** 2),
              "mse_raw": float(np.mean((np.log(np.abs(rp)) - np.log(np.abs(rt))) ** 2))}
print("VOB_SEEDS " + json.dumps(out))
'''


def main():
    if "--cached" in sys.argv and os.path.exists(CACHE):
        st = json.load(open(CACHE))
    else:
        st = site_python(REMOTE, sorted(set(RUNS.values())), "VOB_SEEDS", site="ccin2p3")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(st, open(CACHE, "w"), indent=1)
    res = {}
    for (proc, arm, seed), d in sorted(RUNS.items()):
        r = st.get(d)
        res.setdefault(proc, {}).setdefault(arm, {})[seed] = r and r["mse"]
        print("%-9s %-4s seed %d  %-28s MSE %s  (raw float32 %s)" % (
            proc, arm, seed, d, "%.3g" % r["mse"] if r else "missing", "%.3g" % r["mse_raw"] if r else "-"))
    summ = {}
    for proc, a in res.items():
        v, vb = [x for x in a["V"].values() if x], [x for x in a["V/B"].values() if x]
        if v and vb:
            mv, mvb = sorted(v)[len(v) // 2], sorted(vb)[len(vb) // 2]
            summ[proc] = {"V_median": mv, "VB_median": mvb, "median_V_over_VB": mv / mvb,
                          "all_VB_below_V": int(max(vb) < min(v)), "VB_max": max(vb)}
            print("%s: V %.3g..%.3g, V/B %.3g..%.3g; max(V/B) below min(V): %s; geo-mean V over V/B %.3g" % (
                proc, min(v), max(v), min(vb), max(vb), max(vb) < min(v),
                (_gm(v) / _gm(vb))))
    june = june_scratch()
    print("June from scratch, t=4000:", json.dumps(june))
    figure(res, june)
    print(json.dumps({"summary": summ, "june_scratch_t4000": june, "runs": {"|".join(map(str, k)): d for k, d in RUNS.items()}, "site": st, "mse": res}))


def june_scratch():
    """{process: {arm: MSE of log|V|}} of June's from-scratch solo sweeps at t = 4000 (june_targets.py's conversion)."""
    from analysis.virtual_over_born import june_targets as jt
    st = json.load(open(jt.CACHE))
    opt = json.load(open(os.path.join(ROOT, "analysis", "hpo_optima", "hpo_optima.json")))
    out = {}
    for proc in ("ee_ttbar", "ee_uu"):
        for arm, tgt in (("V", "virt_e4"), ("V/B", "virt_ratio")):
            name = "%s_nlo_%s" % (proc, tgt)
            sw = "scaling_solo_full_%s_t04000" % jt.SWEEP_TAG[name]
            r = [x for x in opt if x["sweep"] == sw and x.get("best_val_loss") is not None]
            if r:
                out.setdefault(proc, {})[arm] = r[0]["best_val_loss"] * st[name]["t_std"] ** 2
    return out


def _gm(x):
    import math
    return math.exp(sum(math.log(y) for y in x) / len(x))


def figure(res, june):
    import matplotlib
    matplotlib.use("Agg")
    import plot_style as ps
    ps.use()
    fig, axes = ps.figure(2, 1)
    for ax, proc in zip(axes, ("ee_ttbar", "ee_uu")):
        for i, (arm, col) in enumerate((("V", ps.C.blue), ("V/B", ps.C.vermillion))):
            pts = [(s, m) for s, m in sorted(res.get(proc, {}).get(arm, {}).items()) if m]
            ax.plot([i + 0.08 * (s - 43) for s, _ in pts], [m for _, m in pts], "o", color=col, ms=4,
                    label="%s, now" % arm)
            if arm in june.get(proc, {}):
                ax.plot([i + 0.3], [june[proc][arm]], "s", mfc="none", color=col, ms=5, label="%s, June" % arm)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["V", "V/B"]); ax.set_xlim(-0.5, 1.6)
        ax.set_yscale("log")
        ps.legend(ax, "upper left"); ps.make_room(ax)
        ps.process_label(ax, r"$e^+e^-\to t\bar t$" if proc == "ee_ttbar" else r"$e^+e^-\to u\bar u$")
    axes[0].set_ylabel(r"MSE of $\log|V|$ (test)")
    ps.layout(fig)
    base = os.path.join(HERE, "figs", "vob_seeds")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png"); fig.savefig(base + ".pdf")
    print("figure", base + ".png")


if __name__ == "__main__":
    main()
