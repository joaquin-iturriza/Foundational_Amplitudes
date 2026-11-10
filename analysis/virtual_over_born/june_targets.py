"""The June 2026 virtual-over-Born attempt read from its own datasets (plan virtual_over_born, D366).

The four datasets the June sweeps trained on (make_ratio_datasets.py and data/parse_dat.py at commit 2b81ee1e; on Jean
Zay's checkout data/, the only copy): ee_{uu,ttbar}_nlo_virt_e4 (the stripped virtual times e^4) and
ee_{uu,ttbar}_nlo_virt_ratio (virt_fin / born from the same .dat rows). Per dataset, on the rows the runs trained on
(data.subsample 100000, the 0.7 train share; the first 70000 rows, the order the loader kept before its shuffle being
immaterial to a mean and a std): the target's range, sign, its transform as the June loader resolved it (positive ->
log, else legacy signedlog with s = 1, preprocessing.resolve_amp_trafos scale_quantile=None), the transformed target's
mean and std (the standardization the loss is measured in), std/|mean| and the number of distinct values.

Then, locally, the June sweeps' best validation losses (analysis/hpo_optima/hpo_optima.json, standardized units)
converted to the raw transformed units, loss x std^2: for log targets the MSE of log|target|, and for the ratio (B
exact) the MSE of the recombined log|V| = log|V/B| + log B.

    python3 analysis/virtual_over_born/june_targets.py [--cached]   (the read on Jean Zay's login node via site_python,
        kept in data/june_targets_site.json; --cached reuses it) -> figs/june_vob.{png,pdf}
The last line printed is the JSON of the numbers quoted.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from analysis.common.readers import site_python

NAMES = ("ee_uu_nlo_virt_e4", "ee_uu_nlo_virt_ratio", "ee_ttbar_nlo_virt_e4", "ee_ttbar_nlo_virt_ratio")
SWEEP_TAG = {"ee_uu_nlo_virt_e4": "eeuunlovirte4", "ee_uu_nlo_virt_ratio": "eeuunlovirtratio",
             "ee_ttbar_nlo_virt_e4": "eettbarnlovirte4", "ee_ttbar_nlo_virt_ratio": "eettbarnlovirtratio"}

REMOTE = r'''
import json, sys, numpy as np
out = {}
for name in sys.argv[1:]:
    a = np.load("data/%s.npy" % name, mmap_mode="r")[:70000, -1].astype(np.float64)
    signed = bool(a.min() <= 0)
    t = np.sign(a) * np.log1p(np.abs(a)) if signed else np.log(a)
    out[name] = {"n": int(a.size), "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                 "rel_std_raw": float(a.std() / abs(a.mean())), "frac_neg": float((a < 0).mean()),
                 "transform": "signedlog(s=1)" if signed else "log", "t_mean": float(t.mean()),
                 "t_std": float(t.std()), "n_distinct": int(np.unique(a).size)}
print("JUNE_TARGETS " + json.dumps(out))
'''


ARMS = (("finetune_scaling_virt_002", "fine-tuned"), ("scaling_solo_full", "scratch"))
CACHE = os.path.join(HERE, "data", "june_targets_site.json")


def curve(opt, arm, tag, s2):
    """(steps, best val loss x std^2) of one June scaling cell family, sorted by steps."""
    pts = sorted((r["best_t_steps"], r["best_val_loss"] * s2) for r in opt
                 if r.get("best_val_loss") is not None and r["sweep"].startswith(arm + "_" + tag + "_t"))
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    if "--cached" in sys.argv and os.path.exists(CACHE):
        st = json.load(open(CACHE))
    else:
        st = site_python(REMOTE, list(NAMES), "JUNE_TARGETS", site="jeanzay")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(st, open(CACHE, "w"), indent=1)
    opt = json.load(open(os.path.join(ROOT, "analysis", "hpo_optima", "hpo_optima.json")))
    conv = {}
    for name in NAMES:
        s2 = st[name]["t_std"] ** 2
        for r in opt:
            sw = r["sweep"]
            if r.get("best_val_loss") is None or not sw.endswith("_t04000") or ("_" + SWEEP_TAG[name] + "_") not in sw:
                continue
            conv[sw] = {"loss_std_units": r["best_val_loss"], "t_std": st[name]["t_std"],
                        "mse_transformed_units": r["best_val_loss"] * s2}
    # matched pairs at t = 4000 (same arm, same process, V against V/B), the V/B raw MSE over V's
    pairs = {}
    for sw in conv:
        if "virtratio" in sw:
            v = sw.replace("virtratio", "virte4")
            if v in conv:
                pairs[sw.replace("virtratio", "")] = {"V": conv[v]["mse_transformed_units"],
                                                      "VB": conv[sw]["mse_transformed_units"],
                                                      "V_over_VB": conv[v]["mse_transformed_units"]
                                                      / conv[sw]["mse_transformed_units"]}
    for k, v in st.items():
        print(k, json.dumps(v))
    for k, v in sorted(pairs.items()):
        print("%-50s V %.3g  V/B %.3g  V over V/B %.3g" % (k, v["V"], v["VB"], v["V_over_VB"]))
    figure(st, opt)
    print(json.dumps({"targets": st, "t4000": conv, "pairs": pairs}))


def figure(st, opt):
    import matplotlib
    matplotlib.use("Agg")
    import plot_style as ps
    ps.use()
    fig, axes = ps.figure(2, 1, sharey=True)
    for ax, proc in zip(axes, ("ee_ttbar", "ee_uu")):
        for (arm, lab), mk in zip(ARMS, ("o", "s")):
            for tgt, col, tl in (("virt_e4", ps.C.blue, "V"), ("virt_ratio", ps.C.vermillion, "V/B")):
                name = "%s_nlo_%s" % (proc, tgt)
                x, y = curve(opt, arm, SWEEP_TAG[name], st[name]["t_std"] ** 2)
                if x:
                    ax.plot(x, y, marker=mk, color=col, ls="-" if lab == "scratch" else "--", ms=3,
                            label="%s, %s" % (tl, lab))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("steps")
    axes[0].set_ylabel(r"MSE of $\log|V|$")
    axes[0].legend(fontsize=7)
    ps.layout(fig)
    base = os.path.join(HERE, "figs", "june_vob")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png"); fig.savefig(base + ".pdf")
    print("figure", base + ".png")


if __name__ == "__main__":
    main()
