"""D83: the catalog pretrainings under normalizations that need no data of a process (A: the per-event factor
(s0/s_hat)^(4-N)/C_p and one shared standardization; B: one shared standardization; scripts/job_catalog_norm.sh), read
on the 139 held-out processes on their own, against hp58 (per-process standardization) read with the statistics map g
(heldout_delta.py). Every number is MSE(log|M|^2) per process: the factor multiplies truth and prediction alike, so the
difference q - t is the same in log|M|^2. "After the best rescale and shift" (an oracle: fitted on the test truth)
for each model too. A run that blew up is read at its best checkpoint and says so.
Inputs: data/heldout_preds.npz (hp58), data/catalog_norm_{A,B}.npz (collect_heldout_preds.py on
$SCRATCH/zz_catalog_norm_{A,B}), data/stats_net.json (g).
Output: the table printed, data/catalog_norm_heldout.json, figs/catalog_norm_mse_{interpolation,extrapolation}.
    python analysis/catalog_daniel/catalog_norm_heldout.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402
from heldout_delta import ORDER, SHORT, proc_label, G  # noqa: E402

BLOWUP = {"B": "diverged at step 8734 (50 non-finite-gradient steps); its best checkpoint"}


def load(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def main():
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    twins = json.load(open(os.path.join(HERE, "heldout_twins.json")))
    z58 = dict(np.load(os.path.join(HERE, "data", "heldout_preds.npz")))
    sn = json.load(open(os.path.join(HERE, "data", "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z58:
            z58[f"{n}__q"] = mu + np.exp(lsd) * z58[f"{n}__h"].astype(float)
    models = {"hp58 + g": z58}
    for arm in ("A", "B"):
        z = load(os.path.join(HERE, "data", f"catalog_norm_{arm}.npz"))
        if z is not None:
            models[arm + (" (blew up)" if arm in BLOWUP else "")] = z
    probes = sorted({k.split("__")[0] for k in z58 if k.endswith("__t")})
    mse = lambda z, p, k: float(np.mean((z[f"{p}__{k}"].astype(float) - z[f"{p}__t"].astype(float)) ** 2)) \
        if f"{p}__{k}" in z else np.nan
    res = {}
    for name, z in models.items():
        for k, lab in (("q", "on its own"), ("q_aff", "after rescale and shift")):
            m = {p: mse(z, p, k) for p in probes}
            for rel in (False, True):
                Q = [m[p] for p in probes if (p in twins) == rel and np.isfinite(m[p])]
                res[f"{name}|{lab}|{'96 relabels' if rel else '43 new'}"] = dict(
                    n=len(Q), median=float(np.median(Q)) if Q else None,
                    p10=float(np.quantile(Q, .1)) if Q else None, p90=float(np.quantile(Q, .9)) if Q else None)
    # the on-its-own error split per process into the squared mean error (level) and the error's variance (shape):
    # medians of each, and the median of the per-process level fraction (medians of the parts do not add)
    for name, z in models.items():
        for rel in (False, True):
            d = [z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in probes
                 if (p in twins) == rel and f"{p}__q" in z]
            lv, sh = np.array([x.mean() ** 2 for x in d]), np.array([x.var() for x in d])
            res[f"{name}|split|{'96 relabels' if rel else '43 new'}"] = dict(
                n=len(d), level=float(np.median(lv)), shape=float(np.median(sh)),
                level_fraction=float(np.median(lv / (lv + sh))))
            v = res[f"{name}|split|{'96 relabels' if rel else '43 new'}"]
            print(f"{name}|split|{'96 relabels' if rel else '43 new'}: level {v['level']:.3g}, shape {v['shape']:.3g}, "
                  f"level fraction (median per process) {v['level_fraction']:.2f}")
    for k, v in res.items():
        if v.get("median") is not None:
            print(f"{k:58s} n={v['n']:3d} median {v['median']:.3g} (10-90% {v['p10']:.3g} to {v['p90']:.3g})")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_norm_heldout.json"), "w"), indent=1)

    import matplotlib.pyplot as plt
    groups = {sp: [g for g in ORDER[sp] if any(meta[p]["test_group"] == g for p in probes)] for sp in ORDER}
    sty = {"hp58 + g": (ps.C.grey, "o"), "A": (ps.C.vermillion, "D"), "B (blew up)": (ps.C.blue, "s"),
           "B": (ps.C.blue, "s")}
    for sp in ("interpolation", "extrapolation"):
        ref = "A" if "A" in models else "hp58 + g"
        P = [p for g in groups[sp] for p in sorted((p for p in probes if meta[p]["test_group"] == g),
                                                   key=lambda p: mse(models[ref], p, "q"))]
        fig, ax = plt.subplots(figsize=(8.0, 2.9))
        x = np.arange(len(P))
        for name, z in models.items():
            c, mk = sty[name]
            ax.plot(x, [mse(z, p, "q") for p in P], mk, color=c, mfc=c, ms=3.2, label=name + ", on its own")
        ax.set_yscale("log"); ax.set_xlim(-0.7, len(P) - 0.3)
        ax.set_xticks(x, [proc_label(meta[p]["process"]) for p in P], rotation=90, fontsize=5.5)
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        i0 = 0
        for g in groups[sp]:
            n = sum(meta[p]["test_group"] == g for p in P)
            if i0:
                ax.axvline(i0 - 0.5, color="grey", lw=0.6)
            ax.text(i0 + n / 2 - 0.5, 1.02, SHORT.get(g, g), transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7)
            i0 += n
        fig.legend(*ax.get_legend_handles_labels(), loc="upper right", fontsize=7, frameon=False, ncol=len(models))
        fig.text(0.08, 0.97, f"held-out: {sp}", fontsize=9, va="top")
        fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.36)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(HERE, "figs", f"catalog_norm_mse_{sp}.{ext}"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
