#!/usr/bin/env python3
"""
zs_plot.py — Aggregate zero-shot eval JSONs (tools/zero_shot_eval.py outputs) into
a comparison table + bar chart: PID vs physics encoding, on held-out processes.

Reads analysis/zero_shot/<label>__<dataset>.json. Labels are free-form; we map the
ones this experiment produces. PID-on-ee_ttbar is structurally un-encodable
(no top token) and is drawn as a hatched "cannot encode" bar.

Usage: python tools/zs_plot.py
"""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZS = os.path.join(ROOT, "analysis", "zero_shot")

# (label -> pretty arm name). Extend as needed.
ARMS = {
    "phys25_500k_recipe": "physics (500k)",
    "physzs_short":       "physics (20k)",
    "pidzs_short":        "PID (20k)",
}
DSETS = {
    "ee_uu_10-1000GeV_test_amplitudes":    "ee->uu  (flavor swap: u<->d)",
    "ee_ttbar_346-1000GeV_test_amplitudes":"ee->ttbar  (novel top quark)",
}
SPLIT = "test"


def load_all():
    rows = {}
    for f in glob.glob(os.path.join(ZS, "*.json")):
        d = json.load(open(f))
        lab = os.path.basename(f).split("__")[0]
        ds = d["dataset"]
        rows[(lab, ds)] = d
    return rows


def main():
    rows = load_all()
    arms = [a for a in ARMS if any(k[0] == a for k in rows)]
    dsets = list(DSETS)

    print(f"\n{'arm':<16} {'process':<32} {'encodable':<10} {'mse_prepd':<10} "
          f"{'med relerr':<11} {'mean relerr':<11}")
    print("-" * 92)
    table = {}
    for a in arms:
        for ds in dsets:
            d = rows.get((a, ds))
            if d is None:
                continue
            if not d.get("encodable", True):
                table[(a, ds)] = None
                print(f"{ARMS[a]:<16} {DSETS[ds]:<32} {'NO':<10} "
                      f"{'cannot encode (unseen PDG ' + str(d.get('unseen_pdgs')) + ')'}")
                continue
            s = d[SPLIT]
            table[(a, ds)] = s
            print(f"{ARMS[a]:<16} {DSETS[ds]:<32} {'yes':<10} "
                  f"{s['mse_prepd']:<10.4f} {s['rel_err_median']*100:<10.1f}% "
                  f"{s['rel_err_mean']*100:<10.1f}%")

    # ---- bar chart: mse_prepd and median rel err, grouped by process ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"physics (500k)": "#1b9e77", "physics (20k)": "#66c2a5", "PID (20k)": "#d95f02"}
    metrics = [("mse_prepd", "val MSE (standardized log-amp)\nlower = better; 1.0 = predict-mean", 1.0),
               ("rel_err_median", "median |rel err| on amplitude\nlower = better", None)]
    for ax, (mkey, mlabel, ref) in zip(axes, metrics):
        x = np.arange(len(dsets)); w = 0.8 / max(len(arms), 1)
        for i, a in enumerate(arms):
            vals, hatch_flags = [], []
            for ds in dsets:
                s = table.get((a, ds))
                if s is None:                # un-encodable
                    vals.append(0.0); hatch_flags.append(True)
                else:
                    v = s[mkey]; vals.append(v * (100 if mkey.startswith("rel") else 1))
                    hatch_flags.append(False)
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=ARMS[a],
                          color=colors.get(ARMS[a], None))
            for b, hf in zip(bars, hatch_flags):
                if hf:
                    b.set_hatch("xxx"); b.set_edgecolor("black"); b.set_facecolor("white")
                    ax.text(b.get_x() + b.get_width() / 2, 0.02, "cannot\nencode",
                            ha="center", va="bottom", fontsize=8, rotation=0)
        if ref is not None:
            ax.axhline(ref, ls="--", c="gray", lw=1)
            ax.text(len(dsets) - 0.5, ref, " predict-mean", va="bottom", ha="right",
                    fontsize=8, color="gray")
        ax.set_xticks(x); ax.set_xticklabels([DSETS[d] for d in dsets], fontsize=9)
        ax.set_ylabel(mlabel); ax.legend(fontsize=8)
        if mkey.startswith("rel"):
            ax.set_ylabel(ax.get_ylabel() + " (%)")
    fig.suptitle("Zero-shot transfer to held-out processes: physics features vs PID one-hot",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(ZS, "zero_shot_comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
