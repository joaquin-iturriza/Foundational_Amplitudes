"""Readout of the validation-with-training test (submit_together.py; design 2026-11-12 after D420). Per cell (target,
D, scratch/fine-tune) and arm (control, wsd, half): best validation MSE in ln units (val mu-MSE x
prepd_std^2 at its minimum = the best checkpoint), test MSE after the sigma stage, and curve_quality's separation and
end_up. Per arm against the cell's control rerun: BETTER (best val <= control/2 and beyond the cell's control-vs-original
spread), WORSE (>= 2x and beyond that spread), else SAME; pooled: G = geometric mean over cells of arm/control, against
the pooled rerun spread S = geometric mean of max(c/o, o/c). Branch FIX (good in >= 8 of 12, G < S), GAIN WITHOUT FIX
(G <= 1/(2 S)), else NOT FIXED AT THE BASELINE'S HPs.
Figures figs/together_{val,sep}: per arm, each cell's best val over its control's, and each cell's separation.

    python analysis/data_saving/together_readout.py [--cached]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from curve_quality import BAR, quality  # noqa: E402
from suspects_readout import load, ps  # noqa: E402

DATA = os.path.join(HERE, "data")
ARMS = ("control", "wsd", "half")
TCOL = {"epem_ttx": ps.C.blue, "ee_uu": ps.C.vermillion, "bbx_zz_nlo": ps.C.green}
TLAB = {"epem_ttx": r"$t\bar t$", "ee_uu": r"$u\bar u$", "bbx_zz_nlo": r"$ZZ$ $V/B$"}
CMARK = {("scr", 5): "o", ("scr", 7): "s", ("ft", 5): "^", ("ft", 7): "D"}


def runs():
    """{(arm, side, target, k): run_dir} of the submitted runs."""
    out = {}
    for r in json.load(open(os.path.join(DATA, "together_runs.json"))):
        _, arm, side, tg, dk = r["note"].split()[:5]
        out[(arm, side, tg, int(dk[1:]))] = r["run_dir"]
    return out


def main():
    R = load("--cached" in sys.argv, runs, "together_runs_read.json")
    orig = {(c["arm"], c["target"], c["k"]): c["val_ln"] for c in
            json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]}
    cells = sorted({k[1:] for k in R})
    res = {"bars": BAR, "cells": []}
    for side, tg, k in cells:
        row = {"side": side, "target": tg, "k": k, "original_val": orig.get((side, tg, k))}
        for arm in ARMS:
            r = R.get((arm, side, tg, k), {})
            v, t = r.get("val_mse_curve"), r.get("train_eval_mse_curve")
            if not v or not t:
                row[arm] = {"error": r.get("error", "no curves"), "run_dir": r.get("run_dir")}
                continue
            row[arm] = {"val": r["val_min"], "test": r.get("test_mse"), "train_end": t[-1] * r["prepd_std"] ** 2,
                        **quality(v, t), "run_dir": r["run_dir"]}
        c = row["control"].get("val")
        spread = max(c / row["original_val"], row["original_val"] / c) if c and row["original_val"] else 1.0
        row["rerun_spread"] = spread
        for arm in ARMS[1:]:
            a = row[arm]
            if a.get("val") and c:
                q = a["val"] / c
                a["over_control"] = q
                a["train_end_over_control"] = a["train_end"] / row["control"]["train_end"]  # a fix by underfitting shows here
                a["verdict"] = "BETTER" if q <= 0.5 and q < 1 / spread else "WORSE" if q >= 2 and q > spread else "SAME"
        res["cells"].append(row)
        print(f"{tg:10s} {side:3s} D={10 ** (k / 2):5.0f} orig {row['original_val'] or np.nan:.2e} spread {spread:.2f}  "
              + "  ".join(f"{arm} {row[arm].get('val', np.nan):.2e} sep {row[arm].get('separation', np.nan):.3g}"
                          f"{'' if arm == 'control' else ' ' + row[arm].get('verdict', '-')}" for arm in ARMS))
    res["arms"] = {}
    S = float(np.exp(np.mean([np.log(row["rerun_spread"]) for row in res["cells"]])))
    res["pooled_rerun_spread"] = S
    for arm in ARMS:
        rr = [row[arm] for row in res["cells"] if row[arm].get("val")]
        s = {"n": len(rr), "good": sum(r["well"] for r in rr),
             "better": sum(r.get("verdict") == "BETTER" for r in rr), "worse": sum(r.get("verdict") == "WORSE" for r in rr)}
        q = [r["over_control"] for r in rr if r.get("over_control")]
        s["G"] = float(np.exp(np.mean(np.log(q)))) if q else None
        s["branch"] = None if arm == "control" or not q else (
            "FIX" if s["good"] >= 8 and s["G"] < S else "GAIN WITHOUT FIX" if s["G"] <= 1 / (2 * S)
            else "NOT FIXED AT THE BASELINE'S HPs")
        res["arms"][arm] = s
        print(f"{arm:8s} {s}")
    json.dump(res, open(os.path.join(DATA, "together_readout.json"), "w"), indent=1)
    figure(res)


def figure(res):
    for name, key, ylab, line in (("val", "over_control", "best val MSE / control's", 1.0),
                                  ("sep", "separation", "separation", BAR["separation"])):
        fig, ax = ps.figure()
        x0 = 0 if name == "sep" else 1
        for row in res["cells"]:
            ys = [(i, row[a].get(key) if a != "control" or name == "sep" else None) for i, a in enumerate(ARMS)]
            ys = [(i, y) for i, y in ys if y]
            ax.plot([i for i, _ in ys], [y for _, y in ys], linestyle="none", marker=CMARK[(row["side"], row["k"])],
                    color=TCOL[row["target"]])
        ax.axhline(line, color="grey", linestyle="--")
        ax.set_yscale("log"); ax.set_xticks(range(x0, len(ARMS))); ax.set_xticklabels(ARMS[x0:])
        ax.set_xlim(x0 - 0.5, len(ARMS) - 0.5); ax.set_ylabel(ylab); ax.set_xlabel("arm")
        h = [ax.plot([], [], color=c, marker="o", linestyle="none", label=TLAB[t])[0] for t, c in TCOL.items()]
        h += [ax.plot([], [], color="black", marker=m, linestyle="none",
                      label=f"{'scr' if s == 'scr' else 'FT'} $D={10 ** (k / 2):.0f}$")[0] for (s, k), m in CMARK.items()]
        h += [ax.plot([], [], color="grey", linestyle="--", label="1" if name == "val" else "bar")[0]]
        ps.legend(ax, "upper left", ncol=2)
        ps.save(fig, os.path.join(HERE, "figs", f"together_{name}"))


if __name__ == "__main__":
    main()
