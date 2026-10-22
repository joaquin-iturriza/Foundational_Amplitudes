"""D75: the het-loss DyHPO sweeps on 8 Delta cells with more than two final-state particles (sweep/gen_het23_sweeps.py),
each sweep's best trial against its MSE cell (best of the MSE sweep, delta_plots.pick) and against the het retrain at the
MSE cell's HPs (D47, het_delta_data). Inputs: het23_data/*.npz (job_het23_eval.sh, collect_het_delta.py), het23_hps.txt
(the searched HPs of every best run, read from their config.yaml on CC).
  het23_delta   median |Delta| per cell: MSE, het at the MSE HPs, het at its own best HPs
  het23_hps     each searched HP of each cell's best run, MSE and het, as its position in the search range
    python analysis/transfer/het23_summary.py
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import load, pick, tag  # noqa: E402

CELLS = ("tp3_scr_ee_ddbarg_d8", "tp3_r1fte_ee_ddbarg_d8", "tp3_scr_ee_ttbarg_d8", "tp3_r1fte_ee_ttbarg_d8",
         "tp_scr_uubar_Zgg_d8", "tp3_r1fte_uubar_Zgg_d8", "tp3_scrh_uubar_Zggg_d8", "tp3_r1fte_uubar_Zggg_d8")
LAB = {"ee_ddbarg": r"$e^+e^-\to d\bar dg$", "ee_ttbarg": r"$e^+e^-\to t\bar tg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$"}
HPS = (("lr", "log", 9.12e-5, 9.12e-3), (r"$\lambda$", "log", 1e-10, 1e-6), ("warmup", "lin", 0.05, 0.2),
       (r"$\eta_{\min}$", "log", 1e-10, 1e-7), ("EMA", "lin", 0.9, 0.999))


def probe(c):
    return c.split("_", 2)[2].rsplit("_d8", 1)[0]


def arm(c):
    return "scratch" if "_scr" in c else "rung 1"


def main():
    mse, hetd = load(), load("het_delta_data")
    h23 = {}
    for f in glob.glob(os.path.join(HERE, "het23_data", "*.npz")):
        with np.load(f) as z:
            h23.update({k[: -len("__delta")]: z[k] for k in z.files if k.endswith("__delta") and "_het__" in k})
    rows = {(r.get("paper_8k") or {}).get("sweep"): r["paper_8k"]
            for r in json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"] if r.get("paper_8k")}
    hp = {}
    for l in open(os.path.join(HERE, "het23_hps.txt")):
        f = l.split()
        if len(f) == 7:
            hp[f[0]] = [float(f[1]), float(f[2]), float(f[3]), float(f[4]), f[5] == "True", float(f[6])]
    med = lambda a: float(np.median(np.abs(a[np.isfinite(a)])))
    out = []
    for c in CELLS:
        p = rows[c]
        t = tag(p["run_dir"])
        k = next(k for k in h23 if k.startswith(c + "_het__"))
        out.append(dict(cell=c, mse=med(mse[pick(mse, p)]), het_mse_hp=med(hetd[t]) if t in hetd else None,
                        het_own=med(h23[k]), mse_run=p["run_dir"], het_run="runs/" + k.replace("__", "/")))
    for o in out:
        print(f"{o['cell']:26s} MSE {o['mse']:.4g}  het@MSE-HPs {o['het_mse_hp'] or float('nan'):.4g}  "
              f"het@own {o['het_own']:.4g}  own/MSE {o['het_own'] / o['mse']:.2f}")
    json.dump(out, open(os.path.join(HERE, "het23_summary.json"), "w"), indent=1)

    fig, ax = ps.figure()
    x = np.arange(len(out))
    ax.plot(x, [o["mse"] for o in out], "o", color=ps.C.blue, label="MSE")
    ax.plot(x, [o["het_mse_hp"] for o in out], "s", color=ps.C.green, mfc="none", label="het, the MSE cell's HPs")
    ax.plot(x, [o["het_own"] for o in out], "D", color=ps.C.vermillion, label="het, its own best HPs")
    ax.set_yscale("log"); ax.set_ylabel(r"median $|\Delta|$")
    ax.set_xticks(x, [f"{i + 1}" for i in x]); ax.set_xlabel("cell")
    ps.shared_legend(fig, ax, ncol=2)          # a corner legend lands on the points
    ps.save(fig, os.path.join(HERE, "figs", "het23_delta"))

    fig, ax = ps.figure()
    for j, (name, sc, lo, hi) in enumerate(HPS):
        col = j if j < 4 else 5
        for o, dx in zip(out, np.linspace(-0.3, 0.3, len(out))):
            for run, mk, c in ((o["mse_run"], "o", ps.C.blue), (o["het_run"], "D", ps.C.vermillion)):
                v = hp[run][col]
                u = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if sc == "log" else (v - lo) / (hi - lo)
                ema_off = not hp[run][4]
                ax.plot([j + dx], [u], mk, color=c, mfc="none" if (col == 5 and ema_off) else c)
    ax.plot([], [], "o", color=ps.C.blue, label="MSE best"); ax.plot([], [], "D", color=ps.C.vermillion, label="het best")
    ax.plot([], [], "o", color="grey", mfc="none", label="EMA off")
    ax.set_xticks(range(len(HPS)), [h[0] for h in HPS]); ax.set_ylabel("position in the search range")
    ax.set_ylim(-0.05, 1.05)
    ps.shared_legend(fig, ax, ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", "het23_hps"))
    print("cells:", "; ".join(f"{i + 1} {LAB[probe(c)]} {arm(c)}" for i, c in enumerate(CELLS)))


if __name__ == "__main__":
    main()
