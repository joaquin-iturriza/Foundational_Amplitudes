"""Transfer study, the accuracy benchmark of arXiv:2603.22407 and the draft's Sec. "Accuracy and precision
benchmark" (decision D42): the distribution of the per-event relative error Delta = (A_theta - A)/A on each probe's
test split, |Delta| on log bins, one line per model, at each cell's largest D on the paper's 8k grid (d8 = D 10^4;
analysis/transfer/delta_audit.json lists every run drawn and why it is the one in the scaling figures).
A_theta and A are the raw |M|^2 (tools/eval_delta.py: the best checkpoint, EMA off with the checkpoint's own weights).

Sets (each panel one probe):
  delta_ladder_<probe>    scratch and every rung 0-9 (fig:ladder's cells), legend strip delta_ladder_legend
  delta_focus_<probe>     the rung_focus arms: scratch, the rung below the first-gaining rung, that rung, the best
  delta_finale_<probe>    the finale_synth arms: scratch, rung 9, synthetic, finale
  delta_star_<arm>        each star arm on its own probe with scratch and rung 1; delta_star_ee_ddbar: every arm
Data: analysis/transfer/delta_data/<site>.npz, one array of test-split Delta per run, keyed <sweep>__<trial>
(collect_delta.py packs them on each site).
    python analysis/transfer/delta_plots.py
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$", "ee_ddbarg": r"$e^+e^-\to d\bar dg$", "ee_ttbarg": r"$e^+e^-\to t\bar tg$",
       "ee_ttbar_nlo_thr": r"$e^+e^-\to t\bar t$ (1-loop, thr.)", "udbar_enu": r"$u\bar d\to e^+\nu_e$",
       "ee_dd_ew_nlo": r"$e^+e^-\to d\bar d$ (EW 1-loop)"}
OUT = os.path.join(HERE, "figs")
BINS = np.logspace(-7, 1, 49)


def tag(run_dir):
    return "__".join(os.path.normpath(run_dir).split(os.sep)[-2:])


def load():
    d = {}
    for f in sorted(glob.glob(os.path.join(HERE, "delta_data", "*.npz"))):
        with np.load(f) as z:
            d.update({k: z[k] for k in z.files if k != "_meta"})
            if "_meta" in z.files:
                d.setdefault("_meta", {}).update(json.loads(z["_meta"].tobytes().decode()))
    return d


def hist(ax, delta, **kw):
    """The |Delta| density per unit log10|Delta| (each line integrates to the fraction of events inside the bins)."""
    a = np.abs(delta[np.isfinite(delta)])
    h, e = np.histogram(a, bins=BINS)
    h = h / max(len(a), 1) / np.diff(np.log10(e))
    ax.stairs(h, e, **kw)
    return float(np.median(a))


def axes_lab(ax):
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\Delta| = |A_\theta - A|/A$")
    ax.set_ylabel(r"events per unit $\log_{10}|\Delta|$")


def main():
    audit = json.load(open(os.path.join(HERE, "delta_audit.json")))
    D = load()
    missing, off = [], []

    def get(row):
        p = row.get("paper_8k") or {}
        t = tag(p.get("run_dir", "?"))
        if t not in D and t + "_rerun" in D:
            t += "_rerun"          # a pruned trial rerun from its config (scripts/job_rerun_trial.sh)
        if t not in D:
            missing.append(f"{row['probe']} {row['arm']} {p.get('site')}:{p.get('run_dir')}")
            return None
        rec, got = p.get("mse_logM2"), D.get("_meta", {}).get(t)
        if rec and got and not t.endswith("_rerun") and abs(np.log10(got / rec)) > 0.02:
            off.append(f"{t}: rebuilt {got:.3g} vs recorded {rec:.3g}")
        return D[t]

    by = {}
    for r in audit["rows"]:
        for f in r["figures"]:
            fam, _, arm = f.partition(":")
            by.setdefault(fam, {}).setdefault(r["probe"], []).append((arm or r["arm"], r))
    medians = {}
    os.makedirs(OUT, exist_ok=True)

    # ladder: scratch black, rungs 0-9 on the sequence
    seq = dict(zip([f"r{i}" for i in range(10)], ps.sequence(10)))
    leg_ax = None
    for probe, rows in by.get("ladder", {}).items():
        fig, ax = ps.figure()
        for arm, r in sorted(rows, key=lambda x: (x[0] != "scratch", x[0])):
            dl = get(r)
            if dl is None:
                continue
            lab = "from scratch" if arm == "scratch" else f"rung {arm[1:]}"
            medians[f"ladder {probe} {arm}"] = hist(ax, dl, color="k" if arm == "scratch" else seq[arm], label=lab)
        axes_lab(ax); ps.process_label(ax, LAB[probe])
        if leg_ax is None and len(ax.get_legend_handles_labels()[0]) == 11:
            leg_ax = ax
            ps.legend_strip(ax, os.path.join(OUT, "delta_ladder_legend"), ncol=6)
        ps.save(fig, os.path.join(OUT, f"delta_ladder_{probe}"))

    def small(fam, colors, labels, base):
        for probe, rows in by.get(fam, {}).items():
            fig, ax = ps.figure()
            for arm, r in sorted(rows, key=lambda x: list(colors).index(x[0])):
                dl = get(r)
                if dl is None:
                    continue
                rung = r["arm"][1:] if r["arm"].startswith("r") else ""
                medians[f"{fam} {probe} {arm}"] = hist(ax, dl, color=colors[arm], label=labels(arm, rung))
            axes_lab(ax); ps.process_label(ax, LAB[probe]); ps.legend(ax, "upper left")
            ps.save(fig, os.path.join(OUT, f"{base}_{probe}"))

    small("rung_focus", {"scratch": "k", "rung below": ps.C.sky, "first rung": ps.C.blue, "best overall": ps.C.vermillion},
          lambda a, n: {"scratch": "from scratch", "rung below": f"rung {n}", "first rung": f"rung {n}",
                        "best overall": f"rung {n} (best)"}[a], "delta_focus")
    small("finale_synth_paper", {"scratch": "k", "rung 9": ps.C.blue, "synthetic": ps.C.green, "finale": ps.C.vermillion},
          lambda a, n: {"scratch": "from scratch", "rung 9": "rung 9", "synthetic": "synthetic", "finale": "finale"}[a],
          "delta_finale")

    # star arms: own probe (scratch, rung 1, the arm); ee_ddbar from every arm with rung 1 dashed
    ARM_C = {"soft": ps.C.vermillion, "isr": ps.C.orange, "deadcone": ps.C.green, "threshold": ps.C.sky,
             "wpole": ps.C.purple, "sudakovew": ps.C.yellow}
    star = by.get("star_arms", {})
    own = {a: p for p, rows in star.items() for a, _ in rows if a.endswith(" on own probe")}
    for a, p in own.items():
        arm = a.split()[0]
        fig, ax = ps.figure()
        for lab_arm, r in star[p]:
            if lab_arm in ("scratch", "rung 1") or lab_arm == a:
                dl = get(r)
                if dl is None:
                    continue
                c, lab = {"scratch": ("k", "from scratch"), "rung 1": (ps.C.blue, "rung 1")}.get(
                    lab_arm, (ARM_C[arm], f"rung 1 + {arm}"))
                medians[f"star {p} {lab_arm}"] = hist(ax, dl, color=c, label=lab)
        axes_lab(ax); ps.process_label(ax, LAB[p]); ps.legend(ax, "upper left")
        ps.save(fig, os.path.join(OUT, f"delta_star_{arm}"))
    fig, ax = ps.figure()
    for lab_arm, r in star.get("ee_ddbar", []):
        dl = get(r)
        if dl is None:
            continue
        if lab_arm.startswith("panel g"):
            medians["star ee_ddbar rung 1"] = hist(ax, dl, color=ps.C.blue, linestyle="--", label="rung 1")
        else:
            arm = lab_arm.split()[0]
            medians[f"star ee_ddbar {arm}"] = hist(ax, dl, color=ARM_C[arm], label=f"+ {arm}")
    axes_lab(ax); ps.process_label(ax, LAB["ee_ddbar"]); ps.legend(ax, "upper left", ncol=2)
    ps.save(fig, os.path.join(OUT, "delta_star_ee_ddbar"))

    json.dump(medians, open(os.path.join(HERE, "delta_medians.json"), "w"), indent=1, sort_keys=True)
    print(f"{len(medians)} lines drawn; {len(missing)} missing")
    for m in missing:
        print("  missing", m)
    print(f"{len(off)} rebuilt validation losses off the recorded one by > 5%")
    for m in off:
        print("  !!", m)


if __name__ == "__main__":
    main()
