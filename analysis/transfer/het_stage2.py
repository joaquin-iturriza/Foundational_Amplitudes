"""Stage 2 of the het fine-tune benchmark (D94): the big-catalog pretraining hp58 (runs/dancat_het_pre/trial_0058) fine-tuned
with the het loss on the seven finale probes that have no twin in its catalog, at D = 10^1, 10^2, 10^3, 10^4 (k = 2, 4, 6, 8),
against the finale fine-tuned with the het loss (stage 1 and 2, k = 4, 8) and the MSE cells of the ladder: scratch, rung 9,
synthetic, finale (finale_synth_scaling.py). Every cell is its DyHPO sweep's best trial, the loss its validation MSE of
log|M|^2 at the best checkpoint (het: the mu MSE, the het cells select on it), in the target's raw units (the standardized
value times prepd_std^2; every arm of a probe and D shares one standardization, het_stage2_stds.json). The two one-loop
probes are positive on every event, so the MSE cells' sign head is inert and their targets equal the het ones.
Inputs: sweeps_local/tp3_{hp58,fin}fte_*_het/results/*.json, het_stage2_stds.json (data_stats.json of every trial, from
CC), scratch_sweeps.json through cells.py / finale_synth.py. Outputs: the table printed, het_stage2.json,
figs/het_stage2_{a..g}.{png,pdf} and figs/het_stage2_legend.{png,pdf}.
    .venv-driver/bin/python analysis/transfer/het_stage2.py
"""
import contextlib, glob, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ROOT, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import cell  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

PROBES = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_nnbar": r"$e^+e^-\to\nu\bar\nu$",
          "ee_ttbar": r"$e^+e^-\to t\bar t$", "uubar_Zggg": r"$u\bar u\to Zggg$",
          "ee_dd_nlo": r"$e^+e^-\to d\bar d$, one loop", "ee_bb_nlo": r"$e^+e^-\to b\bar b$, one loop"}
STD = {k: v[0] for k, v in json.load(open(os.path.join(HERE, "het_stage2_stds.json"))).items()}
# every trial of a sweep shares one standardization (one probe, one D; checked over het_stage2_stds.json), so a trial
# added later (D97's extension) takes its sweep's value
SWEEP_STD = {k.split("/")[1]: v for k, v in STD.items()}


def het(sweep):
    """The sweep's best trial: (raw-unit mu MSE, number of trials), or None"""
    fs = glob.glob(os.path.join(ROOT, "sweeps_local", sweep, "results", "*.json"))
    rows = []
    for f in fs:
        r = json.load(open(f))
        rd = r["run_dir"].split("Foundational_Amplitudes/", 1)[1]
        rows.append(r["val_loss"] * STD.get(rd, SWEEP_STD[sweep]) ** 2)
    return (min(rows), len(rows)) if rows else None


def extension():
    """D97: the 13 hp58 sweeps extended in place to lr 2e-2 (het_hp58_ext_cells.json): the best of the first 5 trials
    (trial_idx < 5, the stage-2 sweep) against the best of all 10, and whether the best is again the highest lr tried"""
    out = {}
    for sw in json.load(open(os.path.join(HERE, "het_hp58_ext_cells.json"))):
        d = os.path.join(ROOT, "sweeps_local", sw)
        # each trial's HPs from the driver's state (no surrogate pickle: it loads only under the driver's numpy)
        hps = {t["hp_idx"]: t["hp"] for t in json.load(open(os.path.join(d, "driver_state.json")))["trials"].values()}
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(d, "results", "*.json"))]
        rs = [r for r in rs if r.get("val_loss") is not None]
        lr = lambda r: float(hps[r["hp_idx"]]["training.lr"])
        old = [r for r in rs if r["trial_idx"] < 5]
        bo, bn = min(old, key=lambda r: r["val_loss"]), min(rs, key=lambda r: r["val_loss"])
        out[sw] = dict(n=len(rs), old=bo["val_loss"], new=bn["val_loss"], ratio=bn["val_loss"] / bo["val_loss"],
                       lr_old=lr(bo), lr_new=lr(bn), new_at_top=lr(bn) == max(lr(r) for r in rs),
                       lr_max_tried=max(lr(r) for r in rs))
        print(f"{sw:32s} best {bo['val_loss']:.3g} -> {bn['val_loss']:.3g} ({out[sw]['ratio']:.2f}), lr {lr(bo):.2e} -> "
              f"{lr(bn):.2e}, highest tried {out[sw]['lr_max_tried']:.2e}{'  (best again at the top)' if out[sw]['new_at_top'] else ''}")
    return out


def main():
    K = (2, 4, 6, 8)
    res = {}
    figs = ps.panels(len(PROBES))
    for (fig, ax), (p, lab) in zip(figs, PROBES.items()):
        D = {k: 10 ** (k / 2) for k in range(2, 9)}
        arms = {"scratch (MSE)": {k: scratch(p, k, steered=False)[0] for k in range(2, 9)},
                "rung 9 (MSE)": {k: cell("tp3_r9fte", p, k) for k in range(2, 9)},
                "synthetic (MSE)": {k: cell("tp3_synfte", p, k) for k in range(2, 9)},
                "finale (MSE)": {k: cell("tp3_finfte", p, k) for k in range(2, 9)},
                "finale (het)": {k: (het(f"tp3_finfte_{p}_d{k}_het") or (None,))[0] for k in (4, 8)},
                "hp58 (het)": {k: (het(f"tp3_hp58fte_{p}_d{k}_het") or (None,))[0] for k in K}}
        style = {"scratch (MSE)": dict(color="k", marker="o", ls="--"), "rung 9 (MSE)": dict(color=ps.C.blue, marker="s"),
                 "synthetic (MSE)": dict(color=ps.C.green, marker="^"), "finale (MSE)": dict(color=ps.C.vermillion, marker="D"),
                 "finale (het)": dict(color=ps.C.vermillion, marker="D", mfc="none", ls=":"),
                 "hp58 (het)": dict(color=ps.C.orange, marker="*", ls="-")}
        res[p] = {a: {str(k): v for k, v in c.items()} for a, c in arms.items()}
        for a, c in arms.items():
            pts = [(D[k], v) for k, v in sorted(c.items()) if v]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], label=a, **style[a])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, lab)
        ps.make_room(ax, max_growth=3.0)
        f = lambda v: f"{v:.3g}" if v else "-"
        print(f"{p:11s} " + "  ".join(f"D=1e{k // 2}: hp58 {f(arms['hp58 (het)'].get(k))} finale het {f(arms['finale (het)'].get(k))} "
                                      f"finale {f(arms['finale (MSE)'].get(k))} r9 {f(arms['rung 9 (MSE)'].get(k))} "
                                      f"scr {f(arms['scratch (MSE)'].get(k))}" for k in K))
    a0 = max((ax for _, ax in figs), key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    ps.legend_strip(a0, os.path.join(HERE, "figs", "het_stage2_legend"), ncol=3)
    ps.save_panels(figs, os.path.join(HERE, "figs", "het_stage2"))
    res["_extension"] = extension()
    json.dump(res, open(os.path.join(HERE, "het_stage2.json"), "w"), indent=1)
    print(json.dumps(numbers(res)))


def numbers(res):
    """results.tex numbers (D94, D97): hp58 over the finale, both het, at D=10^2 and 10^4 (the table); hp58 het and
    scratch MSE at D=10; the extension's counts and ratios (best of all 10 trials over best of the first 5); the
    finale het cells whose best is the lowest lr tried; the stage-2 sweeps and their first-five trials"""
    from het_finale_s1 import cell as s1cell
    ratio = {p: {k: res[p]["hp58 (het)"][k] / res[p]["finale (het)"][k] for k in ("4", "8")} for p in PROBES}
    d10 = {p: dict(hp58=res[p]["hp58 (het)"]["2"], scratch=res[p]["scratch (MSE)"]["2"]) for p in PROBES}
    ext = res["_extension"]
    fin_low = [f"{p}_d{k}" for p in PROBES for k in (4, 8)
               if (lambda rows: rows[0]["training.lr"] == min(x["training.lr"] for x in rows))(s1cell(p, k))]
    sweeps = [f"tp3_hp58fte_{p}_d{k}_het" for p in PROBES for k in (2, 4, 6, 8)] + \
             [f"tp3_finfte_{p}_d{k}_het" for p in ("ee_nnbar", "ee_ttbar", "ee_bb_nlo") for k in (4, 8)]
    first5 = sum(json.load(open(f)).get("trial_idx", 0) < 5 for sw in sweeps
                 for f in glob.glob(os.path.join(ROOT, "sweeps_local", sw, "results", "*.json")))
    lowered = [c for c, v in ext.items() if v["ratio"] < 1]
    top = [c for c, v in ext.items() if v["new_at_top"]]
    return dict(ratio=ratio, d10=d10, stage2_sweeps=len(sweeps), stage2_trials=first5, ext_n=len(ext),
                ext_lowered=len(lowered), ext_unchanged_max=max(abs(1 - v["ratio"]) for c, v in ext.items() if c not in lowered),
                ext_gain={c: 1 / v["ratio"] for c, v in ext.items()},
                bb_d8_old=ext["tp3_hp58fte_ee_bb_nlo_d8_het"]["old"], bb_d8_new=ext["tp3_hp58fte_ee_bb_nlo_d8_het"]["new"],
                ext_bracketed=len(ext) - len(top), ext_top=len(top),
                ext_top_dd_nlo_lr=ext["tp3_hp58fte_ee_dd_nlo_d4_het"]["lr_new"],
                ext_top_below_6e3=sum(ext[c]["lr_max_tried"] < 6e-3 for c in top),
                finale_het_cells=len(PROBES) * 2, finale_het_lowest=len(fin_low))


if __name__ == "__main__":
    main()
