#!/usr/bin/env python3
"""pretrain_data_saving step 6 readout (design passed 2026-11-13 01:52): on b b~ > Z Z tree level, does a WSD-trained
held-out parent keep the pretrained head start through a WSD fine-tune? Arms per D in {100, 316, 3162}: R (stage-2
record, cosine parent + cosine fine-tune, dsave_s2_ft), C (cosine parent + WSD fine-tune, dsave_s5cosparent_ft),
W (WSD parent + WSD fine-tune, dsave_s5wsdparent_ft); reference the stage-2 scratch search best (dsave_s2_scr).
Primary: each search best's v_ft(t) / v_scr(t) (val mu-MSE in ln units, 3-check running median, curve_quality.smooth,
as ft_constraint.py), the run fraction where it first reaches 1, or NEVER; KEPT if NEVER or after 0.5, arm KEPT at
>= 2 of 3 D; branches a-d of the design. Per trial the same fraction and the warmup rise (ft_constraint.rise).
Secondary: best val MSE(ln|M|^2) and test MSE (sigma stage, stage2_readout REMOTE) per arm and D; W/C, C/R, W/R and
each over scratch, pooled (geometric mean over D) against 5^(1/sqrt 3); zero shot = step-0 val of each arm's trials.
Diverged trials listed apart (D398). Figures: figs/step6_curves_{a,b,c} (per D the val curves of the four bests), figs/step6_ratio_{a,b,c} (per D each fine-tune best over the scratch best against the run fraction), figs/step6_best (best val per arm against D).

    python analysis/data_saving/step6_readout.py [--cached]   -> data/step6_readout.json, data/step6_rows.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from curve_quality import smooth, trials  # noqa: E402
from s4_readout import failed  # noqa: E402
import stage2_readout  # noqa: E402
from trial_overfit import driven_rows  # noqa: E402
import plot_style as ps  # noqa: E402

DATA = os.path.join(HERE, "data")
CELLS = {100: (4, 4960), 316: (5, 7840), 3162: (7, 8500)}          # D: (k, T) of the stage-2 twins
NEW = {"C": "s5cosparent", "W": "s5wsdparent", "L": "s6cosparentlongwu"}   # L: the warmup test, C with warmup [0.05, 0.6]
BAR = 5 ** (1 / np.sqrt(3))


def name(arm, D):
    k, T = CELLS[D]
    return f"dsave_{NEW[arm]}_ft_bbx_zz_d{k}_t{T}"


def rows(cached):
    cache = os.path.join(DATA, "step6_rows.json")
    if cached:
        return json.load(open(cache))
    names = [n for n in (name(a, D) for a in NEW for D in CELLS) if os.path.isdir(os.path.join(REPO, "sweeps_local", n))]
    r = {"curves": driven_rows(names),
         "test": driven_rows(names, stage2_readout.REMOTE, "S2T", "'runs/dsave_s5*parent_ft_bbx_zz_d*/trial_*'")}
    json.dump(r, open(cache, "w"))
    return r


def ln(r):
    p = r["prepd_std"] ** 2
    return dict(r, val_ln=[x * p for x in r["val"]], val_min=min(r["val"]) * p)


def lost_at(f, s):
    """The run fraction where v_f / v_s (smoothed) first reaches 1, None if never; the ratio curve."""
    a, b = smooth(f["val_ln"]), smooth(s["val_ln"])
    n = min(len(a), len(b))
    fr, ratio = np.arange(n) / (n - 1), 10 ** (a[:n] - b[:n])
    return next((float(fr[i]) for i in range(n) if ratio[i] >= 1), None), fr, ratio


def rise(r):
    v = smooth(r["val_ln"])
    iw = max(1, int(round(r["hp"]["warmup_frac"] * (len(v) - 1))))
    return float(10 ** (v[:iw + 1].max() - v[0]))


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    rr = rows("--cached" in sys.argv)
    test = {(r["sweep"], r["run"]): r for r in json.load(open(stage2_readout.TEST))}
    test.update({(r["sweep"], r["run"]): r for r in rr["test"]})
    s2 = [ln(r) for r in trials() if r["stage"] == "s2" and r["target"] == "bbx_zz" and r["val"] and len(r["val"]) > 3]
    new = [ln(r) for r in rr["curves"] if r.get("val") and len(r["val"]) > 3]
    out, best = {"cells": {}, "arms": {}, "bar": BAR}, {}
    for D, (k, T) in CELLS.items():
        g = {"scr": [r for r in s2 if r["arm"] == "scr" and round(r["D"]) == D],
             "R": [r for r in s2 if r["arm"] == "ft" and round(r["D"]) == D],
             **{a: [r for r in new if r["sweep"] == name(a, D)] for a in NEW}}
        best[D] = {a: min(v, key=lambda r: r["val_min"]) for a, v in g.items() if v}
        sc = best[D].get("scr")
        cell = {}
        for a, v in g.items():
            if not v:
                continue
            b = best[D][a]
            te = test.get((b["sweep"], os.path.basename(b["run_dir"].rstrip("/"))), {})
            c = {"n": len(v), "best_val": b["val_min"], "best_test": te.get("mse_ln"), "best_run": b["run_dir"],
                 "zero_shot_val": [float(r["val_ln"][0]) for r in v]}
            if a != "scr":
                lost, _, ratio = lost_at(b, sc)
                c.update(start_ft_over_scr=float(ratio[0]), lost_at=lost,
                         kept=lost is None or lost > 0.5, best_rise=rise(b),
                         trials=[{"run": os.path.basename(r["run_dir"].rstrip("/")), "lost_at": lost_at(r, sc)[0],
                                  "rise": rise(r), "val": r["val_min"], "warmup_frac": r["hp"]["warmup_frac"],
                                  "peak_lr": r["hp"]["lr"] * (r["hp"]["lr_scale"] or 1)} for r in v],
                         failed=failed(b["sweep"]) if a in NEW else None)
            cell[a] = c
        for x, y in (("W", "C"), ("C", "R"), ("W", "R"), ("L", "C"), ("R", "scr"), ("C", "scr"), ("W", "scr"), ("L", "scr")):
            if x in cell and y in cell:
                cell[f"{x}/{y}"] = cell[x]["best_val"] / cell[y]["best_val"]
        out["cells"][D] = cell
    for a in ("R", "C", "W", "L"):
        ks = [c[a]["kept"] for c in out["cells"].values() if a in c]
        out["arms"][a] = {"kept_at": sum(ks), "of": len(ks), "kept": len(ks) == 3 and sum(ks) >= 2}
    for q in ("W/C", "C/R", "W/R", "L/C"):
        v = [c[q] for c in out["cells"].values() if q in c]
        if v:
            G = gm(v)
            out["pooled_" + q] = {"G": G, "read": "LOWER" if G < 1 / BAR else "HIGHER" if G > BAR else "WITHIN"}
    # the warmup test's check: does a trial lose the head start where its own warmup ends (the peak lr arrives)?
    # per arm over its trials that did not blow up in the warmup (rise < 10), lost_at (1 if never) against warmup_frac
    for a in ("R", "C", "W", "L"):
        T_ = [t for c in out["cells"].values() if a in c for t in c[a]["trials"] if t["rise"] < 10]
        if len(T_) > 2:
            lo = np.array([t["lost_at"] if t["lost_at"] is not None else 1.0 for t in T_])
            wf = np.array([t["warmup_frac"] for t in T_])
            out["arms"][a]["lost_vs_warmup"] = {
                "n": len(T_), "spearman": float(spearmanr(wf, lo)[0]), "median_lost_minus_wf": float(np.median(lo - wf)),
                "share_within_0.05": float(np.mean(np.abs(lo - wf) <= 0.05))}
    A = out["arms"]
    if all(A.get(a, {}).get("of") == 3 for a in "CW"):
        out["branch"] = {(True, False): "a: WSD parent keeps it", (True, True): "b: WSD fine-tune keeps it",
                         (False, True): "c: WSD parent hurts", (False, False): "d: neither keeps it"}[
            (A["W"]["kept"], A["C"]["kept"])]
    for D, c in out["cells"].items():
        print(f"D={D}: " + "  ".join(
            f"{a} best {c[a]['best_val']:.2e} test {c[a]['best_test'] or float('nan'):.2e}"
            + (f" start {c[a]['start_ft_over_scr']:.3f} lost {c[a]['lost_at']} rise {c[a]['best_rise']:.2f}"
               if a != "scr" else "") for a in ("scr", "R", "C", "W") if a in c))
        print("      " + "  ".join(f"{q} {c[q]:.3g}" for q in c if "/" in q)
              + "  zero shot " + "  ".join(f"{a} {np.median(c[a]['zero_shot_val']):.3g}" for a in ("R", "C", "W") if a in c)
              + "  failed " + str({a: c[a]["failed"] for a in NEW if a in c and c[a]["failed"]}))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}))
    json.dump(out, open(os.path.join(DATA, "step6_readout.json"), "w"), indent=1)
    # figure: per D the val curves of the scratch best and the R, C, W bests (legend: parent schedule -> fine-tune schedule)
    sty = {"scr": ("scratch", "black", "-"), "R": (r"cos$\to$cos", ps.C.grey, "-"),
           "C": (r"cos$\to$WSD", ps.C.blue, "-"), "W": (r"WSD$\to$WSD", ps.C.vermillion, "-"),
           "L": (r"cos$\to$WSD, long warmup", ps.C.green, "-")}
    figs = ps.panels(len(CELLS))
    for (fig, ax), D in zip(figs, CELLS):
        for a, (lab, col, ls) in sty.items():
            r = best[D].get(a)
            if r is None:
                continue
            st = np.arange(len(r["val_ln"])) * (r.get("every") or 1)
            ax.plot(st, r["val_ln"], color=col, linestyle=ls, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(r"val MSE$(\ln|\mathcal{M}|^2)$")
        ps.process_label(ax, rf"$b\bar b\to ZZ$, $D={D}$", loc="upper right")
        ps.legend(ax, "lower left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "step6_curves"))
    # figure: per D each fine-tune best's val over the scratch best's (smoothed) against the run fraction
    figs = ps.panels(len(CELLS))
    for (fig, ax), D in zip(figs, CELLS):
        for a in ("R", "C", "W", "L"):
            if a in best[D] and "scr" in best[D]:
                _, fr, ratio = lost_at(best[D][a], best[D]["scr"])
                ax.plot(fr, ratio, color=sty[a][1], label=sty[a][0])
        ax.axhline(1, color="0.5", ls="--", label="equal")
        ax.set_yscale("log"); ax.set_xlabel("fraction of the run"); ax.set_ylabel("val MSE fine-tune / scratch")
        ps.process_label(ax, rf"$b\bar b\to ZZ$, $D={D}$", loc="upper right")
        ps.legend(ax, "lower left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "step6_ratio"))
    # figure: best val per arm against D
    fig, ax = ps.figure()
    for a, (lab, col, _) in sty.items():
        Ds = [D for D in CELLS if a in out["cells"][D]]
        ax.plot(Ds, [out["cells"][D][a]["best_val"] for D in Ds], color=col, marker="o", label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$D$ (training events)")
    ax.set_ylabel(r"best val MSE$(\ln|\mathcal{M}|^2)$")
    ps.process_label(ax, r"$b\bar b\to ZZ$", loc="upper right")
    ps.legend(ax, "lower left")
    ps.save(fig, os.path.join(HERE, "figs", "step6_best"))
    # figure: per trial (no warmup blow-up, rise < 10) the run fraction where the head start is lost against its own
    # warmup fraction, D = 100, 316, 3162 pooled; never lost drawn at 1
    fig, ax = ps.figure()
    for a in ("R", "C", "W", "L"):
        for D, c in out["cells"].items():
            T_ = [t for t in c.get(a, {}).get("trials", []) if t["rise"] < 10]
            ax.scatter([t["warmup_frac"] for t in T_], [t["lost_at"] if t["lost_at"] is not None else 1.0 for t in T_],
                       color=sty[a][1], label=sty[a][0] if D == 100 else None)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", label="lost at warmup end")
    ax.set_xlim(0, 0.65); ax.set_ylim(0, 1.02)
    ax.set_xlabel("fine-tune warmup fraction"); ax.set_ylabel("run fraction where the head start is lost")
    ps.process_label(ax, r"$b\bar b\to ZZ$", loc="lower right")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "step6_lost_vs_warmup"))


if __name__ == "__main__":
    main()
