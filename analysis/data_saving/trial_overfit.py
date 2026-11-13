#!/usr/bin/env python3
"""pretrain_data_saving step 10 (Joaquin 2026-11-12: 'that blow up is a sign of the training being bad, bad HPs, bad
learning rate'): every trial of every stage-2 search, not only the best, with its own logged mu-MSE curves
(val_mse_curve, train_eval_mse_curve of its result JSON on CC, ln units by prepd_std of data_stats.json) and its HPs
(config.yaml). Per trial: rise = val mu-MSE at the end over its minimum, gap = val over train mu-MSE at the end, and the
peak lr of the head group, training.lr x fine_tune.lr_scale (fine-tune) or training.lr (scratch).
Cache data/trial_overfit.json; figure figs/trial_overfit_<a..d> (rise against peak lr, one panel per target).

    python analysis/data_saving/trial_overfit.py [--cached]
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_readout import run_of, searches  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs/")

REMOTE = r'''
import glob, json, os, sys
import numpy as np, yaml
out = []
for sw in sys.argv[1:]:
    for d in sorted(glob.glob("runs/" + sw + "/trial_*")):
        r = {"sweep": sw, "run_dir": d}
        try:
            c = yaml.safe_load(open(d + "/config.yaml"))
            j = json.load(open(c["training"]["result_path"]))
            r["every"] = j.get("validate_every")
            r["val"], r["train"] = j.get("val_mse_curve"), j.get("train_eval_mse_curve")
            r["prepd_std"] = float(np.ravel(json.load(open(d + "/data_stats.json"))["prepd_std"])[0])
            tr, ft = c["training"], (c.get("fine_tune") or {})
            r["hp"] = {"lr": tr.get("lr"), "lr_scale": ft.get("lr_scale"), "layer_decay": ft.get("layer_decay"),
                       "muon_weight_decay": tr.get("muon_weight_decay"), "warmup_frac": tr.get("cosanneal_warmup_frac"),
                       "reg": tr.get("regularization_lambda"), "ema": c.get("ema"), "batchsize": tr.get("batchsize")}
        except Exception as e:
            r["error"] = repr(e)[:160]
        out.append(r)
print("TOVF " + json.dumps(out))
'''


def driven_rows(names, script=REMOTE, marker="TOVF", pattern=None):
    """The rows `script` prints for drive.py searches, read on every site that ran one of their observed trials (CC
    and lxplus), kept only for the run dir each trial's result record names (sweeps_local/<name>/results/*.json; a
    requeued trial runs in trial_<hp>_r<n> beside its lost first try). `pattern`: the script's one argument (a quoted
    glob of run dirs) in place of the names."""
    keep, sites = set(), {}
    for n in names:
        st = {t["hp_idx"]: t["site"] for t in
              json.load(open(os.path.join(REPO, "sweeps_local", n, "driver_state.json")))["trials"].values()}
        for f in glob.glob(os.path.join(REPO, "sweeps_local", n, "results", "*.json")):
            j = json.load(open(f))
            keep.add((n, os.path.basename(j["run_dir"].rstrip("/"))))
            sites.setdefault(st[j["hp_idx"]], set()).add(n)
    rows = []
    for site, ns in sorted(sites.items()):
        args = [pattern] if pattern else sorted(ns)
        rows += [dict(r, site=site) for r in site_python(script, args, marker, site=site, timeout=1800)]
    return [r for r in rows if (r["sweep"], os.path.basename((r.get("run_dir") or r["run"]).rstrip("/"))) in keep]


def load(cached=False):
    """Every observed trial of the stage-2 searches with its curves, HPs and derived reads; the cache holds the raw
    rows read on CC."""
    cache = os.path.join(DATA, "trial_overfit.json")
    meta = {c["sweep"]: c for c in json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]}
    rows = json.load(open(cache)) if cached else site_python(REMOTE, sorted(meta), "TOVF", timeout=1800)
    if not cached:
        json.dump(rows, open(cache, "w"))
    S = searches()
    # a requeued trial runs in trial_<hp>_r<n> beside its lost first try, which may share its result path: keep only
    # the run dir each observed trial's result record names
    keep = {(sw, run_of(sw, t)) for sw, s in S.items() for t in s["trials"]}
    rows = [r for r in rows if (r["sweep"], os.path.basename(r["run_dir"])) in keep]
    for r in rows:
        c, s = meta[r["sweep"]], S[r["sweep"]]
        best = run_of(r["sweep"], next(t for t in s["trials"] if int(t["hp_idx"]) == c["best_hp"]))
        r.update(arm=c["arm"], target=c["target"], D=c["D"], T=c["T"], best=os.path.basename(r["run_dir"]) == best)
        v, t = r.get("val"), r.get("train")
        if v:
            hp = r["hp"]
            r["rise"] = v[-1] / min(v)
            r["val_min_ln"] = min(v) * r["prepd_std"] ** 2
            r["peak_lr"] = hp["lr"] * (hp["lr_scale"] if r["arm"] == "ft" else 1.0)
            # the sampled lr's place in its search window on a log scale, 0 = low edge, 1 = high edge
            x = hp["lr_scale"] if r["arm"] == "ft" else hp["lr"]
            r["x_window"] = float(np.log(x / s["lo"]) / np.log(s["hi"] / s["lo"]))
    return rows


T = {"epem_ttx": r"$e^+e^-\to t\bar t$", "ee_ttbar_nlo": r"$e^+e^-\to t\bar t$, $V$",
     "ee_uu": r"$e^+e^-\to u\bar u$", "ee_uu_nlo": r"$e^+e^-\to u\bar u$, $V$"}


def figure(rows):
    """trial_overfit_<a..d>: rise against peak lr; trial_lrwindow_<a..d>: each trial's best val mu-MSE over its search's
    best against where its lr sits in the search window; one panel per target, every trial of every D."""
    import plot_style as ps
    ARM = {"scr": ("scratch", ps.C.blue, "o"), "ft": ("fine-tune", ps.C.vermillion, "s")}
    for name_, x_of, y_of, xlab, ylab, xlog in (
            ("trial_overfit", lambda r, rr: r["peak_lr"], lambda r, rr: r["rise"], "peak lr",
             r"val MSE$_{\rm end}$ / val MSE$_{\rm min}$", True),
            ("trial_lrwindow", lambda r, rr: r["x_window"],
             lambda r, rr: r["val_min_ln"] / min(q["val_min_ln"] for q in rr if q["D"] == r["D"]),
             "lr position in search window", r"val MSE$_{\rm min}$ / search best", False)):
        figs = ps.panels(len(T))
        for (fig, ax), (tg, lab) in zip(figs, T.items()):
            for a, (name, col, m) in ARM.items():
                rr = [r for r in rows if r["target"] == tg and r["arm"] == a and r.get("rise")]
                ax.scatter([x_of(r, rr) for r in rr], [y_of(r, rr) for r in rr], c=col, marker=m, s=14, alpha=0.6,
                           label=name)
                bb = [r for r in rr if r["best"]]
                ax.scatter([x_of(r, rr) for r in bb], [y_of(r, rr) for r in bb], facecolors="none", edgecolors="k",
                           marker=m, s=40, label=f"{name}, best")
            if xlog:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ps.process_label(ax, lab)
            ps.legend(ax, "upper left")
        ps.save_panels(figs, FIGS + name_)


def summary(rows):
    for tg in ("epem_ttx", "ee_ttbar_nlo", "ee_uu", "ee_uu_nlo"):
        for a in ("scr", "ft"):
            rr = [r for r in rows if r["target"] == tg and r["arm"] == a]
            ok = [r for r in rr if r.get("rise")]
            if not ok:
                continue
            rise = np.array([r["rise"] for r in ok])
            lr = np.array([r["peak_lr"] for r in ok])
            rho = np.corrcoef(np.argsort(np.argsort(np.log(lr))), np.argsort(np.argsort(np.log(rise))))[0, 1]
            print(f"{tg:13s} {a:3s} trials {len(rr)} with curves {len(ok)}: rise>2 {int((rise > 2).sum())}, "
                  f">10 {int((rise > 10).sum())}; median rise {np.median(rise):.2f}; Spearman(lr, rise) {rho:+.2f}")
            for D in sorted({r['D'] for r in ok}):
                q = sorted((r for r in ok if r["D"] == D), key=lambda r: r["peak_lr"])
                print(f"    D={D:>8.0f}  " + "  ".join(f"{r['peak_lr']:.1e}:{r['rise']:.1f}{'*' if r['best'] else ''}" for r in q))
            # where the best sits in the lr window, and does a lower lr reach a lower val minimum within a search
            xb = [r["x_window"] for r in ok if r["best"]]
            rho_in = []
            for D in sorted({r["D"] for r in ok}):
                q = [r for r in ok if r["D"] == D]
                if len(q) >= 5:
                    rk = [np.argsort(np.argsort(z)) for z in ([r["x_window"] for r in q], [r["val_min_ln"] for r in q])]
                    rho_in.append(np.corrcoef(*rk)[0, 1])
            print(f"    best lr in the window's lowest 10%: {sum(x <= 0.1 for x in xb)}/{len(xb)} (all trials "
                  f"{sum(r['x_window'] <= 0.1 for r in ok)}/{len(ok)}); median best position {np.median(xb):.2f}; "
                  f"Spearman(lr, val min) within searches > 0 in {sum(v > 0 for v in rho_in)}/{len(rho_in)}, "
                  f"median {np.median(rho_in):+.2f}")


if __name__ == "__main__":
    rows = load("--cached" in sys.argv)
    summary(rows)
    figure(rows)
