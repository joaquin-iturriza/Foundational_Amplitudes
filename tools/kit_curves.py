#!/usr/bin/env python3
"""Does a KIT run's validation fall together with its training (kit_benchmark step 15, D421: "Good HPs, in particular
lr, behave like the first ones")? analysis/data_saving/curve_quality.py's measure, read on the KIT runs.

The KIT runs are HETEROSC: the curves they kept are the validation NLL (the log's "Val loss: x | step n" lines, EMA-free)
and the per-batch training NLL (train_steps.npz `loss`, live weights), both of NLL = r^2/(2 sigma^2) + ln sigma. Their
val mu-MSE curve was never written (no training.result_path in job_kit.sh). curve_quality's ratios of MSEs become
differences of NLLs: with sigma tracking the residuals (sigma^2 = MSE) NLL = 1/2 + 1/2 ln MSE, so a factor F in MSE is
1/2 ln F nats. Per run, on running-median-3 curves (the training NLL first reduced to its median over each validation
interval):
  separation  (v - t at the end) - min (v - t), good <= 1/2 ln 3 = 0.55 nats (curve_quality's factor 3);
  end_up      v_end - min v,                   good <= 1/2 ln 2 = 0.35 nats (factor 2).
An estimate: it assumes sigma tracks the residuals; a sigma collapse (sigma far below the residuals) raises the NLL
without the mu-MSE, so a failing NLL curve can still hide a sound mu-MSE curve (the best checkpoint is chosen on it).

    python tools/kit_curves.py collect --out analysis/kit_benchmark/data/kit_curves.json   (reads CC)
    python tools/kit_curves.py read analysis/kit_benchmark/data/kit_curves.json [--figs analysis/kit_benchmark/figs]
    python tools/kit_curves.py sweep kit_hpo_uug_plain kit_hpo_ttvirt_thr --out analysis/kit_benchmark/data/kit_hpo.json
        (step 15's searches, driven by sweep/drive.py: the trials in sweeps_local/<name>/results, their curves read on CC: per trial its HPs, best val mu-MSE and curve_quality on the mu-MSE curves
         val_mse_curve / train_eval_mse_curve; chosen = the lowest best val among the good trials)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BAR = {"separation": 0.5 * np.log(3.0), "end_up": 0.5 * np.log(2.0)}

SITE = r'''
import glob, json, os, re
import numpy as np, yaml
out = {}
for d in sorted(glob.glob("runs/kit/kit_A*")):
    log = os.path.join(d, "out_0.log")
    if not (os.path.exists(log) and os.path.exists(d + "/train_steps.npz") and os.path.exists(d + "/config.yaml")):
        continue
    vs, vv = [], []
    for l in open(log, errors="replace"):
        if "Finished training" in l or "Creating new experiment" in l and vs:
            break
        m = re.search(r"Val loss: (\S+) \| step (\d+)", l)
        if m:
            vs.append(int(m.group(2))); vv.append(float(m.group(1)))
    z = np.load(d + "/train_steps.npz")
    st, lo = z["step"], z["loss"]
    tr, prev = [], 0
    for s in vs:                      # median training NLL over the validation interval (prev, s]
        sel = lo[(st > prev) & (st <= s) & np.isfinite(lo)]
        tr.append(float(np.median(sel)) if len(sel) else None); prev = s
    c = yaml.safe_load(open(d + "/config.yaml"))
    t, f = c.get("training", {}), c.get("fine_tune", {}) or {}
    out[os.path.basename(d)] = {"lr": t.get("lr"), "lr_scale": f.get("lr_scale"), "warmup_frac": t.get("cosanneal_warmup_frac"),
                                "iterations": t.get("iterations"), "select": t.get("heterosc_select"),
                                "steps": vs, "val": vv, "train": tr}
print("KITCURVES " + json.dumps(out))
'''

# the runs step 15 names, by their run directory (registry id in the comment)
NAMED = {"c15b3337": "kit_A_ee_uug_me2_n100000_t8000_softpair",
         "c364a7ae": "kit_A_ee_uugg_me2_n100000_t8000_softpair",
         "f21fa163": "kit_A_ee_tt_virt_finite_n100000_t8000_softpair",
         "3fc1016a": "kit_A_ee_tt_virt_finite_n100000_t8000_softpairthr",
         "203a274f": "kit_A_ee_uug_me2_n100000_t8000_softpair_plain",
         "7b891d10": "kit_A_ee_uug_me2_n100000_t8000_softpair_plain_logskip",
         "eb870ad4": "kit_A_ee_tt_virt_finite_n100000_t8000_softpairthr_logskip",
         "95be8db2": "kit_A_ee_uug_me2_n100000_t8000_softpair_gmax4",
         "88b9334a": "kit_A_ee_uugg_me2_n100000_t8000_softpair_gmax4",
         "c4fc1d6f": "kit_A_ee_uug_me2_n100000_t8000_softpair_plain_logskiplin",
         "6bc05a40": "kit_A_ee_tt_virt_finite_n100000_t8000_softpairthr_logskiplin",
         # step 18: asymptotic-factor arms (A1 on the control's inputs, A2 no log inputs + gamma <= 4) and the uugg control
         "d5fb3d78": "kit_A_ee_uug_me2_n100000_t8000_softpair_asym",
         "3bbda50a": "kit_A_ee_uug_me2_n100000_t8000_asym_cont",
         "fec28926": "kit_A_ee_uugg_me2_n100000_t8000_softpair_plain",
         "1ad20d13": "kit_A_ee_uugg_me2_n100000_t8000_softpair_plain_asym",
         "602ce14f": "kit_A_ee_uugg_me2_n100000_t8000_asym_cont",
         "09576b49": "kit_A_ee_tt_virt_finite_n100000_t8000_softpairthr_asym_plainlog",
         "5ce19348": "kit_A_ee_tt_virt_finite_n100000_t8000_asym_cont_plainlog"}


def smooth(x):
    x = np.asarray(x, float)
    return np.array([np.nanmedian(x[max(i - 1, 0):i + 2]) for i in range(len(x))])


def quality(v, t):
    """separation and end_up in nats on running-median-3 NLL curves; None entries (empty interval) dropped."""
    ok = [i for i in range(len(v)) if t[i] is not None and np.isfinite(v[i])]
    lv, lt = smooth([v[i] for i in ok]), smooth([t[i] for i in ok])
    g = lv - lt
    q = {"separation": float(g[-1] - g.min()), "end_up": float(lv[-1] - lv.min()),
         "v_min": float(lv.min()), "v_min_step": None, "v_end": float(lv[-1])}
    q["well"] = bool(q["separation"] <= BAR["separation"] and q["end_up"] <= BAR["end_up"])
    return q


def read(path, figs=None):
    d = json.load(open(path))
    rows = []
    for run, e in sorted(d.items()):
        if len(e["val"]) < 4:
            continue
        q = quality(e["val"], e["train"])
        q["v_min_step"] = e["steps"][int(np.argmin(smooth(e["val"])))]
        rid = next((k for k, v in NAMED.items() if v == run), "")
        rows.append(dict(run=run, id=rid, lr=e["lr"], lr_scale=e["lr_scale"], **q))
    print(f"good = separation <= {BAR['separation']:.2f} and end_up <= {BAR['end_up']:.2f} nats (NLL)")
    for r in rows:
        print(f"{r['id']:8s} {r['run']:62s} lr {r['lr']} x{r['lr_scale']}  sep {r['separation']:7.2f}  "
              f"end_up {r['end_up']:8.2f}  vmin {r['v_min']:6.2f}@{r['v_min_step']}  " + ("good" if r["well"] else "BAD"))
    lrs = sorted({(r["lr"], r["lr_scale"]) for r in rows}, key=str)
    for lr in lrs:
        g = [r for r in rows if (r["lr"], r["lr_scale"]) == lr]
        print(f"lr {lr[0]} x{lr[1]}: {sum(r['well'] for r in g)} of {len(g)} good")
    json.dump({"bars_nats": BAR, "rows": rows}, open(path.replace(".json", "_read.json"), "w"), indent=1)
    if figs:
        plot(d, figs)
    return rows


def curve_panel(ax, e):
    """one run's validation and training NLL against step (running median of 3), y range 6 nats above the lowest"""
    import plot_style as ps
    st = np.array(e["steps"])
    tr = smooth(np.array([np.nan if x is None else x for x in e["train"]], float))
    va = smooth(e["val"])
    ax.plot(st, va, color=ps.C.vermillion, label="validation")
    ax.plot(st, tr, color=ps.C.blue, linestyle="--", label="training")
    lo = np.nanmin(np.r_[va, tr])
    ax.set_ylim(lo - 0.5, lo + 6)
    ax.set_xlabel("training step"); ax.set_ylabel("NLL")


def plot(d, figs):
    """kit_curves_nll: the four KIT controls' validation and training NLL against step (running median of 3); the y
    range stops 6 nats above the lowest point so the late rises (up to +2800 nats on u u~ g g) leave the box."""
    import plot_style as ps
    four = ("c15b3337", "203a274f", "c364a7ae", "3fc1016a")
    lab = {"c15b3337": r"$e^+e^-\to u\bar u g$, $T$", "203a274f": r"$e^+e^-\to u\bar u g$, $|\mathcal{M}|^2$",
           "c364a7ae": r"$e^+e^-\to u\bar u gg$, $T$", "3fc1016a": r"$e^+e^-\to t\bar t$, $V$"}
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, rid in zip(axs.ravel(), four):
        e = d[NAMED[rid]]
        curve_panel(ax, e)
        ps.process_label(ax, lab[rid])
    ps.legend(axs[1, 1], "lower left")
    ps.save(fig, os.path.join(figs, "kit_curves_nll"))


def plot_runs(d, runs, base):
    """one panel file per run (<base>_a, _b, ... in the order given): its validation and training NLL against step
    (running median of 3), the y range 6 nats above the lowest point as kit_curves_nll"""
    import plot_style as ps
    figs = ps.panels(len(runs))
    for (fig, ax), run in zip(figs, runs):
        e = d[run]
        curve_panel(ax, e)
        ps.legend(ax, "upper right")
    for w in ps.save_panels(figs, base):
        print(w)


SWEEP_SITE = r'''
import base64, json, os, sys
out = {}
# the trials drive.py observed: [name, file, result_path, run_dir] each, from sweeps_local/<name>/results (local)
for name, fname, f, rd in json.loads(base64.b64decode(sys.argv[1])):
    rows = out.setdefault(name, [])
    try:
        j = json.load(open(f))
    except Exception:
        continue
    hp = {}
    if os.path.exists(rd + "/config.yaml"):
        import yaml
        c = yaml.safe_load(open(rd + "/config.yaml")); t = c.get("training", {})
        hp = {f"training.{k}": t.get(k) for k in ("lr", "batchsize", "muon_weight_decay", "regularization_lambda",
                                                  "cosanneal_warmup_frac", "cosanneal_eta_min", "ema_decay", "scheduler")}
        hp["ema"] = c.get("ema")
    rows.append({"file": fname, "hp": hp,
                 "val_loss": j.get("val_loss"), "best_step": j.get("best_step"), "run_dir": rd,
                 "val": j.get("val_mse_curve"), "train": j.get("train_eval_mse_curve")})
print("KITHPO " + json.dumps(out))
'''


def sweep_read(names, out):
    """Per search: each trial's curve_quality (separation, end_up on the mu-MSE curves) and best val; the chosen trial."""
    sys.path.insert(0, os.path.join(ROOT, "analysis", "data_saving"))
    from curve_quality import quality as q_mse
    from analysis.common.readers import site_python
    import base64, glob
    # drive.py keeps each observed trial's RESULT_JSON locally; its result_path (the curves) and run_dir are on the
    # site that ran it (driver_state.json's trial "site")
    todo = {}
    for n in names:
        st = json.load(open(os.path.join(ROOT, "sweeps_local", n, "driver_state.json")))["trials"].values()
        where = {(t["hp_idx"], t["t_steps"]): t["site"] for t in st if t["status"] == "observed"}
        for f in sorted(glob.glob(os.path.join(ROOT, "sweeps_local", n, "results", "*.json"))):
            j = json.load(open(f))
            todo.setdefault(where[(j["hp_idx"], j["t_steps"])], []).append(
                [n, os.path.basename(f), j["result_path"], j["run_dir"]])
    d = {n: [] for n in names}
    for site, rows in todo.items():
        for n, r in site_python(SWEEP_SITE, [base64.b64encode(json.dumps(rows).encode()).decode()], "KITHPO",
                                site=site).items():
            d[n] += [dict(x, site=site) for x in r]
    for name, rows in d.items():
        for r in rows:
            if r["val"] and r["train"]:
                r.update(q_mse(r["val"], r["train"]))
        good = [r for r in rows if r.get("well") and r["val_loss"] is not None]
        ch = min(good, key=lambda r: r["val_loss"]) if good else None
        print(f"{name}: {len(rows)} trials, {len(good)} good; chosen {ch and ch['file']}")
        for r in sorted(rows, key=lambda r: r["val_loss"] if r["val_loss"] is not None else 1e99):
            lr, bs = (r["hp"] or {}).get("training.lr"), (r["hp"] or {}).get("training.batchsize")
            print(f"  {r['file']:28s} lr {lr} batch {bs}  best val {r['val_loss']}  sep {r.get('separation', float('nan')):.3g}  "
                  f"end_up {r.get('end_up', float('nan')):.3g}  " + ("good" if r.get("well") else "BAD")
                  + ("  <- chosen" if r is ch else ""))
        d[name] = {"trials": rows, "chosen": ch and ch["file"]}
    json.dump(d, open(out, "w"), indent=1)


def main():  # dup-ok: argparse boilerplate (collect/read), no shared logic
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    r = sub.add_parser("read"); r.add_argument("data"); r.add_argument("--figs")
    w = sub.add_parser("sweep"); w.add_argument("names", nargs="+"); w.add_argument("--out", required=True)
    p = sub.add_parser("curves"); p.add_argument("data"); p.add_argument("runs", nargs="+"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        from analysis.common.readers import site_python
        json.dump(site_python(SITE, [], "KITCURVES"), open(a.out, "w"))
        print(f"-> {a.out}")
    elif a.cmd == "curves":
        plot_runs(json.load(open(a.data)), a.runs, a.out)
    elif a.cmd == "sweep":
        sweep_read(a.names, a.out)
    else:
        read(a.data, a.figs)


if __name__ == "__main__":
    main()
