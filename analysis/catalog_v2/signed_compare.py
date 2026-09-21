"""The signed one-loop pools on a common footing across target choices: the absolute error of
log|x| over events with |x| above the pool's signed-log scale, and the sign accuracy, from
preds_val.npz (evaluation.save_predictions) plus the run's frozen data_stats.json. Runs where
the project imports (login node, .venv).
    python analysis/catalog_v2/signed_compare.py "label=runs/<run glob>" ["label=..."]
"""
import glob, json, os, sys, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from preprocessing import undo_preprocess_amplitude, signedlog_scale  # noqa: E402
HERE = os.path.join(ROOT, "analysis", "catalog_v2"); sys.path.insert(0, HERE)
import census as C  # noqa: E402
s27, all50 = C.signed_classes()
arms = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
def one_run(d):
    z = np.load(sorted(glob.glob(os.path.join(d, "**", "preds_val.npz"), recursive=True))[-1])
    st = json.load(open(sorted(glob.glob(os.path.join(d, "**", "data_stats.json"), recursive=True))[-1]))
    names = list(z["names"]); pid = z["process_id"]; pred = z["pred"].reshape(-1); raw_truth = z["raw_truth"].reshape(-1)
    out = {}
    for p, n in enumerate(names):
        if n not in all50: continue
        tr = st["amp_trafos_pp"][p]; m, sd = st["prepd_mean"][p], st["prepd_std"][p]
        sel = pid == p
        if not sel.any(): continue
        scale = signedlog_scale(tr[0])
        if tr[0].startswith("abslog"):
            logmag_pred = np.log(undo_preprocess_amplitude(pred[sel].reshape(-1, 1), m, sd, trafos=tr).reshape(-1))
            sign_pred = np.where(z["sign"][sel, 0], 1.0, -1.0) if "sign" in z.files else None
        else:
            raw_pred = undo_preprocess_amplitude(pred[sel].reshape(-1, 1), m, sd, trafos=tr).reshape(-1)
            logmag_pred = np.log(np.maximum(np.abs(raw_pred), scale)); sign_pred = np.sign(raw_pred)
        rt = raw_truth[sel]; big = np.abs(rt) > scale
        err = np.abs(logmag_pred[big] - np.log(np.abs(rt[big])))
        acc = float(np.mean(sign_pred[big] == np.sign(rt[big]))) if sign_pred is not None else float("nan")
        out[n] = (float(np.median(err)), float(np.percentile(err, 90)), acc, float(big.mean()))
    return out
for label, pat in arms:
    runs = [one_run(d) for d in sorted(glob.glob(pat))]
    names = sorted(set.intersection(*[set(r) for r in runs]))
    med = np.array([[r[n][0] for r in runs] for n in names]).mean(1); p90 = np.array([[r[n][1] for r in runs] for n in names]).mean(1)
    acc = np.array([[r[n][2] for r in runs] for n in names]).mean(1); frac = np.mean([r[n][3] for r in runs for n in names])
    print(f"{label} ({len(runs)} seeds, {len(names)} signed pools; {frac:.0%} of events above the scale): "
          f"|Δ log|x|| median over pools {np.median(med):.3f} (90th pct of pools {np.percentile(med,90):.3f}); "
          f"per-pool 90th pct of events, median over pools {np.median(p90):.3f}; sign accuracy median {np.nanmedian(acc):.4f}, min {np.nanmin(acc):.4f}")
