"""llm_compare D567: the shape of the predicted log|M|^2 distribution with each arm's own level removed, on the new-family
held-out processes where the arms differ most. Per process, true log|M|^2 against each arm's prediction shifted by its
bias (pred - mean(pred - true)), so only the shape (scale and floor of tools/llm_score.py's split) is compared; ratio
panel pred/true (ps.true_pred_hist). Processes: the 4 of the 43 new whose MSE - bias^2 differs most across the arms
(largest worst/best ratio, data/split/<tag>_heldout.json).
Events: data/events/<tag>_heldout_events.csv (tools/llm_score.py --events-out, first 5000 test events per process).
    python analysis/llm_compare/shape_dists.py
Output: figs/shape_dists.{png,pdf}.
"""
import csv, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402
from heldout_per_process import ARMS, proc_label  # noqa: E402

DAN = os.path.join(ROOT, "analysis", "catalog_daniel")


def main():
    tags = [t for t in ARMS if os.path.exists(os.path.join(HERE, "data", "events", f"{t}_heldout_events.csv"))]
    assert tags, "no data/events/<tag>_heldout_events.csv"
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(DAN, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(DAN, "heldout_twins.json"))))
    ev = {}
    for t in tags:
        d = {}
        for r in csv.DictReader(open(os.path.join(HERE, "data", "events", f"{t}_heldout_events.csv"))):
            d.setdefault(r["process"], []).append((float(r["true_logm2"]), float(r["pred_logm2"])))
        ev[t] = {n: np.array(v) for n, v in d.items()}
    new = sorted(n for n in ev[tags[0]] if meta[n]["split"] != "train" and n not in rel)
    assert len(new) == 43, len(new)

    def shape(t, n):  # MSE - bias^2 of these events
        y, q = ev[t][n].T
        return float(np.mean((q - y) ** 2) - np.mean(q - y) ** 2)
    sh = {n: [shape(t, n) for t in tags] for n in new}
    pick = sorted(new, key=lambda n: -max(sh[n]) / min(sh[n]))[:4]
    fig, axes = ps.figure(ncols=2, nrows=2)
    for ax, n in zip(np.ravel(axes), pick):
        y = ev[tags[0]][n][:, 0]
        for t in tags[1:]:
            assert np.allclose(ev[t][n][:, 0], y, rtol=1e-5), f"{t}: other test events for {n}"
        sq = {t: ev[t][n][:, 1] - np.mean(ev[t][n][:, 1] - y) for t in tags}
        allv = np.concatenate([y] + list(sq.values()))
        e = np.linspace(allv.min(), allv.max() + 1e-9, 41)   # every event on the frame
        tc = np.histogram(y, e)[0]
        models = [{"label": ARMS[t][0], "edges": e, "pred": np.histogram(sq[t], e)[0], "truth": tc,
                   "style": {"color": ARMS[t][1]}} for t in tags]
        r = ps.true_pred_hist(ax, models, xlabel=r"$\log|\mathcal{M}|^2$, level removed")
        ax.text(0.03, 0.95, proc_label(meta[n]["process"]), transform=ax.transAxes, ha="left", va="top")
        print(meta[n]["process"], meta[n]["test_group"], {t: round(s, 4) for t, s in zip(tags, sh[n])})
    ps.legend(np.ravel(axes)[0], "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "shape_dists"))


if __name__ == "__main__":
    main()
