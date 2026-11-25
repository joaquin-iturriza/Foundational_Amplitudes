"""llm_compare D567: the shape of the predicted log|M|^2 distribution with each arm's own level removed, on the new-family
held-out processes, one representative per family. Per process, true log|M|^2 against each arm's prediction shifted by its
bias (pred - mean(pred - true)), so only the shape (scale and floor of tools/llm_score.py's split) is compared; ratio
panel pred/true (ps.true_pred_hist). Processes: every new family in Suprio's layout order (heldout_per_process.GROUPS),
each by its process at the median (lower median) of the across-arm spread of MSE - bias^2 of these events (worst/best
ratio): a representative example, never the extreme (CLAUDE.md; repo review 2026-11-25). The legend sits in the gg->Wqq'
panel, whose narrow peak leaves its upper corners free.
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
from heldout_per_process import ARMS, GROUPS, proc_label  # noqa: E402

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
    spread = {n: max(sh[n]) / min(sh[n]) for n in new}
    pick = []
    for g, _ in GROUPS["extrap"]:  # one process per family: the median of its spread
        ns = sorted((n for n in new if meta[n]["test_group"] == g), key=lambda n: spread[n])
        pick.append(ns[(len(ns) - 1) // 2])
    fig, axes = ps.figure(ncols=2, nrows=3)
    for ax, n in zip(np.ravel(axes)[:len(pick)], pick):
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
        ps.process_label(ax, proc_label(meta[n]["process"]), "upper left")   # registered: the legend avoids it
        print(meta[n]["process"], meta[n]["test_group"], {t: round(s, 4) for t, s in zip(tags, sh[n])})
    ps.legend(axes[2, 0], "upper right", ncol=2)   # a 7th cell leaves an empty one or overflows the page; ncol=2 brings the font down
    ps.save(fig, os.path.join(HERE, "figs", "shape_dists"))


if __name__ == "__main__":
    main()
