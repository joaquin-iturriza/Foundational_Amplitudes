#!/usr/bin/env python3
"""The t t~ pair input of model.input_pair_threshold (wrappers._pair_log_inputs, threshold=True: log beta^2 / 5) on the
KIT ee_tt events: its range per stratum against the range the training sample (bulk_train) gives it, and its error
against the exact log(1 - 4 m_t^2/s)/5 from the event's own momenta (is the input built right).

    python tools/kit_input_range.py collect [--n 20000] > out.txt       (on the site: prints 'KITINPUT <json>')
    python tools/kit_input_range.py plot <json> --out <stem>            (local: the figure pair)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def collect(n):
    import h5py
    import torch
    from tools import kit_hardps as kh
    from wrappers import _pair_log_inputs
    src, o = kh.h5_path("ee_tt"), {}
    with h5py.File(src, "r") as f:
        for sp, st in [("bulk_train", "bulk")] + [("hard_test", s) for s in f["hard_test"]]:
            p = np.asarray(f[f"{sp}/{st}/momenta"][:n], np.float64)            # e+ e- t t~, the CM frame
            col = _pair_log_inputs(torch.tensor(p.reshape(-1, 4)), torch.arange(0, 4 * len(p) + 1, 4), 4,
                                   threshold=True)[::4, -1].numpy()          # the t t~ pair: the last of the 6
            P = p[:, 2] + p[:, 3]
            s = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1)
            m2 = p[:, 2, 0] ** 2 - (p[:, 2, 1:] ** 2).sum(1)
            exact = np.log(1 - 4 * m2 / s) / 5
            o[f"{sp}/{st}"] = dict(lo=float(col.min()), hi=float(col.max()), q=[float(x) for x in np.quantile(col, [0.005, 0.5, 0.995])],
                                   max_err=float(np.max(np.abs(col - exact))), n=len(p))
    print("KITINPUT " + json.dumps(o))


def plot(path, out):
    import plot_style as ps
    d = json.load(open(path))
    fig, ax = ps.figure()
    b = d.pop("bulk_train/bulk")
    ax.axvspan(b["lo"], b["hi"], color="0.85", lw=0, label="training events (bulk)")
    for i, (k, v) in enumerate(sorted(d.items(), key=lambda kv: kv[1]["q"][1])):
        fam = "threshold" if "threshold" in k else "Sudakov"
        c = ps.C.vermillion if fam == "threshold" else ps.C.blue
        ax.plot([v["lo"], v["hi"]], [i, i], "-", color=c, label=f"{fam} test bins" if fam not in
                [t.get_label().split()[0] for t in ax.lines] else None)
        ax.plot([v["q"][1]], [i], "o", color=c)
    ax.set_xlabel(r"input $\log(\beta^2)/5$ of the $t\bar t$ pair")
    ax.set_ylabel("test bin")
    ax.set_yticks([])
    ps.legend(ax, "upper left")
    ps.process_label(ax, r"$e^+e^-\to t\bar t$")
    print(ps.save(fig, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--n", type=int, default=20000)
    p = sub.add_parser("plot"); p.add_argument("json"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    collect(a.n) if a.cmd == "collect" else plot(a.json, a.out)


if __name__ == "__main__":
    main()
