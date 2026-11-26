"""llm_compare D566 attempt C: the level error on the 43 new held-out processes, on the full pools and on the same
processes re-sampled by restricting each pool (sqrt s_hat below / above its median, a harder cut on the smallest pair
invariant, its singular fifth), for the per-process level from process and diagrams (constant per process) and the
anchored per-event levels c^ + b^.(x - x0) (data/perevent_anchored.json, perevent_level.py --anchored). Points: the
median over ten 3-seed network ensembles of the new-set median level error; bars: the ensembles' min to max; the
pool's own true mean (one number per pool, no network) beside them.
    python analysis/catalog_daniel/perevent_anchored_fig.py
Output: figs/perevent_anchored.{png,pdf}.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

SUBS = (("all", "full pool"), ("s_lo", r"low $\sqrt{\hat s}$"), ("s_hi", r"high $\sqrt{\hat s}$"),
        ("q_hard", "harder cut"), ("q_soft", "singular fifth"))
ARMS = (("regressor_ens", "one level per process", ps.C.blue, "s"),
        ("C_s", r"per event: $\log\hat s$", ps.C.green, "v"),
        ("C_sqE", "per event: + invariants", ps.C.orange, "^"),
        ("C_sqEP", "per event: + propagators", ps.C.vermillion, "D"))


def main():
    d = json.load(open(os.path.join(HERE, "data", "perevent_anchored.json")))
    E, S = d["ensembles"], d["summary"]
    x = np.arange(len(SUBS))
    fig, ax = ps.figure()
    for k, (c, lab, col, mk) in enumerate(ARMS):
        off = (k - len(ARMS) / 2) * 0.14
        m = np.array([E[c]["new"][s]["median"] for s, _ in SUBS])
        lo = np.array([E[c]["new"][s]["min"] for s, _ in SUBS]); hi = np.array([E[c]["new"][s]["max"] for s, _ in SUBS])
        ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt=mk, color=col, label=lab, capsize=2)
    ax.plot(x + (len(ARMS) / 2) * 0.14, [S["oracle_const"]["new"][s] for s, _ in SUBS], "x", color="k",
            label="the pool's own mean")
    ax.set_xticks(x, [t for _, t in SUBS])
    ax.set_xlabel("events scored")
    ax.set_yscale("log")
    ax.set_ylabel(r"level error $\langle\hat y-y\rangle^2/\sigma_p^2$")
    ps.layout(fig)
    ps.legend_free(ax)
    ps.save(fig, os.path.join(HERE, "figs", "perevent_anchored"))


if __name__ == "__main__":
    main()
