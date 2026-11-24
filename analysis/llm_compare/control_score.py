"""llm_compare part (a): pooled test RMSE of log|M|^2 of one of our runs against Suprio's LLM pID run and his LLoCa, on
seen, held out, and the held out split into the 96 relabels of a training process and the 43 that relabel none
(his interp / extrap sets, docs/results.tex 'Its held-out split'); per-process MSE of log|M|^2 on the two held-out sets.
Ours: data/<tag>_{seen,heldout}.json, written by tools/llm_score.py (scripts/job_llm_score.sh) on lxplus, each process
its first 1e4 test events, pooled RMSE = sqrt(mean over events of (pred - true)^2) (equal events per process, so it is
sqrt of the mean of the per-process MSEs). His: the table's pooled RMSEs (LLM 0.321 / 0.721, his LLoCa 0.506 / 0.953),
the LLM's per-process MSEs from talks/progress_oct2026b/data/draft_pid_by_process.json (not committed; sqrt of their
mean over the 139 is 0.7214, his table's 0.721), so the LLM's subset RMSEs are read from it; his LLoCa has none.
    python analysis/llm_compare/control_score.py [--tag control ...] [--label "matched control" ...] [--out NAME]
Several --tag/--label pairs draw several of our arms side by side (output named by --out, default the first tag).
Output: figs/<out>_score.{png,pdf}, data/<out>_score_summary.json.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LLM = {"seen": 0.321, "held out": 0.721}
HIS_LLOCA = {"seen": 0.506, "held out": 0.953}
SETS = ["seen", "held out", "relabels", "new"]


def rmse(proc, names):
    m = np.array([proc[n]["mse_logm2"] for n in names])
    w = np.array([proc[n]["n"] for n in names])
    return float(np.sqrt((m * w).sum() / w.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["control"])
    ap.add_argument("--label", nargs="+", default=["our LLoCa, his optimizer settings"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert len(a.tag) == len(a.label), (a.tag, a.label)
    out_name = a.out or a.tag[0]
    arms = {lab: tuple(json.load(open(os.path.join(HERE, "data", f"{t}_{s}.json")))["processes"]
                       for s in ("seen", "heldout")) for t, lab in zip(a.tag, a.label)}
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    llm_pp = {n: v[1] for g in draft.values() for n, v in g.items()}
    groups = {"relabels": sorted(draft["interp"]), "new": sorted(draft["extrap"])}
    for seen, held in arms.values():
        assert len(seen) == 1352 and set(held) == set(llm_pp) and len(held) == 139, (len(seen), len(held))
    ours = {lab: {"seen": rmse(seen, list(seen)), "held out": rmse(held, list(held)),
                  **{g: rmse(held, ns) for g, ns in groups.items()}} for lab, (seen, held) in arms.items()}
    llm = dict(LLM, **{g: float(np.sqrt(np.mean([llm_pp[n] for n in ns]))) for g, ns in groups.items()})
    assert abs(np.sqrt(np.mean(list(llm_pp.values()))) - LLM["held out"]) < 1e-3   # the draft is the table's run
    out = {"pooled_rmse_logm2": {**ours, "LLM (Suprio)": llm, "LLoCa (Suprio)": HIS_LLOCA},
           "median_mse_logm2": {g: {**{lab: float(np.median([h[n]["mse_logm2"] for n in ns]))
                                       for lab, (_, h) in arms.items()},
                                    "LLM (Suprio)": float(np.median([llm_pp[n] for n in ns]))}
                                for g, ns in groups.items()},
           "ours_below_llm": {lab: {g: int(sum(h[n]["mse_logm2"] < llm_pp[n] for n in ns)) for g, ns in groups.items()}
                              for lab, (_, h) in arms.items()},
           "counts": {"seen": 1352, **{g: len(ns) for g, ns in groups.items()}}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "data", f"{out_name}_score_summary.json"), "w"), indent=1)
    ours_c = list(zip(arms, (ps.C.blue, ps.C.green, ps.C.orange), ("o", "D", "^")))

    fig, axs = ps.figure(ncols=2, nrows=2)
    ax = axs[0][0]
    x = np.arange(len(SETS))
    rows = [(lab, ours[lab], mk, c) for lab, c, mk in ours_c] + [("LLM (Suprio)", llm, "*", "k"),
                                                                 ("LLoCa (Suprio)", HIS_LLOCA, "s", ps.C.vermillion)]
    for j, (lab, d, mk, c) in enumerate(rows):
        dx = 0.3 * (j / (len(rows) - 1) - 0.5)
        xs = [i + dx for i, s in enumerate(SETS) if s in d]
        ax.plot(xs, [d[s] for s in SETS if s in d], mk, color=c, ls="none", label=lab)
    ax.set_xticks(x, SETS)
    ax.set_yscale("log")
    ax.set_ylabel(r"pooled RMSE$(\log|\mathcal{M}|^2)$")
    for ax, (g, ns) in zip((axs[0][1], axs[1][0]), groups.items()):
        ns = sorted(ns, key=lambda n: llm_pp[n])   # ordered by the LLM's error, as the comparison reads
        k = np.arange(1, len(ns) + 1)
        for lab, c, mk in ours_c:
            ax.plot(k, [arms[lab][1][n]["mse_logm2"] for n in ns], mk, color=c, ls="none", label=lab, alpha=0.85)
        ax.plot(k, [llm_pp[n] for n in ns], "*", color="k", ls="none", label="LLM (Suprio)", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("held-out process, by the LLM's MSE")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, f"held out, {g}", loc="lower right")
    ps.legend_cell(axs[1][1], [axs[0][0]])
    ps.save(fig, os.path.join(HERE, "figs", f"{out_name}_score"))


if __name__ == "__main__":
    main()
