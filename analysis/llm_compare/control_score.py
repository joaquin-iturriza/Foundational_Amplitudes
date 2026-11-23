"""llm_compare part (a): pooled test RMSE of log|M|^2 of one of our runs against Suprio's LLM pID run and his LLoCa, on
seen, held out, and the held out split into the 96 relabels of a training process and the 43 that relabel none
(his interp / extrap sets, docs/results.tex 'Its held-out split'); per-process MSE of log|M|^2 on the two held-out sets.
Ours: data/<tag>_{seen,heldout}.json, written by tools/llm_score.py (scripts/job_llm_score.sh) on lxplus, each process
its first 1e4 test events, pooled RMSE = sqrt(mean over events of (pred - true)^2) (equal events per process, so it is
sqrt of the mean of the per-process MSEs). His: the table's pooled RMSEs (LLM 0.321 / 0.721, his LLoCa 0.506 / 0.953),
the LLM's per-process MSEs from talks/progress_oct2026b/data/draft_pid_by_process.json (not committed; sqrt of their
mean over the 139 is 0.7214, his table's 0.721), so the LLM's subset RMSEs are read from it; his LLoCa has none.
    python analysis/llm_compare/control_score.py [--tag control] [--label "matched control"]
Output: figs/<tag>_score.{png,pdf}, data/<tag>_score_summary.json.
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
    ap.add_argument("--tag", default="control")
    ap.add_argument("--label", default="our LLoCa, his optimizer settings")
    a = ap.parse_args()
    seen = json.load(open(os.path.join(HERE, "data", f"{a.tag}_seen.json")))["processes"]
    held = json.load(open(os.path.join(HERE, "data", f"{a.tag}_heldout.json")))["processes"]
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    llm_pp = {n: v[1] for g in draft.values() for n, v in g.items()}
    groups = {"relabels": sorted(draft["interp"]), "new": sorted(draft["extrap"])}
    assert len(seen) == 1352 and set(held) == set(llm_pp) and len(held) == 139, (len(seen), len(held))
    ours = {"seen": rmse(seen, list(seen)), "held out": rmse(held, list(held)),
            **{g: rmse(held, ns) for g, ns in groups.items()}}
    llm = dict(LLM, **{g: float(np.sqrt(np.mean([llm_pp[n] for n in ns]))) for g, ns in groups.items()})
    assert abs(np.sqrt(np.mean(list(llm_pp.values()))) - LLM["held out"]) < 1e-3   # the draft is the table's run
    out = {"pooled_rmse_logm2": {a.label: ours, "LLM (Suprio)": llm, "LLoCa (Suprio)": HIS_LLOCA},
           "median_mse_logm2": {g: {a.label: float(np.median([held[n]["mse_logm2"] for n in ns])),
                                    "LLM (Suprio)": float(np.median([llm_pp[n] for n in ns]))}
                                for g, ns in groups.items()},
           "ours_below_llm": {g: int(sum(held[n]["mse_logm2"] < llm_pp[n] for n in ns)) for g, ns in groups.items()},
           "counts": {"seen": len(seen), **{g: len(ns) for g, ns in groups.items()}}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "data", f"{a.tag}_score_summary.json"), "w"), indent=1)

    fig, axs = ps.figure(ncols=2, nrows=2)
    ax = axs[0][0]
    x = np.arange(len(SETS))
    for (lab, d, mk, c, dx) in ((a.label, ours, "o", ps.C.blue, -0.15), ("LLM (Suprio)", llm, "*", "k", 0.0),
                                ("LLoCa (Suprio)", HIS_LLOCA, "s", ps.C.vermillion, 0.15)):
        xs = [i + dx for i, s in enumerate(SETS) if s in d]
        ax.plot(xs, [d[s] for s in SETS if s in d], mk, color=c, ls="none", label=lab)
    ax.set_xticks(x, SETS)
    ax.set_yscale("log")
    ax.set_ylabel(r"pooled RMSE$(\log|\mathcal{M}|^2)$")
    for ax, (g, ns) in zip((axs[0][1], axs[1][0]), groups.items()):
        ns = sorted(ns, key=lambda n: llm_pp[n])   # ordered by the LLM's error, as the comparison reads
        k = np.arange(1, len(ns) + 1)
        ax.plot(k, [held[n]["mse_logm2"] for n in ns], "o", color=ps.C.blue, ls="none", label=a.label, alpha=0.85)
        ax.plot(k, [llm_pp[n] for n in ns], "*", color="k", ls="none", label="LLM (Suprio)", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("held-out process, by the LLM's MSE")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, f"held out, {g}", loc="lower right")
    ps.legend_cell(axs[1][1], [axs[0][0]])
    ps.save(fig, os.path.join(HERE, "figs", f"{a.tag}_score"))


if __name__ == "__main__":
    main()
