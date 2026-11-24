"""llm_compare part (a): pooled test RMSE of log|M|^2 of one of our runs against Suprio's LLM pID run and his LLoCa, on
seen, held out, and the held out split into the 96 relabels of a training process and the 43 that relabel none
(his interp / extrap sets, docs/results.tex 'Its held-out split'); per-process MSE of log|M|^2 on the two held-out sets.
Ours: data/<tag>_{seen,heldout}.json, written by tools/llm_score.py (scripts/job_llm_score.sh) on lxplus, each process
its first 1e4 test events, pooled RMSE = sqrt(mean over events of (pred - true)^2) (equal events per process, so it is
sqrt of the mean of the per-process MSEs). His: the table's pooled RMSEs (LLM 0.321 / 0.721, his LLoCa 0.506 / 0.953),
the LLM's per-process MSEs from talks/progress_oct2026b/data/draft_pid_by_process.json (not committed; sqrt of their
mean over the 139 is 0.7214, his table's 0.721), so the LLM's subset RMSEs are read from it; his LLoCa has none.
    python analysis/llm_compare/control_score.py [--tag control ...] [--out NAME]
Several --tag give several of our arms, keyed by tag (output named by --out, default the first tag).
Output: data/<out>_score_summary.json (numbers only). The figure of a held-out comparison is
analysis/llm_compare/heldout_per_process.py, Suprio's held-out layout: this script's own layout was rejected.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LLM = {"seen": 0.321, "heldout": 0.721}
HIS_LLOCA = {"seen": 0.506, "heldout": 0.953}


def rmse(proc, names):
    m = np.array([proc[n]["mse_logm2"] for n in names])
    w = np.array([proc[n]["n"] for n in names])
    return float(np.sqrt((m * w).sum() / w.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["control"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or a.tag[0]
    arms = {t: tuple(json.load(open(os.path.join(HERE, "data", f"{t}_{s}.json")))["processes"]
                       for s in ("seen", "heldout")) for t in a.tag}   # keyed by tag: claim keys hold no spaces
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    llm_pp = {n: v[1] for g in draft.values() for n, v in g.items()}
    groups = {"relabels": sorted(draft["interp"]), "new": sorted(draft["extrap"])}
    for seen, held in arms.values():
        assert len(seen) == 1352 and set(held) == set(llm_pp) and len(held) == 139, (len(seen), len(held))
    ours = {lab: {"seen": rmse(seen, list(seen)), "heldout": rmse(held, list(held)),
                  **{g: rmse(held, ns) for g, ns in groups.items()}} for lab, (seen, held) in arms.items()}
    llm = dict(LLM, **{g: float(np.sqrt(np.mean([llm_pp[n] for n in ns]))) for g, ns in groups.items()})
    assert abs(np.sqrt(np.mean(list(llm_pp.values()))) - LLM["heldout"]) < 1e-3   # the draft is the table's run
    out = {"pooled_rmse_logm2": {**ours, "llm": llm, "his_lloca": HIS_LLOCA},
           "median_mse_logm2": {g: {**{lab: float(np.median([h[n]["mse_logm2"] for n in ns]))
                                       for lab, (_, h) in arms.items()},
                                    "llm": float(np.median([llm_pp[n] for n in ns]))}
                                for g, ns in groups.items()},
           "ours_below_llm": {lab: {g: int(sum(h[n]["mse_logm2"] < llm_pp[n] for n in ns)) for g, ns in groups.items()}
                              for lab, (_, h) in arms.items()},
           "counts": {"seen": 1352, **{g: len(ns) for g, ns in groups.items()}}}
    if "control" in ours:   # pooled held-out MSE (RMSE^2) of each arm over the control's, against the 1.38 seed margin
        out["heldout_mse_ratio_to_control"] = {lab: (r["heldout"] / ours["control"]["heldout"]) ** 2
                                               for lab, r in ours.items() if lab != "control"}
    print(json.dumps(out))   # one line: tools/verify_claims.py reads the last {...} line
    json.dump(out, open(os.path.join(HERE, "data", f"{out_name}_score_summary.json"), "w"), indent=1)
    # no figure here: Joaquin rejected this layout (2026-11-23); the figure is heldout_per_process.py's


if __name__ == "__main__":
    main()
