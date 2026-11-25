"""llm_compare step 15 (D564): where on the held-out split Suprio's LLM beats our LLoCa, per family of his figure.
Per family (his test_group; analysis/catalog_daniel/classes_default.json) and per arm: the median over processes of
MSE(log|M|^2), how many processes are below the LLM, the median per-process ratio MSE(ours)/MSE(LLM); per set the
Spearman rank correlation of ours against the LLM across processes (do both find the same processes hard?), the share
of the pooled MSE held by the 3 worst processes, and the pooled RMSE of the new set without the gg -> t b W pair.
LLM points: talks/progress_oct2026b/data/draft_pid_by_process.json (read off his figure, as heldout_per_process.py).
Ours: data/<tag>_heldout.json (tools/llm_score.py).
    python analysis/llm_compare/family_gap.py --tag a_trial5 b_h16 [--out NAME]
Output: data/<out>_family_gap.json (default out: family_gap).
"""
import argparse, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GG_TBW = ("g g > t b~ w-", "g g > b t~ w+")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--out", default="family_gap")
    a = ap.parse_args()
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    meta = {"dc_" + p["folder"]: p for p in
            json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "classes_default.json")))["processes"]}
    ours = {t: json.load(open(os.path.join(HERE, "data", f"{t}_heldout.json")))["processes"] for t in a.tag}
    out = {}
    for sp in ("interp", "extrap"):
        names = sorted(draft[sp], key=lambda n: draft[sp][n][0])
        arms = {"llm": np.array([draft[sp][n][1] for n in names])}
        arms.update({t: np.array([ours[t][n]["mse_logm2"] for n in names]) for t in a.tag})
        fams = {}
        for i, n in enumerate(names):
            fams.setdefault(meta[n]["test_group"], []).append(i)
        rows = {}
        for g, idx in list(fams.items()) + [("all", list(range(len(names))))]:
            r = {"n": len(idx), "llm_median": float(np.median(arms["llm"][idx]))}
            for t in a.tag:
                m, l = arms[t][idx], arms["llm"][idx]
                r[t] = {"median": float(np.median(m)), "below_llm": int((m < l).sum()),
                        "median_ratio_to_llm": float(np.median(m / l))}
            rows[g] = r
        keep = [i for i, n in enumerate(names) if meta[n]["process"] not in GG_TBW]
        out[sp] = {"families": rows,
                   "spearman_vs_llm": {t: spearman(arms[t], arms["llm"]) for t in a.tag},
                   "top3_share": {t: float(np.sort(m)[-3:].sum() / m.sum()) for t, m in arms.items()},
                   "top3": {t: [meta[names[i]]["process"] for i in np.argsort(-m)[:3]] for t, m in arms.items()},
                   "pooled_rmse_without_gg_tbW": {t: float(np.sqrt(m[keep].mean())) for t, m in arms.items()}}
    assert sum(r["n"] for g, r in out["extrap"]["families"].items() if g != "all") == 43
    p = os.path.join(HERE, "data", f"{a.out}_family_gap.json")
    json.dump(out, open(p, "w"), indent=1)
    for sp, o in out.items():
        print(sp, "spearman", {t: round(v, 2) for t, v in o["spearman_vs_llm"].items()},
              "top3 share", {t: round(v, 2) for t, v in o["top3_share"].items()},
              "pooled w/o gg->tbW", {t: round(v, 3) for t, v in o["pooled_rmse_without_gg_tbW"].items()})
        for g, r in o["families"].items():
            print(f"  {g[:40]:40s} n={r['n']:3d} LLM {r['llm_median']:.3g} |" +
                  "".join(f" {t} {r[t]['median']:.3g} ({r[t]['below_llm']}<, r {r[t]['median_ratio_to_llm']:.2g})"
                          for t in a.tag))
    print("wrote", p)


if __name__ == "__main__":
    main()
