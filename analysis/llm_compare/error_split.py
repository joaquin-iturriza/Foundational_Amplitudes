"""llm_compare step 15 (D564): what our error on the held-out processes is made of, beside the LLM's MSE.
Per process, from tools/llm_score.py's exact split MSE(log|M|^2) = bias^2 + scale + floor (bias = mean(pred - true): a
misplaced level; floor = var(true)(1 - rho^2): what the best affine map of the prediction leaves, a wrong shape; scale
= the rest, what the best rescale removes): per arm and per family of his figure the median share of each part and
how many processes have each part as their largest; the processes that carry the pooled new-set gap listed with their
parts and the LLM's whole MSE; how many processes have bias^2 alone above the LLM's whole MSE; the pooled share of each part, sum over processes of
the part over sum of the MSE (what the pooled RMSE is made of), with and without the gg -> t b W pair (gg_tbxwm,
gg_btxwp: the two processes that carry the pooled new-set gap).
Ours: data/split/<tag>_heldout.json (job_llm_score.sh ONLY=heldout, tools/llm_score.py @ae5427d12). LLM:
talks/progress_oct2026b/data/draft_pid_by_process.json (his per-process MSE, read off his figure).
    python analysis/llm_compare/error_split.py --tag a_trial5 b_h16 c_control c_best
Output: data/split/error_split.json, and its JSON on the last line.
"""
import argparse, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PARTS = ("bias2", "scale", "floor")
GGTBW = ("dc_gg_tbxwm", "dc_gg_btxwp")


def pooled(procs, ns):
    tot = sum(procs[n]["mse_logm2"] for n in ns)
    return {k: float(sum(parts(procs[n])[k] for n in ns) / tot) for k in PARTS}


def parts(p):
    return {"bias2": p["bias_logm2"] ** 2, "scale": p["scale_logm2"], "floor": p["floor_logm2"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    a = ap.parse_args()
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    meta = {"dc_" + p["folder"]: p for p in
            json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "classes_default.json")))["processes"]}
    out = {}
    for t in a.tag:
        procs = json.load(open(os.path.join(HERE, "data", "split", f"{t}_heldout.json")))["processes"]
        for n, p in procs.items():
            s = sum(parts(p).values())
            assert abs(s - p["mse_logm2"]) <= 1e-6 * p["mse_logm2"] + 1e-12, (t, n, s, p["mse_logm2"])
        out[t] = {}
        for sp in ("interp", "extrap"):
            names = list(draft[sp])
            fams = {}
            for n in names:
                fams.setdefault(meta[n]["test_group"], []).append(n)
            rows = {}
            for g, ns in list(fams.items()) + [("all", names)]:
                sh = {k: [parts(procs[n])[k] / procs[n]["mse_logm2"] for n in ns] for k in PARTS}
                largest = [max(PARTS, key=lambda k: parts(procs[n])[k]) for n in ns]
                rows[g] = {"n": len(ns), "median_share": {k: float(np.median(v)) for k, v in sh.items()},
                           "largest_count": {k: largest.count(k) for k in PARTS},
                           "bias2_above_llm_mse": sum(parts(procs[n])["bias2"] > draft[sp][n][1] for n in ns),
                           "pooled_share": pooled(procs, ns),
                           "pooled_share_no_ggtbw": pooled(procs, [n for n in ns if n not in GGTBW])}
            worst = sorted(names, key=lambda n: -procs[n]["mse_logm2"])[:6]
            out[t][sp] = {"families": rows, "worst": [
                {"process": meta[n]["process"], "mse": procs[n]["mse_logm2"], "llm_mse": draft[sp][n][1],
                 "bias": procs[n]["bias_logm2"], **{k: parts(procs[n])[k] for k in PARTS}, "rho": procs[n]["rho"],
                 "true_std": procs[n]["true_std_logm2"]} for n in worst]}
    p = os.path.join(HERE, "data", "split", "error_split.json")
    json.dump(out, open(p, "w"), indent=1)
    for t, o in out.items():
        for sp, r in o.items():
            al = r["families"]["all"]
            print(t, sp, "median share", {k: round(v, 2) for k, v in al["median_share"].items()}, "largest",
                  al["largest_count"], "bias2>LLM", al["bias2_above_llm_mse"])
            print("   pooled share", {k: round(v, 3) for k, v in al["pooled_share"].items()}, "without gg->tbW",
                  {k: round(v, 3) for k, v in al["pooled_share_no_ggtbw"].items()})
            if sp == "extrap":
                for g, f in r["families"].items():
                    print(f"   {g[:34]:34s}", {k: round(v, 2) for k, v in f["median_share"].items()}, f["largest_count"])
                for w in r["worst"]:
                    print("   ", w["process"], {k: round(v, 3) for k, v in w.items() if k != "process"})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
