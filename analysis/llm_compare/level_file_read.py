"""llm_compare step 16 (D564): does a network trained with each process's target mean from the frozen diagram level
file keep that level on the 43 new-family held-out processes? Per set of suprio.split() (relabels 96, new 43), for
part (a) trial 5 and the level-file arm: median bias^2, median MSE - bias^2, pooled bias^2 share sum(bias^2)/sum(MSE);
the gg -> t b W pair's MSE and bias; on the new set, |our bias - the level network's own bias| (how far the trained
network moved off the level it was given; both signed mean(pred - true) of log|M|^2) and how many processes have MSE
below the LLM's median over his seeds 0-2.
    python analysis/llm_compare/level_file_read.py [--tags a_trial5 level_file]
Output: data/step16_level_file_read.json, and its JSON on the last line.
"""
import argparse, json, os

import numpy as np

import suprio

HERE = os.path.dirname(os.path.abspath(__file__))
GGTBW = ("dc_gg_tbxwm", "dc_gg_btxwp")


def load(tag):
    return json.load(open(os.path.join(HERE, "data", "split", f"{tag}_heldout.json")))["processes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a_trial5", "level_file"])
    a = ap.parse_args()
    sp, lv = suprio.split(), load("lvl_diag")
    llm = {n: float(np.median(v)) for n, v in suprio.llm("mse").items()}
    out = {"lvl_diag_new_bias2_median": float(np.median([lv[n]["bias_logm2"] ** 2 for n in sp["extrap"]])),
           "llm_new_mse_median": float(np.median([llm[n] for n in sp["extrap"]]))}
    for t in a.tags:
        p, r = load(t), {}
        for s in ("interp", "extrap"):
            b = np.array([p[n]["bias_logm2"] ** 2 for n in sp[s]])
            m = np.array([p[n]["mse_logm2"] for n in sp[s]])
            r[s] = {"mse_median": float(np.median(m)), "bias2_median": float(np.median(b)),
                    "shape_median": float(np.median(m - b)), "pooled_bias2_share": float(b.sum() / m.sum())}
        d = np.abs([p[n]["bias_logm2"] - lv[n]["bias_logm2"] for n in sp["extrap"]])
        r["extrap"].update(off_given_level_median=float(np.median(d)), off_given_level_p90=float(np.quantile(d, 0.9)),
                           below_llm=int(sum(p[n]["mse_logm2"] < llm[n] for n in sp["extrap"])))
        r["ggtbw"] = {n: {"mse": p[n]["mse_logm2"], "bias": p[n]["bias_logm2"], "given_level_bias": lv[n]["bias_logm2"]}
                      for n in GGTBW}
        out[t] = r
    json.dump(out, open(os.path.join(HERE, "data", "step16_level_file_read.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
