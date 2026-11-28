"""llm_compare step 27 (D566): does a trained network keep its level on the 43 new-family held-out processes when the
same process is sampled another way? From a run's per-event held-out CSV (tools/llm_score.py --events-out, with the
log_shat and q columns), per process and per subset of analysis/catalog_daniel/perevent_level.py's subsets() (all,
s_lo/s_hi: s_hat below/above the pool's median; q_hard: q = Lmin - log s_hat above its median; q_soft: its lowest
fifth), the level error mean(pred - true)^2 / sd^2 (sd the std of the true log|M|^2 over the process's events in the
CSV) and bias^2 = mean(pred - true)^2 in nat^2; medians over the 43, with the event-bootstrap sd of each median (30
resamples, subsets redrawn), as attempt E's read. --check-factor FILE: the CSV's log_factor must equal
-b.(x - x0)/sx of data.level_events' FILE at the CSV's own x (the columns are the factor's events). --relevel (step 30,
a level file built as network + the pool's own mean log F): each subset is scored with the level recomputed on that
subset, pred' = pred - mean log F(process) + mean log F(subset), as the level would be under that sampling; also the
new-43 median |mean log F(subset) - mean log F(process)|.
    python analysis/llm_compare/level_subsets.py --runs level_file_s30=<csv> level_events_s30=<csv> \
        [--check-factor analysis/catalog_daniel/data/level_events_E_kin_s30.json]
Output: data/level_subsets.json, and its JSON on the last line.
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

import suprio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "catalog_daniel"))
from perevent_level import SUBSETS, subsets  # noqa: E402


def errs(g, relevel=False):
    d, y = (g.pred_logm2 - g.true_logm2).to_numpy(), g.true_logm2.to_numpy()
    lf = g.log_factor.to_numpy()
    S = subsets({"L": g.log_shat.to_numpy(), "q": g.q.to_numpy()})
    sd = y.std()
    sh = {s: float(lf[m].mean() - lf.mean()) if relevel else 0.0 for s, m in S.items()}
    return {s: (float((d[m].mean() + sh[s]) ** 2 / sd ** 2), float((d[m].mean() + sh[s]) ** 2), abs(sh[s]))
            for s, m in S.items()}


def read(csv, new, check=None, nboot=30, relevel=False):
    ev = pd.read_csv(csv)
    assert {"log_shat", "q"} <= set(ev.columns), f"{csv}: no x columns (tools/llm_score.py --events-out before step 27)"
    G = {n: g for n, g in ev.groupby("process") if n in new}
    assert len(G) == len(new), sorted(set(new) - set(G))
    out = {}
    if check is not None:
        f = json.load(open(check))
        dev = []
        for n, g in ev.groupby("process"):
            co = f["coef"][n]
            x = g[["log_shat", "q", "E"]].to_numpy()
            dev.append(np.abs(g.log_factor.to_numpy() + ((x - co["x0"]) / f["sx"]) @ co["b"]))
        dev = np.concatenate(dev)
        out["factor_check_max_abs"], out["factor_check_p99_abs"] = float(dev.max()), float(np.quantile(dev, 0.99))
        # the p99, not the max: float32 boosted momenta can move a near-collinear Lmin by ~1e-2 (repo review); a
        # misaligned or misscaled column moves every event by O(1)
        assert out["factor_check_p99_abs"] < 1e-2, f"log_factor differs from -b.(x - x0)/sx: {out}"
    E = {n: errs(g, relevel) for n, g in G.items()}
    out["new"] = {s: {"level_err": float(np.median([E[n][s][0] for n in new])),
                      "bias2": float(np.median([E[n][s][1] for n in new])),
                      "dlogF": float(np.median([E[n][s][2] for n in new]))} for s in SUBSETS}
    rng = np.random.default_rng(0)
    bs = {s: [] for s in SUBSETS}
    for _ in range(nboot):
        e = [errs(G[n].iloc[rng.integers(0, len(G[n]), len(G[n]))], relevel) for n in new]
        for s in SUBSETS:
            bs[s].append(np.median([x[s][0] for x in e]))
    for s in SUBSETS:
        out["new"][s]["boot_sd"] = float(np.std(bs[s]))
    out["per_process"] = {n: {s: E[n][s][0] for s in SUBSETS} for n in new}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="tag=csv")
    ap.add_argument("--check-factor", default=None, help="data.level_events file of the runs whose CSV has an E column")
    ap.add_argument("--relevel", action="store_true", help="the level recomputed on each subset (a split level file)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "level_subsets.json"))
    a = ap.parse_args()
    new = suprio.split()["extrap"]
    out = {}
    for r in a.runs:
        t, csv = r.split("=", 1)
        has_e = "E" in pd.read_csv(csv, nrows=1).columns
        out[t] = read(csv, new, a.check_factor if has_e else None, relevel=a.relevel)
        print(t, {s: round(v["level_err"], 4) for s, v in out[t]["new"].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({t: v["new"] for t, v in out.items()}))


if __name__ == "__main__":
    main()
