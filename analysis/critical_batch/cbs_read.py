"""critical_batch step 2: the critical batch from the batch-size scan (McCandlish et al. 2018, arXiv:1812.06162).

    python analysis/critical_batch/cbs_read.py <trials.json> [--out <summary.json>]
    python analysis/critical_batch/cbs_read.py --selfcheck

<trials.json>: {"<batch>": [<trial result.json dict, with "hp_idx">, ...], ...}, the sweeps cbs_b<batch>' trials.
Per batch B and target L (the v4_het_control_wsd run's 25/50/75% stable-phase levels): S_B(L) = the lowest
steps-to-target over B's trials (sweep/run_trial.py steps_to_target, the sweeps' objective), E_B = B S_B; the fit
S = S_min (1 + B_crit / B) by least squares in log S over the batches that reach L; B_crit's spread from a bootstrap
over those batch points. A batch none of whose trials reaches L is listed with its lower bound (its last validation
step), never dropped."""
import argparse, json, os, sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from run_trial import steps_to_target  # noqa: E402

TARGETS = (-1.66, -2.08, -2.30)   # the control's crossings 1023 / 1955 / 2707 steps (critical_batch design)


def fit_bcrit(B, S):
    """(S_min, B_crit) minimising sum (log S - log S_min - log(1 + B_crit / B))^2, B_crit >= 0."""
    B, S = np.asarray(B, float), np.asarray(S, float)
    r = lambda p: np.log(S) - p[0] - np.log1p(np.exp(p[1]) / B)
    p = least_squares(r, [np.log(S.min()), np.log(np.median(B))]).x
    return float(np.exp(p[0])), float(np.exp(p[1]))


def bootstrap(B, S, n=2000, seed=0):
    """B_crit's 16/50/84% quantiles over resamples of the batch points (with at least 2 distinct batches)."""
    rng, B, S, out = np.random.default_rng(seed), np.asarray(B), np.asarray(S), []
    while len(out) < n:
        i = rng.integers(0, len(B), len(B))
        if len(set(B[i])) >= 2:
            out.append(fit_bcrit(B[i], S[i])[1])
    return [float(q) for q in np.quantile(out, [0.16, 0.5, 0.84])]


def read(trials):
    out = {}
    for L in TARGETS:
        rows, reach = [], []
        for b, ts in sorted(trials.items(), key=lambda x: int(x[0])):
            s = [(steps_to_target(t, L), t.get("hp_idx")) for t in ts if t.get("val_curve")]
            hit = sorted((v, h) for v, h in s if v is not None)
            row = {"B": int(b), "trials": len(ts), "reached": len(hit)}
            if hit:
                row.update(S=hit[0][0], E=int(b) * hit[0][0], hp_idx=hit[0][1], top3=[v for v, _ in hit[:3]])
                reach.append((int(b), hit[0][0]))
            else:
                row["S_lower_bound"] = max((len(t["val_curve"]) * t["validate_every"] for t in ts
                                            if t.get("val_curve")), default=None)
            rows.append(row)
        res = {"batches": rows}
        if len(reach) >= 2:
            Bs, Ss = zip(*reach)
            res["S_min"], res["B_crit"] = fit_bcrit(Bs, Ss)
            res["B_crit_q16_50_84"] = bootstrap(Bs, Ss)
            res["S_ratio_smallest_over_largest"] = Ss[0] / Ss[-1]
            # a batch that never reached L while its lower bound exceeds the fit's S there: B_crit is a lower bound
            res["unreached_above_fit"] = [r["B"] for r in rows if r.get("S_lower_bound") and
                                          r["S_lower_bound"] > res["S_min"] * (1 + res["B_crit"] / r["B"])]
        out[str(L)] = res
    return out


def selfcheck():
    """Known answer: S from B_crit = 500 and 20000 with 3% noise at the scan's batches is fitted back within 25%."""
    rng = np.random.default_rng(1)
    B = np.array([2048, 4096, 8192, 16384, 32768])
    for bc in (500.0, 20000.0):
        S = 2000 * (1 + bc / B) * np.exp(rng.normal(0, 0.03, len(B)))
        smin, got = fit_bcrit(B, S)
        assert abs(got / bc - 1) < 0.25 and abs(smin / 2000 - 1) < 0.1, (bc, got, smin)
    # a flat S (B_crit far below the batches) fits a B_crit small against 2048
    assert fit_bcrit(B, np.full(5, 2000.0))[1] < 50
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        sys.exit(0)
    res = read(json.load(open(a.trials)))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
