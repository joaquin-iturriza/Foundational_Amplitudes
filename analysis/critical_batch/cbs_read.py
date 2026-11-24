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
import yaml
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from run_trial import steps_to_target  # noqa: E402

TARGETS = (-1.66, -2.08, -2.30)   # the control's crossings 1023 / 1955 / 2707 steps (critical_batch design)
EDGE_TARGET = -2.08                # the design's edge test reads each batch's best trial on S_B(-2.08), the objective


def lr_window(b):
    """Batch b's searched lr window, from its own sweep config (a re-searched batch has a shifted window)."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, "sweep", f"sweep_config_cbs_b{b}.yaml")))
    p = next(x for x in cfg["search_space"] if x["name"] == "training.lr")
    return p["low"], p["high"]


def lr_at_edge(lr, window):
    """The design's lr-edge rule: lr in the outer tenth of the log window; that batch's S is an upper bound only."""
    lo, hi = np.log(window)
    x = (np.log(lr) - lo) / (hi - lo)
    return bool(x < 0.1 or x > 0.9)


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


def best_trial(ts, L):
    """(S, trial) of the lowest steps-to-target L among ts, (None, None) when none reaches L."""
    hit = [(v, t) for t in ts if t.get("val_curve") for v in [steps_to_target(t, L)] if v is not None]
    if not hit:
        return None, None
    v, t = min(hit, key=lambda x: x[0])
    assert t.get("lr") is not None, f"best trial hp {t.get('hp_idx')} carries no lr: the collector must add training.lr"
    return v, t


def read(trials):
    out, edge = {}, {}
    for b, ts in trials.items():
        _, t = best_trial(ts, EDGE_TARGET)
        if t is not None:
            edge[int(b)] = lr_at_edge(t["lr"], lr_window(b))
    for L in TARGETS:
        rows, reach = [], []
        for b, ts in sorted(trials.items(), key=lambda x: int(x[0])):
            s = [(steps_to_target(t, L), t.get("hp_idx")) for t in ts if t.get("val_curve")]
            hit = sorted((v, h) for v, h in s if v is not None)
            row = {"B": int(b), "trials": len(ts), "reached": len(hit)}
            if hit:
                Sb, best = best_trial(ts, L)   # S, hp_idx and lr of one trial; asserts the trial carries training.lr
                lr = best["lr"]
                # the edge flag is the batch's, from its best trial on S_B(-2.08); a batch never at -2.08 (it can
                # still reach -1.66) is judged on this target's best trial
                row.update(S=Sb, E=int(b) * Sb, hp_idx=best.get("hp_idx"), top3=[v for v, _ in hit[:3]],
                           lr=lr, lr_edge=edge.get(int(b), lr_at_edge(lr, lr_window(b))))
                if row["lr_edge"] is False:   # an edge (or unknown) lr: S an upper bound, out of the fit
                    reach.append((int(b), Sb))
            else:
                row["S_lower_bound"] = max((len(t["val_curve"]) * t["validate_every"] for t in ts
                                            if t.get("val_curve")), default=None)
            rows.append(row)
        res = {"batches": rows}
        # the design's ratio is S(2048) / S(32768) exactly; with either missing (unreached, edge lr) it is not formed
        S = {r["B"]: r.get("S") for r in rows if r.get("lr_edge") is False}
        res["S_2048_over_S_32768"] = S[2048] / S[32768] if S.get(2048) and S.get(32768) else None
        if len(reach) >= 2:
            Bs, Ss = zip(*reach)
            res["S_min"], res["B_crit"] = fit_bcrit(Bs, Ss)
            res["B_crit_q16_50_84"] = bootstrap(Bs, Ss)
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
    # read(): the ratio is S(2048)/S(32768) only; an edge lr leaves its batch out of the ratio and the fit
    curve = lambda n: {"val_curve": [0.0] * (n - 1) + [-3.0], "validate_every": 100}
    tr = {str(b): [dict(curve(n), hp_idx=0, lr=5e-3)] for b, n in ((2048, 40), (4096, 30), (16384, 20), (32768, 20))}
    r = read(tr)["-2.08"]
    assert abs(r["S_2048_over_S_32768"] - 4000 / 2000) < 0.1 and r["B_crit"] > 0, r
    tr["32768"][0]["lr"] = 1.6e-3                      # outer tenth of the window
    r = read(tr)["-2.08"]
    assert r["S_2048_over_S_32768"] is None and r["batches"][-1]["lr_edge"] is True, r
    del tr["2048"]
    assert read(tr)["-2.08"]["S_2048_over_S_32768"] is None   # never the 4096 / 16384 pair
    # the edge flag is the batch's best trial on S_B(-2.08), at every target: trial 1 is first to -1.66 at an edge
    # lr, trial 2 first to -2.08 at a centre lr, so 4096 stays in the fit at -1.66
    tr["4096"] = [{"val_curve": [0.0, -1.7, -1.7, -1.7, -3.0], "validate_every": 100, "hp_idx": 1, "lr": 1.6e-3},
                  {"val_curve": [0.0, 0.0, -3.0], "validate_every": 100, "hp_idx": 2, "lr": 5e-3}]
    row = next(x for x in read(tr)["-1.66"]["batches"] if x["B"] == 4096)
    assert row["hp_idx"] == 1 and row["lr_edge"] is False, row
    tr["4096"][1].pop("lr")                              # a collector that omits lr stops the read
    try:
        read(tr)
        raise AssertionError("missing lr not caught")
    except AssertionError as e:
        assert "carries no lr" in str(e), e
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
