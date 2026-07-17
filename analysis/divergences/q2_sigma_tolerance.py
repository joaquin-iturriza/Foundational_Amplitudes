#!/usr/bin/env python
"""Q2 (offline core): how imperfect can sigma be and still guide reweighting?

The reweighting upweights events by a power of sigma to cut the deep-IR metric
    J = sum_x Q(x) e(x),   e(x) = (true_logamp - pred_logamp)^2 ,
    Q(x) = log-flat per y_min decade  (each IR decade counts equally).
Because pi(x) ∝ Q(x) sigma(x)^alpha renormalizes, sigma's ABSOLUTE scale is
irrelevant; only its ORDERING of events by true error matters. So the tolerance is
set by the rank quality rho = Spearman(sigma, |r|), NOT by calibration.

We have the TRUE per-event error field e(x) for 300k deep-IR events (deep_eval_*.npz),
so we can answer this without training:

  * build a synthetic sigma-quality axis sigma_rho with controlled Spearman rho vs the
    true |r| (rho=1 -> oracle sigma=|r|; rho=0 -> noise), verifying achieved rho;
  * score events by the reweighting priority s(x) = Q(x) * sigma_rho(x)^(2*beta)
    (sigma ~ sqrt(e), so sigma^(2beta) ~ e^beta, the variance-/error-proportional
    proposal; beta=1 default);
  * EFFICIENCY(rho) = how much of the ORACLE's Q-weighted error-capture the sigma_rho
    ranking achieves, relative to uniform:
        capture(frac) = sum of Q*e over the top-`frac` events by score,
        AUC = area under capture(frac);  eff = (AUC_rho - AUC_unif)/(AUC_oracle - AUC_unif).
    eff=1 -> sigma as good as a perfect error oracle; eff<=0 -> no better than uniform.

Measured operating points dropped on the curve: real uug sigma rho~0.46 (single-proc
= within-process), base22 within-process sigma rho~0.29. Reads off the KNEE and the
benefit fraction captured at the achievable sigma quality.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DIV = os.path.join(REPO, "analysis/divergences")


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def _spearman(x, y):
    rx = _rankdata(x); ry = _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def make_sigma_rho(err, target_rho, rng, grid=None):
    """Monotone-in-|err| sigma with Spearman(sigma, err) = target_rho (approx).

    Blend the rank of true error with a random rank in rank space, then map back to a
    positive sigma via a monotone transform of the blended rank. A scan over the blend
    weight w hits the requested rho (achieved rho is monotone in w). rho=1 -> sigma∝err
    ordering (oracle); rho=0 -> random.
    """
    n = len(err)
    r_err = _rankdata(err) / n                     # in (0,1], true-error rank
    if target_rho >= 0.999:
        base = r_err
    elif target_rho <= 0.001:
        base = rng.random(n)
    else:
        grid = grid if grid is not None else np.linspace(0, 1, 41)
        r_noise = rng.random(n)
        best_w, best_gap = 0.0, 1e9
        for w in grid:
            cand = w * r_err + (1 - w) * r_noise
            rho = _spearman(cand, err)
            if abs(rho - target_rho) < best_gap:
                best_gap, best_w = abs(rho - target_rho), w
        base = best_w * r_err + (1 - best_w) * rng.random(n)
    # map blended rank -> positive sigma spanning the true error scale (order is all
    # that matters downstream, but keep it on the error scale for interpretability)
    lo, hi = np.quantile(err[err > 0], [0.01, 0.99])
    s = lo * (hi / lo) ** (_rankdata(base) / n)
    return s, _spearman(s, err)


def qflat_weights(y_min, n_dec_edges=None):
    """Log-flat-per-decade weight: each y_min decade sums to equal mass."""
    ly = np.log10(np.clip(y_min, 1e-30, None))
    edges = np.arange(np.floor(ly.min()), np.ceil(ly.max()) + 1e-9, 1.0)
    b = np.clip(np.digitize(ly, edges) - 1, 0, len(edges) - 2)
    w = np.zeros(len(y_min))
    for k in np.unique(b):
        m = b == k
        cnt = m.sum()
        if cnt:
            w[m] = 1.0 / cnt          # each decade sums to 1 -> equal mass per decade
    w /= w.sum()                       # normalize total mass to 1
    return w, b


def capture_auc(score, qe, n_grid=200):
    """Area under capture(frac): sum of Q*e among the top-`frac` by score."""
    order = np.argsort(-score, kind="mergesort")
    qe_sorted = qe[order]
    cum = np.cumsum(qe_sorted) / qe.sum()
    fracs = (np.arange(1, len(cum) + 1)) / len(cum)
    idx = np.linspace(0, len(cum) - 1, n_grid).astype(int)
    return float(np.trapz(cum[idx], fracs[idx])), fracs[idx], cum[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(DIV, "deep_eval_antenna.npz"),
                    help="per-event deep-IR eval npz (true/pred_logamp, y_min)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="proposal exponent: score ∝ Q*sigma^(2beta) (~Q*e^beta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.eval, allow_pickle=True)
    err2 = (np.asarray(d["true_logamp"], float) - np.asarray(d["pred_logamp"], float)) ** 2
    y_min = np.asarray(d["y_min"], float)
    rng = np.random.RandomState(args.seed)

    Q, _ = qflat_weights(y_min)
    qe = Q * err2                                  # Q-weighted true error mass per event

    # oracle score (sigma = |r| exactly): s* ∝ Q * err2^beta
    err = np.sqrt(err2)
    s_oracle = Q * err2 ** args.beta
    auc_oracle, f_or, c_or = capture_auc(s_oracle, qe)
    auc_unif = 0.5                                 # uniform ranking -> capture is the diagonal

    rhos = np.array([0.0, 0.1, 0.2, 0.29, 0.35, 0.46, 0.6, 0.75, 0.9, 1.0])
    effs, ach = [], []
    curves = {}
    for rho in rhos:
        s_sig, rho_ach = make_sigma_rho(err, rho, rng)
        score = Q * s_sig ** (2 * args.beta)
        auc, f, c = capture_auc(score, qe)
        eff = (auc - auc_unif) / (auc_oracle - auc_unif)
        effs.append(eff); ach.append(rho_ach)
        curves[round(rho, 2)] = (f, c)
    effs = np.array(effs); ach = np.array(ach)

    # report
    def eff_at(target):
        j = int(np.argmin(np.abs(ach - target)))
        return effs[j], ach[j]
    e_uug, _ = eff_at(0.46)
    e_b22, _ = eff_at(0.29)
    print(f"oracle AUC={auc_oracle:.3f} (uniform 0.5)  -> reweighting headroom "
          f"{auc_oracle - 0.5:+.3f}")
    print(f"{'rho_target':>10} {'rho_achieved':>12} {'efficiency':>11}")
    for rt, ra, e in zip(rhos, ach, effs):
        tag = ""
        if abs(rt - 0.46) < 1e-6: tag = "  <- real uug sigma (within-proc)"
        if abs(rt - 0.29) < 1e-6: tag = "  <- base22 within-process sigma"
        print(f"{rt:10.2f} {ra:12.3f} {e:11.3f}{tag}")
    # knee: smallest rho reaching 50% / 80% of oracle
    def knee(thr):
        ok = np.where(effs >= thr)[0]
        return ach[ok[0]] if len(ok) else None
    print(f"\nKNEE: rho for 50% of oracle benefit = {knee(0.5)}, "
          f"80% = {knee(0.8)}")
    print(f"real uug sigma (rho~0.46) captures {100*e_uug:.0f}% of oracle reweighting "
          f"benefit; base22 within-proc (rho~0.29) captures {100*e_b22:.0f}%")

    # figure
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8))
    a0.plot(ach, effs, "-o", color="C0")
    a0.axhline(0, color="k", lw=0.7)
    for xr, lab, col in [(0.29, "base22 within-proc", "C3"),
                         (0.46, "real uug sigma", "C2")]:
        a0.axvline(xr, ls=":", color=col, lw=1.4, label=f"{lab} (rho~{xr})")
    a0.axhline(0.5, ls="--", color="grey", lw=0.8)
    a0.axhline(0.8, ls="--", color="grey", lw=0.8)
    a0.set_xlabel("sigma ranking quality  rho = Spearman(sigma, |r|)")
    a0.set_ylabel("reweighting efficiency\n(fraction of oracle benefit)")
    a0.set_title("Q2 tolerance: how imperfect can sigma be?")
    a0.set_ylim(-0.05, 1.05); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    for rho in [0.0, 0.29, 0.46, 0.75, 1.0]:
        f, c = curves[rho]
        a1.plot(f, c, label=f"rho={rho}")
    a1.plot(f_or, c_or, "k--", lw=1.5, label="oracle (sigma=|r|)")
    a1.plot([0, 1], [0, 1], color="grey", lw=0.8)
    a1.set_xlabel("training-budget fraction (top events by score)")
    a1.set_ylabel("Q-weighted true error captured")
    a1.set_title("error-capture curves")
    a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Q2 sigma-imperfection tolerance for reweighting "
                 f"(deep-IR uug, beta={args.beta})", fontsize=12)
    fig.tight_layout()

    outdir = os.path.join(DIV, "figs")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "q2_sigma_tolerance")
    fig.savefig(base + ".png", dpi=140); fig.savefig(base + ".pdf")
    with open(os.path.join(DIV, "q2_sigma_tolerance.json"), "w") as fh:
        json.dump(dict(rho_target=rhos.tolist(), rho_achieved=ach.tolist(),
                       efficiency=effs.tolist(), auc_oracle=auc_oracle,
                       beta=args.beta, eval=os.path.basename(args.eval)), fh, indent=2)
    print(f"wrote {base}.png/.pdf and q2_sigma_tolerance.json")


if __name__ == "__main__":
    main()
