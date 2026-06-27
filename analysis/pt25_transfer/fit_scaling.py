"""Weighted power-law fit  L(C) = A * C^{-alpha}  with chi2/dof goodness-of-fit.

Per (family, t_steps) cell: ~14-25 DyHPO trials, each (val_loss, test_loss).
Scaling point y_i = held-out test_loss of the best-val trial. Per-point error
sigma_i (in ln L) = std of ln(test_loss of best-val trial) under resampling the
trials -> this is ONLY the best-of-20 *selection* variance (NOT training-seed
noise), so it is a LOWER bound on the true error; chi2/dof computed against it
therefore over-states the fit quality. Weighted log-log fit -> alpha +/- sigma
(from the covariance) and reduced chi2.
"""
import glob, json, re, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RNG  = np.random.default_rng(0)
NBOOT = 4000

def f_step(nh, n_avg, bs):
    d = 16 * nh
    return 3.0 * bs * (n_avg * 131072 + 8 * n_avg * (24 * d**2 + 2 * n_avg * d))
FSTEP = f_step(8, 4, 8192)

def _load(f):
    for _ in range(4):
        try:
            return json.load(open(f))
        except OSError:
            continue
    return None

def cell_trials(glob_pat):
    out = {}
    for d in glob.glob(os.path.join(ROOT, glob_pat)):
        m = re.search(r"_t0*(\d+)$", d)
        if not m:
            continue
        t = int(m.group(1))
        arr = [(r["val_loss"], r["test_loss"])
               for r in (_load(f) for f in glob.glob(d + "/results/*.json"))
               if r and "test_loss" in r and r["test_loss"] > 0]
        if arr:
            out.setdefault(t, []).extend(arr)
    return {t: np.array(v) for t, v in out.items()}

def sel_test(vt):
    return vt[np.argmin(vt[:, 0]), 1]

def point(vt):
    """central best-val test_loss + sigma in ln-space (selection variance only)."""
    central = sel_test(vt)
    n = len(vt)
    boots = np.array([sel_test(vt[RNG.integers(0, n, n)]) for _ in range(NBOOT)])
    sig = np.std(np.log(boots))
    return central, max(sig, 1e-3)            # floor so a degenerate cell doesn't dominate

from scipy.optimize import curve_fit

def wls_powerlaw(C, L, sig_ln):
    """Weighted nonlinear fit of  L = A*C^(-alpha) + C0  (3 params), in ln-space.
    Returns A, alpha, sig_alpha, C0, chi2, dof."""
    lnC, lnL = np.log(C), np.log(L)
    # params: lnA, alpha, lnC0  (exp() keeps A, C0 > 0)
    def logmodel(lnc, lnA, alpha, lnC0):
        return np.log(np.exp(lnA) * np.exp(-alpha * lnc) + np.exp(lnC0))
    p0 = [np.log(L.max()) + 1.8 * lnC.min(), 1.8, np.log(max(L.min() * 0.5, 1e-8))]
    bounds = ([-80, 0.1, np.log(1e-9)], [80, 5.0, np.log(L.min())])  # C0 <= smallest point
    popt, pcov = curve_fit(logmodel, lnC, lnL, p0=np.clip(p0, *bounds), sigma=sig_ln,
                           absolute_sigma=True, bounds=bounds, maxfev=200000)
    lnA, alpha, lnC0 = popt
    sig_alpha = np.sqrt(pcov[1, 1])
    resid = (lnL - logmodel(lnC, *popt)) / sig_ln
    chi2 = float(np.sum(resid ** 2)); dof = len(C) - 3
    return np.exp(lnA), alpha, sig_alpha, np.exp(lnC0), chi2, dof

FAMS = [
    ("pt25 FT (25-proc)",  "C0", lambda k: [f"sweeps/finetune_pt25_*{k}_t*"]),
    ("old FT (8-proc)",    "C1", lambda k: [f"sweeps/finetune_scaling_virt_{k}_t*",
                                            f"sweeps/finetune_scaling_virt_002_{k}_t*"]),
    ("from-scratch (nh8)", "C2", lambda k: [f"sweeps/scaling_solo_nh8_curve_virt_{k}_t*",
                                            f"sweeps/scaling_solo_nh8_lowt_virt_{k}_t*",
                                            f"sweeps/scaling_solo_nh8_anchor_virt_002_{k}_t*",
                                            f"sweeps/scaling_solo_nh8_anchor2_virt_{k}_t*",
                                            f"sweeps/solo_nh8_10k_virt_{k}_t*"]),
]
DATASETS = [("ee_uu NLO-virt", "eeuunlovirte4"), ("ee_ttbar NLO-virt", "eettbarnlovirte4")]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
summary = {}
for ax, (title, key) in zip(axes, DATASETS):
    for label, col, pats in FAMS:
        merged = {}
        for p in pats(key):
            for t, v in cell_trials(p).items():
                merged[t] = np.vstack([merged[t], v]) if t in merged else v
        if len(merged) < 3:
            continue
        ts = sorted(merged)
        C = np.array([2 * FSTEP * t for t in ts])
        cen, sig = [], []
        for t in ts:
            c, s = point(merged[t]); cen.append(c); sig.append(s)
        cen, sig = np.array(cen), np.array(sig)
        A, al, sal, C0, chi2, dof = wls_powerlaw(C, cen, sig)
        red = chi2 / dof if dof > 0 else float("nan")
        summary.setdefault(key, []).append((label, al, sal, C0, chi2, dof, len(ts)))
        # error bars: +/-1sigma in ln -> multiplicative in linear space
        ax.errorbar(C, cen, yerr=[cen - cen*np.exp(-sig), cen*np.exp(sig) - cen],
                    fmt="o", color=col, ms=5, capsize=2, lw=1,
                    label=f"{label}  α={al:.2f}±{sal:.2f}, χ²/dof={red:.1f}")
        cgrid = np.geomspace(C.min(), C.max(), 200)
        ax.plot(cgrid, A * cgrid ** (-al) + C0, color=col, lw=1.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute [FLOP]"); ax.set_ylabel("held-out test_loss")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9, title="weighted fit  L = A·C^(-α)")

fig.suptitle("NLO transfer scaling fits (all nh=8) — L = A·C^(-α) + C₀, weighted, χ²/dof",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "scaling_fit_nh8.pdf"))
fig.savefig(os.path.join(HERE, "scaling_fit_nh8.png"), dpi=130)
print("saved scaling_fit_nh8.{pdf,png}\n")
for key, rows in summary.items():
    print(key)
    for label, al, sal, C0, chi2, dof, n in rows:
        print(f"  {label:22s} alpha={al:5.2f} +/- {sal:.2f}   C0={C0:.2e}   chi2/dof = {chi2:7.1f}/{dof} = {chi2/dof:6.1f}   ({n} levels)")
