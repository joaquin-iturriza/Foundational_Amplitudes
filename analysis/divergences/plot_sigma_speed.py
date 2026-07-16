#!/usr/bin/env python
"""Plot the σ-ranking convergence curve(s) from sigma_rank_curve.json.

Reads one or more runs' sigma_rank_curve.json (written per validation step by the
heterosc_rank_curve probe) and plots the σ-RANKING figures of merit vs iteration:
how few σ-only iters make σ usable as a reweighting signal. The reweighting cares
about σ's ORDER, not its calibration, so ρ (Spearman σ vs |r|) is the headline.

For each run it also reports the "usable" iteration = first step at which ρ_proc
reaches TARGET_FRAC of its own plateau (median of the last few points).

Usage:
  python plot_sigma_speed.py                       # default runs (base22, uug)
  RUNS=base22,uug python plot_sigma_speed.py
  python plot_sigma_speed.py /abs/run_dirA /abs/run_dirB
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
DEFAULT = ["base22", "uug"]
TARGET_FRAC = 0.95   # fraction of plateau ρ_proc that counts as "usable"


def resolve(runs):
    dirs = []
    for r in runs:
        if os.path.isdir(r):
            dirs.append(r)
        else:
            dirs.append(os.path.join(WT, "runs", "heterosc_sigspeed", r))
    return dirs


def load(run_dir):
    fp = os.path.join(run_dir, "sigma_rank_curve.json")
    if not os.path.exists(fp):
        print(f"  MISSING {fp}")
        return None
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def usable_iter(steps, rho, frac=TARGET_FRAC):
    rho = np.asarray(rho, float)
    good = np.isfinite(rho)
    if good.sum() < 3:
        return None, None
    plateau = float(np.median(rho[good][-max(3, good.sum() // 5):]))
    thr = frac * plateau
    for s, v in zip(steps, rho):
        if np.isfinite(v) and v >= thr:
            return int(s), plateau
    return None, plateau


def main():
    args = [a for a in sys.argv[1:]]
    runs = args or os.environ.get("RUNS", "").split(",") or DEFAULT
    runs = [r for r in runs if r] or DEFAULT
    dirs = resolve(runs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(dirs)))
    for run_dir, c in zip(dirs, colors):
        name = os.path.basename(run_dir.rstrip("/"))
        curve = load(run_dir)
        if not curve:
            continue
        step = [m["step"] for m in curve]
        rho_g = [m["spearman_global"] for m in curve]
        rho_p = [m["spearman_proc"] for m in curve]
        slope = [m["slope"] for m in curve]
        wall = [m.get("wall_s") for m in curve]

        uit, plat = usable_iter(step, rho_p)
        # wall time to usable
        wsec = None
        if uit is not None:
            for s, w in zip(step, wall):
                if s >= uit and w is not None:
                    wsec = w
                    break
        lab = f"{name}"
        if uit is not None:
            lab += f"  (ρ_proc→{plat:.2f} plateau; {int(TARGET_FRAC*100)}%@it{uit}"
            if wsec is not None:
                lab += f"={wsec/60:.1f}m"
            lab += ")"
        axes[0].plot(step, rho_g, "-o", ms=3, color=c, label=lab)
        axes[1].plot(step, rho_p, "-o", ms=3, color=c)
        axes[2].plot(step, slope, "-o", ms=3, color=c)
        if uit is not None:
            for ax, series in ((axes[0], rho_g), (axes[1], rho_p)):
                ax.axvline(uit, color=c, ls=":", lw=1, alpha=0.6)
        print(f"{name:>10}: ρ_proc plateau {plat if plat else float('nan'):.3f}, "
              f"{int(TARGET_FRAC*100)}% at iter {uit}"
              + (f" ({wsec/60:.1f} min)" if wsec else ""))

    axes[0].set_title("Spearman ρ(σ, |r|)  —  global")
    axes[1].set_title("Spearman ρ(σ, |r|)  —  per-process median\n(the high-dim signal)")
    axes[2].set_title("reliability slope\n(1 = σ tracks error; secondary)")
    for ax in axes:
        ax.set_xlabel("σ-only iteration")
        ax.grid(alpha=0.3)
    axes[2].axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("σ-head ranking convergence (constant-lr σ-only fit) — "
                 "how few iters make σ usable for reweighting", fontsize=11)
    fig.tight_layout()

    outdir = os.path.join(WT, "analysis/divergences/figs")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "sigma_speed_curve")
    fig.savefig(base + ".png", dpi=140)
    fig.savefig(base + ".pdf")
    print(f"wrote {base}.png / .pdf")


if __name__ == "__main__":
    main()
