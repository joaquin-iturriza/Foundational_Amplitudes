#!/usr/bin/env python3
"""Collect the μP LR-transfer sweep and plot loss-vs-LR per width.

Reads the manifest written by generate_mup_lr_transfer.py and the per-run
result JSONs, then:
  * plots val_loss vs LR, one curve per width (num_heads), on log-log axes,
  * marks each width's empirical-best LR and a parabola-interpolated optimum,
  * prints the optimal LR per width and the max/min spread.

μP works if the per-width optima cluster (small spread) instead of sliding with
width. Without μP you'd expect the optimum to fall ~1/width.

Run anywhere the result JSONs are visible (Jean Zay, or this mount):

    python sweep/analyze_mup_lr_transfer.py --sweep-dir <.../mup_lr_transfer_D1e3_t31623>
"""
import argparse
import json
import os

import numpy as np


def load(sweep_dir):
    with open(os.path.join(sweep_dir, "manifest.json")) as f:
        man = json.load(f)
    rows = []
    for j in man["jobs"]:
        rp = j["result_path"]
        if not os.path.exists(rp):
            rows.append((j["num_heads"], j["lr"], None))
            continue
        try:
            with open(rp) as f:
                val = float(json.load(f)["val_loss"])
        except Exception:
            val = None
        rows.append((j["num_heads"], j["lr"], val))
    return man, rows


def parabola_min(lrs, losses):
    """Quadratic fit in (log10 lr, log10 loss); return interpolated argmin LR."""
    lrs, losses = np.asarray(lrs), np.asarray(losses)
    m = np.isfinite(losses) & (losses > 0)
    if m.sum() < 3:
        return None
    x, y = np.log10(lrs[m]), np.log10(losses[m])
    # use the 3-5 points around the empirical min for a local quadratic
    k = int(np.argmin(y))
    lo, hi = max(0, k - 2), min(len(x), k + 3)
    if hi - lo < 3:
        return None
    a, b, c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if a <= 0:
        return float(lrs[m][k])
    return float(10 ** (-b / (2 * a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out", default=None, help="output PDF (default: <sweep-dir>/lr_transfer.pdf)")
    args = ap.parse_args()

    man, rows = load(args.sweep_dir)
    widths = sorted({nh for nh, _, _ in rows})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    opt_by_width = {}
    n_done = n_total = 0

    print(f"{'width':>6} {'best_lr(grid)':>14} {'best_lr(fit)':>13} {'best_loss':>11} {'#done':>6}")
    for wi, nh in enumerate(widths):
        sub = sorted([(lr, v) for w, lr, v in rows if w == nh], key=lambda t: t[0])
        lrs = [lr for lr, _ in sub]
        losses = [v for _, v in sub]
        n_total += len(losses)
        done = [(lr, v) for lr, v in sub if v is not None and np.isfinite(v)]
        n_done += len(done)
        color = cmap(wi / max(1, len(widths) - 1))
        if done:
            dl, dv = zip(*done)
            ax.plot(dl, dv, "o-", color=color, label=f"nh={nh}")
            k = int(np.argmin(dv))
            grid_best = dl[k]
            fit_best = parabola_min(lrs, [v if v is not None else np.nan for v in losses])
            opt_by_width[nh] = fit_best or grid_best
            ax.axvline(opt_by_width[nh], color=color, ls=":", alpha=0.5)
            print(f"{nh:>6} {grid_best:>14.3e} "
                  f"{(fit_best or float('nan')):>13.3e} {dv[k]:>11.4e} {len(done):>6}")
        else:
            print(f"{nh:>6} {'--':>14} {'--':>13} {'--':>11} {0:>6}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("learning rate"); ax.set_ylabel("best val_loss")
    ax.set_title(f"μP LR transfer — {man['sweep_name']}\n(aligned minima across widths = μP transfers)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    out = args.out or os.path.join(args.sweep_dir, "lr_transfer.pdf")
    fig.tight_layout(); fig.savefig(out)
    print(f"\nprogress: {n_done}/{n_total} runs finished")
    print(f"wrote {out}")

    if len(opt_by_width) >= 2:
        opts = np.array(list(opt_by_width.values()))
        spread = opts.max() / opts.min()
        print(f"\noptimal-LR spread across widths: {spread:.2f}x "
              f"(min {opts.min():.2e}, max {opts.max():.2e})")
        print("→ μP transfers well if this is ~1-2x; a width-scaling failure "
              "would show ~{:.0f}x (≈ widest/narrowest width ratio).".format(
                  max(opt_by_width) / min(opt_by_width)))


if __name__ == "__main__":
    main()
