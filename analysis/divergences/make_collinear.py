#!/usr/bin/env python
"""Collinear-resolved view: log|M|^2 vs a signed collinear coordinate that stretches
both beam directions, so the collinear divergence (crammed against cos->+-1 in linear
cos and washed out by mean-per-bin) becomes a visible ramp.

  eta = sign(cos) * log10( 1 / (1 - |cos|) )
      = -sign(cos) * log10(1 - |cos|)

  eta=0  -> central (cos=0);  eta->+6 forward-collinear (cos->+1);
  eta->-6 backward-collinear (cos->-1).

Per process: top panel shows truth vs model as MEAN and MAX per eta-bin (max reveals
the true peak the mean averages away); bottom panel shows the model error vs eta, i.e.
whether accuracy degrades INTO the divergence. CPU only (reads extract npz)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

EPS = 1e-7  # floors 1-|cos|; |cos|=0.999999 in data -> eta up to ~6


def collinear_coord(cos):
    a = np.clip(1.0 - np.abs(cos), EPS, None)
    return -np.sign(cos) * np.log10(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="comma-separated npz paths")
    ap.add_argument("--labels", required=True, help="comma-separated titles")
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--nbins", type=int, default=49)
    args = ap.parse_args()

    npzs = [x.strip() for x in args.npz.split(",")]
    labels = [x.strip() for x in args.labels.split("|")]
    assert len(npzs) == len(labels), "npz and labels (| separated) length mismatch"
    n = len(npzs)

    edges = np.linspace(-6, 6, args.nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.2), squeeze=False,
                             gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.06,
                                          "wspace": 0.28})
    fig.suptitle(r"Collinear-resolved: $\log|\mathcal{M}|^2$ vs "
                 r"$\eta=\mathrm{sign}(\cos\theta^*)\,\log_{10}\frac{1}{1-|\cos\theta^*|}$",
                 fontsize=13, y=0.99)

    for j, (npz, lab) in enumerate(zip(npzs, labels)):
        d = np.load(npz)
        tl, pl = d["true_logamp"], d["pred_logamp"]
        eta = collinear_coord(d["cos_theta"])

        def stat(v, how):
            r, _, _ = binned_statistic(eta, v, statistic=how, bins=edges)
            return r
        t_mean, p_mean = stat(tl, "mean"), stat(pl, "mean")
        t_max = stat(tl, "max")
        p_max = stat(pl, "max")
        err = stat(np.abs(pl - tl), "mean")

        ax = axes[0][j]
        ax.plot(ctr, t_mean, color="k", lw=2.0, label="truth (mean)")
        ax.plot(ctr, p_mean, color="crimson", lw=1.3, ls="--", label="model (mean)")
        ax.plot(ctr, t_max, color="k", lw=1.0, ls=":", alpha=0.7, label="truth (max)")
        ax.plot(ctr, p_max, color="crimson", lw=1.0, ls=":", alpha=0.7,
                label="model (max)")
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel(r"$\log|\mathcal{M}|^2$")
        ax.axvline(0, color="0.7", lw=0.8, zorder=0)
        ax.tick_params(labelbottom=False)
        if j == 0:
            ax.legend(fontsize=8, loc="upper center")

        axe = axes[1][j]
        axe.plot(ctr, err, color="steelblue", lw=1.4)
        axe.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$", fontsize=9)
        axe.set_xlabel(r"$\eta$   ($\leftarrow$ backward $\cdot$ central $\cdot$ "
                       r"forward $\rightarrow$)")
        axe.axvline(0, color="0.7", lw=0.8, zorder=0)
        axe.set_ylim(bottom=0)
        # secondary top ticks: |cos| for a few eta
        axt = ax.secondary_xaxis("top")
        et = [-6, -4, -2, 0, 2, 4, 6]
        axt.set_xticks(et)
        axt.set_xticklabels(["0" if e == 0 else f"{1.0 - 10.0**(-abs(e)):.6g}"
                             for e in et], fontsize=7)
        axt.set_xlabel(r"$|\cos\theta^*|$", fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png / .pdf")


if __name__ == "__main__":
    main()
