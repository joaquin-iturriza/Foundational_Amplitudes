#!/usr/bin/env python
"""IR-resolved plots for the gluon channels (ee->uug, ee->uugg): how the model
behaves at the genuine SOFT and COLLINEAR singularities.

Main figure (per process), reading the extract_ir npz:
  row 1 : <log|M|^2> truth | model | error, over (log10 y_min, log10 x_gmin)
          [y_min = IR resolution var ->0 soft&collinear; x_gmin = softest gluon frac]
  row 2 : log|M|^2 vs log10 y_min (IR ramp, mean&max) | vs log10 x_gmin (soft)
          | pred-vs-true hexbin.  (blue = mean model error on twin axis)
Plus, for ee->q qbar g, a Dalitz figure: <log|M|^2> over (x_q, x_qbar) truth/model/error.
CPU only.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic, binned_statistic_2d


def _map(x, y, z, xb, yb, stat="mean"):
    s, xe, ye, _ = binned_statistic_2d(x, y, z, statistic=stat, bins=[xb, yb])
    return s.T, xe, ye


def make_ir(npz, label, out_base):
    d = np.load(npz)
    tl, pl = d["true_logamp"], d["pred_logamp"]
    resid = pl - tl
    ly = np.log10(np.clip(d["y_min"], 1e-12, None))     # IR / collinear master var
    lx = np.log10(np.clip(d["x_gmin"], 1e-12, None))    # softest gluon energy fraction
    n = len(tl)

    xb = np.linspace(np.percentile(ly, 0.2), np.percentile(ly, 99.8), 45)
    yb = np.linspace(np.percentile(lx, 0.2), np.percentile(lx, 99.8), 45)
    tmap, xe, ye = _map(ly, lx, tl, xb, yb)
    pmap, _, _ = _map(ly, lx, pl, xb, yb)
    emap, _, _ = _map(ly, lx, np.abs(resid), xb, yb)
    vmin, vmax = np.nanpercentile(tmap, 1), np.nanpercentile(tmap, 99)

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.32, height_ratios=[1.0, 0.95])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    fig.suptitle(f"{label}: model vs truth at the IR singularities  "
                 f"(N={n:,}, RMS Δlog|M|²={rms:.3g})", fontsize=13, y=0.98)

    def draw(ax, M, title, cmap, vmn, vmx, cl):
        pm = ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmn, vmax=vmx, shading="flat")
        ax.set_xlabel(r"$\log_{10} y_{\min}$   (collinear/soft $\to -\infty$)")
        ax.set_ylabel(r"$\log_{10} x_{g,\min}$   (soft $\to -\infty$)")
        ax.set_title(title, fontsize=11)
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.02); cb.set_label(cl, fontsize=9)

    draw(fig.add_subplot(gs[0, 0]), tmap, "truth", "viridis", vmin, vmax,
         r"$\langle\log|\mathcal{M}|^2\rangle$")
    draw(fig.add_subplot(gs[0, 1]), pmap, "model prediction", "viridis", vmin, vmax,
         r"$\langle\log|\mathcal{M}|^2\rangle$")
    draw(fig.add_subplot(gs[0, 2]), emap, "model error", "inferno",
         0.0, np.nanpercentile(emap, 99), r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")

    def ramp(ax, coord, xlabel, title):
        bins = np.linspace(np.percentile(coord, 0.2), np.percentile(coord, 99.8), 55)
        c = 0.5 * (bins[:-1] + bins[1:])
        tm, _, _ = binned_statistic(coord, tl, "mean", bins=bins)
        pm, _, _ = binned_statistic(coord, pl, "mean", bins=bins)
        tx, _, _ = binned_statistic(coord, tl, "max", bins=bins)
        px, _, _ = binned_statistic(coord, pl, "max", bins=bins)
        am, _, _ = binned_statistic(coord, np.abs(resid), "mean", bins=bins)
        ax.plot(c, tm, "k", lw=1.9, label="truth (mean)")
        ax.plot(c, pm, color="crimson", lw=1.3, ls="--", label="model (mean)")
        ax.plot(c, tx, "k", lw=1.0, ls=":", alpha=0.7, label="truth (max)")
        ax.plot(c, px, color="crimson", lw=1.0, ls=":", alpha=0.7, label="model (max)")
        ax.set_xlabel(xlabel); ax.set_ylabel(r"$\log|\mathcal{M}|^2$")
        ax.set_title(title, fontsize=11); ax.legend(fontsize=7, ncol=2, loc="upper right")
        axr = ax.twinx()
        axr.plot(c, am, color="steelblue", lw=1.0, alpha=0.7)
        axr.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$", color="steelblue",
                       fontsize=9)
        axr.tick_params(axis="y", labelcolor="steelblue"); axr.set_ylim(bottom=0)

    ramp(fig.add_subplot(gs[1, 0]), ly,
         r"$\log_{10} y_{\min}$  ($\leftarrow$ deeper IR)", "IR ramp: soft + collinear")
    ramp(fig.add_subplot(gs[1, 1]), lx,
         r"$\log_{10} x_{g,\min}$  ($\leftarrow$ softer gluon)", "soft-gluon limit")

    axsc = fig.add_subplot(gs[1, 2])
    hb = axsc.hexbin(tl, pl, gridsize=55, bins="log", cmap="magma", mincnt=1)
    lo, hi = min(tl.min(), pl.min()), max(tl.max(), pl.max())
    axsc.plot([lo, hi], [lo, hi], color="cyan", lw=1.0, ls=":")
    axsc.set_xlabel(r"truth $\log|\mathcal{M}|^2$"); axsc.set_ylabel(r"model $\log|\mathcal{M}|^2$")
    axsc.set_title("predicted vs true", fontsize=11)
    fig.colorbar(hb, ax=axsc, fraction=0.046, pad=0.02).set_label("count", fontsize=9)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf (N={n}, RMSΔ={rms:.3g})")

    # Dalitz (ee -> q qbar g only)
    if "x_q" in d.files:
        xq, xqb = d["x_q"], d["x_qbar"]
        b = np.linspace(0, 1, 60)
        tm, xe2, ye2 = _map(xq, xqb, tl, b, b)
        pm, _, _ = _map(xq, xqb, pl, b, b)
        em, _, _ = _map(xq, xqb, np.abs(resid), b, b)
        vmn, vmx = np.nanpercentile(tm, 1), np.nanpercentile(tm, 99)
        figd, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        figd.suptitle(f"{label}: Dalitz plane  (collinear at $x\\to1$ edges, "
                      f"soft gluon at the $(1,1)$ corner)", fontsize=12)
        for ax, M, ti, cm, a, bb, cl in [
                (axs[0], tm, "truth", "viridis", vmn, vmx, r"$\langle\log|\mathcal{M}|^2\rangle$"),
                (axs[1], pm, "model", "viridis", vmn, vmx, r"$\langle\log|\mathcal{M}|^2\rangle$"),
                (axs[2], em, "error", "inferno", 0.0, np.nanpercentile(em, 99),
                 r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")]:
            pmesh = ax.pcolormesh(xe2, ye2, M, cmap=cm, vmin=a, vmax=bb, shading="flat")
            ax.set_xlabel(r"$x_q=2E_q/\sqrt{s}$"); ax.set_ylabel(r"$x_{\bar q}=2E_{\bar q}/\sqrt{s}$")
            ax.set_title(ti, fontsize=11)
            figd.colorbar(pmesh, ax=ax, fraction=0.046, pad=0.02).set_label(cl, fontsize=9)
        figd.tight_layout()
        for ext in ("png", "pdf"):
            figd.savefig(f"{out_base}_dalitz.{ext}", dpi=140, bbox_inches="tight")
        plt.close(figd)
        print(f"wrote {out_base}_dalitz.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()
    make_ir(args.npz, args.label, args.out_base)


if __name__ == "__main__":
    main()
