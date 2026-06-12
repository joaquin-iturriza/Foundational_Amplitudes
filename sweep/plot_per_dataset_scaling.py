#!/usr/bin/env python3
"""
plot_per_dataset_scaling.py — Per-dataset loss-vs-compute scaling for the
pretraining (joint, multi-process) sweep, compared against the solo (single-
process) scaling runs.

For every joint cell in sweeps/pretraining_scaling/ (scaling_p1_* and
scaling_p1ext_*) we take the BEST model found in that cell (the HPO trial with
the lowest combined no-reg val loss = geometric mean of proc_val_losses_no_reg),
and read off, from that single model:
  - proc_val_losses_no_reg[ds]  : the per-dataset validation loss
  - compute_ds[ds]              : the number of samples of ds the model has seen

The per-dataset COMPUTE is then computed properly as

    C_ds = compute_ds[ds] * flops_per_sample(num_heads, n_particles(ds))

i.e. samples-seen times the per-sample forward+backward MACs of the model,
using *that dataset's own particle multiplicity* (4/5/6) instead of a fixed
average — bigger models and higher-multiplicity processes cost more per sample.

This is overlaid, per dataset, with the solo scaling runs
(scaling_solo_full_<tag>_t*), whose best-per-cell val_loss is plotted at
C = (t_steps * batchsize) * flops_per_sample(nh_solo, n_particles(ds)).

UNITS.  The joint runs standardise log|A| *globally* (one mean/std pooled over
all 8 processes), while a solo run standardises on its single process.  A raw
standardised MSE is therefore on a different vertical scale for joint vs solo.
By default we rescale both to a common PHYSICAL scale,

    MSE_phys(ln|A|) = run_loss * std(ln|A|)**2 ,

using std over the global pool for joint points and the per-dataset std for
solo points.  (This only shifts curves vertically; the scaling slope/exponent
is invariant.)  Pass --raw-units to plot the as-reported standardised losses.

Run this ON Jean Zay (it scans thousands of result JSONs and reads the .npy
amplitude columns — slow over the mount):

    python sweep/plot_per_dataset_scaling.py
    python sweep/plot_per_dataset_scaling.py --raw-units      # no std rescaling
    python sweep/plot_per_dataset_scaling.py --no-fit         # scatter only
    python sweep/plot_per_dataset_scaling.py --dry-run        # table only, no plot

Output is a multipage PDF plus PNGs:
  - one PAGE per physics dataset: SUBPLOT per data size D, one fitted curve per
    width nh + solo baseline, plus the untrained-loss "anchor" lines
    (solo: sigma_p^2 ; joint: sigma_p^2 + (mu_p - mu_global)^2) and a dotted
    leftward extrapolation of each fit toward them;
  - ratio-vs-D diagnostic (joint/solo loss vs per-process data, by gauge family);
  - de-levered phase space: alpha vs loss at a FIXED in-range compute C_ref
    (set with --c-ref; default = geomean of all measured compute).
A printed "anchor test" table fits the dual line of each process and reports the
concurrency point (C0, L0), testing the hypothesis L0 ~ Var(ln|A|).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEP_BASE  = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")
SOLO_BASE   = os.path.join(LUSTRE_BASE, "sweeps")
DATA_DIR    = os.path.join(LUSTRE_BASE, "data")

# FLOP model — must stay in sync with analyze_pretraining_scaling.py
NUM_BLOCKS = 8
D_PER_HEAD = 16

# Visual style per num_heads value (joint points coloured by width)
NH_STYLE = {
    2:  {"color": "crimson",      "marker": "v"},
    4:  {"color": "mediumpurple", "marker": "s"},
    8:  {"color": "seagreen",     "marker": "D"},
    16: {"color": "steelblue",    "marker": "o"},
    32: {"color": "darkorange",   "marker": "^"},
}
_DEFAULT_STYLE = {"color": "gray", "marker": "x"}
SOLO_STYLE = {"color": "black", "marker": "*"}

SIGMA_FRAC = 0.01  # assumed relative uncertainty on loss, for chi2 of fits


def _nh_style(nh):
    return NH_STYLE.get(nh, _DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def flops_per_sample(num_heads: int, n_avg: float) -> float:
    """Forward+backward MACs to process ONE event of multiplicity n_avg."""
    d = D_PER_HEAD * num_heads
    L = NUM_BLOCKS
    f_transformer = L * n_avg * (24 * d**2 + 2 * n_avg * d)
    f_framesnet   = n_avg * 131_072
    return 3.0 * (f_framesnet + f_transformer)


# ---------------------------------------------------------------------------
# Dataset bookkeeping
# ---------------------------------------------------------------------------

def solo_tag(full_name: str) -> str:
    """'ee_wwz_255-1000GeV_amplitudes' -> 'eewwz255-1000GeV' (solo dir tag)."""
    return full_name.replace("_amplitudes", "").replace("_", "")


def n_particles(full_name: str, cache: dict) -> int:
    """Particle multiplicity of a dataset, read once from the .npy header."""
    if full_name in cache:
        return cache[full_name]
    path = os.path.join(DATA_DIR, f"{full_name}.npy")
    a = np.load(path, mmap_mode="r")
    npart = (a.shape[1] - 1) // 5
    cache[full_name] = npart
    return npart


def log_amp_stats(full_name: str, n_max: int, cache: dict) -> tuple[float, float]:
    """(mean, std) of ln|A| over the first n_max rows of the dataset (signed-log
    if any amplitude is non-positive, matching resolve_amp_trafos). Cached.
    std**2 is the untrained-MSE / target variance used in the anchor test."""
    key = (full_name, n_max)
    if key in cache:
        return cache[key]
    path = os.path.join(DATA_DIR, f"{full_name}.npy")
    a = np.load(path, mmap_mode="r")
    amp = np.asarray(a[:n_max, -1], dtype=np.float64)
    la = np.log(amp) if amp.min() > 0 else np.sign(amp) * np.log1p(np.abs(amp))
    out = (float(la.mean()), float(la.std()))
    cache[key] = out
    return out


def log_amp_std(full_name: str, n_max: int, cache: dict) -> float:
    return log_amp_stats(full_name, n_max, cache)[1]


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _combined_metric(r: dict) -> float | None:
    """Geometric mean of proc_val_losses_no_reg (the joint selection metric),
    falling back to val_loss. None if unusable."""
    pnr = r.get("proc_val_losses_no_reg")
    if pnr:
        vals = [v for v in pnr.values() if v is not None and v > 0]
        if vals:
            return float(np.exp(np.mean(np.log(vals))))
    v = r.get("val_loss")
    return float(v) if v is not None and v > 0 else None


def best_joint_trial(cell_dir: str) -> dict | None:
    """Return the result dict of the best (lowest combined no-reg loss) trial."""
    rdir = os.path.join(cell_dir, "results")
    if not os.path.isdir(rdir):
        return None
    best, best_m = None, math.inf
    for fn in os.listdir(rdir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(rdir, fn)) as f:
                r = json.load(f)
        except Exception:
            continue
        m = _combined_metric(r)
        if m is not None and m < best_m:
            best_m, best = m, r
    return best


def best_solo_val_loss(cell_dir: str) -> float | None:
    """Lowest val_loss across the solo cell's trials."""
    rdir = os.path.join(cell_dir, "results")
    if not os.path.isdir(rdir):
        return None
    best = math.inf
    for fn in os.listdir(rdir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(rdir, fn)) as f:
                r = json.load(f)
            v = r.get("val_loss")
            if v is not None and v > 0:
                best = min(best, float(v))
        except Exception:
            continue
    return best if math.isfinite(best) else None


def _read_cfg(cell_dir: str) -> dict | None:
    p = os.path.join(cell_dir, "sweep_config.yaml")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gather joint (multi-process) points
# ---------------------------------------------------------------------------

def gather_joint(npart_cache):
    """
    Returns dict: ds_full_name -> list of point dicts
      {nh, c_macs, loss_std, samples, cell}
    pooled so that duplicate (nh, subsample, t) cells (_fast / _002 reruns) are
    merged and only the single best model across the pool is kept per cell.
    """
    # pool[(nh, subsample, t)] = (best_combined_metric, best_result, cell_name)
    pool: dict[tuple, tuple] = {}
    if not os.path.isdir(SWEEP_BASE):
        print(f"!! joint sweep base not found: {SWEEP_BASE}")
        return {}
    for name in sorted(os.listdir(SWEEP_BASE)):
        if not (name.startswith("scaling_p1_") or name.startswith("scaling_p1ext_")):
            continue
        cell_dir = os.path.join(SWEEP_BASE, name)
        if not os.path.isdir(cell_dir):
            continue
        cfg = _read_cfg(cell_dir)
        if not cfg:
            continue
        try:
            fp = cfg["fixed_params"]
            nh  = int(fp["model.net.num_heads"])
            bs  = int(fp["training.batchsize"])
            sub = int(fp["data.subsample"])
            t   = int(cfg["fidelity_schedule"]["t_steps"][-1])
        except Exception:
            continue
        r = best_joint_trial(cell_dir)
        if r is None or not r.get("proc_val_losses_no_reg") or not r.get("compute_ds"):
            continue
        m = _combined_metric(r)
        key = (nh, sub, t)
        if key not in pool or m < pool[key][0]:
            pool[key] = (m, r, name)

    points: dict[str, list] = {}
    for (nh, sub, t), (m, r, cell) in pool.items():
        losses  = r["proc_val_losses_no_reg"]
        samples = r["compute_ds"]
        for ds, loss in losses.items():
            n_s = samples.get(ds)
            if loss is None or loss <= 0 or not n_s:
                continue
            c = n_s * flops_per_sample(nh, n_particles(ds, npart_cache))
            points.setdefault(ds, []).append(
                {"nh": nh, "sub": sub, "t": t, "c_macs": c,
                 "loss_std": float(loss), "samples": int(n_s), "cell": cell})
    return points


# ---------------------------------------------------------------------------
# Gather solo (single-process) points
# ---------------------------------------------------------------------------

def gather_solo(joint_datasets, npart_cache):
    """ds_full_name -> list of {nh, c_macs, loss_std, samples, cell} from
    scaling_solo_full_<tag>_t* cells (best val_loss per cell)."""
    tag2ds = {solo_tag(ds): ds for ds in joint_datasets}
    points: dict[str, list] = {}
    if not os.path.isdir(SOLO_BASE):
        return points
    for name in sorted(os.listdir(SOLO_BASE)):
        if not name.startswith("scaling_solo_full_"):
            continue
        rest = name[len("scaling_solo_full_"):]
        if "_t" not in rest:
            continue
        tag = rest.rsplit("_t", 1)[0]
        ds = tag2ds.get(tag)
        if ds is None:
            continue  # solo dataset not among the 8 joint processes
        cell_dir = os.path.join(SOLO_BASE, name)
        cfg = _read_cfg(cell_dir)
        if not cfg:
            continue
        try:
            fp = cfg["fixed_params"]
            nh = int(fp["model.net.num_heads"])
            bs = int(fp["training.batchsize"])
            t  = int(cfg["fidelity_schedule"]["t_steps"][-1])
        except Exception:
            continue
        v = best_solo_val_loss(cell_dir)
        if v is None:
            continue
        samples = t * bs
        c = samples * flops_per_sample(nh, n_particles(ds, npart_cache))
        points.setdefault(ds, []).append(
            {"nh": nh, "c_macs": c, "loss_std": v,
             "samples": samples, "cell": name})
    return points


# ---------------------------------------------------------------------------
# Power-law fit (pure, log-space OLS) for optional guide lines
# ---------------------------------------------------------------------------

def fit_pure(c_vals, l_vals):
    """L = A * C^-alpha (log-space OLS). Returns (A, alpha) or None."""
    c = np.asarray(c_vals, float); l = np.asarray(l_vals, float)
    if len(c) < 2:
        return None
    X = np.column_stack([np.ones(len(c)), np.log(c)])
    (logA, neg_alpha), *_ = np.linalg.lstsq(X, np.log(l), rcond=None)
    return math.exp(logA), -neg_alpha


def fit_alpha_err(c_vals, l_vals):
    """OLS power-law slope in log10-log10 with its standard error, estimating the
    (single, unknown) noise level from the residual scatter — NO per-point sigma
    assumed.  log10 L = b0 - alpha * log10 C.
        s^2 = RSS/(n-2),   SE(alpha) = s / sqrt(Sum (x_i - xbar)^2)
    Returns (alpha, se_alpha, n) or None (needs n>=3 for a finite SE)."""
    x = np.log10(np.asarray(c_vals, float)); y = np.log10(np.asarray(l_vals, float))
    n = len(x)
    if n < 3:
        return None
    xbar = x.mean()
    Sxx = float(((x - xbar) ** 2).sum())
    if Sxx <= 0:
        return None
    b1 = float(((x - xbar) * (y - y.mean())).sum() / Sxx)
    resid = y - (y.mean() + b1 * (x - xbar))
    s2 = float((resid ** 2).sum() / (n - 2))
    return (-b1, math.sqrt(s2 / Sxx), n)   # alpha, SE(alpha), n


def fit_floor(c_vals, l_vals, n_grid: int = 400):
    """L = A * C^-alpha + L_inf, profiled over an L_inf grid (needs >=4 pts).
    Returns (A, alpha, L_inf) or None."""
    c = np.asarray(c_vals, float); l = np.asarray(l_vals, float)
    n = len(c)
    if n < 4:
        return None
    log_c = np.log(c)
    l_min = float(l.min())
    grid = np.unique(np.concatenate([
        np.linspace(0.0, 0.99 * l_min, n_grid // 2),
        l_min * np.geomspace(1e-4, 0.99, n_grid - n_grid // 2)]))
    best, best_chi2 = None, np.inf
    for li in grid:
        r = l - li
        if (r <= 0).any():
            continue
        w = r / l                       # error-propagation weight
        X = np.column_stack([w, w * log_c])
        (logA, neg_alpha), *_ = np.linalg.lstsq(X, w * np.log(r), rcond=None)
        if neg_alpha >= 0:
            continue
        A, alpha = math.exp(logA), -neg_alpha
        pred = A * c ** (-alpha) + li
        chi2 = float(np.sum(((l - pred) / (SIGMA_FRAC * l)) ** 2))
        if chi2 < best_chi2:
            best_chi2, best = chi2, (A, alpha, li)
    return best


# Map per-dataset subsample (data.subsample) -> total joint dataset size D label.
N_JOINT_DATASETS = 8

def d_label(sub: int) -> str:
    """Per-process data budget. In the joint run each of the 8 processes sees
    only `sub` = D_total/8 samples of itself, so label by the per-process size."""
    tot = sub * N_JOINT_DATASETS
    return f"{sub:g} samples/process   (D_total={tot:g})"


# Gauge-family of each process (for the QED-vs-QCD diagnostic).
FAMILY_STYLE = {
    "QCD": {"color": "crimson",    "members": ("ee_uu_", "ee_uug_", "ee_uugg_")},
    "QED": {"color": "royalblue",  "members": ("ee_aa_", "ee_aaa_")},
    "EW":  {"color": "seagreen",   "members": ("ee_WW_", "ee_wwz_", "ee_ttbar_")},
}

def family(ds: str) -> str:
    for fam, spec in FAMILY_STYLE.items():
        if any(ds.startswith(m) for m in spec["members"]):
            return fam
    return "other"


def fit_series(c, l):
    """Unified power-law fit for one (loss vs compute) series.
    Returns dict {A, alpha, Linf, kind, cmin, cmax} or None.
    Prefers the floor form A*C^-alpha + Linf (>=4 pts), falls back to pure."""
    c = np.asarray(c, float); l = np.asarray(l, float)
    if len(c) < 2:
        return None
    base = {"cmin": float(c.min()), "cmax": float(c.max())}
    fr = fit_floor(c, l)
    if fr is not None:
        A, al, li = fr
        return {**base, "A": A, "alpha": al, "Linf": li, "kind": "floor"}
    fp = fit_pure(c, l)
    if fp is None:
        return None
    A, al = fp
    return {**base, "A": A, "alpha": al, "Linf": 0.0, "kind": "pure"}


def predict(fit: dict, c) -> float:
    return fit["A"] * np.asarray(c, float) ** (-fit["alpha"]) + fit["Linf"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-units", action="store_true",
                    help="Plot reported standardised losses (no std rescaling).")
    ap.add_argument("--no-fit", action="store_true",
                    help="Scatter only; do not draw the per-(D,nh) power-law fits.")
    ap.add_argument("--std-nrows", type=int, default=100_000,
                    help="Rows per dataset used to estimate ln|A| stats (default 1e5).")
    ap.add_argument("--c-ref", type=float, default=None,
                    help="Reference compute (MACs) for the (alpha, loss@C_ref) phase "
                         "plot. Default = geomean of all measured compute (in range).")
    ap.add_argument("--out", default=os.path.join(SWEEP_BASE, "per_dataset_scaling.pdf"))
    ap.add_argument("--dry-run", action="store_true", help="Table only, no plot.")
    args = ap.parse_args()

    npart_cache: dict = {}
    std_cache: dict = {}

    joint = gather_joint(npart_cache)
    if not joint:
        print("No joint points found — aborting.")
        return
    datasets = sorted(joint.keys())
    solo = gather_solo(datasets, npart_cache)

    # ── std rescaling factors + global ln|A| mean (for the anchor test) ──────
    rescale = not args.raw_units
    s_global = None
    mu_global = None
    if rescale:
        pooled = []
        for ds in datasets:
            path = os.path.join(DATA_DIR, f"{ds}.npy")
            a = np.load(path, mmap_mode="r")
            amp = np.asarray(a[:args.std_nrows, -1], dtype=np.float64)
            la = np.log(amp) if amp.min() > 0 else np.sign(amp) * np.log1p(np.abs(amp))
            pooled.append(la)
        pooled = np.concatenate(pooled)
        s_global = float(pooled.std())
        mu_global = float(pooled.mean())

    def y_joint(ds, loss):
        return loss * s_global**2 if rescale else loss

    def y_solo(ds, loss):
        if not rescale:
            return loss
        return loss * log_amp_std(ds, args.std_nrows, std_cache)**2

    # Untrained-loss anchors in physical ln|A| MSE units (the hypothesis):
    #   solo  predicts its own mean  -> L0_solo  = sigma_p^2
    #   joint outputs ~0 on globally-standardised target
    #         -> L0_joint = sigma_p^2 + (mu_p - mu_global)^2
    def anchors(ds):
        if not rescale:
            return None, None
        mu_p, s_p = log_amp_stats(ds, args.std_nrows, std_cache)
        return s_p**2, s_p**2 + (mu_p - mu_global)**2   # (solo, joint)

    ylabel = ("val MSE on ln|A| (physical units)" if rescale
              else "val loss (standardised, as reported)")

    # ── Table ────────────────────────────────────────────────────────────────
    print(f"\n{'dataset':30s} np  source {'D':>8s} {'nh':>3s} {'samples':>10s} "
          f"{'C(TMACs)':>10s} {'loss(std)':>11s} {'loss(plot)':>11s}")
    print("-" * 108)
    for ds in datasets:
        npd = n_particles(ds, npart_cache)
        for pt in sorted(joint[ds], key=lambda p: (p["sub"], p["nh"], p["c_macs"])):
            print(f"{ds:30s} {npd:2d}  joint  {pt['sub']*N_JOINT_DATASETS:8g} "
                  f"{pt['nh']:3d} {pt['samples']:10d} {pt['c_macs']/1e12:10.4f} "
                  f"{pt['loss_std']:11.5e} {y_joint(ds, pt['loss_std']):11.5e}")
        for pt in sorted(solo.get(ds, []), key=lambda p: p["c_macs"]):
            print(f"{'':30s} {npd:2d}  solo   {'100000':>8s} {pt['nh']:3d} "
                  f"{pt['samples']:10d} {pt['c_macs']/1e12:10.4f} "
                  f"{pt['loss_std']:11.5e} {y_solo(ds, pt['loss_std']):11.5e}")
    if rescale:
        print(f"\nGlobal ln|A| std (joint y-rescale): {s_global:.4f}")
        print("Per-dataset solo ln|A| std (solo y-rescale):")
        for ds in datasets:
            print(f"  {ds:30s} {log_amp_std(ds, args.std_nrows, std_cache):.4f}")

    if args.dry_run:
        print("\n[dry-run] no plot written.")
        return

    # ── Plot: one PAGE per physics dataset; within a page one SUBPLOT per data
    #    size D; within a subplot one fitted SERIES per model width nh + solo. ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages

    unit_note = "physical ln|A| MSE" if rescale else "raw standardised loss"
    out_dir = os.path.dirname(args.out)

    # global reference compute (in range) for the (alpha, loss@C_ref) plot, and a
    # floor to extrapolate fits toward the untrained anchor on the scaling pages.
    all_c = ([p["c_macs"] for ds in datasets for p in joint[ds]] +
             [p["c_macs"] for ds in datasets for p in solo.get(ds, [])])
    c_ref = args.c_ref or math.exp(float(np.mean(np.log(all_c))))
    c_floor = min(all_c) / 5.0

    def _draw_series(ax, c, l, style, label, extrap_lo=None):
        """Scatter a series + its power-law fit (floor, else pure).
        Returns (legend handle, fit dict or None)."""
        order = np.argsort(c)
        c = np.asarray(c)[order]; l = np.asarray(l)[order]
        ax.scatter(c, l, color=style["color"], marker=style["marker"],
                   s=44, zorder=4, edgecolors="none")
        fit = None if args.no_fit else fit_series(c, l)
        if fit is not None:
            xx = np.logspace(math.log10(c.min()), math.log10(c.max()), 200)
            ls = "-" if fit["kind"] == "floor" else "--"
            ax.plot(xx, predict(fit, xx), color=style["color"], lw=1.6, ls=ls, zorder=3)
            if extrap_lo is not None and extrap_lo < c.min():
                xe = np.logspace(math.log10(extrap_lo), math.log10(c.min()), 60)
                ax.plot(xe, predict(fit, xe), color=style["color"], lw=1.0,
                        ls=":", alpha=0.45, zorder=2)
            if fit["Linf"] > 0:
                ax.axhline(fit["Linf"], color=style["color"], ls=":", alpha=0.3, zorder=1)
                label += rf"  $\alpha$={fit['alpha']:.2f} $L_\infty$={fit['Linf']:.1e}"
            else:
                label += rf"  $\alpha$={fit['alpha']:.2f}"
        return (Line2D([0], [0], color=style["color"], marker=style["marker"],
                       lw=1.6, label=label), fit)

    # series_fits[ds] = {"joint": [(sub, nh, fit)], "solo": fit,
    #                    "joint_raw": [(sub, nh, c[], y[])], "solo_raw": (c[], y[])}
    series_fits: dict[str, dict] = {}

    pdf_path = args.out
    with PdfPages(pdf_path) as pdf:
        # ── Pages 1..N: per-dataset scaling (panel per D, curve per nh) ───────
        for ds in datasets:
            jp = joint[ds]
            npd = n_particles(ds, npart_cache)
            subs = sorted({p["sub"] for p in jp})
            if not subs:
                continue
            series_fits[ds] = {"joint": [], "solo": None,
                               "joint_raw": [], "solo_raw": None}

            sp = sorted(solo.get(ds, []), key=lambda p: p["c_macs"])
            sx = [p["c_macs"] for p in sp]
            sy = [y_solo(ds, p["loss_std"]) for p in sp]
            if sp:
                series_fits[ds]["solo"] = fit_series(sx, sy)
                series_fits[ds]["solo_raw"] = (np.asarray(sx), np.asarray(sy))

            a_solo, a_joint = anchors(ds)   # untrained-loss anchors (physical), or None

            ncols = min(3, len(subs))
            nrows = math.ceil(len(subs) / ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5.2 * ncols, 4.3 * nrows),
                                     squeeze=False)
            for j, sub in enumerate(subs):
                ax = axes[j // ncols][j % ncols]
                pts = [p for p in jp if p["sub"] == sub]
                handles = []
                for nh in sorted({p["nh"] for p in pts}):
                    series = [p for p in pts if p["nh"] == nh]
                    c = [p["c_macs"] for p in series]
                    l = [y_joint(ds, p["loss_std"]) for p in series]
                    h, fit = _draw_series(ax, c, l, _nh_style(nh), f"nh={nh}",
                                          extrap_lo=c_floor)
                    handles.append(h)
                    if fit is not None:
                        series_fits[ds]["joint"].append((sub, nh, fit))
                    series_fits[ds]["joint_raw"].append(
                        (sub, nh, np.asarray(c), np.asarray(l)))
                if sp:
                    h, _ = _draw_series(ax, sx, sy, SOLO_STYLE,
                                        "solo (nh=4, 100k samples/process)",
                                        extrap_lo=c_floor)
                    handles.append(h)
                # untrained-loss anchors (the "meet at the variance" hypothesis)
                if a_solo is not None:
                    ax.axhline(a_joint, color="dimgray", ls="--", lw=1.1, alpha=0.7)
                    ax.axhline(a_solo, color="black", ls="-.", lw=1.1, alpha=0.7)
                    handles.append(Line2D([0], [0], color="dimgray", ls="--",
                                          label=r"joint anchor $\sigma_p^2+\Delta\mu^2$"))
                    handles.append(Line2D([0], [0], color="black", ls="-.",
                                          label=r"solo anchor $\sigma_p^2$"))
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(d_label(sub), fontsize=10)
                ax.set_xlabel("compute on this dataset  [MACs]", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, which="both", alpha=0.15)
                ax.legend(handles=handles, fontsize=7, loc="best")
            for j in range(len(subs), nrows * ncols):
                axes[j // ncols][j % ncols].set_visible(False)

            fig.suptitle(
                f"{ds.replace('_amplitudes','')}  (n_particles={npd})   "
                f"joint pretrain (best model / cell) vs solo\n"
                f"y = {unit_note}   |   x = samples-seen x flops/sample(nh, n)   "
                f"|   panel = per-process data D/8, curve = width nh   "
                f"(solo saw the full 100k of this process)",
                fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=150)
            png = os.path.join(out_dir, f"per_dataset_scaling_{solo_tag(ds)}.png")
            fig.savefig(png, dpi=150)
            plt.close(fig)

        # ── Diagnostic page A: joint/solo loss ratio vs per-process data ──────
        #    starvation -> ratio -> 1 as D/8 grows;  interference -> stays >1.
        fig, ax = plt.subplots(figsize=(9, 6.5))
        fam_in_legend = set()
        for ds in datasets:
            solo_best = min((y_solo(ds, p["loss_std"]) for p in solo.get(ds, [])),
                            default=None)
            if solo_best is None or solo_best <= 0:
                continue
            jp = joint[ds]
            subs = sorted({p["sub"] for p in jp})
            xs, ys = [], []
            for sub in subs:
                jb = min(y_joint(ds, p["loss_std"]) for p in jp if p["sub"] == sub)
                xs.append(sub); ys.append(jb / solo_best)
            fam = family(ds)
            col = FAMILY_STYLE.get(fam, {"color": "gray"})["color"]
            lbl = fam if fam not in fam_in_legend else None
            fam_in_legend.add(fam)
            ax.plot(xs, ys, marker="o", color=col, lw=1.6, alpha=0.9,
                    label=lbl)
            ax.annotate(ds.replace("_amplitudes", "").replace("ee_", ""),
                        (xs[-1], ys[-1]), fontsize=7, color=col,
                        xytext=(4, 0), textcoords="offset points", va="center")
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.7)
        ax.text(ax.get_xlim()[0], 1.02, "joint = solo", fontsize=8, va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("per-process data budget  D/8  [samples of this process]")
        ax.set_ylabel("best joint loss / best solo loss   (>1 = joint worse)")
        ax.set_title("Joint vs solo per-process: does the gap close with data?\n"
                     "(QCD = uu/uug/uugg, QED = aa/aaa, EW = WW/wwz/ttbar; "
                     "y is physical ln|A| MSE)" if rescale else
                     "Joint vs solo per-process loss ratio", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(title="gauge family", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        fig.savefig(os.path.join(out_dir, "per_dataset_ratio_vs_D.png"), dpi=150)
        plt.close(fig)

        # ── Diagnostic page B: de-levered phase space (alpha, loss@C_ref) ─────
        #    y is loss at a FIXED in-range compute, not at C=1, so a same-slope
        #    different-offset (nh) spread shows at full size instead of being
        #    crushed by the C=1 extrapolation. lower-right = better.
        if not args.no_fit:
            n = len(datasets); ncols = 4; nrows = math.ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows),
                                     squeeze=False)
            for idx, ds in enumerate(datasets):
                ax = axes[idx // ncols][idx % ncols]
                jf = series_fits.get(ds, {}).get("joint", [])
                sf = series_fits.get(ds, {}).get("solo")
                if not jf:
                    ax.set_visible(False); continue

                solo_y = predict(sf, c_ref) if sf else None
                solo_a = sf["alpha"] if sf else None
                smin = min(s for s, _, _ in jf)
                for sub, nh, f in jf:
                    st = _nh_style(nh)
                    size = 40 + 35 * math.log10(max(sub / smin, 1) + 1)
                    steeper = (solo_a is not None and f["alpha"] > solo_a)
                    ax.scatter(f["alpha"], predict(f, c_ref), s=size, marker=st["marker"],
                               facecolors=st["color"] if steeper else "none",
                               edgecolors=st["color"], linewidths=1.4, zorder=4)
                if sf is not None:
                    ax.axvline(solo_a, color="black", ls="--", lw=1.3, alpha=0.7)
                    ax.axhline(solo_y, color="black", ls=":", lw=1, alpha=0.3)
                    ax.scatter(solo_a, solo_y, s=240, marker="*",
                               color="black", zorder=6)

                ax.set_yscale("log")
                ax.set_xlabel(r"$\alpha$  (slope; bigger = better)", fontsize=8)
                ax.set_ylabel(f"loss @ C_ref={c_ref:.1e}  (lower = better)", fontsize=8)
                ax.set_title(f"{ds.replace('_amplitudes','').replace('ee_','')}  "
                             f"[{family(ds)}]", fontsize=9)
                ax.grid(True, which="both", alpha=0.15)
            for idx in range(n, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            nh_present = sorted({nh for hs in series_fits.values()
                                 for _, nh, _ in hs.get("joint", [])})
            handles = [Line2D([0], [0], marker=_nh_style(nh)["marker"], color="w",
                              markerfacecolor=_nh_style(nh)["color"], markersize=9,
                              label=f"joint nh={nh}") for nh in nh_present]
            handles += [Line2D([0], [0], marker="*", color="black", lw=0,
                               markersize=13, label="solo fit"),
                        Line2D([0], [0], color="black", ls="--", label=r"solo slope $\alpha$")]
            fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                       fontsize=9, frameon=False)
            fig.suptitle(
                r"De-levered phase space: $\alpha$ vs loss at fixed "
                rf"$C_{{ref}}={c_ref:.2e}$ MACs  —  filled = steeper slope than solo "
                r"($\alpha>\alpha_{solo}$, right of dashed line)", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            fig.savefig(os.path.join(out_dir, "per_dataset_alpha_lossCref.png"), dpi=150)
            plt.close(fig)

        # ── Page C: exponent alpha vs data, with error bars ──────────────────
        #    OLS slope + SE(alpha) from residual scatter (no per-point sigma).
        #    Per process: joint alpha(nh, D) +/- SE (coloured by nh, nh=4 is the
        #    like-for-like vs solo), and the solo alpha +/- SE as a grey band.
        n = len(datasets); ncols = 4; nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows),
                                 squeeze=False)
        alpha_tab = []   # (ds, nh, Dtot, alpha, se, n)
        for idx, ds in enumerate(datasets):
            ax = axes[idx // ncols][idx % ncols]
            raws = series_fits.get(ds, {}).get("joint_raw", [])
            # solo band
            sr = series_fits.get(ds, {}).get("solo_raw")
            if sr is not None:
                fe = fit_alpha_err(sr[0], sr[1])
                if fe is not None:
                    a_s, se_s, _ = fe
                    ax.axhspan(a_s - se_s, a_s + se_s, color="black", alpha=0.12, zorder=0)
                    ax.axhline(a_s, color="black", ls="--", lw=1.2, alpha=0.8)
                    alpha_tab.append((ds, "solo", 100000, a_s, se_s, len(sr[0])))
            # joint points by nh, x = per-process D = sub
            for sub, nh, c, y in raws:
                fe = fit_alpha_err(c, y)
                if fe is None:
                    continue
                a_j, se_j, nn = fe
                st = _nh_style(nh)
                ax.errorbar(sub, a_j, yerr=se_j, marker=st["marker"], ms=6,
                            color=st["color"], ecolor=st["color"], capsize=3,
                            lw=0, elinewidth=1.2, zorder=3)
                alpha_tab.append((ds, f"nh{nh}", sub * N_JOINT_DATASETS, a_j, se_j, nn))
            ax.set_xscale("log")
            ax.set_xlabel("per-process data D/8  [samples]", fontsize=8)
            ax.set_ylabel(r"scaling exponent $\alpha$", fontsize=8)
            ax.set_title(f"{ds.replace('_amplitudes','').replace('ee_','')}  "
                         f"[{family(ds)}]", fontsize=9)
            ax.grid(True, which="both", alpha=0.15)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        nh_present = sorted({nh for hs in series_fits.values()
                             for _, nh, _ in hs.get("joint", [])})
        handles = [Line2D([0], [0], marker=_nh_style(nh)["marker"], color=_nh_style(nh)["color"],
                          lw=0, markersize=7, label=f"joint nh={nh}") for nh in nh_present]
        handles.append(Line2D([0], [0], color="black", ls="--", label=r"solo $\alpha\pm$SE"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False)
        fig.suptitle(r"Scaling exponent $\alpha$ vs per-process data, with SE "
                     r"(OLS slope error from residuals) — joint per-$nh$ vs solo band",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        fig.savefig(os.path.join(out_dir, "per_dataset_alpha_vs_D.png"), dpi=150)
        plt.close(fig)

    # ── alpha table: joint nh=4 vs solo, in sigma units ──────────────────────
    print("\nExponent comparison (alpha +/- SE; nh=4 is the like-for-like vs solo):")
    print(f"{'process':26s} {'series':7s} {'D_total':>9s} {'alpha':>7s} {'SE':>6s} "
          f"{'n':>3s} {'(a-a_solo)/sigma':>16s}")
    print("-" * 80)
    for ds in datasets:
        rows = [r for r in alpha_tab if r[0] == ds]
        solo = next((r for r in rows if r[1] == "solo"), None)
        a_s, se_s = (solo[3], solo[4]) if solo else (None, None)
        for r in sorted(rows, key=lambda r: (r[1] != "solo", r[1], r[2])):
            _, ser, Dt, a, se, nn = r
            if ser == "solo" or a_s is None:
                z = ""
            else:
                z = f"{(a - a_s) / math.hypot(se, se_s):+.1f}"
            print(f"{ds.replace('_amplitudes',''):26s} {ser:7s} {Dt:9g} "
                  f"{a:7.3f} {se:6.3f} {nn:3d} {z:>16s}")

    # ── Meeting test: do the per-process curves actually concur at one C? ─────
    #    For every series (joint (D,nh) + solo) of a process we take its pure
    #    power-law dual point (alpha_i, log10 A_i).  Then we MEASURE, not assume:
    #      RMS_resid[dex] : RMS residual of log10 A_i about the best-fit dual line
    #                       (small => the points really are collinear => a real
    #                        common crossing; large => it's a loose cloud).
    #      C0(MACs)       : 10^slope of that dual line (the implied crossing).
    #      cross med/spread : median and 16-84%ile half-spread (in dex) of the
    #                       ACTUAL pairwise crossing computes  x = (a_i-a_j)/(al_i-al_j)
    #                       (tight => they meet at one compute; broad => they don't).
    #      data range     : where the series' data actually live, to see whether
    #                       the crossing is inside the plotted range or extrapolated.
    print("\nMeeting test — do each process's curves really cross at one compute?")
    print("(dual points = pure-law (alpha, log10 A) of all joint+solo series)")
    print(f"{'process':26s} {'n':>3s} {'data C range':>20s} {'C0(MACs)':>9s} "
          f"{'RMS[dex]':>8s} {'cross_med':>9s} {'cross±[dex]':>11s} {'inrange':>7s}")
    print("-" * 104)
    for ds in datasets:
        duals, cmins, cmaxs = [], [], []
        raws = list(series_fits.get(ds, {}).get("joint_raw", []))
        sr = series_fits.get(ds, {}).get("solo_raw")
        if sr is not None:
            raws.append((None, None, sr[0], sr[1]))
        for _, _, c, y in raws:
            fp = fit_pure(c, y)
            if fp is not None:
                A, al = fp
                duals.append((al, math.log10(A)))
                cmins.append(float(np.min(c))); cmaxs.append(float(np.max(c)))
        nm = ds.replace("_amplitudes", "")
        if len(duals) < 3:
            print(f"{nm:26s} {len(duals):3d}   (too few series)")
            continue
        al = np.array([d[0] for d in duals]); la = np.array([d[1] for d in duals])
        m, k = np.polyfit(al, la, 1)                     # log10 A = m*alpha + k
        rms = float(np.sqrt(np.mean((la - (m * al + k)) ** 2)))   # collinearity [dex]
        C0 = 10 ** m
        # all pairwise crossing computes (skip near-parallel pairs)
        xs = []
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                if abs(al[i] - al[j]) > 0.03:
                    xs.append((la[i] - la[j]) / (al[i] - al[j]))    # = log10 C_cross
        c_lo, c_hi = min(cmins), max(cmaxs)
        if xs:
            xs = np.array(xs)
            med = float(np.median(xs))
            spread = float((np.percentile(xs, 84) - np.percentile(xs, 16)) / 2)
            inrange = "yes" if c_lo <= 10 ** med <= c_hi else "no"
            print(f"{nm:26s} {len(duals):3d} {c_lo:8.1e}..{c_hi:8.1e} {C0:9.1e} "
                  f"{rms:8.3f} {10**med:9.1e} {spread:11.2f} {inrange:>7s}")
        else:
            print(f"{nm:26s} {len(duals):3d} {c_lo:8.1e}..{c_hi:8.1e} {C0:9.1e} "
                  f"{rms:8.3f}   (all ~parallel)")
    print("\nRead: RMS[dex] << 1 and cross±[dex] small  => curves genuinely meet at")
    print("one compute (cross_med); large spread => they do NOT meet, the 'line' is loose.")

    print(f"\nWrote multipage PDF:\n  {pdf_path}")
    print(f"PNGs in: {out_dir}/")
    print("  per_dataset_scaling_<tag>.png    (scaling + anchor lines + extrapolation)")
    print("  per_dataset_ratio_vs_D.png       (joint/solo ratio vs per-process data)")
    print("  per_dataset_alpha_lossCref.png   (de-levered phase: alpha vs loss@C_ref)")
    print("  per_dataset_alpha_vs_D.png       (alpha +/- SE vs data; joint nh vs solo)")


if __name__ == "__main__":
    main()
