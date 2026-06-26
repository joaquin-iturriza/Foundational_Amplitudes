#!/usr/bin/env python3
"""
compare_finetune_methods.py — Overlay the scaling laws of several finetune sweeps.

Reads each sweep's saved outer config (sweeps/<name>/sweep_config.yaml) and plots ALL
methods together — one panel per dataset — comparing standard vs LoRA vs EWC vs
reset-head vs freeze finetuning. Optionally overlays the matched-architecture solo
reference line (nh=8 anchor point + nh=4 slope), the same one fit_scaling_law draws.

Writes THREE figures into --out-dir:
  finetune_methods_comparison.{pdf,png}        — original x = full-model FLOPs, which
        charges EVERY method the SAME per-step cost (misleading: it gives LoRA/freeze
        zero credit for being cheaper). Kept for reference; axis relabelled honestly.
  finetune_methods_comparison_flops.{pdf,png}  — (a) x = METHOD-AWARE FLOPs: LoRA ≈0.65×,
        freeze ≈0.76–0.92× (per the best trial's actual freeze level), EWC + a one-time
        Fisher pre-pass. Best trial + its swept HPs read from each cell's summary.txt.
  finetune_methods_comparison_params.{pdf,png} — (b) x = TRAINABLE params (∝ Adam
        optimizer-state bytes): best val_loss vs #trained params per method — the axis
        where LoRA actually wins. Needs the pretrained checkpoint (run on Jean Zay).
  finetune_methods_comparison_overlay.{pdf,png} — old (×) vs corrected (●) compute with
        the shift connector + solo line, so the per-method left/right move is explicit.
  finetune_methods_comparison_walltime.{pdf,png} — x = MEASURED training wall time
        (traintime_hours of the best trial), no FLOP model at all. NB EWC's Fisher
        pre-pass runs in setup and is NOT in traintime_hours, so EWC's wall time
        excludes it.

Usage:
    python sweep/compare_finetune_methods.py \\
        --configs sweeps/finetune_scaling_virt_002/sweep_config.yaml \\
                  sweeps/finetune_lora_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_ewc_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_resethead_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_freeze_scaling_virt/sweep_config.yaml \\
        --compare-anchor  sweeps/scaling_solo_nh8_anchor_virt_002/sweep_config.yaml \\
        --compare-anchor2 sweeps/scaling_solo_nh8_anchor2_virt/sweep_config.yaml \\
        --compare-slope   sweeps/scaling_solo_full/sweep_config.yaml \\
        --out-dir sweeps/finetune_method_comparison

The solo reference line needs the matched-architecture nh=8 solo. With a single
--compare-anchor it pins one nh=8 point and borrows the slope from --compare-slope
(an nh=4 scan). Passing a second nh=8 anchor at a different fidelity via
--compare-anchor2 instead fits the solo slope DIRECTLY from the two nh=8 points (and
the wall-time figure fits its own slope from the two MEASURED wall times, which is
not ∝ compute across fidelities since the low-t anchor is overhead-dominated).
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import yaml
from scipy.optimize import curve_fit

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)
from sweep.fit_scaling_law import (collect_best_from_config, fit_power_law,
                                   build_solo_reference, _ds_tag, _n_particles,
                                   _sweep_dirs_for, load_results_from_dir)
from sweep.generate_pretraining_scaling_sweeps import flops_per_step
from sweep.finetune_compute import (parse_cell_summary, method_total_flops,
                                    method_trainable_params, load_param_breakdown)


def _method_label(sweep_name):
    """standard / lora / ewc / resethead / freeze from the sweep name."""
    n = sweep_name.replace("finetune_", "").replace("scaling_virt", "").strip("_")
    n = re.sub(r"_?\d+$", "", n).strip("_")     # drop trailing _002 etc.
    return n or "standard"


# y-axis = the mean-squared error of the standardized amplitude prediction, on the
# test or validation set (the same MSE bracket, only the ensemble subscript differs).
def _ylabel(metric):
    return (r"$\left\langle\left(\hat{A}_\mathrm{NN}-\hat{A}_\mathrm{true}\right)^2"
            rf"\right\rangle_\mathrm{{{metric}}}$")
YLABEL = _ylabel("test")

# Pretty dataset titles: map the raw file-stem tokens to LaTeX physics.
_PARTICLE_TEX = {
    "ee": r"e^+e^-", "uu": r"u\bar u", "ddbar": r"d\bar d", "ttbar": r"t\bar t",
    "bbbar": r"b\bar b", "ccbar": r"c\bar c", "ww": r"W^+W^-", "zz": r"ZZ",
    "gg": r"gg", "tt": r"t\bar t", "uubar": r"u\bar u",
}
_ORDER_TEX = {"lo": "LO", "nlo": "NLO", "virt": "virtual", "born": "Born",
              "full": "full", "ratio": "ratio"}


def _pretty_ds(ds):
    """Turn a raw dataset stem like 'ee_uu_nlo_virt_e4' into a LaTeX-y title
    '$e^+e^- \\to u\\bar u$  (NLO virtual)'.  The 'e4' size tag is ignored."""
    name = re.sub(r"_amplitudes$", "", ds)
    name = re.sub(r"_e4$", "", name)            # drop the dataset-size tag
    parts, orders = [], []
    for t in name.split("_"):
        if t in _PARTICLE_TEX:
            parts.append(_PARTICLE_TEX[t])
        elif t:
            orders.append(_ORDER_TEX.get(t, t))
    if parts:
        proc = parts[0] + (r" \to " + " ".join(parts[1:]) if len(parts) > 1 else "")
        title = rf"${proc}$"
    else:
        title = name
    if orders:
        title += "  (" + " ".join(orders) + ")"
    return title


def collect_corrected(cfg, sweep_name, method_key, breakdown=None):
    """Per-(dataset) list of best-trial points with METHOD-AWARE cost, read from each
    cell's summary.txt (best val + the best trial's swept HPs). Returns
        { dataset: [ {t, val, flops, params?, hp}, ... ] }
    flops = method_total_flops(...) (honest per-step × t_steps, + EWC Fisher pre-pass);
    params (if a checkpoint breakdown is given) = method_trainable_params(...)."""
    fp = cfg.get("fixed_params", {})
    nh = int(fp.get("model.net.num_heads", 8))
    bs = int(fp.get("training.batchsize", 512))
    n_fisher = int(fp.get("fine_tune.ewc.n_fisher_batches", 64))
    out = {}
    for ds in cfg["datasets"]:
        n_avg = _n_particles(cfg, ds)
        ds_tag = _ds_tag(ds)
        for t in cfg["t_steps_values"]:
            cell = f"{sweep_name}_{ds_tag}_t{t:05d}"
            _, eos = _sweep_dirs_for(cfg, cell)
            # val from the result JSONs (FULL precision — summary.txt rounds to 6 dp,
            # which collapses saturated points to 0.0 and breaks the log fit). HPs come
            # from summary.txt (the result JSONs don't store them).
            res = load_results_from_dir(os.path.join(eos, "results"))
            pos = [r for r in res
                   if r.get("val_loss") is not None and float(r["val_loss"]) > 0]
            s = parse_cell_summary(eos)
            # model-selection on val_loss, but REPORT the matched test_loss.
            best = min(pos, key=lambda r: float(r["val_loss"])) if pos else None
            val = float(best["val_loss"]) if best else (s["best_val"] if s else None)
            test = (float(best["test_loss"]) if best and best.get("test_loss") is not None
                    else None)
            if val is None or val <= 0:          # no usable (positive) val for this cell
                continue
            if test is None or test <= 0:        # fall back to val if no test recorded
                test = val
            hp = {"freeze_blocks": (s or {}).get("freeze_blocks", []),
                  "lora_rank": (s or {}).get("lora_rank") or 8}
            rec = {"t": t, "val": val, "test": test, "hp": hp,
                   "flops": method_total_flops(method_key, nh, n_avg, bs, t, hp,
                                               n_fisher=n_fisher),
                   # old method-agnostic cost (= the previous plot's x): full 3×fwd × t,
                   # exactly what flops_per_step charged every method.
                   "flops_naive": method_total_flops("standard", nh, n_avg, bs, t, hp)}
            if breakdown is not None:
                rec["params"] = method_trainable_params(method_key, hp, breakdown)
            out.setdefault(ds, []).append(rec)
    return out


def _keep_ds(ds):
    """Drop the virt/born RATIO datasets — only plot the absolute-virt (e4) ones."""
    return "ratio" not in ds


# Reference (solo) line style — black dashed, distinct from the gray methods.
SOLO_STYLE = {"color": "black", "ls": "--", "marker": "s", "lw": 1.4, "alpha": 0.9, "z": 4}
# Assumed per-point uncertainty on the solo loss: ±10%, symmetric in log space (no
# repeat-seed measurement, so this is an assumption). Used for error bars and chi2/ndf.
SOLO_LOG_SIGMA = math.log(1.10)


def _solo_scatter(ax, pts, fitted=True):
    """Scatter solo points with ±10% (log-symmetric) y error bars. fitted=True → filled
    squares (in the fit); fitted=False → open squares (plotted but excluded from fit)."""
    if not pts:
        return
    x = [p[0] for p in pts]; y = np.array([p[1] for p in pts], dtype=float)
    f = math.exp(SOLO_LOG_SIGMA)                          # 1.10
    yerr = np.vstack([y - y / f, y * f - y])              # lower, upper (asymmetric in linear)
    face = SOLO_STYLE["color"] if fitted else "none"
    ax.errorbar(x, y, yerr=yerr, fmt=SOLO_STYLE["marker"], mfc=face,
                mec=SOLO_STYLE["color"], ecolor=SOLO_STYLE["color"],
                ms=5, lw=0, elinewidth=0.8, capsize=2, zorder=SOLO_STYLE["z"],
                label=None if fitted else "solo (excluded from fit)")
# New (second) nh=8 solo anchor, overlaid in 'borrow' mode as an open star so it's
# clearly NOT one of the points the line was fit through.
NEW_SOLO_STYLE = {"edgecolor": "black", "facecolor": "white", "marker": "*",
                  "s": 200, "lw": 1.4, "z": 5}


def build_method_styles(label_keys):
    """standard → a bold colour that stands out; every other method → a gray shade,
    semi-transparent, but mutually distinguishable via (shade, linestyle, marker).
    `label_keys` is the ordered list of (display_label, method_key)."""
    grays = [
        ("0.25", (0, (5, 1)),          "s"),   # dark gray, dashed
        ("0.50", (0, (1, 1)),          "^"),   # mid gray,  dotted
        ("0.62", (0, (3, 1, 1, 1)),    "D"),   # light gray, dash-dot
        ("0.38", (0, (5, 1, 1, 1, 1, 1)), "v"),
    ]
    styles, gi = {}, 0
    for label, key in label_keys:
        if key == "standard":
            styles[label] = {"color": "crimson", "alpha": 1.0, "lw": 2.2,
                             "ls": "-", "marker": "o", "z": 6}
        else:
            c, ls, mk = grays[gi % len(grays)]; gi += 1
            styles[label] = {"color": c, "alpha": 0.6, "lw": 1.3,
                             "ls": ls, "marker": mk, "z": 3}
    return styles


def _fit_pos(xs, ys):
    """fit_power_law on only the strictly-positive (x, y) pairs (log fit can't take
    a 0/neg val_loss — saturated points hit the floor). Returns (A, alpha) or None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return None
    A, alpha, _ = fit_power_law([p[0] for p in pairs], [p[1] for p in pairs])
    return A, alpha


def _solo_curve_y(x, ref_tuple):
    """Evaluate the solo model L = A·x^(-alpha) + C. ref_tuple = (A, alpha, C[, chi2])."""
    A, alpha, C = ref_tuple[:3]
    return A * np.asarray(x) ** (-alpha) + C


def _solo_label(ref_tuple, prefix="solo"):
    A, alpha, C = ref_tuple[:3]
    chi2 = ref_tuple[3] if len(ref_tuple) > 3 else float("nan")
    floor = rf", floor$={C:.1e}$" if C > 0 else ""
    gof = rf", $\chi^2$/ndf$={chi2:.1f}$" if chi2 == chi2 else ""   # NaN check
    return rf"{prefix} ($\alpha$={alpha:.2f}{floor}{gof})"


def fit_saturating(xs, ys):
    """Fit the SATURATING power law  L = A·x^(-alpha) + C  (an irreducible floor C, not
    a straight line in log-log) by nonlinear least squares on log L, so the fit is
    decade-balanced and alpha is a genuine exponent rather than a log-linear average.
    Needs >=4 positive points. Returns (A, alpha, C, chi2_ndf) or None on failure.
    chi2_ndf is the reduced chi^2 of the log-residuals against an assumed per-point
    uncertainty SOLO_LOG_SIGMA (±10%, symmetric in log) — the right goodness measure for
    a nonlinear fit (R^2 is not). ndf = n_points - 3 (A, alpha, C)."""
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    lx, ly = np.log(x), np.log(y)

    def f(lx, logA, alpha, logC):
        return np.logaddexp(logA - alpha * lx, logC)      # log(A·x^-alpha + C)

    p0 = [float(ly[0] + 2 * lx[0]), 2.0, float(np.log(y.min() * 0.8))]
    try:
        popt, _ = curve_fit(f, lx, ly, p0=p0, maxfev=200000)
    except Exception:
        return None
    A, alpha, C = math.exp(popt[0]), float(popt[1]), math.exp(popt[2])
    if not (alpha > 0 and np.isfinite(A) and np.isfinite(C)):
        return None
    ndf = len(pts) - 3
    chi2_ndf = (float(np.sum(((ly - f(lx, *popt)) / SOLO_LOG_SIGMA) ** 2)) / ndf
                if ndf > 0 else float("nan"))
    return A, alpha, C, chi2_ndf


def build_solo_reference_two_anchors(anchor_config_path, anchor2_config_path):
    """Matched-architecture solo reference from TWO measured nh=8 solo anchors at
    different fidelities (e.g. t=126 and t=8000), so the slope is fit directly from
    nh=8 data instead of borrowed from an nh=4 sweep.

    Returns { dataset: (A, alpha, C=0.0) } — a pure power law (only 2 points, so no
    floor term can be fit). Datasets present in only one anchor are skipped."""
    with open(anchor_config_path) as f:
        a1 = yaml.safe_load(f)
    with open(anchor2_config_path) as f:
        a2 = yaml.safe_load(f)
    best1 = collect_best_from_config(a1, a1["sweep_name"])   # {ds:{flops:val}} (nh8)
    best2 = collect_best_from_config(a2, a2["sweep_name"])   # {ds:{flops:val}} (nh8)

    ref = {}
    for ds in set(best1) & set(best2):
        pts = {**best1[ds], **best2[ds]}          # merge the (flops -> val) points
        if len(pts) < 2:
            print(f"  [warn] two-anchor solo: <2 distinct compute points for {ds}, skipping")
            continue
        cs = sorted(pts); vs = [pts[c] for c in cs]
        A, alpha_solo, _ = fit_power_law(cs, vs)
        ref[ds] = (A, alpha_solo, 0.0, float("nan"))
        print(f"  {ds}: solo slope from 2 nh=8 anchors  alpha={alpha_solo:.3f}  "
              f"(c={cs[0]:.2e}->{cs[-1]:.2e})")
    return ref


def collect_best_test_from_config(cfg, sweep_name):
    """{ds: {flops: test_loss}} where, per cell, the best trial is selected on val_loss
    but the TEST loss is reported — the same model-selection the finetune method points
    use (collect_corrected). Reporting best-val instead (collect_best_from_config) is
    biased low and noisy (1% val set, min over trials) and is NOT comparable to the
    method points, which are test. Falls back to val if a trial has no test_loss."""
    fp = cfg.get("fixed_params", {})
    nh = int(fp.get("model.net.num_heads", 8))
    bs = int(fp.get("training.batchsize", 512))
    out = {}
    for ds in cfg["datasets"]:
        n_avg = _n_particles(cfg, ds)
        fps = flops_per_step(nh, n_avg, bs)
        ds_tag = _ds_tag(ds)
        for t in cfg["t_steps_values"]:
            cell = f"{sweep_name}_{ds_tag}_t{t:05d}"
            _, eos = _sweep_dirs_for(cfg, cell)
            res = load_results_from_dir(os.path.join(eos, "results"))
            pos = [r for r in res if r.get("val_loss") is not None and float(r["val_loss"]) > 0]
            if not pos:
                continue
            best = min(pos, key=lambda r: float(r["val_loss"]))
            test = best.get("test_loss")
            test = float(test) if test is not None and float(test) > 0 else float(best["val_loss"])
            out.setdefault(ds, {})[fps * t] = test
    return out


def _merge_best(config_paths):
    """{ds: {flops: test_loss_of_best_val_trial}} merged across solo sweep configs."""
    merged = {}
    for p in config_paths or []:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        for ds, cell in collect_best_test_from_config(cfg, cfg["sweep_name"]).items():
            merged.setdefault(ds, {}).update(cell)
    return merged


def build_solo_curve(config_paths, plot_only_paths=None):
    """Matched-architecture nh=8 solo reference: a PURE power law L = A·c^-alpha fit
    through ALL measured solo cells — the same model `fit_scaling_law` used for the
    nh=4 scaling_solo_full reference (no floor term; these curves are clean power laws,
    r^2~0.95, over the measured range). chi2/ndf uses the assumed ±10% log uncertainty.

    Returns (ref, points, points_excluded):
      ref    = { ds: (A, alpha, 0.0, chi2_ndf) }
      points = { ds: [(flops, loss), ...] }  ;  points_excluded = {} (everything is fit)."""
    merged = _merge_best(config_paths)
    for ds, cell in _merge_best(plot_only_paths).items():
        merged.setdefault(ds, {}).update(cell)        # fold any plot-only cells into the fit
    ref, points = {}, {}
    for ds, pts in merged.items():
        if len(pts) < 2:
            print(f"  [warn] solo curve: <2 compute points for {ds}, skipping")
            continue
        cs = sorted(pts); vs = [pts[c] for c in cs]
        A, alpha, r2 = fit_power_law(cs, vs)
        lc = np.log(cs); lv = np.log(vs)
        ndf = len(cs) - 2
        chi2 = (float(np.sum(((lv - (math.log(A) - alpha * lc)) / SOLO_LOG_SIGMA) ** 2)) / ndf
                if ndf > 0 else float("nan"))
        ref[ds] = (A, alpha, 0.0, chi2)
        points[ds] = [(c, pts[c]) for c in cs]
        print(f"  {ds}: solo pure power law from {len(cs)} cells  alpha={alpha:.3f}  "
              f"r2={r2:.3f}  chi2/ndf={chi2:.2f}  (c={cs[0]:.2e}->{cs[-1]:.2e})")
    return ref, points, {}


def collect_walltime(cfg, sweep_name, metric="test"):
    """Per-(dataset) [(traintime_hours, loss, t_steps)] for the best trial of each
    cell. traintime_hours is the trial's own training wall-clock (base_experiment:
    dt since training_start_time) — here ≈ proportional to t_steps (from-scratch off
    the pretrained model), so it's an honest wall-time-to-reach-this-loss axis.
    Always selects the best trial on val_loss; reports `metric` ('test' or 'val')
    as the y value (test falls back to val if no test_loss was recorded)."""
    out = {}
    for ds in cfg["datasets"]:
        ds_tag = _ds_tag(ds)
        for t in cfg["t_steps_values"]:
            cell = f"{sweep_name}_{ds_tag}_t{t:05d}"
            _, eos = _sweep_dirs_for(cfg, cell)
            res = load_results_from_dir(os.path.join(eos, "results"))
            if not res:
                continue
            best = min(res, key=lambda r: r.get("val_loss", float("inf")))
            tt = best.get("traintime_hours")
            v = best.get("val_loss") if metric == "val" else best.get("test_loss")
            if v is None or (isinstance(v, float) and v <= 0):
                v = best.get("val_loss")          # fall back to val if no test recorded
            if tt is None or v is None or tt <= 0 or v <= 0:
                continue
            out.setdefault(ds, []).append((float(tt), float(v), t))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="saved outer sweep_config.yaml of each finetune sweep")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="method labels (default: derived from each sweep_name)")
    ap.add_argument("--compare-anchor", default=None)
    ap.add_argument("--compare-anchor2", default=None,
                    help="second nh=8 solo anchor sweep (different t_steps). When given, "
                         "the solo slope is fit DIRECTLY from the two nh=8 anchors instead "
                         "of borrowed from --compare-slope (nh=4).")
    ap.add_argument("--solo-configs", nargs="+", default=None,
                    help="one or more nh=8 solo sweep outer configs (anchors + curve). When "
                         "given, the solo line is a single power law fit through ALL their "
                         "measured cells, and every solo point is scattered. Overrides "
                         "--compare-anchor/--compare-anchor2/--solo-mode.")
    ap.add_argument("--solo-plot-only-configs", nargs="+", default=None,
                    help="solo sweep configs whose points are PLOTTED (open squares) but "
                         "EXCLUDED from the fit + chi2 — e.g. the near-init low-t plateau "
                         "that the power-law+floor model does not describe.")
    ap.add_argument("--compare-slope", default=None)
    ap.add_argument("--solo-mode", choices=("fit", "borrow"), default="fit",
                    help="how to build the solo line when --compare-anchor2 is given. "
                         "'fit' (default): slope fit directly from the two nh=8 anchors. "
                         "'borrow': keep the line on the OLD anchor + nh=4 slope and just "
                         "overlay the new anchor point to show where it falls.")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    methods = []      # (label, best {ds:{compute:val}}, params {ds:{A,alpha,r2}})
    raw_methods = []  # (label, method_key, cfg, sweep_name) — for the corrected figures
    for i, cpath in enumerate(args.configs):
        with open(cpath) as f:
            cfg = yaml.safe_load(f)
        name  = cfg["sweep_name"]
        label = args.labels[i] if args.labels else _method_label(name)
        print(f"== {label}  ({name}) ==")
        best = collect_best_from_config(cfg, name)
        params = {}
        for ds, cell in best.items():
            if len(cell) >= 2:
                cs = sorted(cell); vs = [cell[c] for c in cs]
                A, alpha, r2 = fit_power_law(cs, vs)
                params[ds] = {"A": A, "alpha": alpha, "r2": r2}
        methods.append((label, best, params))
        raw_methods.append((label, _method_label(name), cfg, name))

    solo_ref = {}
    solo_extra = {}    # {ds: (flops, val)} new nh=8 anchor point, overlaid in 'borrow' mode
    solo_points = {}   # {ds: [(flops, val), ...]} fitted (scaling-regime) solo cells
    solo_excluded = {} # {ds: [(flops, val), ...]} plotted-but-not-fit plateau points
    solo_curve_mode = bool(args.solo_configs)
    two_anchor_fit = (not solo_curve_mode and args.compare_anchor
                      and args.compare_anchor2 and args.solo_mode == "fit")
    if solo_curve_mode:
        print("\nBuilding solo reference (power law+floor through scaling-regime cells) ...")
        solo_ref, solo_points, solo_excluded = build_solo_curve(
            args.solo_configs, args.solo_plot_only_configs)
    elif two_anchor_fit:
        print("\nBuilding solo reference (two nh=8 anchors, measured slope) ...")
        solo_ref = build_solo_reference_two_anchors(args.compare_anchor, args.compare_anchor2)
    elif args.compare_anchor and args.compare_slope:
        print("\nBuilding solo reference (anchor + borrowed nh=4 slope) ...")
        # build_solo_reference returns (c_a, v_a, alpha); convert to (A, alpha, C=0).
        solo_ref = {ds: (v_a * c_a ** alpha, alpha, 0.0, float("nan"))
                    for ds, (c_a, v_a, alpha) in
                    build_solo_reference(args.compare_anchor, args.compare_slope).items()}
    if args.compare_anchor2 and not two_anchor_fit:
        # 'borrow' mode: line stays on the old anchor + nh=4 slope; the new anchor is
        # only overlaid as a marker so we can see where it falls relative to that line.
        a2 = yaml.safe_load(open(args.compare_anchor2))
        for ds, cell in collect_best_from_config(a2, a2["sweep_name"]).items():
            if cell:
                c = max(cell)               # single (highest-fidelity) anchor point
                solo_extra[ds] = (c, cell[c])

    mstyle = build_method_styles([(l, k) for l, k, _, _ in raw_methods])

    datasets = sorted({ds for _, best, _ in methods for ds in best if _keep_ds(ds)})
    if not datasets:
        sys.exit("No datasets with results across the given sweeps.")
    os.makedirs(args.out_dir, exist_ok=True)

    # alpha summary table
    print(f"\n{'dataset':<28}" + "".join(f"{lab:>12}" for lab, _, _ in methods))
    for ds in datasets:
        row = f"{ds.replace('_amplitudes',''):<28}"
        for _, _, params in methods:
            a = f"{params[ds]['alpha']:.3f}" if ds in params else "-"
            row += f"{a:>12}"
        print(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    ncol = min(len(datasets), 2) or 1
    nrow = math.ceil(len(datasets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
    axflat = axes.flatten()
    for ax in axflat[len(datasets):]:
        ax.axis("off")

    for di, ds in enumerate(datasets):
        ax = axflat[di]
        for mi, (label, best, params) in enumerate(methods):
            if ds not in best:
                continue
            cell = best[ds]; cs = sorted(cell); vs = [cell[c] for c in cs]
            st = mstyle[label]
            ax.scatter(cs, vs, color=st["color"], marker=st["marker"], alpha=st["alpha"],
                       s=22, zorder=st["z"])
            if ds in params:
                p = params[ds]
                cfit = np.logspace(math.log10(cs[0]), math.log10(cs[-1]), 200)
                ax.plot(cfit, p["A"] * cfit ** (-p["alpha"]), color=st["color"], ls=st["ls"],
                        lw=st["lw"], alpha=st["alpha"], label=rf"{label} ($\alpha$={p['alpha']:.3f})")
            else:
                ax.plot(cs, vs, color=st["color"], ls=st["ls"], lw=st["lw"],
                        alpha=st["alpha"], label=label)

        if ds in solo_ref:
            span = [c for _, best, _ in methods if ds in best for c in best[ds]]
            cfit = np.logspace(math.log10(min(span)), math.log10(max(span)), 200)
            ax.plot(cfit, _solo_curve_y(cfit, solo_ref[ds]), color=SOLO_STYLE["color"],
                    ls=SOLO_STYLE["ls"], lw=SOLO_STYLE["lw"], label=_solo_label(solo_ref[ds]))
            _solo_scatter(ax, solo_points.get(ds, []))
            _solo_scatter(ax, solo_excluded.get(ds, []), fitted=False)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("compute  (FLOPs, method-agnostic)"); ax.set_ylabel("val_loss")
        ax.set_title(_pretty_ds(ds), fontsize=10)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("Finetuning methods — scaling laws  (x = full-model FLOPs, "
                 "same per-step cost charged to every method)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"finetune_methods_comparison.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot: {path}")
    plt.close(fig)

    # =====================================================================
    # Corrected figures: (a) method-aware FLOPs, (b) trainable-params.
    # Both read each cell's best trial + its swept HPs from summary.txt.
    # =====================================================================
    # Load the pretrained-checkpoint param breakdown once (figure b only).
    breakdown = None
    ckpt = next((m[2]["fixed_params"].get("fine_tune.pretrained_path")
                 for m in raw_methods if m[2].get("fixed_params", {}).get("fine_tune.pretrained_path")),
                None)
    if ckpt:
        try:
            breakdown = load_param_breakdown(ckpt)
            print(f"\nParam breakdown from {ckpt}: total={breakdown['total']:,}, "
                  f"{len(breakdown['blocks'])} blocks")
        except Exception as e:
            print(f"\n[warn] could not load checkpoint for trainable-param figure ({e}); "
                  f"skipping figure (b).")

    corr = []   # (label, method_key, {ds: [recs]})
    for label, mkey, cfg, name in raw_methods:
        corr.append((label, mkey, collect_corrected(cfg, name, mkey, breakdown)))

    corr_datasets = sorted({ds for _, _, d in corr for ds in d if _keep_ds(ds)})
    if not corr_datasets:
        sys.exit("No summary.txt/results found for the corrected figures.")
    ncol = min(len(corr_datasets), 2) or 1
    nrow = math.ceil(len(corr_datasets) / ncol)

    # ---- Figure (a): loss vs METHOD-AWARE FLOPs (honest scaling) ----------
    # Drawn for both the test set (default file) and the validation set (_val).
    print("\n=== method-aware FLOPs (per-step factor vs standard) ===")
    for metric in ("test", "val"):
        suffix = "" if metric == "test" else "_val"
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
        axflat = axes.flatten()
        for ax in axflat[len(corr_datasets):]:
            ax.axis("off")
        for di, ds in enumerate(corr_datasets):
            ax = axflat[di]
            for mi, (label, mkey, data) in enumerate(corr):
                recs = sorted(data.get(ds, []), key=lambda r: r["flops"])
                if not recs:
                    continue
                cs = [r["flops"] for r in recs]; vs = [r[metric] for r in recs]
                st = mstyle[label]
                ax.scatter(cs, vs, color=st["color"], marker=st["marker"], alpha=st["alpha"],
                           s=22, zorder=st["z"])
                fit = _fit_pos(cs, vs)
                if fit:
                    A, alpha = fit
                    cfit = np.logspace(math.log10(cs[0]), math.log10(cs[-1]), 200)
                    ax.plot(cfit, A * cfit ** (-alpha), color=st["color"], ls=st["ls"],
                            lw=st["lw"], alpha=st["alpha"], label=rf"{label} ($\alpha$={alpha:.3f})")
                else:
                    ax.plot(cs, vs, color=st["color"], ls=st["ls"], lw=st["lw"],
                            alpha=st["alpha"], label=label)
            # solo reference (full from-scratch nh8 training = full 3×fwd, already on this axis)
            if ds in solo_ref:
                span = [r["flops"] for _, _, d in corr if ds in d for r in d[ds]]
                if span:
                    cfit = np.logspace(math.log10(min(span)), math.log10(max(span)), 200)
                    ax.plot(cfit, _solo_curve_y(cfit, solo_ref[ds]), color=SOLO_STYLE["color"],
                            ls=SOLO_STYLE["ls"], lw=SOLO_STYLE["lw"],
                            label=_solo_label(solo_ref[ds]))
                    _solo_scatter(ax, solo_points.get(ds, []))
                    _solo_scatter(ax, solo_excluded.get(ds, []), fitted=False)
            if ds in solo_extra:                # 'borrow' mode: show where the new anchor lands
                ce, ve = solo_extra[ds]
                ax.scatter([ce], [ve], edgecolors=NEW_SOLO_STYLE["edgecolor"],
                           facecolors=NEW_SOLO_STYLE["facecolor"], marker=NEW_SOLO_STYLE["marker"],
                           s=NEW_SOLO_STYLE["s"], linewidths=NEW_SOLO_STYLE["lw"],
                           zorder=NEW_SOLO_STYLE["z"], label="solo nh8 (new anchor)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("training compute  (FLOPs)"); ax.set_ylabel(_ylabel(metric))
            ax.set_title(_pretty_ds(ds), fontsize=10)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(fontsize=7)
        fig.suptitle(f"Finetuning methods — {metric} loss vs compute", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(args.out_dir, f"finetune_methods_comparison_flops{suffix}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot: {path}")
        plt.close(fig)

    # ---- Figure (b): best val_loss vs TRAINABLE PARAMS (LoRA's real win) --
    if breakdown is not None:
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
        axflat = axes.flatten()
        for ax in axflat[len(corr_datasets):]:
            ax.axis("off")
        for di, ds in enumerate(corr_datasets):
            ax = axflat[di]
            for mi, (label, mkey, data) in enumerate(corr):
                recs = data.get(ds, [])
                if not recs:
                    continue
                best = min(recs, key=lambda r: r["val"])   # best loss across t_steps
                st = mstyle[label]
                ax.scatter([best["params"]], [best["val"]], color=st["color"],
                           marker=st["marker"], alpha=st["alpha"], s=90, zorder=st["z"],
                           label=f"{label} ({best['params']:,} params)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("trainable parameters  (∝ Adam optimizer-state bytes)")
            ax.set_ylabel("best val_loss")
            ax.set_title(_pretty_ds(ds), fontsize=10)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(fontsize=7)
        fig.suptitle("Finetuning methods — parameter efficiency  "
                     "(best loss vs #trainable params; lower-left = better)", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(args.out_dir, f"finetune_methods_comparison_params.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot: {path}")
        plt.close(fig)

    # ---- Overlay: old (method-agnostic) vs new (method-aware), shift visible ----
    # Each cell is drawn at its OLD x (faint ×, = the previous plot) and its NEW x
    # (solid ●), joined by a horizontal connector at the same val_loss, so the
    # left/right move is explicit. Solo reference line restored.
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
    axflat = axes.flatten()
    for ax in axflat[len(corr_datasets):]:
        ax.axis("off")
    for di, ds in enumerate(corr_datasets):
        ax = axflat[di]
        for mi, (label, mkey, data) in enumerate(corr):
            recs = sorted(data.get(ds, []), key=lambda r: r["t"])
            if not recs:
                continue
            old = [r["flops_naive"] for r in recs]
            new = [r["flops"] for r in recs]
            vs  = [r["val"] for r in recs]
            st = mstyle[label]; color = st["color"]; al = st["alpha"]
            for o, n, v in zip(old, new, vs):               # horizontal shift connector
                ax.plot([o, n], [v, v], color=color, lw=0.7, alpha=0.3 * al, zorder=2)
            ax.scatter(old, vs, color=color, marker="x", s=26, alpha=0.45 * al, zorder=st["z"])
            ax.scatter(new, vs, color=color, marker=st["marker"], s=26, alpha=al, zorder=st["z"] + 1)
            fo, fn = _fit_pos(old, vs), _fit_pos(new, vs)
            if fo and fn:
                Ao, ao = fo; An, an = fn
                xo = np.logspace(math.log10(min(old)), math.log10(max(old)), 100)
                xn = np.logspace(math.log10(min(new)), math.log10(max(new)), 100)
                ax.plot(xo, Ao * xo ** (-ao), color=color, lw=1.0, ls=":", alpha=0.5 * al)
                ax.plot(xn, An * xn ** (-an), color=color, lw=st["lw"], ls=st["ls"], alpha=al,
                        label=rf"{label} ($\alpha$: {ao:.2f}→{an:.2f})")
        if ds in solo_ref:
            span = [r[k] for _, _, d in corr if ds in d for r in d[ds]
                    for k in ("flops", "flops_naive")]
            if span:
                cfit = np.logspace(math.log10(min(span)), math.log10(max(span)), 200)
                ax.plot(cfit, _solo_curve_y(cfit, solo_ref[ds]), color=SOLO_STYLE["color"],
                        ls=SOLO_STYLE["ls"], lw=SOLO_STYLE["lw"],
                        label=_solo_label(solo_ref[ds]))
                _solo_scatter(ax, solo_points.get(ds, []))
                _solo_scatter(ax, solo_excluded.get(ds, []), fitted=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("compute  (FLOPs)"); ax.set_ylabel("val_loss")
        ax.set_title(_pretty_ds(ds), fontsize=10)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=7)
    # global style legend (× faint = old/method-agnostic, ● solid = new/method-aware)
    style = [Line2D([0], [0], color="0.4", marker="x", ls=":", lw=1.0, label="old (method-agnostic)"),
             Line2D([0], [0], color="0.4", marker="o", ls="-", lw=1.4, label="new (method-aware)")]
    fig.legend(handles=style, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Finetuning methods — old vs corrected compute  "
                 "(× = previous plot, ● = method-aware; connector shows the shift)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"finetune_methods_comparison_overlay.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot: {path}")
    plt.close(fig)

    # ---- Figure: loss vs WALL TIME (measured training hours) --------------
    # x = best trial's traintime_hours per cell — no FLOP model at all.
    # Drawn for both the test set (default file) and the validation set (_val).
    print("\n=== wall-time (measured training hours) ===")

    def _draw(ax, pts, st, label):
        pts = sorted(pts)                      # by traintime
        ts = [p[0] for p in pts]; vs = [p[1] for p in pts]
        ax.scatter(ts, vs, color=st["color"], marker=st.get("marker", "o"),
                   alpha=st.get("alpha", 1.0), s=22, zorder=st.get("z", 3))
        fit = _fit_pos(ts, vs)
        if fit:
            A, alpha = fit
            tf = np.logspace(math.log10(ts[0]), math.log10(ts[-1]), 200)
            ax.plot(tf, A * tf ** (-alpha), color=st["color"], ls=st.get("ls", "-"),
                    lw=st.get("lw", 1.3), alpha=st.get("alpha", 1.0),
                    label=rf"{label} ($\alpha$={alpha:.3f})")
        else:
            ax.plot(ts, vs, color=st["color"], ls=st.get("ls", "-"),
                    lw=st.get("lw", 1.3), alpha=st.get("alpha", 1.0), label=label)

    for metric in ("test", "val"):
        suffix = "" if metric == "test" else "_val"
        wall = [(label, collect_walltime(cfg, name, metric)) for label, _, cfg, name in raw_methods]
        wall_solo = {}                          # {ds: [(traintime_h, loss, t_steps), ...]} fitted
        wall_excl = {}                          # {ds: [...]} plotted but excluded from fit
        solo_cfg_paths = args.solo_configs or [p for p in (args.compare_anchor, args.compare_anchor2) if p]
        for apath in solo_cfg_paths:
            acfg = yaml.safe_load(open(apath))
            for ds, pts in collect_walltime(acfg, acfg["sweep_name"], metric).items():
                wall_solo.setdefault(ds, []).extend(pts)   # nh8 solo, matched arch
        for apath in (args.solo_plot_only_configs or []):
            acfg = yaml.safe_load(open(apath))
            for ds, pts in collect_walltime(acfg, acfg["sweep_name"], metric).items():
                wall_excl.setdefault(ds, []).extend(pts)
        wall_datasets = sorted({ds for _, d in wall for ds in d if _keep_ds(ds)})
        if not wall_datasets:
            continue
        wcol = min(len(wall_datasets), 2) or 1
        wrow = math.ceil(len(wall_datasets) / wcol)
        fig, axes = plt.subplots(wrow, wcol, figsize=(6 * wcol, 4.2 * wrow), squeeze=False)
        axflat = axes.flatten()
        for ax in axflat[len(wall_datasets):]:
            ax.axis("off")

        for di, ds in enumerate(wall_datasets):
            ax = axflat[di]
            for mi, (label, data) in enumerate(wall):
                if ds in data:
                    _draw(ax, data[ds], mstyle[label], label)
            # solo scaling line. 'fit' mode (≥2 anchors): fit the slope DIRECTLY in
            # wall-time space (wall time is NOT ∝ compute across fidelities — the low-t
            # anchor is overhead-dominated, so the wall-time exponent differs from the
            # compute one). 'borrow' mode / single anchor: anchor the line on the OLD
            # (lowest-t) point + the borrowed compute exponent from solo_ref, and overlay
            # any newer anchor as an open star to show where it lands relative to it.
            solo_pts = sorted(wall_solo.get(ds, []), key=lambda p: p[2])   # by t_steps, old first
            ts_pts = [p[0] for p in solo_pts]; vs_pts = [p[1] for p in solo_pts]
            # curve mode: fit the SATURATING model L=A·t^-a+C in wall-time space (its own
            # floor/exponent — wall time is not ∝ compute across fidelities). two-anchor
            # 'fit' mode: pure 2-point power law. Else: anchor + borrowed slope.
            wall_ref = None
            if solo_curve_mode:
                sat = fit_saturating(ts_pts, vs_pts)
                if sat:
                    wall_ref = sat
            elif two_anchor_fit:
                pf = _fit_pos(ts_pts, vs_pts)
                if pf:
                    wall_ref = (pf[0], pf[1], 0.0, float("nan"))
            if wall_ref:
                tf = np.logspace(math.log10(min(ts_pts)), math.log10(max(ts_pts)), 200)
                ax.plot(tf, _solo_curve_y(tf, wall_ref), color=SOLO_STYLE["color"],
                        ls=SOLO_STYLE["ls"], lw=SOLO_STYLE["lw"],
                        label=_solo_label(wall_ref, prefix="solo nh8"))
                _solo_scatter(ax, list(zip(ts_pts, vs_pts)))
                _solo_scatter(ax, [(p[0], p[1]) for p in wall_excl.get(ds, [])], fitted=False)
            elif solo_pts and ds in solo_ref:
                t_a, v_a = solo_pts[0][0], solo_pts[0][1]      # OLD anchor (lowest t_steps)
                alpha_s = solo_ref[ds][1]
                span = [p[0] for _, d in wall if ds in d for p in d[ds]] + [p[0] for p in solo_pts]
                tf = np.logspace(math.log10(min(span)), math.log10(max(span)), 200)
                ax.plot(tf, v_a * (tf / t_a) ** (-alpha_s), color=SOLO_STYLE["color"],
                        ls=SOLO_STYLE["ls"], lw=SOLO_STYLE["lw"],
                        label=rf"solo nh8 ($\alpha$={alpha_s:.3f})")
                ax.scatter([t_a], [v_a], color=SOLO_STYLE["color"], marker="s", s=40, zorder=4)
                for tn, vn, _ in solo_pts[1:]:                 # newer anchor(s) as open stars
                    ax.scatter([tn], [vn], edgecolors=NEW_SOLO_STYLE["edgecolor"],
                               facecolors=NEW_SOLO_STYLE["facecolor"], marker=NEW_SOLO_STYLE["marker"],
                               s=NEW_SOLO_STYLE["s"], linewidths=NEW_SOLO_STYLE["lw"],
                               zorder=NEW_SOLO_STYLE["z"], label="solo nh8 (new anchor)")
            elif solo_pts:
                _draw(ax, solo_pts, SOLO_STYLE, "solo (nh8)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("training wall time  (hours)"); ax.set_ylabel(_ylabel(metric))
            ax.set_title(_pretty_ds(ds), fontsize=10)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(fontsize=7)
        title_metric = metric.capitalize()
        fig.suptitle(f"Finetuning methods — {title_metric} loss vs measured training wall time",
                     fontsize=12)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(args.out_dir, f"finetune_methods_comparison_walltime{suffix}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
