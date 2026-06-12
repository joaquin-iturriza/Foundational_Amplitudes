#!/usr/bin/env python3
"""
analyze_pretraining_scaling.py — Analyze Phase 1 compute-scaling results and produce
the phase2_compute.json needed to generate Phase 2 sweeps.

For each dataset size D:
  1. Loads best val_loss per cell from sweeps/pretraining_scaling/scaling_p1_* and
     scaling_p1ext_* directories (any num_heads; nh read from sweep_config.yaml).
  2. Computes FLOPs per cell: C = F_step(nh, n_avg=5, BS) × t_steps
  3. Fits L = A × C^{-alpha} + L_inf  (or pure power law if floor fitting fails)
     Fitting uses combined Phase 1 (nh=16) + all Phase 1 ext widths.
  4. Identifies the plateau compute C*(D) where the curve has flattened to within
     --plateau-frac of the fitted floor L_inf
  5. Writes phase2_compute.json

Usage:
    python sweep/analyze_pretraining_scaling.py
    python sweep/analyze_pretraining_scaling.py --plateau-frac 0.05 --output my_c_star.json
    python sweep/analyze_pretraining_scaling.py --dry-run   # print table, no JSON written
    python sweep/analyze_pretraining_scaling.py --no-ext    # ignore Phase 1 ext cells
    python sweep/analyze_pretraining_scaling.py --skip-cells scaling_p1_nh16_D1e3_t10000 scaling_p1ext_nh8_D1e4_t316
      # skip specific cells from both plots and fits (use ctag names from the printed table)
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

# Must stay in sync with generate_pretraining_scaling_sweeps.py
DATASET_SIZES = {
    "1e3":   1_000,
    "1e3p5": 3_162,
    "1e4":   10_000,
    "1e4p5": 31_623,
    "1e5":   100_000,
}
PHASE1_T_STEPS = {
    "1e3":   [316, 1000, 3162, 10000, 31623],
    "1e3p5": [100, 316,  1000, 3162,  10000],
    "1e4":   [32,  100,  316,  1000,  3162],
    "1e4p5": [10,  32,   100,  316,   1000,  3162],
    "1e5":   [10,  32,   100,  316,   1000,  3162],
}

N_DATASETS = 8
NUM_BLOCKS  = 8
D_PER_HEAD  = 16
N_AVG       = 5.0

# Step times (ms/step) from benchmark on V100-SXM2-16GB at n=6 (worst case).
# Used to convert t_steps → GPU-hours for the wall-time plot.
# None = OOM on 16 GB; at runtime we fall back to BS/2 * 2.
STEP_TIME_MS = {
    4:  {256: 37.78,  1024: 37.93,  4096: 116.34, 8192: 222.74},
    8:  {256: 38.71,  1024: 42.35,  4096: 218.54,  8192: 426.36},
    16: {256: 43.58,  1024: 65.18,  4096: 450.69,  8192: 889.11},
    32: {256: 74.75,  1024: 138.56, 4096: 1015.07, 8192: None},
}
STEP_TIME_MS[2] = {bs: t * 0.95 if t is not None else None
                   for bs, t in STEP_TIME_MS[4].items()}

# Visual style per num_heads value (color + marker)
# Assumed relative uncertainty on val_loss from run-to-run variability.
# Used to compute reduced chi-squared for all fits.
SIGMA_FRAC = 0.01   # 1 % of the loss value

NH_STYLE = {
    2:  {"color": "crimson",      "marker": "v", "bg": "mistyrose"},
    4:  {"color": "mediumpurple", "marker": "s", "bg": "thistle"},
    8:  {"color": "seagreen",     "marker": "D", "bg": "honeydew"},
    16: {"color": "steelblue",    "marker": "o", "bg": "lightsteelblue"},
    32: {"color": "darkorange",   "marker": "^", "bg": "moccasin"},
}
_DEFAULT_STYLE = {"color": "gray", "marker": "x", "bg": "whitesmoke"}


def _nh_style(nh: int) -> dict:
    return NH_STYLE.get(nh, _DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# FLOPs + wall-time helpers
# ---------------------------------------------------------------------------

def _batch_size(d_total: int) -> int:
    half = d_total / 2
    p = 1
    while p * 2 < half:
        p *= 2
    return min(8192, p)


def flops_per_step(num_heads: int, n_avg: float, batch_size: int) -> float:
    d = D_PER_HEAD * num_heads
    L = NUM_BLOCKS
    f_transformer = L * n_avg * (24 * d**2 + 2 * n_avg * d)
    f_framesnet   = n_avg * 131_072
    return 3.0 * batch_size * (f_framesnet + f_transformer)


def _step_ms(nh: int, bs: int) -> float:
    ms = STEP_TIME_MS.get(nh, {}).get(bs)
    if ms is None:
        ms = STEP_TIME_MS.get(nh, {}).get(bs // 2, 1200.0) * 2.0
    return ms


def gpu_hours(c_macs: float, nh: int, bs: int) -> float:
    """Convert a compute budget (MACs) to GPU-hours using benchmark step times."""
    fps = flops_per_step(nh, N_AVG, bs)
    t_steps = c_macs / fps
    return t_steps * _step_ms(nh, bs) / 1000.0 / 3600.0


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _find_cell_dir(prefix: str) -> str | None:
    candidates = []
    if os.path.isdir(SWEEP_BASE):
        for name in os.listdir(SWEEP_BASE):
            if name == prefix or name.startswith(prefix + "_"):
                candidates.append(os.path.join(SWEEP_BASE, name))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _find_p1_cell(d_key: str, t: int) -> str | None:
    return (_find_cell_dir(f"scaling_p1_nh16_D{d_key}_t{t}")
            or _find_cell_dir(f"scaling_p1_D{d_key}_t{t}"))


def _load_p1_cells_for_d(
    d_key: str,
) -> tuple[dict[float, list[float]], dict[float, float], dict[float, str]]:
    """Scan SWEEP_BASE for ALL nh=16 Phase 1 cells matching this D key.

    Naming-convention agnostic: picks up scaling_p1_nh16_*, scaling_p1_D*,
    and any other scaling_p1_* cell whose config says num_heads=16.
    Returns:
      - {c_macs: [val_losses]}
      - {c_macs: mean_traintime_hours}
      - {c_macs: cell_dir_basename}
    """
    result: dict[float, list[float]] = {}
    actual_hours: dict[float, float] = {}
    cell_dir_names: dict[float, str] = {}
    if not os.path.isdir(SWEEP_BASE):
        return result, actual_hours, cell_dir_names
    for name in sorted(os.listdir(SWEEP_BASE)):
        if not name.startswith("scaling_p1_"):
            continue
        if f"_D{d_key}_" not in name:
            continue
        cell_dir = os.path.join(SWEEP_BASE, name)
        if not os.path.isdir(cell_dir):
            continue
        cfg_path = os.path.join(cell_dir, "sweep_config.yaml")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            nh = int(cfg["fixed_params"]["model.net.num_heads"])
            if nh != 16:
                continue
            t_steps = int(cfg["fidelity_schedule"]["t_steps"][-1])
            bs = int(cfg["fixed_params"]["training.batchsize"])
        except Exception:
            continue
        vals = load_all_val_losses(cell_dir)
        if not vals:
            continue
        vals.sort()
        fps   = flops_per_step(16, N_AVG, bs)
        c_mac = fps * t_steps
        existing = result.get(c_mac, [])
        result[c_mac] = sorted(existing + vals)
        cell_dir_names[c_mac] = name
        h = load_mean_traintime(cell_dir)
        if h is not None:
            actual_hours[c_mac] = h
    return result, actual_hours, cell_dir_names


def _best_val_loss_from_result(r: dict) -> float:
    """Return the best non-regularized combined val_loss from a result dict.

    val_loss is written by base_experiment.train() as smallest_val_loss_no_reg
    — the best no-reg combined loss aggregated per the run's loss_aggregation
    config.  For old runs that may not have this, fall back to the geometric
    mean of proc_val_losses_no_reg.
    """
    pnr = r.get("proc_val_losses_no_reg")
    if pnr:
        vals = [v for v in pnr.values() if v is not None and v > 0]
        if vals:
            return float(np.exp(np.mean(np.log(vals))))
    return float(r["val_loss"])


def load_all_val_losses(cell_dir: str) -> list[float]:
    results_dir = os.path.join(cell_dir, "results")
    if not os.path.isdir(results_dir):
        return []
    vals = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fname)) as f:
                r = json.load(f)
            vals.append(_best_val_loss_from_result(r))
        except Exception:
            continue
    return vals


def load_mean_traintime(cell_dir: str) -> float | None:
    """Return mean traintime_hours across all result JSONs in cell_dir, or None."""
    results_dir = os.path.join(cell_dir, "results")
    if not os.path.isdir(results_dir):
        return None
    hours = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fname)) as f:
                r = json.load(f)
            h = r.get("traintime_hours")
            if h is not None and h > 0:
                hours.append(float(h))
        except Exception:
            continue
    return float(np.mean(hours)) if hours else None


def _load_ext_cells_for_d(
    d_key: str,
) -> tuple[dict[int, dict[float, list[float]]], dict[tuple[int, float], float]]:
    """
    Scan SWEEP_BASE for all Phase 1 ext cells matching this D key.
    Reads sweep_config.yaml to determine num_heads — naming-convention agnostic.
    Returns:
      - {num_heads: {c_macs: [val_losses]}}
      - {(num_heads, c_macs): mean_traintime_hours}  (only for cells with measured times)
    """
    result: dict[int, dict[float, list[float]]] = {}
    actual_hours: dict[tuple[int, float], float] = {}
    cell_dir_names: dict[tuple[int, float], str] = {}
    if not os.path.isdir(SWEEP_BASE):
        return result, actual_hours, cell_dir_names
    for name in sorted(os.listdir(SWEEP_BASE)):
        if not name.startswith("scaling_p1ext_"):
            continue
        if f"_D{d_key}_" not in name:
            continue
        cell_dir = os.path.join(SWEEP_BASE, name)
        if not os.path.isdir(cell_dir):
            continue
        cfg_path = os.path.join(cell_dir, "sweep_config.yaml")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            nh = int(cfg["fixed_params"]["model.net.num_heads"])
            t_steps = int(cfg["fidelity_schedule"]["t_steps"][-1])
            bs = int(cfg["fixed_params"]["training.batchsize"])
        except Exception:
            continue
        vals = load_all_val_losses(cell_dir)
        if not vals:
            continue
        vals.sort()
        fps   = flops_per_step(nh, N_AVG, bs)
        c_mac = fps * t_steps
        if nh not in result:
            result[nh] = {}
        existing = result[nh].get(c_mac, [])
        result[nh][c_mac] = sorted(existing + vals)
        cell_dir_names[(nh, c_mac)] = name
        h = load_mean_traintime(cell_dir)
        if h is not None:
            key = (nh, c_mac)
            if key in actual_hours:
                actual_hours[key] = (actual_hours[key] + h) / 2.0
            else:
                actual_hours[key] = h
    return result, actual_hours, cell_dir_names


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_power_law_with_floor(c_vals, l_vals, n_grid: int = 500):
    """
    Fit L = A * C^{-alpha} + L_inf.

    Profile over a grid of L_inf values.  For each candidate L_inf the inner
    problem is solved via WLS in log-space: log(l - L_inf) = log(A) - alpha*log(c).
    Weights come from error propagation of σ_i = SIGMA_FRAC * l_i through
    log(r_i), giving sqrt_w_i = r_i / l_i.

    The best L_inf minimises reduced chi-squared in the original space:
        χ²_r = [Σ ((l_i - l_pred_i) / (SIGMA_FRAC * l_i))²] / (n - 3)

    Returns (A, alpha, L_inf, chi2_red) or None if no valid fit found.
    """
    c = np.array(c_vals, dtype=float)
    l = np.array(l_vals, dtype=float)
    n = len(c)
    dof = n - 3   # 3 free parameters
    if dof <= 0:
        return None

    l_min   = float(l.min())
    log_c   = np.log(c)
    ones    = np.ones(n)

    best_chi2r  = np.inf
    best_params = None

    grid_lin   = np.linspace(0.0, 0.99 * l_min, n_grid // 2)
    grid_log   = l_min * np.geomspace(1e-4, 0.99, n_grid - n_grid // 2)
    l_inf_grid = np.unique(np.concatenate([grid_lin, grid_log]))

    for l_inf in l_inf_grid:
        r = l - l_inf
        if (r <= 0).any():
            continue
        log_r   = np.log(r)
        sqrt_w  = r / l          # from error propagation: σ(log r) ∝ σ_l/r = SIGMA_FRAC*l/r
        X_w     = np.column_stack([sqrt_w, sqrt_w * log_c])
        y_w     = sqrt_w * log_r
        coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
        log_A, neg_alpha = coeffs
        if neg_alpha >= 0:
            continue
        A      = math.exp(log_A)
        alpha  = -neg_alpha
        l_pred = A * c ** (-alpha) + l_inf
        chi2r  = float(np.sum(((l - l_pred) / (SIGMA_FRAC * l)) ** 2)) / dof
        if chi2r < best_chi2r:
            best_chi2r  = chi2r
            best_params = (A, alpha, l_inf, chi2r)

    return best_params


def fit_power_law_pure(c_vals, l_vals):
    """
    Fit L = A * C^{-alpha} via OLS in log-space.

    Log-space OLS is equivalent to WLS with σ_i ∝ l_i (relative errors), so
    it is already the correct estimator for SIGMA_FRAC * l_i uncertainties.

    Returns (A, alpha, chi2_red) where chi2_red uses original-space residuals
    and σ_i = SIGMA_FRAC * l_i, with dof = n - 2.
    """
    c     = np.array(c_vals, dtype=float)
    l     = np.array(l_vals, dtype=float)
    n     = len(c)
    log_c = np.log(c)
    log_l = np.log(l)
    X     = np.column_stack([np.ones(n), log_c])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_l, rcond=None)
    log_A, neg_alpha = coeffs
    A     = math.exp(log_A)
    alpha = -neg_alpha
    l_pred = A * c ** (-alpha)
    dof    = n - 2
    chi2r  = float(np.sum(((l - l_pred) / (SIGMA_FRAC * l)) ** 2)) / dof if dof > 0 else float("inf")
    return A, alpha, chi2r


def plateau_compute(A, alpha, l_inf, plateau_frac: float) -> float:
    if l_inf <= 0 or alpha <= 0:
        return float("inf")
    return (A / (l_inf * plateau_frac)) ** (1.0 / alpha)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Phase 1 results and write phase2_compute.json."
    )
    parser.add_argument("--plateau-frac", type=float, default=0.05,
                        help="Plateau criterion: C* where L = L_inf * (1 + frac). "
                             "Default 0.05 = 5%% above the floor.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: sweeps/pretraining_scaling/phase2_compute.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print table without writing JSON")
    parser.add_argument("--no-ext", action="store_true",
                        help="Ignore all Phase 1 ext cells")
    parser.add_argument("--skip-cells", nargs="+", default=[],
                        metavar="CELL_DIR",
                        help="Cell directory basenames to exclude from plots and fits "
                             "(the ctag names shown in the printed table, e.g. "
                             "scaling_p1_nh16_D1e3_t10000). Can be repeated.")
    args = parser.parse_args()

    skip_cells: set[str] = set(args.skip_cells)

    output_path = args.output or os.path.join(SWEEP_BASE, "phase2_compute.json")

    # data_by_d[d_key] = {
    #   "p1":       {c_macs: [val_losses]},            # nh=16
    #   "ext_by_nh": {nh: {c_macs: [val_losses]}},     # any other nh
    # }
    data_by_d: dict[str, dict] = {}

    print(f"{'D':8s}  {'series':10s}  {'t':8s}  {'n':>4s}  "
          f"{'C (TMACs)':10s}  {'best':10s}  {'median':10s}  cell_dir")
    print("-" * 115)

    for d_key, d_total in DATASET_SIZES.items():
        bs     = _batch_size(d_total)
        p1_cells: dict[float, list[float]] = {}
        p1_actual_hours: dict[float, float] = {}

        p1_cells, p1_actual_hours, p1_cell_dirs = _load_p1_cells_for_d(d_key)
        if skip_cells:
            for c in [c for c, name in p1_cell_dirs.items() if name in skip_cells]:
                p1_cells.pop(c, None)
                p1_actual_hours.pop(c, None)
                p1_cell_dirs.pop(c, None)

        for c_macs, vals in sorted(p1_cells.items()):
            fps_16 = flops_per_step(16, N_AVG, bs)
            t_s    = round(c_macs / fps_16)
            flag   = "  *** incomplete" if len(vals) < 20 else ""
            ctag   = p1_cell_dirs.get(c_macs, "?")
            print(f"  D={d_key:6s}  p1(nh=16)   t={t_s:6d}  {len(vals):>4d}  "
                  f"C={c_macs/1e12:8.3f} TMACs  {vals[0]:.6f}  "
                  f"{vals[len(vals)//2]:.6f}  {ctag}{flag}")

        ext_by_nh: dict[int, dict[float, list[float]]] = {}
        ext_actual_hours: dict[tuple[int, float], float] = {}
        ext_cell_dirs: dict[tuple[int, float], str] = {}
        if not args.no_ext:
            ext_by_nh, ext_actual_hours, ext_cell_dirs = _load_ext_cells_for_d(d_key)
            if skip_cells:
                for nh, c in [(nh, c) for (nh, c), name in ext_cell_dirs.items() if name in skip_cells]:
                    ext_by_nh.get(nh, {}).pop(c, None)
                    if nh in ext_by_nh and not ext_by_nh[nh]:
                        del ext_by_nh[nh]
                    ext_actual_hours.pop((nh, c), None)
                    ext_cell_dirs.pop((nh, c), None)
            for nh in sorted(ext_by_nh):
                for c_macs, vals in sorted(ext_by_nh[nh].items()):
                    fps   = flops_per_step(nh, N_AVG, bs)
                    t_s   = round(c_macs / fps)
                    flag  = "  *** incomplete" if len(vals) < 20 else ""
                    ctag  = ext_cell_dirs.get((nh, c_macs), "?")
                    print(f"  D={d_key:6s}  ext(nh={nh:2d})  t={t_s:6d}  {len(vals):>4d}  "
                          f"C={c_macs/1e12:8.3f} TMACs  {vals[0]:.6f}  "
                          f"{vals[len(vals)//2]:.6f}  {ctag}{flag}")

        data_by_d[d_key] = {"p1": p1_cells, "ext_by_nh": ext_by_nh,
                             "p1_actual_hours": p1_actual_hours,
                             "ext_actual_hours": ext_actual_hours}

    # ── Fitting ──────────────────────────────────────────────────────────────
    print()
    print(f"{'D':8s}  {'A':>10s}  {'alpha':>7s}  {'L_inf':>10s}  "
          f"{'chi2r':>7s}  {'C* (TMACs)':>12s}  notes")
    print("-" * 95)

    c_star: dict[str, float] = {}

    for d_key, series in data_by_d.items():
        p1_cells  = series["p1"]
        ext_by_nh = series["ext_by_nh"]

        # Combine all series for fitting (merge at same C → keep all vals)
        combined: dict[float, list[float]] = {}
        for c, vals in p1_cells.items():
            combined[c] = sorted(combined.get(c, []) + vals)
        for nh_cells in ext_by_nh.values():
            for c, vals in nh_cells.items():
                combined[c] = sorted(combined.get(c, []) + vals)

        n_total = len(combined)
        if n_total < 2:
            print(f"  {d_key:8s}  (too few cells: {n_total} — skip)")
            continue

        c_vals = sorted(combined)
        l_vals = [combined[c][0] for c in c_vals]

        fit_result = fit_power_law_with_floor(c_vals, l_vals) if n_total >= 3 else None
        src_note = (f"p1={len(p1_cells)} "
                    + " ".join(f"nh{nh}={len(cells)}" for nh, cells in sorted(ext_by_nh.items())))

        if fit_result is not None:
            A, alpha, l_inf, chi2r = fit_result
            sigma_eff_pct = SIGMA_FRAC * math.sqrt(chi2r) * 100
            c_star_d = min(plateau_compute(A, alpha, l_inf, args.plateau_frac), max(c_vals))
            c_star[d_key] = c_star_d
            plateau_reached = c_star_d < max(c_vals)
            note = "floor+power-law" if plateau_reached else "floor+power-law (plateau beyond range)"
            print(f"  {d_key:8s}  {A:10.3e}  {alpha:7.4f}  {l_inf:10.3e}  "
                  f"{chi2r:7.2f}  σ_HPO~{sigma_eff_pct:4.1f}%  {c_star_d/1e12:10.3f} TMACs  "
                  f"{note}  [{src_note}]")
        else:
            A, alpha, chi2r = fit_power_law_pure(c_vals, l_vals)
            sigma_eff_pct = SIGMA_FRAC * math.sqrt(chi2r) * 100
            c_star_d = max(c_vals)
            c_star[d_key] = c_star_d
            print(f"  {d_key:8s}  {A:10.3e}  {alpha:7.4f}  {'—':>10s}  "
                  f"{chi2r:7.2f}  σ_HPO~{sigma_eff_pct:4.1f}%  {c_star_d/1e12:10.3f} TMACs  "
                  f"pure power-law (fallback)  [{src_note}]")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        def _draw_scaling(ax, d_key, p1_cells, ext_by_nh, c_star_d,
                          show_all_trials, x_axis="compute",
                          actual_hours=None):
            """
            Plot val_loss vs compute (MACs) or vs wall time (GPU-hours).
            actual_hours: {(nh, c_macs): mean_hours} — uses measured times when
            available, falls back to benchmark estimate otherwise.
            Each width is fitted independently.  x_axis: "compute" | "wall_time"
            """
            d_total = DATASET_SIZES[d_key]
            bs = _batch_size(d_total)

            def _x(c_macs, nh):
                if x_axis == "wall_time":
                    if actual_hours and (nh, c_macs) in actual_hours:
                        return actual_hours[(nh, c_macs)]
                    return gpu_hours(c_macs, nh, bs)
                return c_macs

            # Merge p1 (nh=16) with any ext nh=16 cells into one series
            merged_nh16 = dict(p1_cells)
            for c, vals in ext_by_nh.get(16, {}).items():
                merged_nh16[c] = sorted(merged_nh16.get(c, []) + vals)
            all_series = [(16, merged_nh16)] + [(nh, cells) for nh, cells in sorted(ext_by_nh.items()) if nh != 16]

            # Combined dict (c_macs key) for y-axis range
            combined: dict[float, list[float]] = {}
            for nh, cells in all_series:
                for c, vals in cells.items():
                    combined[c] = sorted(combined.get(c, []) + vals)
            if len(combined) < 2:
                ax.set_visible(False)
                return

            # Individual trial dots (no legend entry)
            if show_all_trials:
                for nh, cells in all_series:
                    color = _nh_style(nh)["color"]
                    for c, vals in cells.items():
                        ax.scatter(np.full(len(vals), _x(c, nh)), vals,
                                   color=color, s=12, alpha=0.45, zorder=2, linewidths=0)

            # Per-series: fit independently, put fit info in scatter label
            for nh, cells in all_series:
                if not cells:
                    continue
                style   = _nh_style(nh)
                c_sorted = sorted(cells)
                x_vals  = [_x(c, nh) for c in c_sorted]
                y_vals  = [cells[c][0] for c in c_sorted]

                label   = f"nh={nh}"
                fit_x   = np.logspace(math.log10(min(x_vals)), math.log10(max(x_vals)), 200)
                fit_y   = None

                if len(c_sorted) >= 4:   # need dof > 0 for 3-param fit
                    fr = fit_power_law_with_floor(x_vals, y_vals)
                    if fr is not None:
                        A_f, alpha_f, l_inf_f, chi2r_f = fr
                        label  = (rf"nh={nh}  $\alpha$={alpha_f:.3f}  "
                                  rf"$L_\infty$={l_inf_f:.2e}  $\tilde{{\chi}}^2$={chi2r_f:.2f}")
                        fit_y  = A_f * fit_x ** (-alpha_f) + l_inf_f
                        if l_inf_f > 0:
                            ax.axhline(l_inf_f, color=style["color"], linestyle=":", alpha=0.35, zorder=1)

                if fit_y is None and len(c_sorted) >= 2:
                    A_p, alpha_p, chi2r_p = fit_power_law_pure(x_vals, y_vals)
                    label  = rf"nh={nh}  $\alpha$={alpha_p:.3f}  $\tilde{{\chi}}^2$={chi2r_p:.2f}"
                    fit_y  = A_p * fit_x ** (-alpha_p)

                if fit_y is not None:
                    ax.plot(fit_x, fit_y, color=style["color"], linestyle="-", alpha=0.7, zorder=3)

                ax.scatter(x_vals, y_vals, color=style["color"], marker=style["marker"],
                           s=40, zorder=4, label=label)

            # C* line on compute axis only
            if x_axis == "compute" and c_star_d is not None and c_star_d < max(combined):
                ax.axvline(c_star_d, color="purple", linestyle=":",
                           label=f"C*={c_star_d/1e12:.3f} TMACs")

            ax.set_xscale("log")
            ax.set_yscale("log")

            # Y-axis range: only best points when trials aren't shown (tighter range)
            if show_all_trials:
                v_flat = [v for vals in combined.values() for v in vals]
            else:
                v_flat = [vals[0] for vals in combined.values()]
            l_lo = min(v_flat)
            l_hi = max(v_flat)
            pad  = (l_hi / l_lo) ** 0.12
            ax.set_ylim(l_lo / pad, l_hi * pad)
            if x_axis == "compute":
                ax.set_xlabel("FLOPs (MACs)")
            elif actual_hours:
                ax.set_xlabel("GPU-hours (actual)")
            else:
                ax.set_xlabel("GPU-hours (V100, benchmark)")
            ax.set_ylabel("val_loss")

        plot_path = os.path.join(SWEEP_BASE, "phase1_scaling.pdf")
        with PdfPages(plot_path) as pdf:
            # ── Page 1: overview — one panel per D, compute x-axis ──────────
            n_cols = len(data_by_d)
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
            for ax, (d_key, series) in zip(axes[0], data_by_d.items()):
                _draw_scaling(ax, d_key, series["p1"], series["ext_by_nh"],
                              c_star.get(d_key), show_all_trials=False, x_axis="compute")
                ax.set_title(f"D={d_key}", fontsize=10)
                ax.legend(fontsize=6)
            nh_legend = "  ".join(f"nh={nh}" for nh in sorted(NH_STYLE))
            fig.suptitle(f"Phase 1 overview: val_loss vs compute  [{nh_legend}]", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ── Pages 2+: one per D, compute + wall-time side by side ───────
            for d_key, series in data_by_d.items():
                p1_cells  = series["p1"]
                ext_by_nh = series["ext_by_nh"]
                combined  = dict(p1_cells)
                for cells in ext_by_nh.values():
                    for c, vals in cells.items():
                        combined[c] = sorted(combined.get(c, []) + vals)
                if len(combined) < 2:
                    continue

                d_total = DATASET_SIZES[d_key]
                bs      = _batch_size(d_total)
                fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(14, 6))

                # Build unified (nh, c_macs) → hours dict from both p1 and ext runs
                actual_hours: dict[tuple[int, float], float] = {
                    (16, c): h for c, h in series.get("p1_actual_hours", {}).items()
                }
                actual_hours.update(series.get("ext_actual_hours", {}))
                _draw_scaling(ax_c, d_key, p1_cells, ext_by_nh, c_star.get(d_key),
                              show_all_trials=True, x_axis="compute")
                _draw_scaling(ax_w, d_key, p1_cells, ext_by_nh, c_star.get(d_key),
                              show_all_trials=True, x_axis="wall_time",
                              actual_hours=actual_hours)

                # Annotate n_done on the compute panel
                _nh16_merged = dict(p1_cells)
                for _c, _v in ext_by_nh.get(16, {}).items():
                    _nh16_merged[_c] = sorted(_nh16_merged.get(_c, []) + _v)
                for nh, cells in [(16, _nh16_merged)] + [(nh, c) for nh, c in ext_by_nh.items() if nh != 16]:
                    color = _nh_style(nh)["color"]
                    for c in sorted(cells):
                        ax_c.annotate(f"n={len(cells[c])}", (c, cells[c][0]),
                                      textcoords="offset points", xytext=(4, 4),
                                      fontsize=7, color=color)

                nh_present = [16] + sorted(ext_by_nh)
                nh_str = ", ".join(f"nh={h}" for h in nh_present)
                fig.suptitle(
                    f"D={d_key}  (subsample={d_total//N_DATASETS}/ds, BS={bs})  [{nh_str}]\n"
                    f"light dots=all trials  |  left: vs FLOPs  |  right: vs GPU-hours",
                    fontsize=10
                )
                ax_c.legend(fontsize=8)
                ax_w.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        print(f"\nPlot: {plot_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping plot.")

    # ── Write JSON ────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] phase2_compute.json not written.")
        print("C* values (MACs):")
        for k, v in c_star.items():
            print(f"  {k}: {v:.6e}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(c_star, f, indent=2)
        print(f"\nphase2_compute.json written to: {output_path}")
        print("Run Phase 2 generation with:")
        print(f"  python sweep/generate_pretraining_scaling_sweeps.py --phase 2 "
              f"--phase2-compute {output_path}")


if __name__ == "__main__":
    main()
