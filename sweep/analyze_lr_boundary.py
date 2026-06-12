#!/usr/bin/env python3
"""
analyze_lr_boundary.py — Detect and fix lr search-range boundary issues in DyHPO sweeps.

Usage:
    # Print table of all sweeps + boundary status
    python sweep/analyze_lr_boundary.py

    # Generate PDF with val_loss vs lr plots (all sweeps, or only flagged)
    python sweep/analyze_lr_boundary.py --plot [--flagged-only] [--output lr_boundary.pdf]

    # Extend lr range for specific sweeps
    python sweep/analyze_lr_boundary.py --extend sweep_name_1 sweep_name_2

    # Extend all flagged sweeps automatically
    python sweep/analyze_lr_boundary.py --extend-all [--n-jobs 20] [--extend-factor 3]
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

LR_PARAM             = 'training.lr'
DEFAULT_SWEEP_BASE   = os.path.join(_project_dir, 'sweeps', 'pretraining_scaling')
DEFAULT_BOUNDARY_FRAC = 0.15   # fraction of log-range that counts as "near boundary"
DEFAULT_EXTEND_FACTOR = 3.0
DEFAULT_N_NEW_JOBS    = 20
DEFAULT_N_NEW_CANDS   = 50
TOP_K = 5


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def load_state(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_state(pkl_path, state):
    tmp = pkl_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pkl_path)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _quadratic_min(records):
    """
    Fit a quadratic in (log lr, log val_loss) to all records.
    Returns (poly_coeffs, lr_minimum) or (None, None).
    lr_minimum is in original lr space; None if the parabola opens downward.
    """
    if len(records) < 4:
        return None, None
    xs = np.array([math.log(lr) for lr, _ in records])
    ys = np.array([math.log(vl) for _, vl in records])
    try:
        p = np.polyfit(xs, ys, 2)
    except Exception:
        return None, None
    if p[0] <= 0:
        return p, None  # opens downward — no minimum
    lr_min_log = -p[1] / (2.0 * p[0])
    return p, math.exp(lr_min_log)


def analyze_sweep(state, boundary_frac=DEFAULT_BOUNDARY_FRAC, top_k=TOP_K, lr_param=LR_PARAM):
    """
    Compute lr boundary metrics for one sweep.

    Returns a dict, or {'n_obs': 0} if no observations, or None if no lr param.

    Edge detection uses the actual min/max of the tested lr values (not the
    declared search space bounds), so a minimum at the edge of what was explored
    is flagged even if the declared range extends further.
    """
    lr_entry = next((e for e in state['hp_space'] if e['name'] == lr_param), None)
    if lr_entry is None:
        return None

    low, high = lr_entry['low'], lr_entry['high']

    candidates = state['candidates_raw']
    vh = state.get('val_loss_history', {})
    if not vh:
        return {'n_obs': 0, 'low': low, 'high': high}

    records = []
    for hp_idx, obs_dict in vh.items():
        lr_val = candidates[hp_idx].get(lr_param)
        if lr_val is None:
            continue
        for combo, vl in obs_dict.items():
            records.append((float(lr_val), float(vl)))
    if not records:
        return {'n_obs': 0, 'low': low, 'high': high}

    records.sort(key=lambda r: r[1])   # ascending val_loss

    best_lr  = records[0][0]

    # Use the actual tested lr range for edge detection, not the declared bounds.
    # A minimum at the edge of what was explored is flagged even if the declared
    # range extends further.
    lr_tested_low  = min(r[0] for r in records)
    lr_tested_high = max(r[0] for r in records)
    log_tested_low  = math.log(lr_tested_low)
    log_tested_high = math.log(lr_tested_high)
    log_tested_range = log_tested_high - log_tested_low

    if log_tested_range > 0:
        best_pct = (math.log(best_lr) - log_tested_low) / log_tested_range
    else:
        best_pct = 0.5
    edge_pct  = min(best_pct, 1.0 - best_pct)
    direction = 'up' if best_pct >= 0.5 else 'down'

    top_recs = records[:min(top_k, len(records))]
    if direction == 'up':
        n_at_bdy = sum(1 for lr, _ in top_recs
                       if log_tested_range > 0 and
                       (math.log(lr) - log_tested_low) / log_tested_range >= 1.0 - boundary_frac)
    else:
        n_at_bdy = sum(1 for lr, _ in top_recs
                       if log_tested_range > 0 and
                       (math.log(lr) - log_tested_low) / log_tested_range <= boundary_frac)

    # Quadratic fit to all records in log-log space.
    quad_p, quad_lr_min = _quadratic_min(records)

    # Boundary slope: 5 observations closest in lr to the flagged boundary.
    # Positive slope near the upper boundary → loss rising → minimum inside → clear.
    # Negative slope near the upper boundary → loss falling → may be outside → inconclusive.
    # (mirrored for lower boundary)
    if direction == 'up':
        border_pts = sorted(records, key=lambda r: -r[0])[:5]
    else:
        border_pts = sorted(records, key=lambda r:  r[0])[:5]

    trend = None
    if len(border_pts) >= 3:
        xs = np.array([math.log(lr) for lr, _ in border_pts])
        ys = np.array([math.log(vl) for _, vl in border_pts])
        A  = np.column_stack([np.ones_like(xs), xs])
        coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        trend = float(coeffs[1])   # d(log val_loss) / d(log lr)

    # Inconclusive if:
    #   - slope near the tested boundary still points outward (primary signal), OR
    #   - global quadratic minimum lands outside the tested range on that side
    slope_inconclusive = trend is not None and (
        (direction == 'up'   and trend < 0) or
        (direction == 'down' and trend > 0)
    )
    quad_min_outside = (
        quad_lr_min is not None and (
            (direction == 'up'   and quad_lr_min > lr_tested_high) or
            (direction == 'down' and quad_lr_min < lr_tested_low)
        )
    )
    inconclusive = slope_inconclusive or quad_min_outside

    is_flagged = edge_pct < boundary_frac

    return {
        'n_obs':            len(records),
        'low':              low,
        'high':             high,
        'lr_tested_low':    lr_tested_low,
        'lr_tested_high':   lr_tested_high,
        'best_lr':          best_lr,
        'best_pct':         best_pct,
        'edge_pct':         edge_pct,
        'direction':        direction,
        'n_at_bdy':         n_at_bdy,
        'trend':            trend,
        'quad_lr_min':      quad_lr_min,
        'inconclusive':     inconclusive,
        'is_flagged':       is_flagged,
        'all_lrs':          [r[0] for r in records],
        'all_losses':       [r[1] for r in records],
    }


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def print_table(sweep_metrics, boundary_frac=DEFAULT_BOUNDARY_FRAC):
    rows = [(n, m) for n, m in sweep_metrics.items() if m and m.get('n_obs', 0) > 0]
    # Flagged first (sorted by edge_pct), then ok (sorted by edge_pct)
    rows.sort(key=lambda x: (not x[1]['is_flagged'], x[1]['edge_pct']))

    hdr = (f"{'sweep':<48} {'n':>4}  {'lr_low':>9} {'lr_high':>9} {'best_lr':>9}  "
           f"{'edge%':>5} {'dir':>4}  {'n@bdy':>5}  {'slope':>7}  {'quad_min':>9}  status")
    print(hdr)
    print('-' * len(hdr))

    for name, m in rows:
        slope_s   = f"{m['trend']:+.3f}"    if m['trend']       is not None else "    n/a"
        qmin_s    = f"{m['quad_lr_min']:.2e}" if m['quad_lr_min'] is not None else "      n/a"
        if m['is_flagged'] and m['inconclusive']:
            status = "INCONCLUSIVE !"
        elif m['is_flagged']:
            status = "flagged"
        else:
            status = "ok"
        print(
            f"{name[:48]:<48} {m['n_obs']:>4}  "
            f"{m['low']:9.3e} {m['high']:9.3e} {m['best_lr']:9.3e}  "
            f"{m['edge_pct']*100:5.1f}% {m['direction']:>4}  "
            f"{m['n_at_bdy']:>5}  {slope_s:>7}  {qmin_s:>9}  {status}"
        )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweeps(sweep_metrics, output_path, flagged_only=False, boundary_frac=DEFAULT_BOUNDARY_FRAC):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = [(n, m) for n, m in sweep_metrics.items() if m and m.get('n_obs', 0) > 0]
    if flagged_only:
        rows = [(n, m) for n, m in rows if m['is_flagged']]
    rows.sort(key=lambda x: (not x[1]['is_flagged'], x[1]['edge_pct']))

    with PdfPages(output_path) as pdf:
        for name, m in rows:
            fig, ax = plt.subplots(figsize=(8, 5))

            lrs    = np.array(m['all_lrs'])
            losses = np.array(m['all_losses'])

            # All observations coloured by val_loss rank
            rank_norm = np.argsort(np.argsort(losses)) / max(len(losses) - 1, 1)
            sc = ax.scatter(lrs, losses, c=rank_norm, cmap='viridis',
                            s=30, alpha=0.75, zorder=3)
            plt.colorbar(sc, ax=ax, label='val_loss rank (0=best)')

            # Best point
            best_idx = np.argmin(losses)
            ax.scatter([lrs[best_idx]], [losses[best_idx]],
                       color='red', s=140, marker='*', zorder=5,
                       label=f"best lr={m['best_lr']:.2e}")

            # Declared search boundaries (dashed grey)
            ax.axvline(m['low'],  color='gray', ls='--', lw=0.8, alpha=0.6,
                       label=f"declared lo={m['low']:.2e}")
            ax.axvline(m['high'], color='gray', ls='--', lw=0.8, alpha=0.6,
                       label=f"declared hi={m['high']:.2e}")

            # Boundary zone shading based on the actual tested range
            t_lo  = m['lr_tested_low']
            t_hi  = m['lr_tested_high']
            log_tested_range = math.log(t_hi) - math.log(t_lo)
            upper_thresh = math.exp(math.log(t_hi) - boundary_frac * log_tested_range)
            lower_thresh = math.exp(math.log(t_lo) + boundary_frac * log_tested_range)
            ax.axvspan(upper_thresh, t_hi, alpha=0.15, color='red',
                       label=f"tested upper {boundary_frac*100:.0f}%")
            ax.axvspan(t_lo, lower_thresh, alpha=0.15, color='orange',
                       label=f"tested lower {boundary_frac*100:.0f}%")

            # Quadratic fit to all observations; curve shown only within the tested range
            all_records = list(zip(m['all_lrs'], m['all_losses']))
            if len(all_records) >= 4:
                try:
                    xs = np.array([math.log(lr) for lr, _ in all_records])
                    ys = np.array([math.log(vl) for _, vl in all_records])
                    p  = np.polyfit(xs, ys, 2)
                    # Extend a bit beyond the tested range to show the minimum
                    margin = 0.3 * log_tested_range
                    x_fit = np.linspace(math.log(t_lo) - margin,
                                        math.log(t_hi) + margin, 300)
                    y_fit = np.exp(np.polyval(p, x_fit))
                    # Clip to the data y-range so the curve doesn't dominate the axes
                    y_lo_data = losses.min()
                    y_hi_data = losses.max()
                    y_fit = np.where((y_fit >= y_lo_data * 0.5) & (y_fit <= y_hi_data * 3),
                                     y_fit, np.nan)
                    ax.plot(np.exp(x_fit), y_fit, 'k-', lw=1, alpha=0.45,
                            label='quad fit (log-log, all pts)')
                    if p[0] > 0:
                        lr_min_log = -p[1] / (2 * p[0])
                        lr_min     = math.exp(lr_min_log)
                        ax.axvline(lr_min, color='green', ls=':', lw=1.2,
                                   label=f"fit min={lr_min:.2e}")
                except Exception:
                    pass

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(m['low'] * 0.8, m['high'] * 1.25)
            ax.set_ylim(losses.min() * 0.85, losses.max() * 1.2)
            ax.set_xlabel('lr')
            ax.set_ylabel('val_loss')

            slope_s = f"slope={m['trend']:+.3f}" if m['trend'] is not None else "slope=n/a"
            qmin_s  = (f"  quad_min={m['quad_lr_min']:.2e}" if m['quad_lr_min'] is not None
                       else "")
            status  = ("INCONCLUSIVE" if (m['is_flagged'] and m['inconclusive']) else
                       "FLAGGED"      if m['is_flagged'] else "ok")
            ax.set_title(
                f"{name}\n"
                f"n={m['n_obs']}  edge={m['edge_pct']*100:.1f}%  dir={m['direction']}  "
                f"{slope_s}{qmin_s}  [{status}]",
                fontsize=9,
            )
            ax.legend(fontsize=7, loc='upper right')
            fig.tight_layout()
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"Plot saved: {output_path}  ({len(rows)} sweeps)")


# ---------------------------------------------------------------------------
# Extension (core)
# ---------------------------------------------------------------------------

def _apply_extension(sweep_dir, direction, new_bound,
                     n_new_cands=DEFAULT_N_NEW_CANDS,
                     n_new_jobs=DEFAULT_N_NEW_JOBS,
                     lr_param=LR_PARAM,
                     submit=False):
    """
    Apply a decided extension: update hp_space, add candidates, write job scripts.

    direction : 'up' or 'down'
    new_bound : the new lr_high (if 'up') or new lr_low (if 'down')

    How it works
    ------------
    The lr bounds in hp_space are updated, then n_new_cands new HP configs are
    sampled with lr restricted to the extension zone [old_bound, new_bound].
    All other HPs are drawn freely from their (possibly already extended) ranges.
    The full candidate pool is re-encoded so the surrogate sees consistent
    features.  On the next run_trial.py call, DyHPO's surrogate assigns high
    uncertainty (→ high EI) to the new untested candidates and will naturally
    pick them early.
    """
    from sweep.dyhpo_sampler import _sample_candidates, _encode_candidates

    pkl_path = os.path.join(sweep_dir, 'dyhpo_state.pkl')
    state    = load_state(pkl_path)

    lr_entry  = next(e for e in state['hp_space'] if e['name'] == lr_param)
    old_low   = lr_entry['low']
    old_high  = lr_entry['high']
    ext_count = state.get('extension_count', {}).get(lr_param, 0)

    if direction == 'up':
        zone_low, zone_high = old_high, new_bound
        lr_entry['high']    = new_bound
    else:
        zone_low, zone_high = new_bound, old_low
        lr_entry['low']     = new_bound

    # Sample new candidates in the extension zone only
    ext_hp_space = [
        {**e, 'low': zone_low, 'high': zone_high} if e['name'] == lr_param else dict(e)
        for e in state['hp_space']
    ]
    ext_seed = (state['seed'] * 997 + ext_count + 1) % (2 ** 31)
    new_raw  = _sample_candidates(ext_hp_space, n_new_cands, ext_seed)
    state['candidates_raw'].extend(new_raw)

    arr, log_ind = _encode_candidates(state['hp_space'], state['candidates_raw'])
    state['candidates_array'] = arr
    state['log_indicator']    = log_ind

    ec = state.get('extension_count', {})
    ec[lr_param] = ext_count + 1
    state['extension_count'] = ec

    save_state(pkl_path, state)
    print(f"    Saved: +{n_new_cands} candidates  total={len(state['candidates_raw'])}  "
          f"extension #{ext_count+1}")

    # Generate new SLURM job scripts
    cfg_path = os.path.join(sweep_dir, 'sweep_config.yaml')
    if not os.path.exists(cfg_path):
        print(f"    (no sweep_config.yaml — job scripts not generated)")
        return []

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    jobs_dir    = os.path.join(sweep_dir, 'jobs')
    existing    = len([fn for fn in os.listdir(jobs_dir)
                       if fn.startswith('trial_') and fn.endswith('.sh')])
    cluster     = cfg['cluster']
    project_dir = cfg['paths']['project_dir']
    lustre_dir  = os.path.join(cfg['paths']['sweep_dir'], cfg['sweep_name'])
    setup_lines = '\n'.join(cfg['paths'].get('setup_commands', []))
    trial_script = os.path.join(project_dir, 'sweep', 'run_trial.py')

    new_scripts = []
    for i in range(existing, existing + n_new_jobs):
        content = f"""\
#!/bin/bash
#SBATCH --job-name=trial_{i:04d}
#SBATCH --partition={cluster["partition"]}
#SBATCH --account={cluster["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{cluster["request_gpus"]}
#SBATCH --cpus-per-task={cluster.get("cpus_per_task", 8)}
#SBATCH --time={cluster.get("time", "02:00:00")}
#SBATCH --output={lustre_dir}/output/trial_{i:04d}_%j.out
#SBATCH --error={lustre_dir}/error/trial_{i:04d}_%j.err

{setup_lines}

python {trial_script} \\
    --sweep-config {lustre_dir}/sweep_config.yaml \\
    --trial-idx {i}
"""
        path = os.path.join(jobs_dir, f'trial_{i:04d}.sh')
        with open(path, 'w') as fh:
            fh.write(content)
        os.chmod(path, 0o755)
        new_scripts.append(path)

    last = existing + n_new_jobs - 1
    print(f"    Jobs: trial_{existing:04d}…trial_{last:04d}  ({lustre_dir}/jobs/)")

    if submit:
        for path in new_scripts:
            result = subprocess.run(['sbatch', path], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    sbatch: {result.stdout.strip()}")
            else:
                print(f"    sbatch FAILED {path}: {result.stderr.strip()}", file=sys.stderr)

    return new_scripts


# ---------------------------------------------------------------------------
# Interactive extension session
# ---------------------------------------------------------------------------

def _prompt(msg, valid=None):
    """Read a line of input, optionally validating against a set of choices."""
    while True:
        ans = input(msg).strip().lower()
        if valid is None or ans in valid:
            return ans
        print(f"  (enter one of: {', '.join(valid)})")


def _parse_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def _propose(direction, old_low, old_high, extend_factor):
    """Return (new_bound, description string)."""
    if direction == 'up':
        nb = old_high * extend_factor
        return nb, f"lr_high  {old_high:.3e}  →  {nb:.3e}  (×{extend_factor})"
    else:
        nb = old_low / extend_factor
        return nb, f"lr_low   {old_low:.3e}  →  {nb:.3e}  (÷{extend_factor})"


def interactive_extend(rows, sweep_dirs, sweep_metrics, n_new_cands, n_new_jobs,
                       extend_factor, skip_file=None, submit=False, lr_param=LR_PARAM):
    """
    Step through every sweep in table order and prompt for extension decisions.

    For flagged/inconclusive sweeps: show proposed extension, ask y/manual/skip.
    For ok sweeps: offer to manually flag; if flagged, ask direction + confirm.
    Skipped sweeps are saved to skip_file (with n_obs) and auto-skipped on future
    runs unless new observations have arrived since the skip.
    """
    print("\nExtension mechanism: new candidates are sampled with lr in the extended zone.")
    print("DyHPO assigns high uncertainty to untested candidates → high EI → picked early.")
    print("Press Enter or 's' to skip a sweep (remembered until new obs arrive).\n")

    # Load existing skips: {name: {"n_obs": N}}
    skips = {}
    if skip_file and os.path.exists(skip_file):
        with open(skip_file) as f:
            skips = json.load(f)
        if skips:
            print(f"  ({len(skips)} sweep(s) previously skipped — re-asked if new obs arrive)\n")

    def _save_skips():
        if skip_file:
            with open(skip_file, 'w') as f:
                json.dump(skips, f, indent=2)

    decided = []   # list of (name, direction, new_bound)

    for name, m in rows:
        if m is None or m.get('n_obs', 0) == 0:
            continue

        if name in skips:
            stored_n_obs = skips[name].get('n_obs', 0)
            if m['n_obs'] <= stored_n_obs:
                print(f"  (skipping {name} — previously skipped at n_obs={stored_n_obs})")
                continue
            print(f"  (previously skipped at n_obs={stored_n_obs}, now n_obs={m['n_obs']} — re-asking)")

        is_flagged    = m['is_flagged']
        is_incon      = m['inconclusive']
        slope_s       = f"{m['trend']:+.3f}" if m['trend'] is not None else "n/a"
        qmin_s        = f"{m['quad_lr_min']:.2e}" if m['quad_lr_min'] is not None else "n/a"
        status_label  = ("INCONCLUSIVE" if (is_flagged and is_incon) else
                         "flagged"      if is_flagged else "ok")

        print(f"{'─'*60}")
        print(f"{name}")
        print(f"  [{status_label}]  edge={m['edge_pct']*100:.1f}%  dir={m['direction']}  "
              f"slope={slope_s}  quad_min={qmin_s}")

        lr_entry = None
        d = sweep_dirs.get(name)
        if d:
            state = load_state(os.path.join(d, 'dyhpo_state.pkl'))
            lr_entry = next((e for e in state['hp_space'] if e['name'] == lr_param), None)
        if lr_entry is None:
            print("  (no lr param in state, skipping)")
            continue

        old_low, old_high = lr_entry['low'], lr_entry['high']

        if is_flagged:
            # Propose extension in the detected direction
            direction = m['direction']
            new_bound, desc = _propose(direction, old_low, old_high, extend_factor)
            print(f"  Proposed: {desc}")
            ans = _prompt("  [y]es / [s]kip / [m]anual value : ", {'y', 's', 'm', ''})
            if ans in ('s', ''):
                skips[name] = {'n_obs': m['n_obs']}; _save_skips()
                print("  Skipped (remembered).")
                continue
            if ans == 'm':
                raw = input(f"  New {'lr_high' if direction=='up' else 'lr_low'}"
                            f" (current={old_high if direction=='up' else old_low:.3e}): ")
                new_bound = _parse_float(raw)
                if new_bound is None:
                    print("  Invalid value, skipping.")
                    continue
                print(f"  Will use: {desc.split('→')[0].strip()} → {new_bound:.3e}")
                if _prompt("  Confirm? [y]es / [s]kip : ", {'y', 's', ''}) in ('s', ''):
                    skips[name] = {'n_obs': m['n_obs']}; _save_skips()
                    print("  Skipped (remembered).")
                    continue
            decided.append((name, direction, new_bound))

        else:
            ans = _prompt("  [s]kip / [f]lag and extend : ", {'s', 'f', ''})
            if ans in ('s', ''):
                skips[name] = {'n_obs': m['n_obs']}; _save_skips()
                print("  Skipped (remembered).")
                continue
            # Manually flag: ask direction
            d_ans = _prompt("  Direction [u]p / [d]own : ", {'u', 'd'})
            direction = 'up' if d_ans == 'u' else 'down'
            new_bound, desc = _propose(direction, old_low, old_high, extend_factor)
            print(f"  Proposed: {desc}")
            raw = input(f"  Enter for default, or type new value: ").strip()
            if raw:
                v = _parse_float(raw)
                if v is None:
                    print("  Invalid value, skipping.")
                    continue
                new_bound = v
                print(f"  Will use: {desc.split('→')[0].strip()} → {new_bound:.3e}")
            if _prompt("  Confirm? [y]es / [s]kip : ", {'y', 's', ''}) in ('s', ''):
                skips[name] = {'n_obs': m['n_obs']}; _save_skips()
                print("  Skipped (remembered).")
                continue
            decided.append((name, direction, new_bound))

    if not decided:
        print("\nNo extensions applied.")
        return

    print(f"\nApplying {len(decided)} extension(s)…")
    for name, direction, new_bound in decided:
        d = sweep_dirs[name]
        bound_s = (f"lr_high → {new_bound:.3e}" if direction == 'up'
                   else f"lr_low → {new_bound:.3e}")
        print(f"\n  {name}  ({bound_s})")
        _apply_extension(d, direction, new_bound,
                         n_new_cands=n_new_cands, n_new_jobs=n_new_jobs,
                         lr_param=lr_param, submit=submit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and fix lr boundary issues in DyHPO sweeps."
    )
    parser.add_argument('--sweep-dir', default=DEFAULT_SWEEP_BASE,
                        help="Directory containing sweep subdirectories")
    parser.add_argument('--boundary-frac', type=float, default=DEFAULT_BOUNDARY_FRAC,
                        help=f"Fraction of tested log-range that counts as near-boundary "
                             f"(default {DEFAULT_BOUNDARY_FRAC})")

    # Plot
    parser.add_argument('--plot',         action='store_true', help="Generate a PDF of val_loss vs lr")
    parser.add_argument('--flagged-only', action='store_true', help="Plot only flagged sweeps")
    parser.add_argument('--hide-skipped', action='store_true', help="Exclude previously skipped sweeps from the plot")
    parser.add_argument('--output', default=None,
                        help="Output PDF path (default: sweep_dir/lr_boundary.pdf)")

    # Extension
    parser.add_argument('--extend', nargs='+', metavar='SWEEP',
                        help="Non-interactively extend specific sweep(s) using auto-detected direction")
    parser.add_argument('--extend-all', action='store_true',
                        help="Interactive step-through of all sweeps to decide extensions")
    parser.add_argument('--extend-factor', type=float, default=DEFAULT_EXTEND_FACTOR,
                        help=f"Default extension factor (default {DEFAULT_EXTEND_FACTOR})")
    parser.add_argument('--n-new-cands', type=int, default=DEFAULT_N_NEW_CANDS,
                        help=f"New candidates to sample in the extension zone (default {DEFAULT_N_NEW_CANDS})")
    parser.add_argument('--n-new-jobs', type=int, default=DEFAULT_N_NEW_JOBS,
                        help=f"New SLURM job scripts to generate per sweep (default {DEFAULT_N_NEW_JOBS})")
    parser.add_argument('--reset-skips', action='store_true',
                        help="Clear the list of previously skipped sweeps")
    parser.add_argument('--submit', action='store_true',
                        help="sbatch new job scripts immediately after extension")

    args = parser.parse_args()
    sweep_base = args.sweep_dir
    skip_file  = os.path.join(sweep_base, 'lr_boundary_skips.json')

    if args.reset_skips:
        if os.path.exists(skip_file):
            os.remove(skip_file)
            print("Skips cleared.")
        else:
            print("No skips file found.")

    # Discover sweep dirs
    sweep_dirs = {}
    for name in sorted(os.listdir(sweep_base)):
        d   = os.path.join(sweep_base, name)
        pkl = os.path.join(d, 'dyhpo_state.pkl')
        if os.path.isdir(d) and os.path.exists(pkl):
            sweep_dirs[name] = d

    # Load and analyze
    sweep_metrics = {}
    for name, d in sweep_dirs.items():
        try:
            state = load_state(os.path.join(d, 'dyhpo_state.pkl'))
            sweep_metrics[name] = analyze_sweep(state, boundary_frac=args.boundary_frac)
        except Exception as e:
            sweep_metrics[name] = None
            print(f"Warning: could not analyze {name}: {e}", file=sys.stderr)

    # Table (always printed)
    print_table(sweep_metrics, boundary_frac=args.boundary_frac)
    n_flagged = sum(1 for m in sweep_metrics.values() if m and m.get('is_flagged'))
    n_inc     = sum(1 for m in sweep_metrics.values() if m and m.get('is_flagged') and m.get('inconclusive'))
    n_obs     = sum(1 for m in sweep_metrics.values() if m and m.get('n_obs', 0) > 0)
    print(f"\n{n_flagged} flagged ({n_inc} inconclusive) out of {n_obs} sweeps with observations.")

    # Plot
    if args.plot:
        skipped_names = set()
        if args.hide_skipped and os.path.exists(skip_file):
            with open(skip_file) as f:
                skipped_names = set(json.load(f).keys())
        plot_metrics = {n: m for n, m in sweep_metrics.items() if n not in skipped_names}
        out = args.output or os.path.join(sweep_base, 'lr_boundary.pdf')
        plot_sweeps(plot_metrics, out, flagged_only=args.flagged_only,
                    boundary_frac=args.boundary_frac)

    # Non-interactive batch extension
    if args.extend:
        for name in args.extend:
            d = sweep_dirs.get(name)
            m = sweep_metrics.get(name)
            if d is None:
                print(f"\n{name}: not found in {sweep_base}")
                continue
            if m is None or m.get('n_obs', 0) == 0:
                print(f"\n{name}: no observations, skipping")
                continue
            direction = m['direction']
            old_low   = m['low']
            old_high  = m['high']
            new_bound, desc = _propose(direction, old_low, old_high, args.extend_factor)
            print(f"\n{name}  [{('INCONCLUSIVE' if m['inconclusive'] else 'flagged') if m['is_flagged'] else 'ok'}]")
            print(f"  {desc}")
            _apply_extension(d, direction, new_bound,
                             n_new_cands=args.n_new_cands, n_new_jobs=args.n_new_jobs,
                             submit=args.submit)

    # Interactive step-through
    elif args.extend_all:
        rows = [(n, m) for n, m in sweep_metrics.items() if m and m.get('n_obs', 0) > 0]
        rows.sort(key=lambda x: (not x[1]['is_flagged'], x[1]['edge_pct']))
        interactive_extend(rows, sweep_dirs, sweep_metrics,
                           n_new_cands=args.n_new_cands,
                           n_new_jobs=args.n_new_jobs,
                           extend_factor=args.extend_factor,
                           skip_file=skip_file,
                           submit=args.submit)


if __name__ == "__main__":
    main()
