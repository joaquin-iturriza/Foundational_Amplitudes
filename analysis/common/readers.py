"""Readers, pickers and statistics shared by the analysis scripts (analysis/common/__init__.py). Each function names
the script whose rule it carries; tests: tests/test_analysis_common.py."""
import glob, json, os, sys

import numpy as np


# ---------------------------------------------------------------- run and sweep reads
def read_jsonl(path):
    """The records of a JSON-lines file, one dict per line; [] when it does not exist.

    The rule of analysis/transfer/lowd_ft.py load()."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_metrics(path):
    """A run's per_process_metrics.json, the last one under `path` (the run's own).

    The rule of analysis/catalog_v2/census.py metrics()."""
    js = sorted(glob.glob(os.path.join(path, "**", "per_process_metrics.json"), recursive=True))
    if not js:
        raise SystemExit(f"no per_process_metrics.json under {path}")
    with open(js[-1]) as f:
        return json.load(f)


def best_checkpoint(d):
    """A run's best checkpoint: (index, combined val_loss_no_reg, {process: val_loss_no_reg}).

    The validation with the lowest aggregate val_loss_no_reg, every process at that same step (the checkpoint
    selection, experiment._result_extra; CLAUDE.md "Reported values"). `d` is run_metrics()'s dict."""
    i = int(np.argmin(d["val_loss_no_reg"]))
    return i, float(d["val_loss_no_reg"][i]), {n: float(v[i]) for n, v in d["proc_val_losses_no_reg"].items()
                                               if len(v) > i}


def sweep_trials(sweeps, name, continuation=True):
    """A sweep's trial records, its <name>_002 continuation included.

    From a {sweep name: [trial]} dict (analysis/transfer/scratch_sweeps.json and its kin); a sweep finished on
    another site continues as <name>_002, one search with it (analysis/transfer/cells.py)."""
    return list(sweeps.get(name, [])) + (list(sweeps.get(name + "_002", [])) if continuation else [])


def trial_mse(t):
    """A trial's MSE of log|M|^2 at its best checkpoint, None when it has not converged.

    val_loss (val_loss_no_reg on the standardised target) times prepd_std^2; None without val_loss or prepd_std."""
    if t.get("val_loss") is None or not t.get("prepd_std"):
        return None
    return t["val_loss"] * t["prepd_std"] ** 2


def best_trial(trials, value=trial_mse, label=""):
    """(value, trial) of the lowest-value trial, (None, None) when none has a value.

    Trials without a value are left out and never silently: those with a val_loss but no value (a result not
    converted to MSE of log|M|^2, no prepd_std) are counted on stderr, as analysis/transfer/cells.py best() lists
    them; `label` names the search in that line."""
    scored = [(v, t) for t in trials for v in [value(t)] if v is not None]
    unconv = sum(1 for t in trials if value(t) is None and t.get("val_loss") is not None)
    if unconv:
        print(f"best_trial{' ' + label if label else ''}: {unconv} of {len(trials)} trials have a result but no "
              f"value (no prepd_std), left out", file=sys.stderr)
    if not scored:
        return None, None
    return min(scored, key=lambda x: x[0])


# ---------------------------------------------------------------- pools
def find_pool(name, sqrts=None, role="val", data_dir=None, any_window=False):  # dup-ok: the canonical one
    """The path of a process's frozen pool of one role, None when there is none.

    `<name>_<lo>-<hi>GeV_<role>_*amplitudes.npy` in data_dir (default $DATA_DIR), the window anchored right after the
    name, so a process whose name extends this one (<name>_nlo, <name>__E10) never matches: the window `sqrts` =
    (lo, hi) when given; with no window given, or with any_window=True when the given one is absent, the first pool of
    any window (the rule of analysis/catalog_v3/prep_shape.py pool(), which takes any_window=True)."""
    d = data_dir or os.environ["DATA_DIR"]
    g = sorted(glob.glob(os.path.join(d, f"{name}_{sqrts[0]}-{sqrts[1]}GeV_{role}_*amplitudes.npy"))) if sqrts else []
    if not g and (not sqrts or any_window):
        g = sorted(glob.glob(os.path.join(d, f"{name}_[0-9]*-[0-9]*GeV_{role}_*amplitudes.npy")))
    return g[0] if g else None


def split_pool(rows, n_particles):
    """A pool's rows split into momenta (N, n, 4), PDG ids (N, n) and |M|^2 (N,).

    A row is the flat 4-momenta, then the PDG ids, then the amplitude (CLAUDE.md Data): 5 n + 1 columns."""
    rows = np.asarray(rows)
    n = n_particles
    if rows.ndim != 2 or rows.shape[1] != 5 * n + 1:
        raise ValueError(f"a pool row of {n} particles has {5 * n + 1} columns, got shape {rows.shape}")
    return rows[:, :4 * n].reshape(-1, n, 4), rows[:, 4 * n:5 * n].astype(np.int64), rows[:, -1]


# ---------------------------------------------------------------- statistics
def read_errors(q, t):
    """{mse, level, shape} of a prediction q against the truth t, both log|M|^2.

    mse the mean (q - t)^2, level the mean of q - t squared (what one number per process removes), shape the
    variance of q - t (what is left); mse = level + shape (analysis/catalog_v3/short_reads.py read())."""
    d = np.asarray(q, float) - np.asarray(t, float)
    return {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2), "shape": float(np.var(d))}


def curve_stats(curve, every=1, scale=1.0, T=None):  # dup-ok: canonical; ft_harm.shape (a claim source) kept
    """First, best, where the best sits and last over best of one validation curve.

    The curve is val_loss_no_reg per validation, one every `every` steps, times `scale` (prepd_std^2 gives MSE of
    log|M|^2); best_frac is the best's step over T (default the curve's length in steps); last_over_best above 1 is
    a curve that turned back up (analysis/transfer/ft_harm.py shape())."""
    y = np.asarray(curve, float) * scale
    i = int(np.argmin(y))
    T = T or every * len(y)
    return {"first": float(y[0]), "first_step": every, "best": float(y[i]), "best_frac": (i + 1) * every / T,
            "last_over_best": float(y[-1] / y[i]), "steps": T}


def gmean(x):
    """Geometric mean of positive values (analysis/catalog_v4/compare_v3.py)."""
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def ratio_stats(num, den):
    """Median, 10% and 90% quantiles of paired ratios num / den, how many are below 1, the median |log10 ratio|.

    Over the same items (two arms' per-process MSEs): n, ratio (a median of log ratios), ratio_q10_q90,
    n_num_lower, gap; the rule of analysis/catalog_v3/pp_compare.py pp_vs(). analysis/catalog_v4/compare_v3.py
    compare() takes the median and the quantiles of the ratios themselves: the same median for an odd count,
    quantiles that differ by the interpolation between two items."""
    lr = np.log10(np.asarray(num, float) / np.asarray(den, float))
    return {"n": int(lr.size), "ratio": float(10 ** np.median(lr)),
            "ratio_q10_q90": [float(10 ** v) for v in np.quantile(lr, [0.1, 0.9])],
            "n_num_lower": int((lr < 0).sum()), "gap": float(np.median(np.abs(lr)))}
