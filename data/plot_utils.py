"""
Histogram utilities with symlog support.

plot_histogram(data, ax, ...)
    General-purpose histogram that handles:
    - purely positive data  -> linear or log x-bins
    - mixed-sign data       -> symlog x-bins (linear near zero, log away from zero)
    - logy option for the counts axis
"""

import numpy as np
import matplotlib.pyplot as plt


def _symlog_bins(data, n_bins, linthresh):
    """
    Build bin edges for a symlog-scaled histogram.
    The range [-linthresh, +linthresh] gets n_lin evenly-spaced bins.
    Outside that, each side gets n_log log-spaced bins.
    """
    finite = data[np.isfinite(data)]
    xmin, xmax = finite.min(), finite.max()

    has_neg = xmin < -linthresh
    has_pos = xmax >  linthresh

    if not has_neg and not has_pos:
        return np.linspace(xmin, xmax, n_bins + 1)

    n_log = max(4, int(n_bins * 0.45))
    n_lin = max(4, n_bins - 2 * n_log if (has_neg and has_pos) else n_bins - n_log)

    parts = []

    if has_neg:
        neg = -np.logspace(np.log10(-xmin), np.log10(linthresh), n_log + 1)
        parts.append(neg[:-1])

    lin_lo = -linthresh if has_neg else max(xmin, -linthresh)
    lin_hi =  linthresh if has_pos else min(xmax,  linthresh)
    parts.append(np.linspace(lin_lo, lin_hi, n_lin + 1))

    if has_pos:
        pos = np.logspace(np.log10(linthresh), np.log10(xmax), n_log + 1)
        parts.append(pos[1:])

    return np.concatenate(parts)


def _log_bins(data, n_bins):
    """Log-spaced bins for strictly positive data."""
    finite = data[np.isfinite(data) & (data > 0)]
    return np.logspace(np.log10(finite.min()), np.log10(finite.max()), n_bins + 1)


def plot_histogram(
    data,
    ax=None,
    n_bins=60,
    logx=False,
    logy=False,
    linthresh=None,
    density=False,
    label=None,
    color=None,
    alpha=0.7,
    histtype="stepfilled",
    **kwargs,
):
    """
    Plot a histogram with optional log/symlog x-axis.

    Parameters
    ----------
    data : array-like
        1-D array of values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if None.
    n_bins : int
        Approximate number of bins.
    logx : bool
        If True, use log (positive data) or symlog (mixed-sign data) x-scale.
    logy : bool
        If True, use log y-scale.
    linthresh : float, optional
        Linear threshold for symlog. Auto-set to 1% of max|x| if None and
        the data contains negative values.
    density : bool
        Normalise to unit area.
    label, color, alpha, histtype, **kwargs
        Passed through to ax.hist().

    Returns
    -------
    n, bins, patches  (from ax.hist)
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if ax is None:
        _, ax = plt.subplots()

    has_neg = data.min() < 0

    if logx:
        if has_neg:
            if linthresh is None:
                # Set linthresh just below the smallest |x| present, so all
                # data falls in the log-scaled regions with no linear zone.
                abs_nonzero = np.abs(data[data != 0])
                linthresh = abs_nonzero.min() * 0.9 if len(abs_nonzero) else 1e-10
            bins = _symlog_bins(data, n_bins, linthresh)
            ax.set_xscale("symlog", linthresh=linthresh)
        else:
            bins = _log_bins(data, n_bins)
            ax.set_xscale("log")
    else:
        bins = n_bins

    result = ax.hist(
        data,
        bins=bins,
        density=density,
        label=label,
        color=color,
        alpha=alpha,
        histtype=histtype,
        **kwargs,
    )

    if logy:
        ax.set_yscale("log")

    return result
