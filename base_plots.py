import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = False
#plt.rcParams[
#    "text.latex.preamble"
#] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12


def plot_loss(
    file,
    losses,
    lr=None,
    labels=None,
    logy=True,
    title=None,
    losses_no_reg=None,
    labels_no_reg=None,
):
    """Plot one or more loss curves.
 
    Parameters
    ----------
    file : str
        Output PDF path.
    losses : list[list[float]]
        Primary loss curves (e.g. [train_loss, val_loss]).
    lr : list[float] | None
        Learning-rate schedule plotted on the right y-axis.
    labels : list[str] | None
        Legend labels for *losses*.
    logy : bool
        If True, also save a log-scale version alongside the linear one.
    title : str | None
        Figure suptitle.
    losses_no_reg : list[list[float]] | None
        Same curves but *without* the regularization term.  Drawn dashed on the
        same axes so the reader can see the regularization contribution.
    labels_no_reg : list[str] | None
        Legend labels for *losses_no_reg*.  Defaults to "<label> (no reg)".
    """
    # ── drop empty val curve ──────────────────────────────────────────────────
    if len(losses) > 1 and len(losses[1]) == 0:
        losses = [losses[0]]
        if labels:
            labels = [labels[0]]
        if losses_no_reg and len(losses_no_reg) > 1:
            losses_no_reg = [losses_no_reg[0]]
        if labels_no_reg and len(labels_no_reg) > 1:
            labels_no_reg = [labels_no_reg[0]]
 
    n = len(losses)
    labels = [None] * n if labels is None else labels
 
    if losses_no_reg is not None:
        labels_no_reg = (
            [f"{l} (no reg)" if l else "no reg" for l in labels]
            if labels_no_reg is None
            else labels_no_reg
        )
 
    iterations = np.arange(1, len(losses[0]) + 1)
 
    # ── helper: compute x-axis for a curve that may be sub-sampled ───────────
    def _its(loss):
        if len(loss) == len(iterations):
            return iterations
        frac = len(losses[0]) / len(loss)
        return np.arange(1, len(loss) + 1) * frac
 
    # ── shift negative losses for the log-scale panel ────────────────────────
    all_vals = [v for curve in losses for v in curve]
    if losses_no_reg:
        all_vals += [v for curve in losses_no_reg for v in curve]
    min_val = min(all_vals)
    need_shift = min_val < 0
    shift = (-min_val + abs(1.0 / min_val)) if need_shift else 0.0
 
    def _shifted(curve):
        return [v + shift for v in curve] if need_shift else curve
 
    # ── one helper that draws onto a given Axes ───────────────────────────────
    def _draw_curves(ax, use_log, shifted):
        curves = [_shifted(c) for c in losses] if shifted else losses
        for i, (curve, label) in enumerate(zip(curves, labels)):
            ax.plot(_its(curve), curve, label=label, color=f"C{i}")
 
        if losses_no_reg is not None:
            nr_curves = (
                [_shifted(c) for c in losses_no_reg] if shifted else losses_no_reg
            )
            for i, (curve, label) in enumerate(zip(nr_curves, labels_no_reg)):
                ax.plot(
                    _its(curve),
                    curve,
                    label=label,
                    color=f"C{i}",
                    linestyle="--",
                    alpha=0.7,
                )
 
        if use_log:
            ax.set_yscale("log")
            if need_shift:
                ax.set_ylabel(
                    f"Loss + {shift:.3g} (shifted for log scale)", fontsize=FONTSIZE
                )
            else:
                ax.set_ylabel("Loss", fontsize=FONTSIZE)
        else:
            ax.set_ylabel("Loss", fontsize=FONTSIZE)
 
        ax.set_xlabel("Iteration", fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
 
        if lr is not None:
            axr = ax.twinx()
            axr.plot(iterations, lr, label="learning rate", color="crimson", alpha=0.6)
            axr.set_ylabel("Learning rate", fontsize=FONTSIZE)
            axr.legend(
                fontsize=FONTSIZE_LEGEND, frameon=False, loc="lower right"
            )
 
    # ── build figure: always linear; optionally add log panel ─────────────────
    ncols = 2 if logy else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
    fig.subplots_adjust(wspace=0.35)
 
    _draw_curves(axes[0, 0], use_log=False, shifted=False)
    axes[0, 0].set_title("Linear scale", fontsize=FONTSIZE)
 
    if logy:
        _draw_curves(axes[0, 1], use_log=True, shifted=need_shift)
        axes[0, 1].set_title(
            "Log scale" + (" (shifted)" if need_shift else ""), fontsize=FONTSIZE
        )
 
    if title:
        fig.suptitle(title, fontsize=FONTSIZE + 1, y=1.02)
 
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_mse(file, mse_losses, labels=None, title=None):
    """Plot raw MSE curves collected during HETEROSC training.
 
    Parameters
    ----------
    file : str
        Output PDF path.
    mse_losses : list[list[float]]
        MSE curves (e.g. [train_mse, val_mse]).
    labels : list[str] | None
        Legend labels.
    title : str | None
        Figure suptitle.
    """
    if len(mse_losses) > 1 and len(mse_losses[1]) == 0:
        mse_losses = [mse_losses[0]]
        if labels:
            labels = [labels[0]]
 
    labels = [None] * len(mse_losses) if labels is None else labels
    iterations = np.arange(1, len(mse_losses[0]) + 1)
 
    fig, ax = plt.subplots(figsize=(6, 4))
 
    for i, (curve, label) in enumerate(zip(mse_losses, labels)):
        if len(curve) == len(iterations):
            its = iterations
        else:
            frac = len(mse_losses[0]) / len(curve)
            its = np.arange(1, len(curve) + 1) * frac
        ax.plot(its, curve, label=label, color=f"C{i}")
 
    ax.set_xlabel("Iteration", fontsize=FONTSIZE)
    ax.set_ylabel("MSE (no regularization)", fontsize=FONTSIZE)
    ax.set_title("MSE evolution (heteroscedastic model)", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.set_yscale("log")
 
    if title:
        fig.suptitle(title, fontsize=FONTSIZE + 1, y=1.02)
 
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric(file, metrics, metric_label, labels=None, logy=False):
    labels = [None for _ in range(len(metrics))] if labels is None else labels
    iterations = range(1, len(metrics[0]) + 1)
    fig, ax = plt.subplots()
    for i, metric, label in zip(range(len(metrics)), metrics, labels):
        if len(metric) == len(iterations):
            its = iterations
        else:
            frac = len(metrics[0]) / len(metric)
            its = np.arange(1, len(metric) + 1) * frac
        ax.plot(its, metric, label=label)

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric_label, fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper left")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()