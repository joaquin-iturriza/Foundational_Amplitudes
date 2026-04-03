import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from base_plots import plot_loss, plot_mse

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = False
#plt.rcParams["text.latex.preamble"] = (
#    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"
#)

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, title, plot_dict):
    """Dispatch all requested plots based on cfg flags."""
 
    if cfg.plotting.loss and cfg.train:
        logy = cfg.plotting.get("loss_log_scale", True)
 
        # ── primary loss curves ───────────────────────────────────────────────
        losses = [plot_dict["train_loss"], plot_dict["val_loss"]]
        labels = ["train loss", "val loss"]
 
        # ── optional: loss without regularization ────────────────────────────
        losses_no_reg = None
        labels_no_reg = None
        if cfg.plotting.get("plot_without_regularization", False):
            losses_no_reg = [
                plot_dict.get("train_loss_no_reg", []),
                plot_dict.get("val_loss_no_reg", []),
            ]
            labels_no_reg = ["train loss (no reg)", "val loss (no reg)"]
 
        loss_title = title[0] if isinstance(title, list) else title
        plot_loss(
            file=f"{plot_path}/loss.pdf",
            losses=losses,
            lr=plot_dict["train_lr"],
            labels=labels,
            logy=logy,
            title=loss_title,
            losses_no_reg=losses_no_reg,
            labels_no_reg=labels_no_reg,
        )
 
        # ── optional: per-process val loss curves ────────────────────────────
        proc_val_losses = plot_dict.get("proc_val_losses", {})
        if proc_val_losses:
            proc_losses_list  = list(proc_val_losses.values())
            proc_labels_list  = [f"val {name}" for name in proc_val_losses]
            # combined val loss for reference
            combined_val = plot_dict["val_loss"]
            plot_loss(
                file=f"{plot_path}/loss_per_process.pdf",
                losses=[combined_val] + proc_losses_list,
                lr=None,
                labels=["val loss (combined)"] + proc_labels_list,
                logy=logy,
                title=loss_title,
            )

        # ── optional: MSE plot for HETEROSC ───────────────────────────────────
        if (
            cfg.training.loss == "HETEROSC"
            and cfg.plotting.get("plot_mse_het", False)
        ):
            mse_losses = [
                plot_dict.get("train_mse", []),
                plot_dict.get("val_mse", []),
            ]
            plot_mse(
                file=f"{plot_path}/mse.pdf",
                mse_losses=mse_losses,
                labels=["train MSE", "val MSE"],
                title=loss_title,
            )

    if cfg.plotting.histograms and cfg.evaluate:
        out = f"{plot_path}/histograms.pdf"
        with PdfPages(out) as file:
            labels = ["Test", "Train", "Prediction"]

            dataset = "combined" if len(cfg.data.dataset) > 1 else cfg.data.dataset[0]
            data = [
                np.log(plot_dict["results_test"][dataset]["raw"]["truth"]),
                np.log(plot_dict["results_train"][dataset]["raw"]["truth"]),
                np.log(plot_dict["results_test"][dataset]["raw"]["prediction"]),
            ]
            plot_histograms(
                file,
                data,
                labels,
                title=title[0],
                xlabel=r"$\log A$",
                logx=False,
                logy=True,
            )

            # Per-dataset pages
            for ds_name, ds_results in plot_dict.get("results_per_proc", {}).items():
                ds_data = [
                    np.log(ds_results["test"]["raw"]["truth"]),
                    np.log(ds_results["train"]["raw"]["truth"]),
                    np.log(ds_results["test"]["raw"]["prediction"]),
                ]
                plot_histograms(
                    file,
                    ds_data,
                    labels,
                    title=f"{title[0].split(':')[0]}: {ds_name}",
                    xlabel=r"$\log A$",
                    logx=False,
                    logy=True,
                )
        
            if cfg.training.loss == "HETEROSC":
                labels = ["Test", "Train"]
                sigmas_test = plot_dict["results_test"][dataset]["preprocessed"]["sigmas"]
                pull_test = plot_dict["results_test"][dataset]["preprocessed"]["pull"]
                sigmas_train = plot_dict["results_train"][dataset]["preprocessed"]["sigmas"]
                pull_train = plot_dict["results_train"][dataset]["preprocessed"]["pull"]
                sigmas = [sigmas_test, sigmas_train]
                pulls = [pull_test, pull_train]
                plot_histogram_single_output(
                    file,
                    sigmas,
                    labels,
                    title=f'{title[0]} - $\sigma$',
                    xlabel=r"$\sigma$",
                    logy=False,
                    plot_ratios=False
                )
                plot_histogram_single_output(
                    file,
                    [sigmas_test],
                    labels=["Test"],
                    title=f'{title[0]} - $\sigma$',
                    xlabel=r"$\sigma$",
                    logy=False,
                    plot_ratios=False
                )
                plot_histogram_single_output(
                    file,
                    [sigmas_test],
                    labels=["Test"],
                    title=f'{title[0]} - $\sigma$',
                    xlabel=r"$\sigma$",
                    logy=True,
                    plot_ratios=False
                )
                plot_histogram_single_output(
                    file,
                    pulls,
                    labels,
                    xrange=(-5,5),
                    title=f'{title[0]} - Pull',
                    xlabel=r"$\mathrm{pull} = \frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma}$",
                    logy=False,
                    plot_ratios=False,
                    pull=True
                )
                plot_histogram_single_output(
                    file,
                    [pull_test],
                    xrange=(-5,5),
                    labels=["Test"],
                    title=f'{title[0]} - Pull',
                    xlabel=r"$\mathrm{pull} = \frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma}$",
                    logy=False,
                    plot_ratios=False,
                    pull=True
                )
                plot_histogram_single_output(
                    file,
                    [pull_test],
                    xrange=(-5,5),
                    labels=["Test"],
                    title=f'{title[0]} - Pull',
                    xlabel=r"$\mathrm{pull} = \frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma}$",
                    logy=True,
                    plot_ratios=False,
                    pull=True
                )
                # Test vs 1% largest 
                truth_test = plot_dict["results_test"][dataset]["raw"]["truth"]
                plot_histogram_single_output(
                    file,
                    [pull_test, pull_test], # Will be masked internally
                    xrange=(-5,5),  
                    labels=["Test", "Largest 1\%"],
                    title=f'{title[0]} - Pull',
                    xlabel=r"$\mathrm{pull} = \frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma}$",
                    logy=True,
                    reference_truth=truth_test,  # Pass truth values for selection
                    pull=True,
                    plot_ratios=False
                )
                plot_histogram_single_output(
                    file,
                    [sigmas_test, sigmas_test],  # Will be masked internally
                    labels=["Test", "Largest 1\%"],
                    title=f'{title[0]} - $\sigma$',
                    xlabel=r"$\sigma$",
                    logy=True,
                    plot_ratios=False,
                    reference_truth=truth_test  # Pass truth values for selection
                )

    if cfg.plotting.delta and cfg.evaluate:
        out = f"{plot_path}/delta.pdf"
        with PdfPages(out) as file:
            dataset = "combined" if len(cfg.data.dataset) > 1 else cfg.data.dataset[0]
            delta_test = (
                plot_dict["results_test"][dataset]["raw"]["prediction"]
                - plot_dict["results_test"][dataset]["raw"]["truth"]
            ) / plot_dict["results_test"][dataset]["raw"]["truth"]
            delta_train = (
                plot_dict["results_train"][dataset]["raw"]["prediction"]
                - plot_dict["results_train"][dataset]["raw"]["truth"]
            ) / plot_dict["results_train"][dataset]["raw"]["truth"]

            # determine 1% largest amplitudes
            scale = plot_dict["results_test"][dataset]["raw"]["truth"]
            largest_idx = round(0.01 * len(scale))
            sort_idx = np.argsort(scale)
            largest_min = scale[sort_idx][-largest_idx - 1]
            largest_mask = scale > largest_min

            xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]  # in %
            binss = [100, 50, 50]
            for xrange, bins in zip(xranges, binss):
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_train * 100],
                    labels=["Test", "Train"],
                    title=title[0],
                    xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=False,
                )
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_test[largest_mask] * 100],
                    labels=["Test", "Largest 1\%"],
                    title=title[0],
                    xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=False,
                )
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_test[largest_mask] * 100],
                    labels=["Test", "Largest 1\%"],
                    title=title[0],
                    xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=True,
                )
        out = f"{plot_path}/delta_abs.pdf"
        with PdfPages(out) as file:
            delta_test = np.abs(
                (
                    plot_dict["results_test"][dataset]["raw"]["prediction"]
                    - plot_dict["results_test"][dataset]["raw"]["truth"]
                )
                / plot_dict["results_test"][dataset]["raw"]["truth"]
            )
            delta_train = np.abs(
                (
                    plot_dict["results_train"][dataset]["raw"]["prediction"]
                    - plot_dict["results_train"][dataset]["raw"]["truth"]
                )
                / plot_dict["results_train"][dataset]["raw"]["truth"]
            )

            # determine 1% largest amplitudes
            scale = plot_dict["results_test"][dataset]["raw"]["truth"]
            largest_idx = round(0.01 * len(scale))
            sort_idx = np.argsort(scale)
            largest_min = scale[sort_idx][-largest_idx - 1]
            largest_mask = scale > largest_min

            xrange = (1e-8, 1)
            bins = 60
            plot_delta_histogram(
                file,
                [delta_test, delta_train],
                labels=["Test", "Train"],
                title=title[0],
                xrange=xrange,
                xlabel=r"$|\Delta| = |\frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}|$ [\%]",
                bins=bins,
                logx=True,
                logy=False,
            )
            plot_delta_histogram(
                file,
                [delta_test, delta_test[largest_mask]],
                labels=["Test", "Largest 1\%"],
                title=title[0],
                xrange=xrange,
                xlabel=r"$|\Delta| = |\frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}|$ [\%]",
                bins=bins,
                logx=True,
                logy=False,
            )
            

    if cfg.plotting.delta_prepd and cfg.evaluate:
        out = f"{plot_path}/delta_prepd.pdf"
        with PdfPages(out) as file:
            dataset = "combined" if len(cfg.data.dataset) > 1 else cfg.data.dataset[0]
            delta_test = (
                plot_dict["results_test"][dataset]["preprocessed"]["prediction"]
                - plot_dict["results_test"][dataset]["preprocessed"]["truth"]
            ) / plot_dict["results_test"][dataset]["preprocessed"]["truth"]
            delta_train = (
                plot_dict["results_train"][dataset]["preprocessed"]["prediction"]
                - plot_dict["results_train"][dataset]["preprocessed"]["truth"]
            ) / plot_dict["results_train"][dataset]["preprocessed"]["truth"]

            # determine 1% largest amplitudes
            scale = plot_dict["results_test"][dataset]["preprocessed"]["truth"]
            largest_idx = round(0.01 * len(scale))
            sort_idx = np.argsort(scale)
            largest_min = scale[sort_idx][-largest_idx - 1]
            largest_mask = scale > largest_min

            xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]  # in %
            binss = [100, 50, 50]
            for xrange, bins in zip(xranges, binss):
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_train * 100],
                    labels=["Test", "Train"],
                    title=title[0],
                    xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=False,
                )
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_test[largest_mask] * 100],
                    labels=["Test", "Largest 1\%"],
                    title=title[0],
                    xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=False,
                )
                plot_delta_histogram(
                    file,
                    [delta_test * 100, delta_test[largest_mask] * 100],
                    labels=["Test", "Largest 1\%"],
                    title=title[0],
                    xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                    xrange=xrange,
                    bins=bins,
                    logy=True,
                )


def plot_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
):
    hists = []
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        axs[0].step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            axs[0].fill_between(
                bins,
                dup_last(hist) * scale,
                0.0 * dup_last(hist),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = np.divide(
            hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
        )  # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")
    if logy:
        axs[0].set_yscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(
        0.04,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_delta_histogram(
    file,
    datas,
    labels,
    title,
    xrange,
    bins=60,
    xlabel=None,
    logy=False,
    logx=False,
):
    assert len(datas) == 2
    dup_last = lambda a: np.append(a, a[-1])
    if logx:
        bins = np.logspace(np.log(xrange[0]), np.log(xrange[1]), bins)
    else:
        _, bins = np.histogram(datas[0], bins=bins - 1, range=xrange)
    hists, scales, mses = [], [], []
    for data in datas:
        mse = np.mean(data**2)

        data = np.clip(data, xrange[0], xrange[1])
        hist, _ = np.histogram(data, bins=bins, range=xrange)
        scale = 1 / np.sum((bins[1:] - bins[:-1]) * hist)
        mses.append(mse)
        hists.append(hist)
        scales.append(scale)

    fig, axs = plt.subplots(figsize=(6, 4))
    for hist, scale, mse, label, color in zip(
        hists, scales, mses, labels, colors[1:3][::-1]
    ):
        axs.step(
            bins,
            dup_last(hist) * scale,
            color,
            where="post",
            label=label + r" ($\overline{\Delta^2} = {%.2g})$" % (mse * 1e-4),
        )  # need 1e-4 to compensate for initial *100
        axs.fill_between(
            bins,
            dup_last(hist) * scale,
            0.0 * dup_last(hist) * scale,
            facecolor=color,
            alpha=0.1,
            step="post",
        )

    if logy:
        axs.set_yscale("log")
    if logx:
        axs.set_xscale("log")
    ymin, ymax = axs.get_ylim()
    if not logy:
        ymin = 0.0
    axs.vlines(0.0, ymin, ymax, color="k", linestyle="--", lw=0.5)
    axs.set_ylim(ymin, ymax)
    axs.set_xlim(xrange)

    axs.set_xlabel(xlabel, fontsize=FONTSIZE)
    axs.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs.legend(frameon=False, loc="upper left", fontsize=FONTSIZE * 0.7)
    axs.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_gradients(file, model, iteration):
    layer_names = []
    grad_values = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_values.append(param.grad.detach().cpu().view(-1).numpy())

    # Plotting
    n_layers = len(layer_names)
    if n_layers == 0:
        # print("No gradients to plot.")
        return
        
    fig, axs = plt.subplots(n_layers, 1, figsize=(6, 2 * n_layers))

    if n_layers == 1:
        axs = [axs]

    for ax, name, grads in zip(axs, layer_names, grad_values):
        ax.hist(grads, range=(grads.min(),grads.max()), bins=50, alpha=0.7)
        ax.set_title(f'Gradient Histogram: {name}')
        ax.set_xlabel("Gradient value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(grads.min(), grads.max())
        # print(name, grads.min(), grads.max(), grads.mean(), grads.std())
    plt.suptitle(f"Gradients at iteration {iteration}", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f'{file}/gradients_{iteration}.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def plot_weights(file, model, iteration):
    layer_names = []
    weight_values = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            layer_names.append(name)
            weight_values.append(param.data.detach().cpu().view(-1).numpy())

    n_layers = len(layer_names)
    if n_layers == 0:
        # print("No weights to plot.")
        return

    fig, axs = plt.subplots(n_layers, 1, figsize=(6, 2 * n_layers))

    if n_layers == 1:
        axs = [axs]

    for ax, name, weights in zip(axs, layer_names, weight_values):
        ax.hist(weights, range=(weights.min(), weights.max()), bins=50, alpha=0.7)
        ax.set_title(f'Weight Histogram: {name}')
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(weights.min(), weights.max())
        # print(name, weights.min(), weights.max(), weights.mean(), weights.std())

    plt.suptitle(f"Weights at iteration {iteration}", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f'{file}/weights_{iteration}.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def plot_histogram_single_output(
    file,
    data,  # List of arrays with shape (N,) or list of batches with shape (M,)
    labels,
    n_bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
    plot_ratios=True,
    pull=False,
    reference_truth=None,
):
    
    restructured_data = [np.concatenate([np.array(batch) for batch in dataset]) for dataset in data]
    flat_reference = np.concatenate([np.array(batch) for batch in reference_truth]) if reference_truth is not None else None
    
    if reference_truth is not None and len(restructured_data) == 2:
        truth_values = flat_reference
        largest_idx = max(1,round(0.01 * len(truth_values)))
        sort_idx = np.argsort(truth_values)
        largest_min = truth_values[sort_idx][-largest_idx - 1]
        largest_mask = truth_values > largest_min
        restructured_data[1] = restructured_data[1][largest_mask]
    
    if xrange is None:
        min_val = min(np.min(dat) for dat in restructured_data)
        max_val = max(np.max(dat) for dat in restructured_data)
        bins = np.linspace(min_val, max_val, n_bins + 1)
    else:
        bins = np.linspace(xrange[0], xrange[1], n_bins + 1)

    hists = []
    for dat in restructured_data:
        hist, _ = np.histogram(dat, bins=bins)
        hists.append(hist)

    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    # Plotting
    fig = plt.figure(figsize=(6, 6 if not plot_ratios else 8))
    if plot_ratios:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
        ax_ratio.set_ylabel("Ratio", fontsize=FONTSIZE)
    else:
        ax_main = fig.add_subplot(111)
        ax_ratio = None

    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        ax_main.step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            ax_main.fill_between(
                bins,
                dup_last(hist) * scale,
                0.0,
                facecolor=color,
                alpha=0.1,
                step="post",
            )
        elif plot_ratios:
            ratio = np.divide(
                hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
            )
            ax_ratio.step(
                bins, dup_last(ratio), linewidth=1.0, where="post", color=color
            )

    if logx:
        ax_main.set_xscale("log")
    if logy:
        ax_main.set_yscale("log")
    if pull:
        x = np.linspace(xrange[0], xrange[1], 1000) if xrange else np.linspace(min_val, max_val, 1000)
        gaussian = norm.pdf(x, 0, 1)
        ax_main.plot(x, gaussian, 'r--', linewidth=2, label='Gaussian (0, 1)')

    ax_main.set_ylabel("Normalized", fontsize=FONTSIZE)
    ax_main.set_title(title, fontsize=FONTSIZE + 2)
    ax_main.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax_main.legend(loc="best", frameon=False, fontsize=FONTSIZE_LEGEND)

    if xlabel:
        (ax_ratio if plot_ratios else ax_main).set_xlabel(xlabel, fontsize=FONTSIZE)

    if plot_ratios:
        ax_ratio.set_yticks(ratio_ticks)
        ax_ratio.set_ylim(ratio_range)
        for tick in ratio_ticks:
            ax_ratio.axhline(y=tick, c="black", ls="--" if tick == 1.0 else "dotted", lw=0.7)
        ax_ratio.tick_params(axis="both", labelsize=FONTSIZE_TICK)

    fig.tight_layout()
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()