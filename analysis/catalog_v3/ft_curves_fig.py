"""Validation curves of the separation-probe fine-tunes (zz_catft_sep, Fig. 15's runs): MSE of log|M|^2 against the
fine-tuning step for the Muon and the AdamW pretraining at D_train = 32 and 316, each beside its zero-shot error
(draft_rewrite step 5: does the error rise above the zero-shot start early in fine-tuning?).
Input analysis/catalog_v3/data/catft_curves.json (collect_catft_curves.py); val_curve[i] is the loss at step
(i + 1) * validate_every in standardized units, times prepd_std^2 the MSE of log|M|^2.
    python3 analysis/catalog_v3/ft_curves_fig.py   -> analysis/catalog_v3/figs/ft_curves_sep.png/.pdf
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps

PROBES = (("v3_gg_ttx", r"$gg\to t\bar t$"), ("v3_dbx_txu", r"$d\bar d\to t\bar u$"),
          ("v3_bbx_hhhz", r"$b\bar b\to HHHZ$"), ("v3_uux_wpwm", r"$u\bar u\to W^+W^-$"))
PARENTS = (("catft_parentstats_dimensionless_muon_MuonAdamW", "Muon", ps.C.blue),
           ("catft_parentstats_dimensionless_AdamW", "AdamW", ps.C.vermillion))
DS = (("d3", "32", "--"), ("d5", "316", "-"))


def curve(r):
    s = r["prepd_std"] ** 2
    v = np.array(r["val_curve"], float) * s
    return (np.arange(len(v)) + 1) * r["validate_every"], v, r["val_loss_step0"] * s


def main():
    d = json.load(open(os.path.join(HERE, "data", "catft_curves.json")))
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, (p, lab) in zip(axs.flat, PROBES):
        for par, name, c in PARENTS:
            for dk, dlab, ls in DS:
                st, v, z0 = curve(d[f"zz_catft_sep/{par}/{p}_{dk}_result.json"])
                ax.plot(st, v, ls, color=c, label=fr"{name} pretraining, $D_{{\rm train}}={dlab}$")
                print(f"{p} {name} D={dlab}: zero shot {z0:.3g}, first {v[0]:.3g} at step {st[0]}, min {v.min():.3g},"
                      f" last {v[-1]:.3g}, share above zero shot {np.mean(v > z0):.2f}")
            ax.axhline(z0, color=c, ls=":", label=f"{name} pretraining, zero shot")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("fine-tuning step"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, lab)
    ps.legend(axs[1][1], "lower left", fontsize=ps.LEGEND_MIN_PT)
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    ps.save(fig, os.path.join(HERE, "figs", "ft_curves_sep"))


if __name__ == "__main__":
    main()
