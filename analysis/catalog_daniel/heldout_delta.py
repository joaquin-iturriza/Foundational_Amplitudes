"""The catalog pretraining (hp58) on the 139 held-out processes of arXiv:2606.23791's split, per event (D74; data from
scripts/job_catalog_heldout_preds.sh via collect_heldout_preds.py -> data/heldout_preds.npz). The parent standardizes
per process, so a held-out process has a prediction in log|M|^2 itself only when it relabels a training process (its
twin's statistics, analysis/catalog_daniel/heldout_twins.json); the 43 others are scored after the best affine map
a + b h only. "On its own" (q_own): the output read in the probe's own standardized units, its train split's mean
and std of log|M|^2 (two numbers from its data, nothing fitted to the predictions). Delta = |M|^2_pred / |M|^2 - 1 = exp(q - t) - 1 (natural-log target).
  heldout_delta_{a,b}   fraction of events with |Delta| below x, one thin line per process (a: raw, the 96
                        relabels; b: after the affine map, all 139), the median process bold
  heldout_mse_{interpolation,extrapolation}   the user's figure (arXiv:2606.23791's held-out figure): MSE(log|M|^2)
                        per process, named, grouped by test group, on its own and after the map
    python analysis/catalog_daniel/heldout_delta.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

X = np.logspace(-6, 1, 141)
# short names of the test groups for the per-process figure (classes_default.json test_group)
SHORT = {"ss~ annihilation processes (flavor-degenerate with dd~ and uu~)": r"$s\bar s$ annihilation",
         "Muon final states from bb~ annihilation": r"$b\bar b\to\mu,\nu_\mu$",
         "g+s processes from gluon + light-quark families": r"$gs$",
         "b+s(~) scattering from b+light-quark families": r"$b\bar s$, $sb$",
         "c+u~ scattering from c+light-quark families": r"$c\bar u$",
         "ds and s~s~ from same-sign light quark families": r"$ds$, $\bar s\bar s$",
         "Same-sign charm-bottom scattering (cb, c~b~)": r"$cb$, $\bar c\bar b$",
         "Multi-Higgs production (hh, hhh, Xhh)": r"multi-$H$", "Higgs + tau lepton processes": r"$H+\tau$",
         "Triple-W production (W+W+W-, W+W-W-)": r"$WWW$", "Triple-Z production (ZZZ)": r"$ZZZ$",
         "gg -> W + qq' (W production from gluon fusion)": r"$gg\to Wq\bar q'$",
         "gg -> tt~ + boson (ttX via gluon fusion)": r"$gg\to t\bar tX$",
         "Single-particle production (2 -> 1 resonances)": r"$2\to1$"}


SYM = {"a": r"\gamma", "g": "g", "h": "H", "z": "Z", "w+": "W^+", "w-": "W^-", "e+": "e^+", "e-": "e^-",
       "mu+": r"\mu^+", "mu-": r"\mu^-", "ta+": r"\tau^+", "ta-": r"\tau^-", "ve": r"\nu_e", "vm": r"\nu_\mu",
       "vt": r"\nu_\tau", "ve~": r"\bar\nu_e", "vm~": r"\bar\nu_\mu", "vt~": r"\bar\nu_\tau"}


def proc_label(proc):
    r"""MadGraph syntax ("b b~ > a h h") as mathtext ($b\bar b\to\gamma HH$)."""
    def one(tok):
        if tok in SYM:
            return SYM[tok]
        return (r"\bar " + tok[:-1]) if tok.endswith("~") else tok
    ini, fin = proc.split(" > ")
    return "$" + " ".join(one(t) for t in ini.split()) + r"\to " + " ".join(one(t) for t in fin.split()) + "$"


def cdf(d):
    d = np.sort(np.abs(d[np.isfinite(d)]))
    return np.searchsorted(d, X, side="right") / len(d)


G = "phys_gbr"   # the statistics map used for "on its own" (stats_net.py: physics features + trees, fit on the training)


def main():
    z = dict(np.load(os.path.join(HERE, "data", "heldout_preds.npz")))
    # hp58 read with g's predicted statistics: q_g = mu_hat + sigma_hat h, nothing from the held-out process's data
    sn = json.load(open(os.path.join(HERE, "data", "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z:
            z[f"{n}__q_g"] = mu + np.exp(lsd) * z[f"{n}__h"].astype(float)
    probes = sorted({k.split("__")[0] for k in z})
    twins = json.load(open(os.path.join(HERE, "heldout_twins.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    groups = {sp: sorted({meta[p]["test_group"] for p in probes if meta[p]["split"] == sp})
              for sp in ("interpolation", "extrapolation")}

    LAB = {"q_g": "on its own (statistics from g)", "q_own": "with its own data's statistics",
           "q_aff": "after the best rescale and shift"}
    figs = ps.panels(3)
    for (fig, ax), kind in zip(figs, ("q_g", "q_own", "q_aff")):
        for rel, c in ((True, ps.C.blue), (False, ps.C.vermillion)):
            P = [p for p in probes if (p in twins) == rel and f"{p}__{kind}" in z]
            if not P:
                continue
            C = np.array([cdf(np.exp(z[f"{p}__{kind}"].astype(float) - z[f"{p}__t"].astype(float)) - 1) for p in P])
            for row in C:
                ax.plot(X, row, "-", color=c, alpha=0.15, lw=0.6)    # weight encodes "one process" vs "median"
            ax.plot(X, np.median(C, 0), "-", color=c, label=f"{'relabel' if rel else 'no relabel'} ({len(P)}), median")
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_xlabel(r"$|\Delta|$"); ax.set_ylabel(r"fraction of events below")
        ps.process_label(ax, LAB[kind], loc="lower right")
    ax = figs[0][1]
    ax.plot([], [], "-", color="grey", alpha=0.3, lw=0.6, label="one process")
    ps.shared_legend(figs[0][0], ax, ncol=2)      # a corner legend would stretch the 0..1 axis
    ps.save_panels(figs, os.path.join(HERE, "figs", "heldout_delta"))

    # the user's figure (arXiv:2606.23791's held-out figure, interpolation and extrapolation): one point per process,
    # named on the x axis, its test group named above. Wider than the house plot box on purpose: it is that figure.
    import matplotlib.pyplot as plt
    for sp in ("interpolation", "extrapolation"):
        P = [p for g in groups[sp] for p in sorted(p for p in probes if meta[p]["split"] == sp and meta[p]["test_group"] == g)]
        fig, ax = plt.subplots(figsize=(8.0, 2.9))
        x = np.arange(len(P))
        for kind, c, mfc, lab in (("q_g", ps.C.vermillion, None, LAB["q_g"]),
                                  ("q_own", ps.C.green, None, LAB["q_own"]),
                                  ("q_aff", ps.C.blue, "none", LAB["q_aff"])):
            y = [np.mean((z[f"{p}__{kind}"].astype(float) - z[f"{p}__t"].astype(float)) ** 2) for p in P]
            ax.plot(x, y, "o", color=c, mfc=mfc or c, ms=3.5, label=lab)
        ax.set_yscale("log"); ax.set_xlim(-0.7, len(P) - 0.3)
        ax.set_xticks(x, [proc_label(meta[p]["process"]) for p in P], rotation=90, fontsize=5.5)
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        i0 = 0
        for g in groups[sp]:
            n = sum(meta[p]["test_group"] == g for p in P)
            if i0:
                ax.axvline(i0 - 0.5, color="grey", lw=0.6)
            ax.text(i0 + n / 2 - 0.5, 1.02, SHORT.get(g, g), transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7)
            i0 += n
        fig.legend(*ax.get_legend_handles_labels(), loc="upper right", fontsize=7, frameon=False, ncol=3)
        fig.text(0.08, 0.97, f"held-out: {sp}", fontsize=9, va="top")
        fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.36)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(HERE, "figs", f"heldout_mse_{sp}.{ext}"), dpi=200)
        plt.close(fig)
    for sp in groups:
        print(sp, "groups:", "; ".join(f"{i} {g}" for i, g in enumerate(groups[sp], 1)))
    for kind in ("q_g", "q_own", "q_aff", "q"):
        P = [p for p in probes if f"{p}__{kind}" in z]
        m = {p: float(np.mean((z[f"{p}__{kind}"] - z[f"{p}__t"]) ** 2)) for p in P}
        md = {p: float(np.median(np.abs(np.exp(z[f"{p}__{kind}"].astype(float) - z[f"{p}__t"].astype(float)) - 1)))
              for p in P}
        for rel in (True, False):
            Q = [p for p in P if (p in twins) == rel]
            if Q:
                print(f"{kind} {'relabel' if rel else 'no relabel'} ({len(Q)}): MSE median {np.median([m[p] for p in Q]):.3g}"
                      f" (10-90% {np.quantile([m[p] for p in Q], .1):.3g} to {np.quantile([m[p] for p in Q], .9):.3g}),"
                      f" median |Delta| per process median {np.median([md[p] for p in Q]):.3g}")
        for p in sorted(P, key=lambda p: -m[p])[:6]:
            print(f"   worst {kind}: {p} MSE {m[p]:.3g} median|Delta| {md[p]:.3g} ({meta[p]['test_group']})")


if __name__ == "__main__":
    main()
