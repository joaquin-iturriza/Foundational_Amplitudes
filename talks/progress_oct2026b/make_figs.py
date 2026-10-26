"""Figures of the late-October weekly deck (talks/progress_oct2026b), every label in plain words.

    .venv-driver/bin/python talks/progress_oct2026b/make_figs.py      -> talks/progress_oct2026b/figs/*.{pdf,png}

Inputs, all read-only: analysis/catalog_daniel/data/*.npz (the zero-shot reads on Daniel's catalog),
analysis/catalog_v3/data/{short_reads,muon_ft,muon_hp_ft}.json, analysis/transfer/data/lowd_dyhpo.json, the transfer
study's cells (analysis/transfer/cells.py), and data/draft_pid_by_process.json: the per-process MSE of the draft's
"pID emb. only" run (seed 0) read off the vector paths of its figure mse_per_process_two_rows.pdf, matched to our
process names by the labels printed under each point (not committed: it is the collaborators' unpublished result).
"""
import contextlib, io, json, os, sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
import plot_style as ps  # noqa: E402

FIG = os.path.join(HERE, "figs")
DAN = os.path.join(ROOT, "analysis", "catalog_daniel")
V3 = os.path.join(ROOT, "analysis", "catalog_v3", "data")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


# ------------------------------------------------------------------ Daniel's catalog reads
def load(f):
    return dict(np.load(os.path.join(DAN, "data", f + ".npz")))


def hp58():
    """The per-process pretraining read with the statistics predicted from the process (zero_shot_compare.hp58)."""
    from heldout_delta import G
    z = load("heldout_preds")
    sn = json.load(open(os.path.join(DAN, "data", "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z:
            z[f"{n}__q"] = mu + np.exp(lsd) * z[f"{n}__h"].astype(float)
    return z


def parts(z):
    """Per process: (MSE, level = squared mean error, shape = variance of the error) of log|M|^2, no map fitted."""
    out = {}
    for k in z:
        if k.endswith("__t") and k[:-3] + "__q" in z:
            e = z[k[:-3] + "__q"].astype(float) - z[k].astype(float)
            out[k[:-3]] = (float(np.mean(e ** 2)), float(np.mean(e) ** 2), float(np.var(e)))
    return out


TWINS = set(json.load(open(os.path.join(DAN, "heldout_twins.json"))))
META = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(DAN, "classes_default.json")))["processes"]}
DRAFT = json.load(open(os.path.join(HERE, "data", "draft_pid_by_process.json")))   # {"interp"|"extrap": {name: [x, mse]}}

ARMS_FULL = [  # (label, reader); the full-length runs, 1e5 events per process, 32k steps
    ("per-process standardization\n(full run)", hp58),
    ("one shared standardization (full run,\nbest checkpoint before it diverged\nat step 8734 of 32000)", lambda: load("catalog_norm_B")),
    ("$\\hat s$ factor and coupling product\n(full run)", lambda: load("catalog_norm_A")),
    ("dimensionless target and inputs,\ncoupling orders (short run)", lambda: load("catalog_short_DlessInOrd_sigstage")),
]
GROUPS = {"interp": [("ss~ annihilation processes (flavor-degenerate with dd~ and uu~)", r"$s\bar s$ annihilation"),
                     ("Muon final states from bb~ annihilation", r"$b\bar b\to\mu,\nu_\mu$"),
                     ("g+s processes from gluon + light-quark families", r"$gs$"),
                     ("b+s(~) scattering from b+light-quark families", r"$b\bar s$, $sb$"),
                     ("c+u~ scattering from c+light-quark families", r"$c\bar u$"),
                     ("ds and s~s~ from same-sign light quark families", r"$ds$, $\bar s\bar s$"),
                     ("Same-sign charm-bottom scattering (cb, c~b~)", r"$cb$, $\bar c\bar b$")],
          "extrap": [("Multi-Higgs production (hh, hhh, Xhh)", r"multi-$H$"),
                     ("Higgs + tau lepton processes", r"$H+\tau$"),
                     ("Triple-W production (W+W+W-, W+W-W-)", r"$WWW$"), ("Triple-Z production (ZZZ)", r"$ZZZ$"),
                     ("gg -> W + qq' (W production from gluon fusion)", r"$gg\to W$ + quarks"),
                     ("gg -> tt~ + boson (ttX via gluon fusion)", r"$gg\to t\bar tX$")]}
PANEL_TITLE = {"interp": "held out, interpolation: 96 relabels of a training process",
               "extrap": "held out, extrapolation: 43 with no relabel in training"}


def proc_label(name):
    from heldout_delta import proc_label as pl
    return pl(META[name]["process"])


def fig_per_process():
    """The draft's mse_per_process_two_rows form: one point per held-out process, the draft's order and groups."""
    arms = [("draft, particle-ID embedding only (one run, seed 0)", None, "*", "k", 6),
            ("ours, one shared standardization (full run, best checkpoint before it diverged at step 8734 of 32000)", parts(ARMS_FULL[1][1]()), "o", ps.C.blue, 3.5),
            ("ours, dimensionless target and inputs, coupling orders (short run)", parts(ARMS_FULL[3][1]()), "D",
             ps.C.vermillion, 3.2)]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.2))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.73, bottom=0.10, hspace=1.0)
    for ax, sp in zip(axes, ("interp", "extrap")):
        names = sorted(DRAFT[sp], key=lambda n: DRAFT[sp][n][0])
        x = np.arange(len(names))
        for i, n in enumerate(names):
            if META[n]["n_out"] == 2:
                ax.axvspan(i - 0.5, i + 0.5, color="0.88", lw=0, zorder=0)
        for lab, m, mk, c, s in arms:
            y = [DRAFT[sp][n][1] for n in names] if m is None else [m[n][0] for n in names]
            ax.plot(x, y, mk, color=c, ms=s, ls="none", label=lab, zorder=3 if m is None else 2,
                    mfc=c, mec="none" if m is None else c, alpha=1 if m is None else 0.85)
        # group separators and names, in the draft's order
        g_of = [META[n]["test_group"] for n in names]
        start = 0
        for g, short in GROUPS[sp]:
            idx = [i for i, gg in enumerate(g_of) if gg == g]
            if not idx:
                continue
            lo, hi = min(idx), max(idx)
            if lo > 0:
                ax.axvline(lo - 0.5, color="0.4", lw=0.7)
            ax.text((lo + hi) / 2, 1.02, short, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([proc_label(n) for n in names], rotation=90, fontsize=3.0)
        ax.tick_params(axis="x", length=1.5, pad=1)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3e2)
        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        ax.set_yticklabels(["$10^{-4}$", "0.001", "0.01", "0.1", "1", "10", "100"], fontsize=7)
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ax.grid(False)
        ax.text(0.0, 1.16, PANEL_TITLE[sp], transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color="0.88")); l.append(r"grey band: $2\to2$ process (the others $2\to3$)")
    fig.legend(h, l, loc="upper center", ncol=1, fontsize=8, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
               columnspacing=1.5)
    save(fig, "dan_per_process")


def fig_grid():
    """The draft's mse_grid_by_type form: per-process MSE histograms, one row per run, medians dashed."""
    rows = [("draft: particle-ID\nembedding only\n(one run, seed 0)", None, "0.35")] + \
           [("ours: " + lab, parts(f()), c) for (lab, f), c in
            zip(ARMS_FULL, (ps.C.grey, ps.C.blue, ps.C.green, ps.C.vermillion))]
    edges = 10 ** (np.log10(2e-4) + 0.2 * np.arange(-5, 31))   # the draft's bins (0.2 decades on its grid from 2e-4), five more below for our smallest
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.0, 5.6), sharex=True)
    fig.subplots_adjust(left=0.23, right=0.93, top=0.94, bottom=0.10, hspace=0.12, wspace=0.16)
    out = {}
    for r, (lab, m, c) in enumerate(rows):
        for j, sp in enumerate(("interp", "extrap")):
            ax = axes[r, j]
            v = np.array([DRAFT[sp][n][1] for n in DRAFT[sp]]) if m is None else \
                np.array([m[n][0] for n in m if (n in TWINS) == (sp == "interp")])
            h, _ = np.histogram(v, bins=edges)
            dens = h / len(v) / 0.2
            ax.stairs(dens, edges, fill=True, color=c, alpha=0.35)
            ax.stairs(dens, edges, color=c, lw=1.4)
            med = float(np.median(v))
            out[f"{lab}|{sp}"] = dict(n=int(len(v)), median=med, below=int((v < edges[0]).sum()),
                                      above=int((v > edges[-1]).sum()))
            ax.axvline(med, color=c if m is not None else "k", ls="--", lw=1.3)
            ax.text(0.97, 0.88, f"median = {med:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color=c if m is not None else "k")
            ax.set_xscale("log")
            ax.set_xlim(edges[0], 150)
            ax.yaxis.tick_right(); ax.tick_params(labelsize=8)
            ax.set_ylim(0, max(1.0, dens.max() * 1.45))
            ax.grid(False)
            if r == 0:
                ax.set_title("held out, interpolation (96 relabels)" if sp == "interp"
                             else "held out, extrapolation (43 new)", fontsize=11)
            if j == 0:
                ax.text(-0.04, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
        pass
    for ax in axes[-1]:
        ax.set_xlabel(r"per-process MSE$(\log|\mathcal{M}|^2)$")
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        ax.set_xticklabels(["$10^{-4}$", "0.001", "0.01", "0.1", "1", "10", "100"])
    fig.text(0.985, 0.52, "fraction of the column's processes per decade of MSE", rotation=90, ha="center",
             va="center", fontsize=9)
    save(fig, "dan_grid")
    return out


# ------------------------------------------------------------------ what moves the zero-shot error
SHORT_ARMS = [  # (label, file stem, has 2->4 / 100 TeV reads)
    ("per-process standardization (full run)", "hp58", False),
    ("one shared standardization (short run)", "B", True),
    ("$\\hat s$ factor, fixed scale (short run)", "Adim", True),
    ("$\\hat s$ factor and coupling product (short run)", "A", False),
    ("dimensionless target (short run)", "Dless", True),
    ("dimensionless target, pair scale (short run)", "Dpair_seed1", True),
    ("dimensionless target and inputs (short run)", "DlessIn", True),
    ("dimensionless target and inputs,\nwithout diagrams (short run)", "DlessInNoDiag", True),
    ("dimensionless target and inputs,\ncoupling orders (short run)", "DlessInOrd", True),
    ("emission scales with coupling orders (short run)", "DlessInOrdQ", True),
    ("emission scales with coupling orders,\ncontext vector (short run)", "DlessInOrdQc", False),
]


def short_parts(stem, pre="catalog_short"):
    if stem == "hp58":
        return parts(hp58()) if pre == "catalog_short" else None
    f = f"{pre}_{stem}" + ("_sigstage" if pre == "catalog_short" else "")
    if not os.path.exists(os.path.join(DAN, "data", f + ".npz")):
        return None
    return parts(load(f))


def dotplot(name, sets):
    """One panel per held-out set, one row per run: the median per-process MSE and its level and shape parts."""
    ARMS = [a for a in SHORT_ARMS if any(short_parts(a[1], pre) is not None for _, pre, _ in sets)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.99, top=0.83, bottom=0.10, wspace=0.06)
    y = np.arange(len(ARMS))[::-1]
    out = {}
    for ax, (title, pre, sel) in zip(axes, sets):
        for (lab, stem, _), yy in zip(ARMS, y):
            p = short_parts(stem, pre)
            if p is None:
                ax.text(0.02, yy, "not evaluated", transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                        color="0.4")
                continue
            v = np.array([p[n] for n in p if sel(n)])
            if not len(v):
                continue
            med = np.median(v, axis=0)
            out[f"{title}|{stem}"] = dict(n=len(v), mse=med[0], level=med[1], shape=med[2])
            ax.plot([med[1]], [yy + 0.18], "^", color=ps.C.blue, ms=6, label="level part")
            ax.plot([med[2]], [yy - 0.18], "s", color=ps.C.orange, ms=5.5, label="shape part")
            ax.plot([med[0]], [yy], "o", color="k", ms=7, label="zero-shot MSE")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"median over processes of MSE$(\log|\mathcal{M}|^2)$")
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
        for yy in y[::2]:
            ax.axhspan(yy - 0.5, yy + 0.5, color="0.95", lw=0, zorder=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([a[0] for a in ARMS], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(ARMS) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    keep = {lab: hh for hh, lab in zip(h, l)}
    fig.legend([keep[k] for k in ("zero-shot MSE", "level part", "shape part")],
               ["zero-shot MSE", "level part: squared mean error", "shape part: variance of the error"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0), fontsize=10)
    save(fig, name)
    return out


def fig_moves():
    a = dotplot("dan_moves_heldout", [("96 relabels", "catalog_short", lambda n: n in TWINS),
                                      ("43 new processes", "catalog_short", lambda n: n not in TWINS)])
    b = dotplot("dan_moves_ood", [(r"40 probes, one more particle ($2\to4$)" + "\n" + r"(trained up to $2\to3$)", "catalog_2to4_short", lambda n: True),
                                  (r"40 training $2\to3$ processes at 100 TeV" + "\n" + "(trained at 13 TeV)", "catalog_2to4E100_short",
                                   lambda n: True)])
    return {**a, **b}


# ------------------------------------------------------------------ why a new catalog
def fig_why():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.12, wspace=0.30)
    ax = axes[0]
    cats = [r"$2\to2$", r"$2\to3$", r"$2\to4$", "one loop"]
    dan_str = [254, 1237, 0, 0]
    dan_cls = [65, 340, 0, 0]
    v3_cls = [65, 321, 1200, 150]
    x = np.arange(4)
    ax.bar(x - 0.27, dan_str, 0.27, color="0.75", label="Daniel's catalog, as listed (1491)")
    ax.bar(x, dan_cls, 0.27, color=ps.C.grey, label="Daniel's catalog, distinct processes (relabels merged)")
    ax.bar(x + 0.27, v3_cls, 0.27, color=ps.C.blue, label="new catalog, distinct SM processes (relabels merged; heavy-quark,\nBSM, extra $\\alpha_s$ and loop-induced pools not counted)")
    for xx, v in zip(x, dan_str):
        ax.text(xx - 0.27, v + 15, str(v), ha="center", fontsize=8)
    for xx, v in zip(x, dan_cls):
        ax.text(xx, v + 15, str(v), ha="center", fontsize=8)
    for xx, v in zip(x, v3_cls):
        ax.text(xx + 0.27, v + 15, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("number of processes")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9)
    ax.set_ylim(0, 1400)
    ax.grid(False)
    ax.set_title(r"Daniel's catalog has no $2\to4$ and no one loop", fontsize=10, pad=62)
    ax = axes[1]
    labs = ["Daniel's catalog", "new catalog"]
    rel = [96, 0]
    new = [43, 617]
    ax.barh([1, 0], rel, color="0.75", label="relabel of a training process")
    ax.barh([1, 0], new, left=rel, color=ps.C.vermillion, label="new: no relabel in training")
    ax.text(96 / 2, 1, "96", ha="center", va="center", fontsize=9)
    ax.text(96 + 43 + 8, 1, "43", ha="left", va="center", fontsize=9)
    ax.text(617 / 2, 0, "617 over 8 axes (the twin run adds 151 on 2 more)", ha="center", va="center", fontsize=8.5, color="w")
    ax.set_yticks([1, 0]); ax.set_yticklabels(labs)
    ax.set_xlabel("held-out processes")
    ax.set_ylim(-0.6, 1.9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9)
    ax.grid(False)
    ax.set_title("96 of his 139 held-out processes are relabels", fontsize=10, pad=40)
    save(fig, "why_new_catalog")


# ------------------------------------------------------------------ catalog_v3 zero shot
AXES = [("H1", "new combinations"), ("H2", r"$2\to4$ of new families"), ("H2p5", r"$2\to5$"),
        ("H3", "the physical top"), ("H3p", r"$\tau$, $\nu_\tau$ never seen"), ("H4", "quartic vertex never seen"),
        ("H5", "one loop"), ("H6", r"$\alpha_s=0.13$ (trained at 0.09, 0.15)"), ("H7c", r"$p_T\geq30$ GeV (trained at 10)"),
        ("H7E", r"$\sqrt{\hat s}$ 1 to 10 TeV (trained up to 1)")]


def v3_dots(name, arms, title_l, title_r, right="level", AXES=AXES, legend_title=None):
    R = json.load(open(os.path.join(V3, "short_reads.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.74 if len(arms) > 2 else 0.80, bottom=0.10, wspace=0.06)
    y = np.arange(len(AXES))[::-1]
    out = {}
    for ax, key, title in zip(axes, ("mse", right), (title_l, title_r)):
        for k, (arm, lab, c, mk) in enumerate(arms):
            off = (k - (len(arms) - 1) / 2) * 0.17
            for (h, _), yy in zip(AXES, y):
                r = R[arm].get(h)
                if r is None:
                    continue
                ax.plot([r[key]], [yy + off], mk, color=c, ms=6, label=lab)
                out[f"{arm}|{h}|{key}"] = r[key]
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"median over the axis's processes")
        ax.grid(True, axis="x", alpha=0.3); ax.grid(False, axis="y")
        for yy in y[::2]:
            ax.axhspan(yy - 0.5, yy + 0.5, color="0.95", lw=0, zorder=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([a[1] for a in AXES], fontsize=10)
    axes[0].set_ylim(-0.6, len(AXES) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    keep = dict(zip(l, h))
    fig.legend(list(keep.values()), list(keep.keys()), loc="upper center", ncol=1 if len(keep) > 2 else 2,
               fontsize=9.5, bbox_to_anchor=(0.6, 1.0), title=legend_title, title_fontsize=9.5)
    save(fig, name)
    return out


def fig_v3():
    a = v3_dots("v3_zero_shot",
                [("baseline", "one shared standardization (short run)", ps.C.grey, "o"),
                 ("dimensionless", "emission scales with coupling orders (short run)", ps.C.blue, "s"),
                 ("dimensionless_context", "emission scales with coupling orders, context vector (zeros for a new process)", ps.C.green, "D"),
                 ("dimensionless_twin", "emission scales with coupling orders, twin trained without $\\tau$, $\\nu_\\tau$, quartic vertex", ps.C.orange, "^")],
                r"zero-shot MSE$(\log|\mathcal{M}|^2)$", "its level part (squared mean error)")
    b = v3_dots("v3_muon",
                [("muon_search_best_adamw", "AdamW, best of its 8-trial search (not the run on the previous slide)", ps.C.grey, "o"),
                 ("muon_search_best_muon", "Muon, best of its 8-trial search", ps.C.vermillion, "s")],
                r"zero-shot MSE$(\log|\mathcal{M}|^2)$", "its level part (squared mean error)",
                AXES=[a for a in AXES if a[0] not in ("H3p", "H4")],
                legend_title="emission scales with coupling orders, short runs of 4264 steps, seed 42")
    return {**a, **b}


# ------------------------------------------------------------------ fine-tunes on catalog_v3 probes
PROBES = [("v3_dg_dgwpwm", r"$dg\to dgW^+W^-$, new combination"), ("v3_usx_cxuwpz", r"$u\bar s\to\bar c uW^+Z$, $2\to4$ of a new family"),
          ("v3_gg_attx", r"$gg\to\gamma t\bar t$, the top"), ("v3_udx_agwp_nlo", r"$u\bar d\to\gamma gW^+$, one loop")]


def fig_ft():
    M = json.load(open(os.path.join(V3, "muon_ft.json")))
    H = json.load(open(os.path.join(V3, "muon_hp_ft.json")))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.08, wspace=0.22, hspace=0.42)
    Ds = [10, 32, 100, 316]
    tag = {10: "d2", 32: "d3", 100: "d4", 316: "d5"}
    out = {}
    for ax, (p, plab) in zip(axes.flat, PROBES):
        def get(kind, parent, opt):
            return {r["D"]: r for r in M if r["kind"] == kind and r["probe"] == p and r["parent"] == parent
                    and r["optimizer"] == opt}
        sc = get("scratch", "dimensionless", "AdamW")
        fa = get("catft_parentstats", "dimensionless", "MuonAdamW")
        fm = get("catft_parentstats", "dimensionless_muon", "MuonAdamW")
        tuned = {D: fm[D]["mse_log"] / H["best"]["cell_gain"][f"{p}_{tag[D]}"] for D in Ds}
        ax.plot(Ds, [sc[D]["mse_log"] for D in Ds], "-o", color=ps.C.grey, label="from scratch (AdamW, the pretraining's HPs)")
        ax.plot(Ds, [fa[D]["mse_log"] for D in Ds], "-s", color=ps.C.blue, label="fine-tune of the AdamW pretraining (Muon, untuned)")
        ax.plot(Ds, [fm[D]["mse_log"] for D in Ds], "-^", color=ps.C.vermillion,
                label="fine-tune of the Muon pretraining (Muon, untuned)")
        ax.plot(Ds, [tuned[D] for D in Ds], "--v", color=ps.C.vermillion, mfc="w",
                label="fine-tune of the Muon pretraining, HPs of a 12-trial search")
        ax.axhline(fa[10]["mse_log_step0"], color=ps.C.blue, ls=":", label="zero shot, AdamW pretraining")
        ax.axhline(fm[10]["mse_log_step0"], color=ps.C.vermillion, ls=":", label="zero shot, Muon pretraining")
        out[p] = {D: dict(scratch=sc[D]["mse_log"], ft_adamw=fa[D]["mse_log"], ft_muon=fm[D]["mse_log"],
                          ft_muon_tuned=tuned[D]) for D in Ds}
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(Ds); ax.set_xticklabels([str(d) for d in Ds])
        ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        lo, hi = ax.get_ylim()
        tk = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3) if lo <= t <= hi]
        ax.set_yticks(tk); ax.set_yticklabels([f"{t:g}" for t in tk]); ax.set_ylim(lo, hi)
        ax.set_title(plab, fontsize=10.5)
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ax.set_xlabel("training events of the probe")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.53, 1.0),
               title="fine tunes keep the parent's standardization (no refit); untuned = the parent's learning rate, no layer decay; one seed per point",
               title_fontsize=9)
    save(fig, "v3_finetune")
    return out


def fig_lowd():
    L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "data", "lowd_dyhpo.json")))
    D = {"2": 10, "3": 10 ** 1.5, "4": 100, "5": 10 ** 2.5}
    meth = [("lpft", "linear probe, then fine-tune", ps.C.vermillion, "s"),
            ("l2sp", "pulled toward the pretrained weights", ps.C.green, "^"),
            ("freeze", "lower blocks frozen", ps.C.orange, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.66, bottom=0.12, wspace=0.36)
    k = sorted(D)
    x = [D[i] for i in k]
    C = L["cells"]
    def band(ax, num, den, j, c, mk, lab):
        med, lo, hi = [], [], []
        for i in k:
            r = np.array([c_[num] / c_[den] for c_ in C if str(c_["k"]) == i])
            q = np.percentile(r, [25, 50, 75]); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        xx = np.array(x) * 10 ** (0.025 * j)
        ax.errorbar(xx, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt="-" + mk, color=c, label=lab,
                    capsize=2.5, lw=1.4, elinewidth=0.9)
        return med
    band(axes[0], "scratch", "recorded_ft", -1.5, ps.C.blue, "o", "default fine-tune")
    for j, (m, lab, c, mk) in enumerate(meth):
        band(axes[0], "scratch", m, j - 0.5, c, mk, lab)
        band(axes[1], "recorded_ft", m, j - 1, c, mk, lab)
    axes[0].set_ylabel("MSE from scratch over MSE of the method\n(above 1: the method beats scratch)", fontsize=10)
    axes[1].set_ylabel("MSE of the default fine-tune over\nMSE of the method (above 1: method better)", fontsize=10)
    for ax in axes:
        ax.axhline(1, color="0.4", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(["10", "32", "100", "316"])
        ax.set_xlabel("training events of the probe")
        ax.minorticks_off()
    axes[0].set_yticks([0.5, 1, 2, 4, 8, 16]); axes[0].set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
    axes[1].set_yticks([0.25, 0.5, 1, 2, 4]); axes[1].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.53, 1.0),
               title="transfer study: pretrainings on 3, 15 and 19 processes; probes $e^+e^-\\to Z\\gamma$, $e^+e^-\\to d\\bar d$, $ud\\to ud$, $u\\bar u\\to gg$, one-loop $e^+e^-\\to b\\bar b$\n"
                     "points: median over the 15 cells (5 probes x 3 pretrainings), bars: middle half of the cells; each method at the best\n"
                     "of its own 5-trial search per cell. Default: full retraining with layer decay. Pulled toward the pretrained weights:\n"
                     "an L2 penalty on the distance to them, its strength searched. Frozen: 2, 4 or 6 of the 8 blocks, searched",
               title_fontsize=9)
    save(fig, "lowd_methods")


# ------------------------------------------------------------------ diagrams
def fig_diagrams_zs():
    from scipy.stats import wilcoxon
    sets = [("96 relabels", "catalog_short", lambda n: n in TWINS, ps.C.blue, "o"),
            ("40 training processes at 100 TeV", "catalog_2to4E100_short", None, ps.C.green, "s"),
            ("43 new processes", "catalog_short", lambda n: n not in TWINS, ps.C.vermillion, "D"),
            (r"40 with one more particle ($2\to4$)", "catalog_2to4_short", None, ps.C.purple, "^")]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.12)
    out = {}
    lo, hi = np.inf, -np.inf
    for lab, pre, sel, c, mk in sets:
        a = short_parts("DlessInNoDiag", pre); b = short_parts("DlessIn", pre)
        pp = [p for p in a if p in b and (sel is None or sel(p))]
        xa = np.array([a[p][0] for p in pp]); yb = np.array([b[p][0] for p in pp])
        better = int((yb < xa).sum())
        out[lab] = dict(n=len(pp), better=better, med_off=float(np.median(xa)), med_on=float(np.median(yb)),
                        p=float(wilcoxon(np.log(xa / yb)).pvalue))
        ax.plot(xa, yb, mk, color=c, ms=4.5, alpha=0.8, ls="none",
                label=f"{lab}: better with diagrams on {better} of {len(pp)}")
        lo, hi = min(lo, xa.min(), yb.min()), max(hi, xa.max(), yb.max())
    ax.plot([lo / 2, hi * 2], [lo / 2, hi * 2], "--", color="0.4", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo / 2, hi * 2); ax.set_ylim(lo / 2, hi * 2)
    ax.set_xlabel(r"zero-shot MSE$(\log|\mathcal{M}|^2)$, no diagrams")
    ax.set_ylabel(r"zero-shot MSE$(\log|\mathcal{M}|^2)$, with diagrams")
    ax.legend(loc="upper left", fontsize=8.5, title="dimensionless target and inputs (short run),\nwith and without diagrams", title_fontsize=8.5)
    save(fig, "diagrams_zero_shot")
    return out


def fig_diagrams_ft():
    sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from cells import S  # noqa: F401
        from finale_synth import LAB, P, cell
        from levdiagl import cell32
    rows = [(4, r"100 events", lambda p: cell("tp3_levdiaglfte", p, 4), lambda p: cell("tp3_r9fte", p, 4), "o", ps.C.blue),
            (6, r"1000 events", lambda p: cell("tp3_levdiaglfte", p, 6), lambda p: cell("tp3_r9fte", p, 6), "s",
             ps.C.green),
            (8, r"10000 events", lambda p: cell32("tp3_levdiaglfte", p, 8), lambda p: cell32("tp3_r9fte", p, 8), "D",
             ps.C.vermillion)]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.70, bottom=0.12)
    out = {}
    for k, (_, lab, arm, ref, mk, c) in enumerate(rows):
        xs, ys = [], []
        for i, p in enumerate(P):
            a, r = arm(p), ref(p)
            if a and r:
                xs.append(r / a); ys.append(i + (k - 1) * 0.22)
        v = np.array(xs)
        out[lab] = dict(n=len(v), median=float(np.median(v)), better=int((v > 1).sum()))
        ax.plot(xs, ys, mk, color=c, ms=5.5, ls="none",
                label=f"{lab}: diagrams better on {int((v > 1).sum())} of {len(v)} probes")
    ax.axvline(1, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(P))); ax.set_yticklabels([LAB[p].replace("1-loop", "one loop") for p in P])
    ax.invert_yaxis()
    ax.set_xlabel("MSE without diagrams over MSE with diagrams\n(right of 1: the diagrams help)")
    ax.set_xticks([0.3, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(False, axis="y")
    fig.legend(loc="upper center", ncol=1, fontsize=9, bbox_to_anchor=(0.62, 1.0), title="fine tunes of the transfer study's 19-process pretraining (64k steps), with and\nwithout the diagram encoder of Daniel's paper; 100 and 1000 events: best of a\n5-trial search per probe, 10000 events: best of 2 chosen points at 32k steps",
               title_fontsize=9)
    save(fig, "diagrams_finetune")
    return out


if __name__ == "__main__":
    res = {}
    fig_per_process()
    res["grid"] = fig_grid()
    res["moves"] = fig_moves()
    fig_why()
    res["v3"] = fig_v3()
    res["ft"] = fig_ft()
    fig_lowd()
    res["diag_zs"] = fig_diagrams_zs()
    res["diag_ft"] = fig_diagrams_ft()
    json.dump(res, open(os.path.join(HERE, "data", "deck_numbers.json"), "w"), indent=1, default=float)
    for k, v in res.items():
        print(k, json.dumps(v, default=float)[:1500])
