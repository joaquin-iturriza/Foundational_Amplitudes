"""Figures of the paper draft's Sec. 3 (plan optimizations_recap), from reads already on this machine; no new numbers.
Each scatter is one point per class: its MSE of ln|M|^2 (geometric mean over the class's probes, and over seeds where
an arm has two) under one setup against another, every set of classes in its own colour on one panel.

    python3 analysis/catalog_v4/paper_sec3_figs.py [pidqn|muon|examples|ft|pp|target|all]
  pidqn     one-hot PDG token against quantum numbers (mixture_ab.PIDQN, the 1x width-8 cell)   -> figs/paper_pidqn
  muon      AdamW against Muon, final models of the 68224-step pair (muon_long.RUNS)            -> figs/paper_muon
  examples  true and predicted ln|M|^2 on three pp test classes at Muon's 10/50/90th percentile of per-probe MSE,
            ratio panel below                                                                   -> figs/paper_muon_examples_{a,b,c}
  ft        fine-tunes of the AdamW and Muon parents at the parent's standardization against D, each parent's
            zero shot (step 0) and scratch (analysis/catalog_v3/data/muon_ft.json)              -> figs/paper_muon_ft
  pp        catalog_v4 with a 0.25 default-sampled share against the arm that trains its hadronic tree classes on
            13 TeV pp events only (mixture_ab.MIX / BOUND_DEFAULT, MSE runs, two seeds each)    -> figs/paper_pp
  unseen    zero-shot MSE per process of the pretraining without tau/nu_tau and quartic-gauge pools (catalog_v3 twin,
            short_reads.json dimensionless_twin): tau processes, quartic-vertex processes, new combinations -> figs/paper_unseen
  pp_direct per class on the 13 TeV pp test pools, the catalog pretraining (Muon, geometric mean of seeds 42 and 43) against
            the pretraining whose hadronic classes train on pp events only (AdamW, one seed); classes without b whose label
            is the same function in both (pp_physics_rescore 'agree'), trained in both and held out in both
            (compare_pp_trained.rows, par:v4_vs_pp)                                              -> figs/paper_pp_direct
  pp_bound  the same axes at one optimizer and length: pp-only bound arm against the catalog control on PP4, the classes
            the bound arm trains and the others (mixture_ab.BOUND, par:v4_pp_bound)              -> figs/paper_pp_bound
  target    zero-shot MSE per held-out probe, dimensionless target against ln|M|^2 in GeV (catalog_v3 short arms,
            analysis/catalog_v3/data/short_reads.json)                                          -> figs/paper_target
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import mixture_ab as m
import plot_style as ps

FIGS = os.path.join(HERE, "figs")
MSE_LAB = r"MSE$(\log|\mathcal{M}|^2)$"


def by_class(x, cls):
    """{class: geometric mean over its probes} of {probe key: mse}."""
    d = {}
    for k, v in x.items():
        while isinstance(k, tuple):      # (read, probe) or (read, (axis, probe))
            k = k[-1]
        d.setdefault(cls(k), []).append(v)
    return {c: m.gmean(v) for c, v in d.items()}


def pair_sets(run_a, run_b, sets, cls):
    """[(label, x, y)] per set: classes read by both, x from run_a, y from run_b (a run may be a seed pair)."""
    out = []
    for lab, reads in sets:
        xa, xb = {}, {}
        for r in reads:
            for run, acc in ((run_a, xa), (run_b, xb)):
                seeds = run if isinstance(run, tuple) else (run,)
                acc.update({(r, k): v for k, v in m.side([m.read_of(s, r) for s in seeds]).items()})
        keys = set(xa) & set(xb)
        a, b = by_class({k: xa[k] for k in keys}, cls), by_class({k: xb[k] for k in keys}, cls)
        cs = sorted(set(a) & set(b))
        out.append((lab, np.array([a[c] for c in cs]), np.array([b[c] for c in cs])))
    return out


COLS = (ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.purple, ps.C.orange)


def strip(row, base, span=1, **kwargs):
    """The legend of a row of the paper's panels, `(fig, ax)` each saved, as one strip across the row's width
    (`span` times it where the row's other panel, without a legend, is not in `row`)."""
    ps.legend_strip([ax for _, ax in row], base, width=span * sum(f.get_size_inches()[0] for f, _ in row), **kwargs)


def scatter(groups, xlab, ylab, base, row=(), span=1):
    """One point per class and set, saved alone with its legend strip (the figures outside the draft)."""
    fig, ax = ps.figure(width="half")
    scatter_ax(ax, groups, xlab, ylab, base)
    ps.save(fig, os.path.join(FIGS, base))
    strip([(fig, ax)] + list(row), os.path.join(FIGS, base + "_legend"), span=span, markerscale=2.5)


def scatter_ax(ax, groups, xlab, ylab, name, legend="lower right"):
    """One point per class and set on `ax`, the y = x line, the legend inside (Joaquin 2026-11-17: one file per
    figure, legend inside, entries in plain words)."""
    lo = min(min(g[1].min(), g[2].min()) for g in groups)
    hi = max(max(g[1].max(), g[2].max()) for g in groups)
    ax.plot([lo, hi], [lo, hi], color=ps.C.grey, ls="--", label=r"$y=x$")
    for (lab, x, y), c in zip(groups, COLS):
        ax.scatter(x, y, color=c, alpha=0.45, s=6, linewidths=0, label=lab, rasterized=True,
                   zorder=2 + 1 / len(x))   # smaller sets on top: a large set drawn later would cover them
        print(f"  {name} {lab!r}: {len(x)} classes, median y/x {np.median(y / x):.3g}, "
              f"medians x {np.median(x):.3g} y {np.median(y):.3g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    # equal x and y limits, so y = x runs corner to corner (Joaquin 2026-11-17); the range is the data's, opened only
    # as far as the legend needs (Joaquin 2026-11-17, third round: 'compact the range as little as possible'): the
    # compact legend goes in whichever off-diagonal corner covers fewer points, then both limits open in steps of
    # 5% of the data's span (in decades) until it covers none
    L, H = lo / (hi / lo) ** 0.04, hi * (hi / lo) ** 0.04
    ax._ps_equal_limits = True     # ps.make_room leaves these limits alone

    def lim(L, H):
        ax.set_xlim(L, H); ax.set_ylim(L, H)   # the house plot box (2.40x1.92in) stays: y = x is its diagonal
        ax.lines[0].set_data([L, H], [L, H])
    lim(L, H)
    if not legend:
        return
    ps.layout(ax.figure)
    kw = dict(fontsize=ps.LEGEND_MIN_PT, markerscale=1.5, handlelength=1.2, handletextpad=0.3, labelspacing=0.2,
              borderpad=0.2, borderaxespad=0.2)
    hs, ls = ax.get_legend_handles_labels()
    # a legend in the corner of an equal-limit scatter clears the diagonal only if (its width / the box's) + (its
    # height / the box's) < 1, whatever the range: one legend if it clears with at most half the data's span opened,
    # else the entries split between the two off-diagonal corners (both empty when the points follow y = x)
    k = (len(hs) + 1) // 2
    for parts in ([range(len(hs))], [range(k), range(k, len(hs))]):
        for a in ax.artists[:]:
            a.remove()
        L, H = L0, H0 = lo / (hi / lo) ** 0.04, hi * (hi / lo) ** 0.04
        lim(L, H)
        legs = []
        for i, idx in enumerate(parts):
            if legs:
                ax.add_artist(legs[-1])
            h, l = [hs[j] for j in idx], [ls[j] for j in idx]
            locs = ("upper left", "lower right") if len(parts) == 1 else (("upper left", "lower right")[i],)
            legs.append(ps.legend_free(ax, locs, handles=h, labels=l, **kw))
        f = (hi / lo) ** 0.05
        while any(ps.covered(ax, g, lines=False) for g in legs) and np.log(H / L) < 1.5 * np.log(H0 / L0):
            L, H = L / f, H * f
            lim(L, H)
        if not any(ps.covered(ax, g, lines=False) for g in legs):
            break
    print(f"  {name}: {len(legs)} legend(s), limits {L:.3g} to {H:.3g}, data {lo:.3g} to {hi:.3g}"
          + ("" if not any(ps.covered(ax, g, lines=False) for g in legs) else " -- STILL COVERS POINTS"))


SETS = (("$pp$, in pretraining,\nsampled from PDFs", ("PP4",)),
        ("$e^+e^-$, in\npretraining", ("EE4",)),
        ("held out, sampled\nby our mixture", ("held",)),
        ("held out, sampled\nby MadEvent", ("HD4", "HD5")))


def pidqn():
    """Particle encoding: the per-class scatter (paper_pidqn) and, as its own figure (Joaquin 2026-11-17: 'what is the
    relation between them?' -- two quantities, two figures), the zero-shot errors of the pretraining without tau,
    nu_tau and quartic vertices (paper_unseen)."""
    g = pair_sets(m.PIDQN["qn"], m.PIDQN["pid"], SETS, m.class_of())
    fig, a = ps.figure(width="half")
    scatter_ax(a, g, f"{MSE_LAB}, particles encoded\nby their quantum numbers",
               f"{MSE_LAB}, particles encoded\nby particle IDs", "paper_pidqn", legend="upper left")
    ps.save(fig, os.path.join(FIGS, "paper_pidqn"))
    fig, b = ps.figure(width="half")
    unseen(b)
    ps.save(fig, os.path.join(FIGS, "paper_unseen"))


def muon():
    """Optimizer: one file, the per-class scatter and the three true-against-predicted examples (2x2)."""
    import muon_long as ml
    g = pair_sets(ml.RUNS["muon"], ml.RUNS["adamw"], SETS, m.class_of())
    fig, axs = ps.figure(ncols=2, nrows=2)
    scatter_ax(axs[0][0], g, f"{MSE_LAB}, Muon optimizer", f"{MSE_LAB}, AdamW optimizer", "paper_muon",
               legend="upper left")
    examples([axs[0][1], axs[1][0], axs[1][1]])
    ps.save(fig, os.path.join(FIGS, "paper_muon"))




def examples(axes):
    import muon_long as ml
    col = {"muon": ps.C.blue, "adamw": ps.C.vermillion}
    lab = {"muon": "Muon optimizer", "adamw": "AdamW optimizer"}
    h = {k: json.load(open(os.path.join(m.READS, f"{run}_hists.json"))) for k, run in ml.RUNS.items()}
    mu = m.read_of(ml.RUNS["muon"], "PP4")
    keys = sorted((p for p in mu if f"PP4:{p}" in h["muon"] and f"PP4:{p}" in h["adamw"]), key=lambda p: mu[p])
    pick = [keys[int(q * (len(keys) - 1))] for q in (0.1, 0.5, 0.9)]
    def rebin(e, c):
        # the records' bins merged to ~40, as the other true-against-predicted figures (Joaquin 2026-11-17: Fig. 14
        # had fewer bins than the others; merged by a fixed four it kept 25)
        REB = max(1, len(c) // 40)
        n = (len(c) // REB) * REB
        return e[:n + 1:REB], np.add.reduceat(np.asarray(c[:n], float), np.arange(0, n, REB))
    for i, (ax, p) in enumerate(zip(axes, pick)):
        models = []
        for k in ("muon", "adamw"):     # each run's record bins truth and prediction on its own edges
            rec = h[k][f"PP4:{p}"]
            ek, t = rebin(np.array(rec["edges"]), rec["truth"])
            _, q = rebin(np.array(rec["edges"]), rec["pred"])
            models.append(dict(label=lab[k], edges=ek, truth=t, pred=q, style=dict(color=col[k])))
        ps.true_pred_hist(ax, models, xlabel=r"$\log|\mathcal{M}|^2$")
        ps.process_label(ax, ml.process_tex(p))
        if i == 0:
            ps.legend(ax, "upper left")


FT_MEDIAN = {"v3_dg_dgwpwm": r"$dg\to dgW^+W^-$", "v3_usx_cxuwpz": r"$u\bar s\to \bar c uW^+Z$",
             "v3_gg_attx": r"$gg\to\gamma t\bar t$", "v3_udx_agwp_nlo": r"$u\bar d\to\gamma gW^+$, one loop"}


# the Fig. 15 rerun (Joaquin 2026-11-17, 'processes that have good losses'): the 10th-percentile probe per held-out axis
# of the two parents' geometric-mean zero-shot error (muon_compare.py low-probes), same cells and protocol
FT_LOW = {"v3_dbx_bxdgg": r"$d\bar d\to b\bar bgg$", "v3_udx_gwpzz": r"$u\bar d\to gW^+ZZ$",
          "v3_epem_mupmumttx": r"$e^+e^-\to\mu^+\mu^-t\bar t$", "v3_uux_wpwm_nlo": r"$u\bar u\to W^+W^-$, one loop"}


# the Fig. 15 probes after Joaquin 2026-11-17 ('relatively good performance, but with also good separation between
# muon and adam'): muon_compare.py sep-probes, same cells and protocol
FT_SEP = {"v3_gg_ttx": r"$gg\to t\bar t$", "v3_dbx_txu": r"$d\bar d\to t\bar u$",
          "v3_bbx_hhhz": r"$b\bar b\to HHHZ$", "v3_uux_wpwm": r"$u\bar u\to W^+W^-$"}


def ft(src="muon_ft_sep.json", probes=FT_SEP, base="paper_muon_ft"):
    """Fine-tunes at small D_train of the Muon and AdamW pretrainings: the low-loss probes (the draft's figure);
    `ft_median` draws the first version, on the median-error probes (muon_ft.json)."""
    d = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", src)))
    FT_PROBES = probes
    arms = (("dimensionless", "AdamW", ps.C.vermillion, "AdamW optimizer"),
            ("dimensionless_muon", "MuonAdamW", ps.C.blue, "Muon optimizer"))
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, (p, name) in zip(axs.flat, FT_PROBES.items()):
        for parent, opt, c, lab in arms:
            rs = sorted((r for r in d if r["kind"] == "catft_parentstats" and r["parent"] == parent
                         and r["optimizer"] == opt and r["probe"] == p), key=lambda r: r["D"])
            ax.plot([r["D"] for r in rs], [r["mse_log"] for r in rs], "o-", color=c, label=f"{lab}, fine-tuned")
            ax.axhline(rs[0]["mse_log_step0"], color=c, ls=":", label=f"{lab}, before fine-tuning")
            print(f"  ft {p} {lab}: zero shot {rs[0]['mse_log_step0']:.3g}, D=10 {rs[0]['mse_log']:.3g}, "
                  f"D=316 {rs[-1]['mse_log']:.3g}")
        sc = sorted((r for r in d if r["kind"] == "scratch" and r["probe"] == p), key=lambda r: r["D"])
        ax.plot([r["D"] for r in sc], [r["mse_log"] for r in sc], "s--", color=ps.C.grey, label="from scratch")
        ax.set_xscale("log"); ax.set_yscale("log")
        ys = [l.get_ydata() for l in ax.lines]
        ax.set_ylim(0.7 * min(np.min(y) for y in ys), 4 * max(np.max(y) for y in ys))   # headroom: the process label sits above the zero-shot lines
        ax.set_xlabel(r"training events $D_{\rm train}$"); ax.set_ylabel(MSE_LAB)
        ps.process_label(ax, name)
    ps.legend(axs.flat[0], "lower left")
    ps.save(fig, os.path.join(FIGS, base))


def pp():
    import yaml
    cls = m.class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    bound = {"PP4": m.BOUND_DEFAULT["PP4"], "EE4": m.BOUND_DEFAULT["EE4"], "HD4": m.BOUND_DEFAULT["HD"], "HD5": m.BOUND_DEFAULT["HD"]}
    share = {"PP4": m.MIX["default25"]["PP4"], "EE4": m.MIX["default25"]["EE4"],
             "HD4": m.HD4_MSE["default25"], "HD5": m.HD5_MSE["default25"]}
    groups = []
    for lab, reads, keep in (("pp, trained", ("PP4",), lambda c: c in trained),
                             ("pp, held out", ("PP4",), lambda c: c not in trained),
                             (r"$e^+e^-$", ("EE4",), lambda c: True),
                             ("held out, MadEvent", ("HD4", "HD5"), lambda c: True)):
        x, y = {}, {}
        for r in reads:
            x.update({(r, k): v for k, v in m.side([m.read_of(s, r) for s in bound[r]]).items()})
            y.update({(r, k): v for k, v in m.side([m.read_of(s, r) for s in share[r]]).items()})
        keys = set(x) & set(y)
        a, b = by_class({k: x[k] for k in keys}, cls), by_class({k: y[k] for k in keys}, cls)
        cs = sorted(c for c in set(a) & set(b) if keep(c))
        groups.append((lab, np.array([a[c] for c in cs]), np.array([b[c] for c in cs])))
    scatter(groups, f"{MSE_LAB}, pp events only", f"{MSE_LAB}, catalog", "paper_pp")


def target():
    res = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads.json")))
    extra = ("H2p5", "H7E")
    groups = []
    for grp, lab in ((None, "held out, multiplicity\nand energy as pretrained"), ("H2p5", "held out, $2\\to5$\n(pretrained $\\leq2\\to4$)"),
                     ("H7E", "held out, $1\\!-\\!10$ TeV\n(pretrained $\\leq1$ TeV)")):
        xs, ys = [], []
        for a in res["baseline"]:
            if (a if a in extra else None) != grp or a not in res["dimensionless"]:
                continue
            px, py = res["baseline"][a].get("probes", {}), res["dimensionless"][a].get("probes", {})
            for p in sorted(set(px) & set(py)):
                xs.append(px[p]["mse"]); ys.append(py[p]["mse"])
        groups.append((lab, np.array(xs), np.array(ys)))
    fig, (a, b) = ps.figure(ncols=2)
    scatter_ax(a, groups, "zero-shot " + MSE_LAB + ",\n" + r"trained on $\log|\mathcal{M}|^2$ in GeV",
               "zero-shot " + MSE_LAB + ",\n" + r"trained on dimensionless $\log|\mathcal{M}|^2$", "paper_target",
               legend="upper left")
    inputs(b)
    ps.save(fig, os.path.join(FIGS, "paper_target"))


def pp_direct():
    import compare_pp_trained as cp
    C = json.load(open(os.path.join(HERE, "data", "pp_physics_check.json")))["classes"]
    R, p, _ = cp.rows()
    role, b = p["_role"], p["_b"]
    groups = []
    for lab, g in (("trained", "train"), ("held out", "heldout")):
        ns = [n for n in role if not b[n] and C[n]["cls"] == "none" and role[n]["v3"] == g and role[n]["v4"] == g
              and all(n in R["mse"][r] for r in cp.V4 + ("dimensionless_pp",))]
        groups.append((lab, np.array([R["mse"]["dimensionless_pp"][n] for n in ns]),
                       np.array([cp.gm(R, "mse", cp.V4, n) for n in ns])))
    scatter(groups, f"{MSE_LAB}, pp events only (AdamW)", f"{MSE_LAB}, catalog (Muon)", "paper_pp_direct")


def pp_bound():
    """Sampling: per class on the 13 TeV pp test sets, the pretraining with our sampling against the one whose pp
    classes train only on PDF-sampled MadEvent events; one file, legend inside."""
    cls = m.class_of()
    import yaml
    trained = {cls(q["name"]) for q in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    x = m.side([m.read_of(s, "PP4") for s in m.BOUND["arm"]])
    y = m.side([m.read_of(s, "PP4") for s in m.BOUND["control"]])
    keys = set(x) & set(y)
    a, c = by_class({k: x[k] for k in keys}, cls), by_class({k: y[k] for k in keys}, cls)
    groups = []
    for lab, keep in (("$pp$, in pretraining,\nsampled from PDFs", lambda q: q in trained),
                      ("$pp$, held out,\nsampled from PDFs", lambda q: q not in trained)):
        cs = sorted(q for q in set(a) & set(c) if keep(q))
        groups.append((lab, np.array([a[q] for q in cs]), np.array([c[q] for q in cs])))
    from analysis.common import ratio_stats
    for lab, xs, ys in groups:   # catalog over pp-only per class: n_num_lower = classes where the catalog is lower
        r = ys / xs
        print(f"  pp_bound {lab}: catalog/pp-only {ratio_stats(ys, xs)}, min {r.min():.3g} at pp-only MSE {xs[r.argmin()]:.3g}")
    fig, ax = ps.figure(width="half")
    scatter_ax(ax, groups, f"{MSE_LAB}, pretrained on $pp$\nevents sampled from PDFs only",
               f"{MSE_LAB}, pretrained on\nour mixture of sampling techniques", "paper_pp_bound", legend="upper left")
    ps.save(fig, os.path.join(FIGS, "paper_pp_bound"))


def unseen(ax):
    """Zero-shot MSE per held-out process of the pretraining without tau, nu_tau and quartic vertices, one column per
    kind of held-out process, every process a point, the median a bar (Joaquin 2026-11-17: the histogram did not show
    which held-out processes do worse)."""
    t = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads.json")))["dimensionless_twin"]
    groups = (("H1", "only seen\nparticles,\nvertices"), ("H4", "unseen\nquartic\nvertex"), ("H3p", "unseen\n" r"$\tau$ or $\nu_\tau$"))
    rng = np.random.default_rng(0)
    for i, ((a, lab), c) in enumerate(zip(groups, COLS)):
        v = np.array([x["mse"] for x in t[a]["probes"].values()])
        ax.scatter(i + rng.uniform(-0.25, 0.25, v.size), v, color=c, alpha=0.5, s=8, linewidths=0, rasterized=True)
        ax.plot([i - 0.35, i + 0.35], [np.median(v)] * 2, color="k", lw=1.5)
        print(f"  unseen {lab!r}: {v.size} processes, median {np.median(v):.3g}")
    ax.set_xticks(range(len(groups)), [lab for _, lab in groups]); ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_yscale("log")
    ax.set_ylabel(f"zero-shot {MSE_LAB}, pretrained\n" + r"without $\tau$, $\nu_\tau$, quartic vertices")
    ax.plot([], [], color="k", lw=1.5, label="median")
    ps.legend(ax, "upper left")


# The 448-process pretraining's single-seed arms at the baseline's best HPs, 8601 steps: geometric mean over processes
# of the per-process validation loss (MSE of the standardized log|M|^2) at the best checkpoint, as docs/results.tex
# tab:bigrun_arms (analysis/bigrun/arms_best.py, run on the site holding the runs) and the pair-bias table of
# sec:bigrun. Rows name the inputs each network has (Joaquin 2026-11-17, third round: symbols, not words that need a
# legend): o the propagator off-shellness, R_c the colour one-hot, G the diagram encoder, MLP or W x_i the particle
# embedding (an MLP or a linear map), b_ij the pair bias on the attention (|b_ij| <= 8 its capped version, v2);
# the symbols are defined in the caption and the text.
BIGRUN = {r"$o,\,R_c,\,Wx_i$": (3.85e-4, 0.12),
          r"$o,\,R_c$, MLP": (3.88e-4, 0.10),
          r"$o,\,R_c,\,G,\,Wx_i$": (4.86e-4, 0.056),
          r"$o,\,R_c,\,Wx_i,\,b_{ij}$": (5.26e-4, 0.084),
          r"$o,\,R_c,\,Wx_i,\,|b_{ij}|\leq8$": (5.74e-4, 0.077),
          r"$o,\,R_c,\,G$, MLP": (5.79e-4, 0.085),
          r"$o,\,G$, MLP": (6.81e-4, 0.094),
          r"$R_c,\,G$, MLP": (8.63e-4, 0.067)}


def inputs(ax):
    rows = sorted(BIGRUN.items(), key=lambda kv: kv[1][0])
    y = np.arange(len(rows))
    ax.plot([v[0] for _, v in rows], y, "o", color="k")   # black: the left panel's colours name its sets
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xscale("log")
    ax.set_xticks([4e-4, 6e-4, 8e-4], [r"$4\cdot10^{-4}$", r"$6\cdot10^{-4}$", r"$8\cdot10^{-4}$"]); ax.minorticks_off()
    ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)/\sigma^2$")


if __name__ == "__main__":
    fns = {"pidqn": pidqn, "muon": muon, "ft": ft, "ft_median": lambda: ft("muon_ft.json", FT_MEDIAN, "paper_muon_ft_median"), "ft_low": lambda: ft("muon_ft_lowloss.json", FT_LOW, "paper_muon_ft_lowloss"), "pp": pp, "target": target, "pp_direct": pp_direct, "pp_bound": pp_bound}
    for k in (fns if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]):
        fns[k]()
