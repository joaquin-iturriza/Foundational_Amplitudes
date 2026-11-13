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


def scatter(groups, xlab, ylab, base):
    fig, ax = ps.figure(width="half")
    lo = min(min(g[1].min(), g[2].min()) for g in groups)
    hi = max(max(g[1].max(), g[2].max()) for g in groups)
    ax.plot([lo, hi], [lo, hi], color=ps.C.grey, ls="--", label=r"$y=x$")
    for (lab, x, y), c in zip(groups, COLS):
        ax.scatter(x, y, color=c, alpha=0.45, s=6, linewidths=0, label=lab, rasterized=True)
        print(f"  {os.path.basename(base)} {lab}: {len(x)} classes, median y/x {np.median(y / x):.3g}, "
              f"medians x {np.median(x):.3g} y {np.median(y):.3g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    pad = (hi / lo) ** 0.04
    ax.set_xlim(lo / pad, hi * pad); ax.set_ylim(lo / pad, hi * pad)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    leg = ps.shared_legend(fig, ax, ncol=2, markerscale=2.5)   # five entries fit in no corner of a 1.9 in box
    for h in leg.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
    ps.save(fig, os.path.join(FIGS, base))


SETS = (("pp test", ("PP4",)), (r"$e^+e^-$ test", ("EE4",)), ("held out", ("held",)),
        ("held out, MadEvent", ("HD4", "HD5")))


def pidqn():
    g = pair_sets(m.PIDQN["qn"], m.PIDQN["pid"], SETS, m.class_of())
    scatter(g, f"{MSE_LAB}, quantum numbers", f"{MSE_LAB}, one-hot", "paper_pidqn")


def muon():
    import muon_long as ml
    g = pair_sets(ml.RUNS["muon"], ml.RUNS["adamw"], SETS, m.class_of())
    scatter(g, f"{MSE_LAB}, Muon", f"{MSE_LAB}, AdamW", "paper_muon")


def hist_ratio(ax):
    """The ratio panel under a histogram, a quarter of the plot box's height, sharing its x axis (the papers' layout)."""
    r = ax.inset_axes([0, -0.31, 1, 0.25], sharex=ax)
    ax.tick_params(labelbottom=False)
    return r


def examples():
    import muon_long as ml
    col = {"muon": ps.C.blue, "adamw": ps.C.vermillion}
    lab = {"muon": "Muon", "adamw": "AdamW"}
    h = {k: json.load(open(os.path.join(m.READS, f"{run}_hists.json"))) for k, run in ml.RUNS.items()}
    mu = m.read_of(ml.RUNS["muon"], "PP4")
    keys = sorted((p for p in mu if f"PP4:{p}" in h["muon"] and f"PP4:{p}" in h["adamw"]), key=lambda p: mu[p])
    pick = [keys[int(q * (len(keys) - 1))] for q in (0.1, 0.5, 0.9)]
    figs = ps.panels(3, width="half")
    REB = 4                              # the records' bins merged by four: the ratio is otherwise counting noise
    def rebin(e, c):
        n = (len(c) // REB) * REB
        return e[:n + 1:REB], np.add.reduceat(np.asarray(c[:n], float), np.arange(0, n, REB))
    for (fig, ax), p in zip(figs, pick):
        r = hist_ratio(ax)
        lo, hi = np.inf, -np.inf
        for k in ("muon", "adamw"):     # each run's record bins truth and prediction on its own edges
            rec = h[k][f"PP4:{p}"]
            ek, t = rebin(np.array(rec["edges"]), rec["truth"])
            _, q = rebin(np.array(rec["edges"]), rec["pred"])
            if k == "muon":
                ax.stairs(t, ek, color="k", label="truth", zorder=5)   # on top of the predictions
            ax.stairs(q, ek, color=col[k], label=lab[k])
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = np.where(t > 0, q / t, np.nan)   # every bin with true events
            r.stairs(np.clip(rat, 0, 2), ek, color=col[k])   # beyond the limits: drawn at the edge
            filled = np.nonzero((t > 0) | (q > 0))[0]
            lo, hi = min(lo, ek[filled[0]]), max(hi, ek[filled[-1] + 1])
        r.axhline(1, color="k", lw=0.8)
        r.set_ylim(-0.06, 2.06); r.set_yticks([0, 1, 2]); r.set_yticklabels(["0", "1", "2"])
        ax.set_ylim(bottom=0.5)              # one-event bins above the frame, not on it
        pad = 0.08 * (hi - lo)               # margin beyond the outermost filled bin
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yscale("log"); ax.set_ylabel("events")
        r.set_xlabel(r"$\log|\mathcal{M}|^2$"); r.set_ylabel("pred/true")
        ps.process_label(ax, ml.process_tex(p))
        ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(FIGS, "paper_muon_examples"))


FT_PROBES = {"v3_dg_dgwpwm": r"$dg\to dgW^+W^-$", "v3_usx_cxuwpz": r"$u\bar s\to \bar c uW^+Z$",
             "v3_gg_attx": r"$gg\to\gamma t\bar t$", "v3_udx_agwp_nlo": r"$u\bar d\to\gamma gW^+$, one loop"}


def ft():
    d = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "muon_ft.json")))
    arms = (("dimensionless", "AdamW", ps.C.vermillion, "AdamW"),
            ("dimensionless_muon", "MuonAdamW", ps.C.blue, "Muon"))
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax, (p, name) in zip(axs.flat, FT_PROBES.items()):
        for parent, opt, c, lab in arms:
            rs = sorted((r for r in d if r["kind"] == "catft_parentstats" and r["parent"] == parent
                         and r["optimizer"] == opt and r["probe"] == p), key=lambda r: r["D"])
            ax.plot([r["D"] for r in rs], [r["mse_log"] for r in rs], "o-", color=c, label=f"{lab}, fine-tuned")
            ax.axhline(rs[0]["mse_log_step0"], color=c, ls=":", label=f"{lab}, zero shot")
            print(f"  ft {p} {lab}: zero shot {rs[0]['mse_log_step0']:.3g}, D=10 {rs[0]['mse_log']:.3g}, "
                  f"D=316 {rs[-1]['mse_log']:.3g}")
        sc = sorted((r for r in d if r["kind"] == "scratch" and r["probe"] == p), key=lambda r: r["D"])
        ax.plot([r["D"] for r in sc], [r["mse_log"] for r in sc], "s--", color=ps.C.grey, label="scratch")
        ax.set_xscale("log"); ax.set_yscale("log")
        ys = [l.get_ydata() for l in ax.lines]
        ax.set_ylim(0.7 * min(np.min(y) for y in ys), 4 * max(np.max(y) for y in ys))   # headroom: the process label sits above the zero-shot lines
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(MSE_LAB)
        ps.process_label(ax, name)
    ps.shared_legend(fig, axs.flat[0], ncol=3)
    ps.save(fig, os.path.join(FIGS, "paper_muon_ft"))


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
    for grp, lab in ((None, "in range"), ("H2p5", r"$2\to5$"), ("H7E", r"$1$-$10$ TeV")):
        xs, ys = [], []
        for a in res["baseline"]:
            if (a if a in extra else None) != grp or a not in res["dimensionless"]:
                continue
            px, py = res["baseline"][a].get("probes", {}), res["dimensionless"][a].get("probes", {})
            for p in sorted(set(px) & set(py)):
                xs.append(px[p]["mse"]); ys.append(py[p]["mse"])
        groups.append((lab, np.array(xs), np.array(ys)))
    scatter(groups, f"{MSE_LAB}, " + r"$\log|\mathcal{M}|^2$ in GeV", f"{MSE_LAB}, " + r"target $T$", "paper_target")


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
    cls = m.class_of()
    import yaml
    trained = {cls(q["name"]) for q in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    x = m.side([m.read_of(s, "PP4") for s in m.BOUND["arm"]])
    y = m.side([m.read_of(s, "PP4") for s in m.BOUND["control"]])
    keys = set(x) & set(y)
    a, c = by_class({k: x[k] for k in keys}, cls), by_class({k: y[k] for k in keys}, cls)
    groups = []
    for lab, keep in (("trained", lambda q: q in trained), ("held out", lambda q: q not in trained)):
        cs = sorted(q for q in set(a) & set(c) if keep(q))
        groups.append((lab, np.array([a[q] for q in cs]), np.array([c[q] for q in cs])))
    scatter(groups, f"{MSE_LAB}, pp events only", f"{MSE_LAB}, catalog", "paper_pp_bound")


def unseen():
    t = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads.json")))["dimensionless_twin"]
    fig, ax = ps.figure(width="half")
    vals = {a: np.array([v["mse"] for v in t[a]["probes"].values()]) for a in ("H1", "H4", "H3p")}
    bins = np.logspace(np.log10(min(v.min() for v in vals.values())) - 0.05,
                       np.log10(max(v.max() for v in vals.values())) + 0.05, 16)
    for (a, lab), c in zip((("H1", "new combinations"), ("H4", "quartic vertex unseen"),
                            ("H3p", r"$\tau$, $\nu_\tau$ unseen")), COLS):
        ax.hist(vals[a], bins=bins, weights=np.full(len(vals[a]), 1 / len(vals[a])), histtype="step", color=c, label=lab)
        print(f"  unseen {lab}: {len(vals[a])} processes, median {np.median(vals[a]):.3g}")
    ax.set_xscale("log"); ax.set_xlabel(f"zero-shot {MSE_LAB}"); ax.set_ylabel("fraction of processes")
    ps.shared_legend(fig, ax, ncol=2)
    ps.save(fig, os.path.join(FIGS, "paper_unseen"))


# The 448-process pretraining's single-seed arms at the baseline's best HPs, 8601 steps: geometric mean and 90th
# percentile over processes of the per-process validation loss at the best checkpoint, as docs/results.tex
# tab:bigrun_arms (analysis/bigrun/arms_best.py, run on the site holding the runs) and the pair-bias table of
# sec:bigrun (whose reference is the 'no diagram encoder, linear embedding' arm, GM 3.85e-4 in both tables)
# adopted = linear embedding, no diagram encoder; pair bias = added to the adopted configuration
BIGRUN = {"adopted": (3.85e-4, 0.12),
          "no diagrams": (3.88e-4, 0.10),
          "linear embedding": (4.86e-4, 0.056),
          "pair bias": (5.26e-4, 0.084),
          "pair bias, capped": (5.74e-4, 0.077),
          "all inputs": (5.79e-4, 0.085),
          "no colour one-hot": (6.81e-4, 0.094),
          "no off-shellness": (8.63e-4, 0.067)}


def inputs():
    rows = sorted(BIGRUN.items(), key=lambda kv: kv[1][0])
    fig, ax = ps.figure(width="half")
    y = np.arange(len(rows))
    ax.plot([v[0] for _, v in rows], y, "o", color=ps.C.blue)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xscale("log")
    ax.set_xticks([4e-4, 6e-4, 8e-4], [r"$4\cdot10^{-4}$", r"$6\cdot10^{-4}$", r"$8\cdot10^{-4}$"]); ax.minorticks_off()
    ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$, standardized")
    ps.save(fig, os.path.join(FIGS, "paper_inputs"))


if __name__ == "__main__":
    fns = {"inputs": inputs, "pidqn": pidqn, "muon": muon, "examples": examples, "ft": ft, "pp": pp, "target": target, "unseen": unseen, "pp_direct": pp_direct, "pp_bound": pp_bound}
    for k in (fns if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]):
        fns[k]()
