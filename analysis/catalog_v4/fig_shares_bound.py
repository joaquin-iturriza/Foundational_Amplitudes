"""The default-sampled shares against the pure-default bound (docs/results.tex par:v4_mixture, tab:v4_bound_shares):
R (median over classes of the arm/control MSE ratio, widened 95% class-bootstrap interval) per read group, one panel per
training loss. MSE: pure default (the bound arm), share 0.50, share 0.25 against the MSE control pair
(data/bound_default.json); het: shares 0.50 and 0.25 against the het control pair (data/mixture_ab_het.json; no het
bound arm was trained). Every read on MadEvent default-sampled test events. Lines at R = 1 (the control) and R = 0.90
(the design's GAIN bar).

    python3 analysis/catalog_v4/fig_shares_bound.py  ->  analysis/catalog_v4/figs/shares_bound_{a,b}.{png,pdf}

With `arms`: the zero-shot reads per arm, each seed its own point (seed 42 filled, 43 open), no error bars
(data/arms_vs_het_control.json "seeds", mixture_ab.py arms), every arm at 4264 steps but the last, het loss, its epochs
(data/epochs.json) in its label. The arms: the control (share 0, width 8, 1e4 train events per process, in memory),
share 0.25 at width 8 on 1e4 events per process (step 15), and share 0.25 on the whole training set (1e5 per process,
memmap path) at widths 8, 16 and 32 (step 33), and the best trial of the 17056-step DyHPO sweep (width 8, whole set,
lr and HPs tuned, seed 42 only, not converged; step 16). Panel a: R per read (a seed's MSE over the control's two-seed geometric
mean, class median; the control's own seeds read the same way); b: the absolute MSE of log|M|^2 per read (class median
of the class's geometric mean over its probes); c and d: the same on the held-out pool by axis (default-sampled).

    python3 analysis/catalog_v4/fig_shares_bound.py arms  ->  analysis/catalog_v4/figs/arms_zero_shot_{a,b,c,d}.{png,pdf}

With `merged` (the deck): one panel, the MSE-trained arms (share 1 on pp, 0.50, 0.25) and the het-trained shares (0.50,
0.25), each R against the share-0 control trained with the MSE loss, on four disjoint-labelled rows.

    python3 analysis/catalog_v4/fig_shares_bound.py merged  ->  analysis/catalog_v4/figs/shares_deck.{png,pdf}
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

DATA = os.path.join(HERE, "data")
# (label, path into an arm's result)
# labels in words: pp 13 TeV MadEvent probes of the trained / untrained classes, the e+e- probes, the held-out pool,
# its physical-top classes and its one-loop classes; class counts from data/bound_default.json
READS = [("pp 13 TeV, trained classes (797)", ("PP4", "trained")),
         ("pp 13 TeV, untrained classes (504)", ("PP4", "not_trained")), (r"$e^+e^-$ (169)", ("EE4", "all")),
         ("held out, default-sampled (700)", ("held_default", "pooled")),
         ("held out, physical top (279)", ("held_default", "H3")), ("held out, one-loop (28)", ("held_default", "H5"))]
ARMS = [("bound", "pure default", ps.C.vermillion, "s"), ("default50", "share 0.50", ps.C.green, "o"),
        ("default25", "share 0.25", ps.C.blue, "D")]


def panel(ax, arms_res, present, reads=READS, arms=ARMS, lines=((1.0, "--", r"$R=1$"), (0.90, ":", r"$R=0.90$")),
          ticks=(0.2, 0.5, 1.0, 2.0), counts=False, xlabel=r"$R$ = arm / control"):
    n = len(reads)
    shown = [a for a in arms if a[0] in present]
    off = {a[0]: (i - (len(shown) - 1) / 2) * (0.22 if len(shown) < 4 else 0.18) for i, a in enumerate(shown)}
    labels = []
    for j, (lab, (r, part)) in enumerate(reads):
        labels.append(f"{lab} ({arms_res[shown[0][0]][r][part]['n_classes']})" if counts else lab)
        for key, label, col, mk in shown:
            d = arms_res[key][r][part]
            y = n - 1 - j + off[key]
            lo, hi = d["widened"]
            ax.errorbar([d["R"]], [y], xerr=[[d["R"] - lo], [hi - d["R"]]], color=col, marker=mk, linestyle="none",
                        label=label if j == 0 else None)
    for x, ls, lab in lines:
        ax.axvline(x, color=ps.C.grey, linestyle=ls, label=lab)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)


# the arms figure: reads, held-out axes in words, arms (label without its epochs, colour, marker)
ARM_READS = [("pp, processes in training", "PP4_trained"), (r"$e^+e^-$, processes in training", "EE4_trained"),
             ("not in training, default sampling", "held_default"), ("not in training, catalog sampling", "held"),
             ("one-loop, not in training", "held_default/H5")]
ARM_AXES = [("H1", "new combinations"), ("H2", r"$2\to4$ families"), ("H2p5", r"$2\to5$"),
            ("H3", "physical top"), ("H6", r"$\alpha_s=0.13$"), ("H7E", r"$\sqrt{\hat s}$ 6 to 20 TeV"),
            ("H7c", r"$p_T\geq30$ GeV"), ("H5", "one-loop")]
# one colour per arm on every slide of the deck (talks/progress_nov2026); {ep} is the arm's epochs (data/epochs.json)
ARM_ARMS = [("control", r"control: share 0, width 8, $10^4$ events per process, {ep} epochs", "k", "s"),
            ("default25_1e4", r"share 0.25, width 8, $10^4$ events per process, {ep} epochs", ps.C.grey, "v"),
            ("w8_full", r"share 0.25, width 8, $10^5$ events per process, {ep} epochs", ps.C.blue, "o"),
            ("w16_full", r"share 0.25, width 16, $10^5$ events per process, {ep} epochs", ps.C.green, "D"),
            ("w32_full", r"share 0.25, width 32, $10^5$ events per process, {ep} epochs", ps.C.vermillion, "^"),
            ("long4x_best", r"share 0.25, width 8, $10^5$ events per process, {ep} epochs, best tuned trial, "
             "not converged", ps.C.purple, "P")]


def seed_panel(ax, seeds, reads, arms, key, xlabel, ticks, ref=None):
    """One row per read, one marker per arm and seed (seed 42 filled, seed 43 open), no error bars"""
    from matplotlib.lines import Line2D
    n = len(reads)
    off = {a[0]: (i - (len(arms) - 1) / 2) * 0.17 for i, a in enumerate(arms)}
    for j, (lab, r) in enumerate(reads):
        for a, label, col, mk in arms:
            d = seeds[a][r]
            y = n - 1 - j + off[a]
            ax.plot([d["seed42"][key]], [y + 0.035], ls="none", marker=mk, color=col)
            if "seed43" in d:          # a one-seed arm (a sweep's best trial) has seed 42 only
                ax.plot([d["seed43"][key]], [y - 0.035], ls="none", marker=mk, color=col, mfc="none")
    if ref is not None:
        ax.axvline(ref[0], color=ps.C.grey, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lab} ({seeds[arms[0][0]][r]['n_probes']})" for lab, r in reads][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)
    hs = [Line2D([], [], ls="none", marker=mk, color=col, label=label) for _, label, col, mk in arms]
    hs += [Line2D([], [], ls="none", marker="o", color="k", label="seed 42"),
           Line2D([], [], ls="none", marker="o", color="k", mfc="none", label="seed 43")]
    if ref is not None:
        hs.append(Line2D([], [], color=ps.C.grey, ls="--", label=ref[1]))
    for h in hs:               # empty proxies, so the legend strip (ps.shared_legend reads the axes) names them
        ax.add_line(h)


def point_panel(ax, pts, reads, arms, key, xlabel, ticks, ref=None, step=0.16):
    """One row per read, one point per arm (its seeds' geometric mean) with its bar (the per-seed 95% class-bootstrap
    intervals' lowest lower to highest upper end)"""
    n = len(reads)
    off = {a[0]: (i - (len(arms) - 1) / 2) * step for i, a in enumerate(arms)}   # rows apart by a marker or more
    for j, (lab, r) in enumerate(reads):
        for a, label, col, mk in arms:
            d = pts[a][r]
            y = n - 1 - j + off[a]
            lo, hi = d[key + "_bar"]
            ax.errorbar([d[key]], [y], xerr=[[d[key] - lo], [hi - d[key]]], ls="none", marker=mk, color=col,
                        label=label if j == 0 else None)
    if ref is not None:
        ax.axvline(ref[0], color=ps.C.grey, linestyle="--", label=ref[1])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lab} ({pts[arms[0][0]][r]['n_classes']})" for lab, r in reads][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)


def arms_figure():
    d = json.load(open(os.path.join(DATA, "arms_vs_het_control.json")))
    pts, ep = d["points"], d["epochs"]
    eps = {a: sorted({round(v["epochs"], 2) for v in ep[a].values()}) for a in ep}
    assert all(len(v) == 1 for v in eps.values()), eps
    arms = [(a, lab.format(ep=f"{eps[a][0]:.2g}"), c, m) for a, lab, c, m in ARM_ARMS]
    axes = [(lab, f"held_default/{ax}") for ax, lab in ARM_AXES]
    ref = None           # the control is its own point at R = 1 (a line too named it twice)
    rlab = r"$R$ = MSE(arm) / MSE(control)"
    mlab = r"MSE of $\log|\mathcal{M}|^2$"
    for name, reads, rt, mt, step in (
            ("arms_zero_shot", ARM_READS, (0.05, 0.1, 0.2, 0.5, 1.0), (0.01, 0.1, 1.0, 10.0), 0.16),
            ("arms_zero_shot_axes", axes, (0.02, 0.05, 0.1, 0.2, 0.5, 1.0), (0.001, 0.01, 0.1, 1.0, 10.0), 0.14)):
        fig, (a, b) = ps.figure(ncols=2)
        point_panel(a, pts, reads, arms, "R", rlab, rt, ref, step)
        point_panel(b, pts, reads, arms, "mse", mlab, mt, None, step)
        b.set_yticklabels([])
        ps.shared_legend(fig, a, ncol=1)     # two columns of these names run off the canvas
        ps.save(fig, os.path.join(HERE, "figs", name))



# the deck's one figure of the share study (Joaquin 2026-11-03: "The point is to show the advantage of het over MSE"):
# every arm's R is over the share-0 control trained with the MSE loss, whatever the arm's own loss; rows and arm names
# as on the zero-shot figure (ARM_READS, ARM_ARMS)
SHARE_READS = [("pp, processes in training", ("PP4", "trained")),
               (r"$e^+e^-$, processes in training", ("EE4", "trained")),
               ("not in training, default sampling", ("held_default", "pooled")),
               ("one-loop, not in training", ("held_default", "H5"))]
# one colour per share fraction for both losses (share 1 orange, 0.50 sky, 0.25 vermillion), the loss told apart by the
# marker fill (open MSE, filled het.); the MSE control a black square at R = 1 with its seed-spread bar, the het control
# (share 0, het loss) a black filled circle over the MSE control
SHARE_ARMS = [("mse", "bound", r"share 1 on pp, MSE loss", ps.C.orange, "s", "none"),
              ("mse", "default50", r"share 0.50, MSE loss", ps.C.sky, "o", "none"),
              ("mse", "default25", r"share 0.25, MSE loss", ps.C.vermillion, "D", "none"),
              ("het", "default50", r"share 0.50, het. loss", ps.C.sky, "o", None),
              ("het", "default25", r"share 0.25, het. loss", ps.C.vermillion, "D", None)]


def het_over_mse_control():
    """R of the het-loss arms and the het control over the MSE control (same per-class construction as mixture_ab's
    compare: geometric mean over seeds per probe, class geometric mean, median over classes, class bootstrap widened by
    the MSE control's seed ratio), on the classes both have (the intersection) -> data/shares_deck_mse_ref.json"""
    sys.path.insert(0, HERE)
    import mixture_ab as m
    cls = m.class_of()
    ax_of = m._held_axes(cls)
    trained = {cls(p["name"]) for p in m.yaml.safe_load(open(os.path.join(ROOT, "recipes",
                                                                      "catalog_v4_base_train.yaml")))["processes"]}
    names = ("default50", "default25", "hetctl")
    table = {a: {r: m.MIX_HET[a][r] for r in ("PP4", "EE4")} for a in names[:2]}
    table["hetctl"] = {r: m.MIX_HET["control"][r] for r in ("PP4", "EE4")}
    table["control"] = {r: m.MIX["control"][r] for r in ("PP4", "EE4")}
    reads, missing = m._reads(table)
    assert not missing, missing
    out = {"design": "R over the share-0 MSE-loss control (MIX['control'] runs), arms from MIX_HET, het control "
                     "MIX_HET['control']; classes = intersection of the arm's and the MSE control's", "arms": {},
           "class_counts": {}}
    for a in names:
        extra = {ax: {a: (m.HD4_HET["control"] if a == "hetctl" else m.HD4_HET[a]), "control": m.HD4_MSE["control"]}
                 for ax in ("HD4",)}
        extra["HD5"] = {a: (m.HD5_HET["control"] if a == "hetctl" else m.HD5_HET[a]), "control": m.HD5_MSE["control"]}
        res = {}
        c = reads["control"]
        for r in ("PP4", "EE4"):
            arm, ctl = m.side(reads[a][r]), m.side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"trained": [k for k in keys if cls(m.probe_name(k)) in trained]}
            res[r] = {n: m.compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            out["class_counts"][f"{a} {r} trained"] = {
                "arm_classes": len({cls(m.probe_name(k)) for k in arm if cls(m.probe_name(k)) in trained}),
                "control_classes": len({cls(m.probe_name(k)) for k in ctl if cls(m.probe_name(k)) in trained}),
                "used": res[r]["trained"]["n_classes"]}
        res["held_default"], _ = m._held_default(reads, a, cls, ax_of, extra)
        assert not res["held_default"]["default_missing"], res["held_default"]["default_missing"]
        out["arms"][a] = res
    json.dump(out, open(os.path.join(DATA, "shares_deck_mse_ref.json"), "w"), indent=1)
    return out


def merged_figure():
    mse = json.load(open(os.path.join(DATA, "bound_default.json")))["arms"]
    het = het_over_mse_control()["arms"]
    res = {"mse": mse, "het": {"default50": het["default50"], "default25": het["default25"]}}
    fig, ax = ps.figure()
    n = len(SHARE_READS)
    k = len(SHARE_ARMS) + 2
    off = [(i - (k - 1) / 2) * 0.12 for i in range(k)]
    for j, (lab, (r, part)) in enumerate(SHARE_READS):
        w = abs(math.log(mse["default25"][r][part]["R_seed"]))   # the MSE control's seed 43 over 42, class median
        lo_c, hi_c = math.exp(-w), math.exp(w)
        print("MSE control bar", lab, round(lo_c, 4), round(hi_c, 4))
        ax.errorbar([1.0], [n - 1 - j + off[0]], xerr=[[1 - lo_c], [hi_c - 1]], color="k", marker="s",
                    linestyle="none", label="control: share 0, MSE loss" if j == 0 else None)
        d = het["hetctl"][r][part]
        lo, hi = d["widened"]
        print("het control over MSE control", lab, d["R"], d["widened"], d["n_classes"])
        ax.errorbar([d["R"]], [n - 1 - j + off[1]], xerr=[[d["R"] - lo], [hi - d["R"]]], color="k", marker="o",
                    linestyle="none", label="share 0, het. loss" if j == 0 else None)
        for i, (loss, a, label, col, mk, mfc) in enumerate(SHARE_ARMS):
            d = res[loss][a][r][part]
            lo, hi = d["widened"]
            if loss == "het":
                print("het", a, lab, d["R"], d["widened"], d["n_classes"])
            ax.errorbar([d["R"]], [n - 1 - j + off[i + 2]], xerr=[[d["R"] - lo], [hi - d["R"]]],
                        color=col, marker=mk, linestyle="none", mfc=mfc or col, label=label if j == 0 else None)
    ticks = (0.2, 0.5, 1.0, 2.0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lab} ({mse['default25'][r][p]['n_classes']})" for lab, (r, p) in SHARE_READS][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"$R$ = MSE(arm) / MSE(control: share 0, MSE loss)")
    ps.shared_legend(fig, ax, ncol=3)   # every arm here at width 8 and 10^4 events per process, as the control
    ps.save(fig, os.path.join(HERE, "figs", "shares_deck"))


def main():
    mse = json.load(open(os.path.join(DATA, "bound_default.json")))["arms"]
    het = json.load(open(os.path.join(DATA, "mixture_ab_het.json")))["shares"]
    # one file per loss, stacked: the read groups' labels in words make a two-across row wider than the paper
    (fa, a), (fb, b) = ps.panels(2)
    panel(a, mse, {"bound", "default50", "default25"})
    panel(b, het, {"default50", "default25"})
    ps.process_label(a, "MSE")
    ps.process_label(b, "het")
    ps.shared_legend(fa, a, ncol=3)     # one strip above the first panel: the dot rows fill every corner
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    ps.save_panels([(fa, a), (fb, b)], os.path.join(HERE, "figs", "shares_bound"))

if __name__ == "__main__":
    {"arms": arms_figure, "merged": merged_figure}.get((sys.argv[1:] or ["main"])[0], main)()
