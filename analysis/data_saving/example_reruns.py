"""Representative examples for the draft's data-saving figure (Sec. 2, D444): the 24 cells of example_cells.json
(4 probes x D = 10^2, 10^3, 10^4 x {scratch, every-structure fine-tune}) rerun cold from their configs
(scripts/job_rerun_trial.sh), per-event predictions on each cell's own validation split (preds_val.npz from
tools/eval_delta.py at the best checkpoint).

    python analysis/data_saving/example_reruns.py collect   # read-only site run on lxplus and CC-IN2P3
                                                            #   -> data/example_reruns.npz (ln|M|^2 true, pred per cell)
    python analysis/data_saving/example_reruns.py plot      # -> figs/example_delta_{a..d}, example_dist_{a..d} (+ _legend)

Per cell it prints the rerun's MSE of ln|M|^2 = mean (ln|M|^2_pred - ln|M|^2_true)^2 against the cell's recorded
value (the design's reproduction check: within 2x). Delta = |M|^2_pred / |M|^2_true - 1.
"""
import base64, io, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
DATA = os.path.join(ROOT, "analysis", "data_saving", "data")
NPZ = os.path.join(DATA, "example_reruns.npz")
CELLS = json.load(open(os.path.join(DATA, "example_cells.json")))
# reruns on CC (the rest on lxplus): the one cell rerun earlier (design: reused, 0.05% of its record), and the
# repeat of tp3_finfte_udbar_enu_d8 trial_0187 (FA-ccin2p3-4fa87893, lxplus full)
ON_CC = {"tp3_finfte_uubar_Zggg_d8__trial_0096_rerun", "tp3_finfte_udbar_enu_d8__trial_0187_rerun2"}
# the design's one repeat of a rerun off its record by more than 2x (job_rerun_trial.sh third argument)
REPEAT = {("tp3_scr_ee_bb_nlo_d8", 154): "2", ("tp3_finfte_udbar_enu_d8", 187): "2"}
NAME = {"uubar_Zggg": r"$u\bar u\to Zggg$", "ee_ddbarg": r"$e^+e^-\to d\bar dg$",
        "udbar_enu": r"$u\bar d\to e^+\nu_e$", "ee_bb_nlo": r"$e^+e^-\to b\bar b$, one loop"}
PROBES = list(NAME)


def tag(c, rep=""):
    """The rerun's zz_delta directory name: tools/eval_delta.py joins the run dir's last two path parts."""
    return (f"runs__{c['sweep']}_rerun" if c["hp"] < 0 else f"{c['sweep']}__trial_{c['hp']:04d}_rerun") + rep


def tags(c):
    """The cell's reruns, the first one and its repeat if it has one."""
    r = REPEAT.get((c["sweep"], c["hp"]))
    return [tag(c)] + ([tag(c, r)] if r else [])


REMOTE = r'''
import base64, io, json, os, sys
import numpy as np
out = {}
for t in sys.argv[1:]:
    f = os.path.join(os.environ["SCRATCH"], "zz_delta", t, "preds_val.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    b = io.BytesIO()
    np.savez_compressed(b, true=np.log(np.abs(z["raw_truth"].ravel())), pred=np.log(np.abs(z["raw_pred"].ravel())))
    out[t] = base64.b64encode(b.getvalue()).decode()
print("EXAMPLE_PREDS " + json.dumps(out))
'''


def collect():
    from common.readers import site_python
    have = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
    for site in ("lxplus", "ccin2p3"):
        want = [t for c in CELLS for t in tags(c) if (t in ON_CC) == (site == "ccin2p3")]
        got = site_python(REMOTE, want, "EXAMPLE_PREDS", site=site)
        for t, s in got.items():
            z = np.load(io.BytesIO(base64.b64decode(s)))
            have[t + "/true"], have[t + "/pred"] = z["true"], z["pred"]
        print(f"{site}: {len(got)} of {len(want)} reruns have preds_val.npz; missing: {sorted(set(want) - set(got))}")
    np.savez_compressed(NPZ, **have)


def cells_with_preds():
    """Each cell with its rerun's predictions; a repeated rerun is shown by its repeat, both MSEs kept."""
    z = np.load(NPZ)
    out = []
    for c in CELLS:
        c = dict(c, mses=[])
        for t in tags(c):
            if t + "/true" in z:
                c["true"], c["pred"] = z[t + "/true"], z[t + "/pred"]
                c["mses"].append(float(np.mean((c["pred"] - c["true"]) ** 2)))
        if c["mses"]:
            c["mse"] = c["mses"][-1]
        out.append(c)
    return out


D_STYLE = {100: ":", 1000: "--", 10000: "-"}
D_LAB = {100: r"$10^2$", 1000: r"$10^3$", 10000: r"$10^4$"}


def hist_ratio(ax):
    """The ratio panel under a histogram, a quarter of the plot box's height, sharing its x axis (as
    analysis/catalog_v4/paper_sec3_figs.hist_ratio, the papers' layout)."""
    r = ax.inset_axes([0, -0.31, 1, 0.25], sharex=ax)
    ax.tick_params(labelbottom=False)
    return r


def plot():
    import plot_style as ps
    figs = os.path.join(ROOT, "analysis", "data_saving", "figs")
    cells = cells_with_preds()
    for c in cells:
        if "mse" not in c:
            print(f"CELL {c['probe']} D={c['D']} {c['arm']}: no rerun predictions yet")
            continue
        d = np.exp(c["pred"] - c["true"]) - 1
        r = [m / c["loss"] for m in c["mses"]]
        print(f"CELL {c['probe']} D={c['D']} {c['arm']} n={c['true'].size}: rerun MSE(ln|M|^2) "
              + ", ".join(f"{m:.3g}" for m in c["mses"]) + f" (repeat last), recorded {c['loss']:.3g}, rerun/recorded "
              + ", ".join(f"{x:.2f}" for x in r) + ("" if 0.5 <= r[-1] <= 2 else "  !! outside 2x")
              + f"; Delta rms {np.sqrt(np.mean(d ** 2)):.3g}, median |Delta| {np.median(np.abs(d)):.3g}, "
              f"|Delta| > 10: {np.mean(np.abs(d) > 10):.2%}, < 1e-6: {np.mean(np.abs(d) < 1e-6):.2%}")
    by = {(c["probe"], c["D"], c["arm"]): c for c in cells if "mse" in c}
    col = {"scratch": ps.C.blue, "fine-tune": ps.C.vermillion}
    lab = {"scratch": "scratch", "fine-tune": "fine-tune"}
    edges = np.logspace(-6, 3, 73)   # |Delta| from 10^-6 to 10^3, eight bins a decade; events beyond go in the end bins
    # |Delta| histograms on a log axis, one panel per probe: colour the arm, line style D
    fd = ps.panels(4)
    for (fig, ax), p in zip(fd, PROBES):
        for a in ("scratch", "fine-tune"):
            for D in D_STYLE:
                if (p, D, a) in by:
                    c = by[p, D, a]
                    x = np.clip(np.abs(np.exp(c["pred"] - c["true"]) - 1), edges[0], edges[-1] * 0.999)
                    ax.stairs(np.histogram(x, edges)[0], edges, color=col[a],
                              linestyle=D_STYLE[D], label=f"{lab[a]}, $D=${D_LAB[D]}")
        ax.set_xscale("log"); ax.set_xlim(edges[0], edges[-1])
        ax.set_yscale("log"); ax.set_ylabel("events")
        ax.set_xlabel(r"$|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}-1$")
        ps.process_label(ax, NAME[p])
    # legends in rows, scratch above fine-tune (entries 0-2 scratch, 3-5 fine-tune; matplotlib fills by column)
    ps.legend_strip(fd[0][1], os.path.join(figs, "example_delta_legend"), ncol=3, order=[0, 3, 1, 4, 2, 5])
    print("FIGURE", ps.save_panels(fd, os.path.join(figs, "example_delta")))
    # true and predicted log|M|^2 with the ratio below, one panel per probe
    fh = ps.panels(4)
    for k, ((fig, ax), p) in enumerate(zip(fh, PROBES)):
        r = hist_ratio(ax)
        have = [(a, D) for a in ("scratch", "fine-tune") for D in D_STYLE if (p, D, a) in by]
        if not have:
            continue
        tr = by[p, have[0][1], have[0][0]]["true"]   # every cell of a probe scores the same validation split
        every = np.concatenate([tr] + [by[p, D, a]["pred"] for a, D in have])
        e = np.linspace(every.min(), every.max(), 41)   # the full range: no true or predicted event outside it
        t = np.histogram(tr, e)[0].astype(float)
        ax.stairs(t, e, color="k", label="truth", zorder=5)   # on top: the best fine-tune otherwise hides it
        for a, D in have:
            h = np.histogram(by[p, D, a]["pred"], e)[0]
            ax.stairs(h, e, color=col[a], linestyle=D_STYLE[D], label=f"{lab[a]}, $D=${D_LAB[D]}")
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = np.where(t > 0, h / t, np.nan)   # every bin with true events
            r.stairs(np.clip(rat, 0.5, 1.5), e, color=col[a], linestyle=D_STYLE[D])   # beyond the limits: drawn at the edge
        r.axhline(1, color="k", lw=0.8)   # the truth line, as thin as the papers' ratio reference
        r.set_ylim(0.47, 1.53); r.set_yticks([0.5, 1, 1.5]); r.set_yticklabels(["0.5", "1", "1.5"])
        ax.set_xlim(e[0], e[-1])
        ax.set_yscale("log"); ax.set_ylabel("events")
        r.set_xlabel(r"$\log|\mathcal{M}|^2$")
        if k % 2 == 0:   # right-hand panels: the label would run into the left panel in the paper's pairs
            r.set_ylabel(r"$\frac{\rm pred}{\rm true}$")
        ps.process_label(ax, NAME[p])
    # scratch row, fine-tune row, truth below (entry 0 truth, 1-3 scratch, 4-6 fine-tune; filled by column)
    ps.legend_strip(fh[0][1], os.path.join(figs, "example_dist_legend"), ncol=3, order=[1, 4, 0, 2, 5, None, 3, 6, None])
    print("FIGURE", ps.save_panels(fh, os.path.join(figs, "example_dist")))


if __name__ == "__main__":
    {"collect": collect, "plot": plot}[sys.argv[1]]()
