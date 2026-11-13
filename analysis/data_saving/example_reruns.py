"""Representative examples for the draft's data-saving figure (Sec. 2, D444): the 24 cells of example_cells.json
(4 probes x D = 10^2, 10^3, 10^4 x {scratch, every-structure fine-tune}) rerun cold from their configs
(scripts/job_rerun_trial.sh), per-event predictions on each cell's own validation split (preds_val.npz from
tools/eval_delta.py at the best checkpoint).

    python analysis/data_saving/example_reruns.py collect   # read-only site run on lxplus and CC-IN2P3
                                                            #   -> data/example_reruns.npz (ln|M|^2 true, pred per cell)
    python analysis/data_saving/example_reruns.py plot      # -> figs/example_delta_d{2,3,4}, example_dist_d{2,3,4}{a,b}

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
# the one cell rerun earlier, on CC (design: reused, 0.05% of its record)
ON_CC = {("tp3_finfte_uubar_Zggg_d8", 96)}
NAME = {"uubar_Zggg": r"$u\bar u\to Zggg$", "ee_ddbarg": r"$e^+e^-\to d\bar dg$",
        "udbar_enu": r"$u\bar d\to e^+\nu_e$", "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)"}
PROBES = list(NAME)


def tag(c):
    """The rerun's zz_delta directory name: tools/eval_delta.py joins the run dir's last two path parts."""
    return f"runs__{c['sweep']}_rerun" if c["hp"] < 0 else f"{c['sweep']}__trial_{c['hp']:04d}_rerun"


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
        tags = [tag(c) for c in CELLS if ((c["sweep"], c["hp"]) in ON_CC) == (site == "ccin2p3")]
        got = site_python(REMOTE, tags, "EXAMPLE_PREDS", site=site)
        for t, s in got.items():
            z = np.load(io.BytesIO(base64.b64decode(s)))
            have[t + "/true"], have[t + "/pred"] = z["true"], z["pred"]
        print(f"{site}: {len(got)} of {len(tags)} reruns have preds_val.npz; missing: {sorted(set(tags) - set(got))}")
    np.savez_compressed(NPZ, **have)


def cells_with_preds():
    z = np.load(NPZ)
    out = []
    for c in CELLS:
        t = tag(c)
        if t + "/true" in z:
            c = dict(c, true=z[t + "/true"], pred=z[t + "/pred"])
            c["mse"] = float(np.mean((c["pred"] - c["true"]) ** 2))
        out.append(c)
    return out


def plot():
    import plot_style as ps
    figs = os.path.join(ROOT, "analysis", "data_saving", "figs")
    cells = cells_with_preds()
    for c in cells:
        if "mse" not in c:
            print(f"CELL {c['probe']} D={c['D']} {c['arm']}: no rerun predictions yet")
            continue
        d = np.exp(c["pred"] - c["true"]) - 1
        r = c["mse"] / c["loss"]
        print(f"CELL {c['probe']} D={c['D']} {c['arm']} n={c['true'].size}: rerun MSE(ln|M|^2) {c['mse']:.3g}, "
              f"recorded {c['loss']:.3g}, rerun/recorded {r:.2f}{'' if 0.5 <= r <= 2 else '  !! outside 2x'}; "
              f"Delta rms {np.sqrt(np.mean(d ** 2)):.3g}, median |Delta| {np.median(np.abs(d)):.3g}")
    by = {(c["probe"], c["D"], c["arm"]): c for c in cells if "mse" in c}
    col = {"scratch": ps.C.blue, "fine-tune": ps.C.vermillion}
    lab = {"scratch": "from scratch", "fine-tune": "fine-tuned"}
    for k, D in ((2, 100), (3, 1000), (4, 10000)):
        # Delta histograms: one panel per probe, both arms on the panel's own range
        fig, axs = ps.figure(ncols=2, nrows=2)
        for ax, p in zip(np.ravel(axs), PROBES):
            arms = [a for a in ("scratch", "fine-tune") if (p, D, a) in by]
            if not arms:
                continue
            ds = {a: np.exp(by[p, D, a]["pred"] - by[p, D, a]["true"]) - 1 for a in arms}
            allv = np.concatenate(list(ds.values()))
            edges = np.linspace(*np.quantile(allv, [0.0005, 0.9995]), 61)   # the central 99.9% of both arms' events
            for a, v in ds.items():
                ax.stairs(np.histogram(v, edges)[0], edges, color=col[a], label=lab[a])
            ax.set_yscale("log"); ax.set_ylabel("events")
            ax.set_xlabel(r"$\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}-1$")
            ps.process_label(ax, NAME[p])
        ps.legend_strip(np.ravel(axs)[0], os.path.join(figs, f"example_delta_d{k}_legend"), ncol=2)
        print("FIGURE", ps.save(fig, os.path.join(figs, f"example_delta_d{k}")))
        # distributions: per canvas two probes, top true and predicted ln|M|^2, bottom predicted / true per bin
        for part, sel in (("a", PROBES[:2]), ("b", PROBES[2:])):
            fig, axs = ps.figure(ncols=2, nrows=2)
            for j, p in enumerate(sel):
                arms = [a for a in ("scratch", "fine-tune") if (p, D, a) in by]
                if not arms:
                    continue
                tr = by[p, D, arms[0]]["true"]
                edges = np.linspace(*np.quantile(tr, [0.001, 0.999]), 41)
                t = np.histogram(tr, edges)[0].astype(float)
                top, bot = axs[0][j], axs[1][j]
                top.stairs(t, edges, color="k", label="truth")
                for a in arms:
                    h = np.histogram(by[p, D, a]["pred"], edges)[0]
                    top.stairs(h, edges, color=col[a], label=lab[a])
                    bot.stairs(np.where(t > 0, h / np.where(t > 0, t, 1), np.nan), edges, color=col[a])
                top.set_yscale("log"); top.set_ylabel("events")
                bot.set_ylabel("pred / truth")
                for ax in (top, bot):
                    ax.set_xlabel(r"$\log|\mathcal{M}|^2$")
                ps.process_label(top, NAME[p])
            ps.legend_strip(axs[0][0], os.path.join(figs, f"example_dist_d{k}{part}_legend"), ncol=3)
            print("FIGURE", ps.save(fig, os.path.join(figs, f"example_dist_d{k}{part}")))


if __name__ == "__main__":
    {"collect": collect, "plot": plot}[sys.argv[1]]()
