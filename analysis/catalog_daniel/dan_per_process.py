"""Zero shot per held-out process of Daniel's catalog split (plan dan_zeroshot_v4, D488): MSE of log|M|^2 per process,
four arms, on the 96 interpolation relabels and on the extrapolation processes that are truly held out from both
trainings (no relabel of them in catalog_v4's training: v4_heldout_overlap.py). Redraws the October deck's
dan_per_process (talks/progress_oct2026b/make_figs.py fig_per_process) with our catalog_v4 pretraining added.

Arms, all read on the same pools (first 2000 val events of recipes/catalog_daniel_heldout.yaml, no map fitted):
  Suprio's particle-ID embedding: talks/progress_oct2026b/data/draft_pid_by_process.json (the draft figure's numbers,
    seed 0, 20 epochs; not committed, the collaborators' unpublished result)
  shared standardization: data/catalog_norm_B.npz (trained on Daniel's catalog, diverged after 1.1 epochs, best ckpt)
  dimensionless + coupling orders: data/catalog_short_DlessInOrd_sigstage.npz (Daniel's catalog, 5 epochs of 1e4)
  catalog_v4 pretraining: data/v4_trial1262_dan.npz, trial_1262 of cov_v4_long_hpo8_cc_68224 (its sigma stage), axis
    DAN of scripts/job_catalog_v3_short.sh v4_het_long_best eval; brought home with
    site run ccin2p3 FA -- python analysis/catalog_daniel/collect_heldout_preds.py \
        '$SCRATCH/zz_v4_long_best_cov_v4_long_hpo8_cc_68224_trial_1262/DAN'   (base64 after __B64__)
Output: figs/dan_per_process.{png,pdf}, data/dan_per_process.json (per process MSE per arm, per-group medians).
    python analysis/catalog_daniel/dan_per_process.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "data")


def mse(f):
    """{process: MSE of log|M|^2} of a zero_shot_parent npz pack; non-finite kept (listed, never dropped)"""
    z = np.load(os.path.join(D, f + ".npz"))
    return {k[:-3]: float(np.mean((z[k[:-3] + "__q"].astype(float) - z[k].astype(float)) ** 2))
            for k in z.files if k.endswith("__t") and k[:-3] + "__q" in z.files}


def main():
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    ov = json.load(open(os.path.join(D, "v4_heldout_overlap.json")))["processes"]
    arms = [("particle-ID embedding (Suprio)", {n: v[1] for g in draft.values() for n, v in g.items()}, "*", "k"),
            ("shared standardization", mse("catalog_norm_B"), "o", ps.C.blue),
            ("dimensionless + coupling orders", mse("catalog_short_DlessInOrd_sigstage"), "D", ps.C.vermillion),
            ("pretrained on the new catalog", mse("v4_trial1262_dan"), "s", ps.C.green)]
    sets = [("interp", "relabels", lambda n: True),
            ("extrap", "truly held out", lambda n: not ov[n]["relabel"])]
    fig, axs = ps.figure(ncols=2)
    out = {"arms": [a[0] for a in arms], "sets": {}}
    for ax, (g, lab, keep) in zip(axs, sets):
        names = sorted((n for n in draft[g] if keep(n)), key=lambda n: draft[g][n][0])
        x = np.arange(1, len(names) + 1)
        rec = {"processes": names, "mse": {}, "median": {}, "unscored": {}}
        for a, m, mk, c in arms:
            y = np.array([m.get(n, np.nan) for n in names])
            rec["mse"][a] = [None if not np.isfinite(v) else v for v in y]
            rec["unscored"][a] = [n for n, v in zip(names, y) if not np.isfinite(v)]
            rec["median"][a] = float(np.median(y[np.isfinite(y)])) if np.isfinite(y).any() else None
            ax.plot(x, y, mk, color=c, ls="none", label=a, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("held-out process (draft order)")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, lab, loc="lower right" if g == "interp" else "upper right")   # the legend holds the interp panel's top
        out["sets"][g] = rec
        print(f"{g} ({len(names)}): median " + ", ".join(f"{a} {rec['median'][a]:.3g}" for a in rec["median"])
              + "; unscored " + str({a: v for a, v in rec["unscored"].items() if v}))
    ps.legend(axs[0], "upper left")
    ps.make_room(axs[0])
    json.dump(out, open(os.path.join(D, "dan_per_process.json"), "w"), indent=1)
    ps.save(fig, os.path.join(HERE, "figs", "dan_per_process"))


if __name__ == "__main__":
    main()
