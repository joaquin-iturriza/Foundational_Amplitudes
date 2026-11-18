"""Zero shot per held-out process of Daniel's catalog split (plan dan_zeroshot_v4, D488): MSE of log|M|^2 per process,
five arms, on the interpolation relabels and on the extrapolation processes truly held out from both trainings (no
relabel of them in catalog_v4's training: v4_heldout_overlap.py). Redraws the October deck's dan_per_process
(talks/progress_oct2026b/make_figs.py fig_per_process) with our catalog_v4 pretraining added.

Arms, all read on the same pools (first 2000 val events of recipes/catalog_daniel_heldout.yaml, no map fitted):
  Suprio's particle-ID embedding: talks/progress_oct2026b/data/draft_pid_by_process.json (the draft figure's numbers,
    seed 0, 20 epochs; not committed, the collaborators' unpublished result)
  per-process standardization: hp58 of dancat_het_pre (D59) with the statistics predicted by g (D81),
    data/heldout_preds.npz + data/stats_net.json (heldout_delta.G), as make_figs.hp58
  shared standardization: data/catalog_norm_B.npz (trained on Daniel's catalog, diverged after 1.1 epochs, best ckpt)
  dimensionless + coupling orders: data/catalog_short_DlessInOrd_sigstage.npz (Daniel's catalog, 5 epochs of 1e4)
  catalog_v4 pretraining: data/v4_trial1262_dan.npz, trial_1262 of cov_v4_long_hpo8_cc_68224 (its sigma stage), axis
    DAN of scripts/job_catalog_v3_short.sh v4_het_long_best eval; and data/v4_trial1262_danid.npz, axis DANID (its
    own test pools of the catalog_v4 training trees identical to one of Daniel's processes, the calibration of the
    cross-catalog read); each brought home with
    site run ccin2p3 FA -- python analysis/catalog_daniel/collect_heldout_preds.py \
        '$SCRATCH/zz_v4_long_best_cov_v4_long_hpo8_cc_68224_trial_1262/<DAN|DANID>'   (base64 after __B64__)
Sets: interpolation relabels without the 5 processes identical to a catalog_v4 training entry (listed apart);
extrapolation truly held out, those without a b first (a b is massive in Daniel's pools, massless in catalog_v4's,
so its Yukawa coupling is one the pretraining never saw). Per set and arm: the median MSE, the median after the best
affine map where the pack has it, and the process-bootstrap 68% interval of the median per-process log10 ratio of
catalog_v4 over each other arm.
Output: figs/dan_per_process.{png,pdf}, data/dan_per_process.json.
    python analysis/catalog_daniel/dan_per_process.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "data")
NEW = "pretrained on the new catalog"


def pack(f):
    return dict(np.load(os.path.join(D, f + ".npz")))


def hp58():
    """hp58's read with the statistics predicted from the process (make_figs.hp58)"""
    from heldout_delta import G
    z = pack("heldout_preds")
    sn = json.load(open(os.path.join(D, "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z:
            z[f"{n}__q"] = mu + np.exp(lsd) * z[f"{n}__h"].astype(float)
    return z


def mse(z, key="q"):
    """{process: MSE of log|M|^2 of prediction `key`}; non-finite kept (listed, never dropped)"""
    return {k[:-3]: float(np.mean((z[k[:-3] + "__" + key].astype(float) - z[k].astype(float)) ** 2))
            for k in z if k.endswith("__t") and k[:-3] + "__" + key in z}


def boot_median_log_ratio(a, b, n=2000, seed=0):
    """median over processes of log10(a/b) and its process-bootstrap 16/84% quantiles"""
    r = np.log10(np.asarray(a) / np.asarray(b))
    r = r[np.isfinite(r)]
    if not len(r):
        return None
    rng = np.random.default_rng(seed)
    bs = np.median(r[rng.integers(0, len(r), (n, len(r)))], axis=1)
    return [float(np.median(r)), float(np.quantile(bs, 0.16)), float(np.quantile(bs, 0.84)), len(r)]


KEYS = ["pid", "perproc", "shared", "dimless", "new"]   # --arms names, in the order of the arms list


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(KEYS), help="comma list of " + ",".join(KEYS) + " drawn (all computed)")
    shown = ap.parse_args().arms.split(",")
    assert set(shown) <= set(KEYS), shown
    full = shown == KEYS
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    ov = json.load(open(os.path.join(D, "v4_heldout_overlap.json")))["processes"]
    packs = {"per-process standardization": hp58(), "shared standardization": pack("catalog_norm_B"),
             "dimensionless + coupling orders": pack("catalog_short_DlessInOrd_sigstage"), NEW: pack("v4_trial1262_dan")}
    arms = [("particle-ID embedding (Suprio)", {n: v[1] for g in draft.values() for n, v in g.items()}, "*", "k"),
            ("per-process standardization", mse(packs["per-process standardization"]), "^", ps.C.grey),
            ("shared standardization", mse(packs["shared standardization"]), "o", ps.C.blue),
            ("dimensionless + coupling orders", mse(packs["dimensionless + coupling orders"]), "D", ps.C.vermillion),
            (NEW, mse(packs[NEW]), "s", ps.C.green)]
    aff = {a: mse(z, "q_aff") for a, z in packs.items()}
    has_b = lambda n: any(t in ("b", "b~") for t in ov[n]["process"].replace(">", " ").split())
    sets = [("interp", "relabels", lambda n: not ov[n]["identical"], lambda n: draft["interp"][n][0]),
            ("extrap", "truly held out", lambda n: not ov[n]["relabel"], lambda n: (has_b(n), draft["extrap"][n][0]))]
    fig, axs = ps.figure(ncols=2)
    out = {"arms": [a[0] for a in arms], "sets": {}}
    new = arms[-1][1]
    for ax, (g, lab, keep, order) in zip(axs, sets):
        names = sorted((n for n in draft[g] if keep(n)), key=order)
        x = np.arange(1, len(names) + 1)
        rec = {"processes": names, "with_b": [n for n in names if has_b(n)], "mse": {}, "median": {},
               "median_affine": {}, "unscored": {}, "new_over_arm_log10": {}}
        for k, (a, m, mk, c) in zip(KEYS, arms):
            y = np.array([m.get(n, np.nan) for n in names])
            rec["mse"][a] = [None if not np.isfinite(v) else v for v in y]
            rec["unscored"][a] = [n for n, v in zip(names, y) if not np.isfinite(v)]
            rec["median"][a] = float(np.median(y[np.isfinite(y)])) if np.isfinite(y).any() else None
            if a in aff:
                ya = np.array([aff[a].get(n, np.nan) for n in names])
                rec["median_affine"][a] = float(np.median(ya[np.isfinite(ya)])) if np.isfinite(ya).any() else None
            if a != NEW:
                for sub, sel in (("all", lambda n: True), ("no_b", lambda n: not has_b(n)), ("b", has_b)):
                    ns = [n for n in names if sel(n) and n in m and n in new]
                    rec["new_over_arm_log10"].setdefault(a, {})[sub] = boot_median_log_ratio(
                        [new[n] for n in ns], [m[n] for n in ns])
            if k in shown:
                ax.plot(x, y, mk, color=c, ls="none", label=a, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("held-out process")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, lab, loc="lower right" if g == "interp" else "upper right")   # the legend holds the interp panel's top
        out["sets"][g] = rec
        print(f"{g} ({len(names)}, {len(rec['with_b'])} with a b): median " +
              ", ".join(f"{a} {v:.3g}" for a, v in rec["median"].items()) +
              "; after the best affine map " + ", ".join(f"{a} {v:.3g}" for a, v in rec["median_affine"].items() if v) +
              "; unscored " + str({a: v for a, v in rec["unscored"].items() if v}))
        for a, v in rec["new_over_arm_log10"].items():
            print(f"   log10(new / {a}): " + ", ".join(f"{s} {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] n={r[3]}"
                                                     for s, r in v.items() if r))
    # the 5 interpolation processes identical to a catalog_v4 training entry, apart
    ident = [n for n in draft["interp"] if ov[n]["identical"]]
    out["identical_interp"] = {n: {a: m.get(n) for a, m, *_ in arms} for n in ident}
    print("interp identical to a catalog_v4 training entry: " +
          "; ".join(f"{n} new {new.get(n, float('nan')):.3g}" for n in ident))
    # calibration: catalog_v4's model on Daniel's pool against its own test pool of the identical training tree
    zid = mse(pack("v4_trial1262_danid"))
    cal = {n: {"v4_entry": m, "on_daniel": new.get(n), "on_own": zid.get(m)}
           for n, v in ov.items() for m in v["identical"] if m in zid}
    out["calibration"] = cal
    for n, c in cal.items():
        print(f"calibration {n} ({ov[n]['process']}): on Daniel's pool {c['on_daniel']:.3g}, "
              f"on its own pool {c['v4_entry']} {c['on_own']:.3g}")
    ps.legend(axs[0], "upper left")
    ps.make_room(axs[0])
    if full:
        json.dump(out, open(os.path.join(D, "dan_per_process.json"), "w"), indent=1)
    ps.save(fig, os.path.join(HERE, "figs", "dan_per_process" if full else "dan_per_process_" + "_vs_".join(shown)))


if __name__ == "__main__":
    main()
