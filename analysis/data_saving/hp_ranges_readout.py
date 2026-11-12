"""pretrain_data_saving: is the stage-2 search template right (Joaquin 2026-11-12: the separation of validation from
training is "the result of a bad combination of HPs ... it's either a bad choice of which HPs to search, bad ranges
for them, too few trials")? Read on every observed stage-2 trial at D < 1e4 with curves (trial_overfit.json), each joined to its
full HPs in sweeps_local/<sweep>/driver_state.json and its search range in sweep/sweep_config_<sweep>.yaml:
  per HP and arm  the HP's place in its range (0 = low edge, 1 = high edge; log for float_log) of the trials
                  curve_quality.py calls good against the bad ones; within-search Spearman of the HP with
                  log separation and with log best val (each search's median removed)
  edges           how many search bests sit in the outer 10% of each range
  trials          per search, best val after the first n observed trials (submission order) over the final best
Figure figs/hp_ranges_<arm>: per searched HP, separation against the HP's place in its range, good trials marked.

    python analysis/data_saving/hp_ranges_readout.py
"""
import glob, json, os, re, sys
import numpy as np
import yaml
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from curve_quality import BAR, trials  # noqa: E402

DATA = os.path.join(HERE, "data")
DMAX = 1e4   # where the separation is: D >= 1e4 search bests are good in 13 of 23 searches, below in 2 of 66


def place(x, lo, hi, kind):
    if isinstance(x, str) or kind == "categorical":
        return float(str(x).lower() == "true")
    return float(np.log(x / lo) / np.log(hi / lo)) if kind in ("float_log", "int_log") else float((x - lo) / (hi - lo))


def joined():
    rows, spaces, order = [], {}, {}
    for r in trials():
        if r["stage"] != "s2":
            continue
        sw = r["sweep"]
        if sw not in spaces:
            cfg = yaml.safe_load(open(os.path.join(REPO, "sweep", f"sweep_config_{sw}.yaml")))
            spaces[sw] = {e["name"]: (e.get("low"), e.get("high"), e["type"]) for e in cfg["search_space"]}
            tr = json.load(open(os.path.join(REPO, "sweeps_local", sw, "driver_state.json")))["trials"]
            tr = list(tr.values()) if isinstance(tr, dict) else tr
            order[sw] = {int(t["hp_idx"]): t for t in tr}
        idx = int(re.search(r"trial_(\d+)", r["run_dir"]).group(1))
        t = order[sw].get(idx)
        if t is None:
            continue
        r = dict(r, hpfull=t["hp"], submitted=t.get("submitted") or 0,
                 place={k: place(t["hp"][k], *spaces[sw][k]) for k in spaces[sw] if k in t["hp"]})
        rows.append(r)
    return rows


def main():
    rows = joined()
    out = {"bars": BAR, "n": len(rows), "hp": {}, "edges": {}, "trials": {}}
    print(f"{len(rows)} stage-2 trials joined to their HPs and ranges")
    for arm in ("scr", "ft"):
        g = [r for r in rows if r["arm"] == arm and r["D"] < DMAX]
        hps = sorted({k for r in g for k in r["place"]})
        by = {}
        for r in g:
            by.setdefault(r["sweep"], []).append(r)
        for hp in hps:
            x, ds, dv = [], [], []
            for sw, t in by.items():
                t = [r for r in t if hp in r["place"]]
                ms = np.median([np.log10(r["separation"]) for r in t])
                mv = np.median([np.log10(r["val_min_ln"]) for r in t])
                for r in t:
                    x.append(r["place"][hp]); ds.append(np.log10(r["separation"]) - ms); dv.append(np.log10(r["val_min_ln"]) - mv)
            good = [r["place"][hp] for r in g if r["well"] and hp in r["place"]]
            bad = [r["place"][hp] for r in g if not r["well"] and hp in r["place"]]
            best = [r["place"][hp] for r in g if r.get("best") and hp in r["place"]]
            e = {"n": len(x), "rho_separation": float(spearmanr(x, ds)[0]), "rho_val": float(spearmanr(x, dv)[0]),
                 "good_place_median": float(np.median(good)) if good else None, "n_good": len(good),
                 "bad_place_median": float(np.median(bad)),
                 "bests_low_edge": sum(p <= 0.1 for p in best), "bests_high_edge": sum(p >= 0.9 for p in best),
                 "n_bests": len(best)}
            out["hp"][f"{arm}/{hp}"] = e
            print(f"{arm:3s} {hp:34s} n={e['n']:3d} rho sep {e['rho_separation']:+.2f} val {e['rho_val']:+.2f}  place good "
                  f"{e['good_place_median'] if good else float('nan'):.2f} ({len(good)}) bad {e['bad_place_median']:.2f}  "
                  f"bests at low/high edge {e['bests_low_edge']}/{e['bests_high_edge']} of {len(best)}")
        # trial count: best after n observed trials over the final best, per search
        R = {n: [] for n in range(1, 9)}
        for sw, t in by.items():
            t = sorted(t, key=lambda r: r["submitted"])
            v = np.minimum.accumulate([r["val_min_ln"] for r in t])
            for n in R:
                if n <= len(v):
                    R[n].append(v[n - 1] / v[-1])
        out["trials"][arm] = {n: {"median": float(np.median(q)), "q75": float(np.percentile(q, 75)), "searches": len(q)}
                              for n, q in R.items() if q}
        print(f"{arm} best after n trials / final best (median, 75%): " + " ".join(
            f"{n}:{out['trials'][arm][n]['median']:.2f}/{out['trials'][arm][n]['q75']:.2f}" for n in out["trials"][arm]))
    json.dump(out, open(os.path.join(DATA, "hp_ranges_readout.json"), "w"), indent=1)
    figure(rows)


LAB = {"training.lr": "lr", "fine_tune.lr_scale": "lr scale", "fine_tune.layer_decay": "layer decay",
       "training.muon_weight_decay": "weight decay", "training.regularization_lambda": r"L2 $\lambda$",
       "training.cosanneal_warmup_frac": "warmup fraction", "training.cosanneal_eta_min": r"lr floor",
       "ema": "EMA on", "training.ema_decay": "EMA decay"}


def figure(rows):
    """hp_ranges_<arm>_<a..>: per searched HP, each trial's separation against the HP's place in its search range."""
    from suspects_readout import ps
    for arm in ("scr", "ft"):
        g = [r for r in rows if r["arm"] == arm and r["D"] < DMAX]
        hps = [h for h in LAB if any(h in r["place"] for r in g) and h != "ema"]
        figs = ps.panels(len(hps))
        for (fig, ax), hp in zip(figs, hps):
            for well, col, lab in ((False, ps.C.grey, "bad"), (True, ps.C.vermillion, "good")):
                h = [r for r in g if r["well"] == well and hp in r["place"]]
                ax.scatter([r["place"][hp] for r in h], [r["separation"] for r in h], color=col, s=8, label=lab)
            ax.axhline(BAR["separation"], color="black", linestyle="--", label="bar")
            ax.set_yscale("log"); ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel(f"{LAB[hp]}, place in search range"); ax.set_ylabel("separation")
            ps.process_label(ax, "scratch" if arm == "scr" else "fine-tune")
            ps.legend(ax, "upper left")
        ps.save_panels(figs, os.path.join(HERE, "figs", f"hp_ranges_{arm}"))


if __name__ == "__main__":
    main()
