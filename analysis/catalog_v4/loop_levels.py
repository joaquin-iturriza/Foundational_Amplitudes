"""What sets catalog_v4's one-loop zero-shot level (v3_prep_explore, CPU, no new run). par:v4_kshot_level found a third
of the one-loop axis's zero-shot error is one number per pool (the offset, mean of q - t over the pool's events),
largest on e+e- -> b b~, b b~ -> c c~ H and g g -> Z H. Here, as analysis/catalog_daniel/level_range.py did on Daniel's
catalog: is a held-out one-loop pool's level (the mean of its target t = log(|M|^2 F) on its train split, F the
pretraining's emission-scale factor) outside the range of the training one-loop pools' levels, or inside it and
mispredicted; and is it the pool's properties (born coupling orders, pure-QCD or full-order born, multiplicity, initial
state, loop-induced or virtual) that its level departs from.

Two modes.
  --dump (on CC, through `site run ccin2p3 FA --item v3_prep_explore -- python analysis/catalog_v4/loop_levels.py
  --dump`): the pretraining default's own config (runs/catalog_v4_short/v4_dimensionless_muon), its recipe cut to the
  one-loop pools of catalog_v4's training recipe and of the one-loop held-out axis (recipes/catalog_v4_heldout_H5.yaml),
  per-pool statistics (shared standardization off, as tools/zero_shot_parent.py reads a probe's own): built through
  init_data only (no model), 10000 train events per pool as the zero-shot reads; prints one LOOP_LEVELS JSON line
  {pool: [train mean of t, its std, transform, particles]}. Saved home as data/loop_levels_cc.json.
  default (laptop): reads that file, the held-out offsets of both seeds from data/kshot_level.json
  (analysis/catalog_v4/kshot_level.py), checks the held-out levels against the saved reads' own train mean (mu_sd[0]
  of each short_reads npz, the same estimator), and per held-out pool: its level against the training range overall
  and at its multiplicity, the distance to the nearest training level in units of the training spread, its predicted
  level (level + offset), and the matched training pools (same multiplicity, initial state and born orders; failing
  that, same multiplicity and initial state): the departure of its level from their mean, against its offset.
    python3 analysis/catalog_v4/loop_levels.py   -> data/loop_levels.json, figs/loop_levels.{png,pdf}
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import argparse, json, os, sys, tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
PARENT = "runs/catalog_v4_short/v4_dimensionless_muon"
RUNS = ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2")
SR = os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads")


def is_loop(name):
    return name.endswith("_nlo") or name.endswith("_loop")


def entries():
    tr = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]
          if is_loop(p["name"])]
    ho = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_heldout_H5.yaml")))["processes"]
    return tr, ho


def dump():
    import siteconf  # noqa: F401  (site paths)
    import torch
    from omegaconf import OmegaConf, open_dict
    tr, ho = entries()
    parent = os.path.join(os.environ.get("PROJECT_DIR", ROOT), PARENT)
    rec = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))
    rec = {k: v for k, v in rec.items() if k != "processes"}
    rec["processes"] = tr + ho
    rdir = tempfile.mkdtemp(prefix="zz_loop_levels_", dir=os.environ["SCRATCH"])
    rpath = os.path.join(rdir, "recipe.yaml")
    yaml.safe_dump(rec, open(rpath, "w"))
    cfg = OmegaConf.load(os.path.join(parent, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False; cfg.warm_start_idx = None
        cfg.run_dir = rdir
        cfg.data.source = "recipes"; cfg.data.processes_file = rpath
        cfg.data.train_subsample = 10000; cfg.data.eval_subsample = 200; cfg.data.memmap = False
        cfg.data.shared_standardization = False; cfg.data.affine_multiplicity_std = False
        cfg.fine_tune.pretrained_path = None
    torch.set_default_dtype(torch.float32)
    from experiment import AmplitudeExperiment
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra(); exp.init_data()
    names = [str(n) for n in exp.cfg.data.dataset]
    trs = exp._amp_trafos_pp or [exp.cfg.data.amp_trafos] * len(names)
    out = {n: [float(exp.prepd_mean[i]), float(exp.prepd_std[i]), str(trs[i][0]), int(exp._proc_npart[i])]
           for i, n in enumerate(names)}
    print("LOOP_LEVELS " + json.dumps(out), flush=True)


def props(name):
    import mg5_pipeline_final as mg
    c = mg.PROCESSES[name]
    o = mg.order_vector(c)
    ini = sorted(abs(x) for x in c["pdg_ids"][:2])
    st = ("ee" if ini[0] == 11 else "gg" if ini == [21, 21] else
          ("bg" if 5 in ini else "qg") if 21 in ini else "bb" if ini == [5, 5] else "b q" if 5 in ini else "qq")
    return {"n": len(c["pdg_ids"]), "initial": st, "born": f"as{o[2]} a{o[3]}",
            "qcd_born": "_qcdborn" in name, "loop_induced": bool(c.get("loopind")), "order": o}


def analyse():
    cc = json.load(open(os.path.join(HERE, "data", "loop_levels_cc.json")))
    tr, ho = entries()
    kl = json.load(open(os.path.join(HERE, "data", "kshot_level.json")))["probes"]
    T = {p["name"]: cc[p["name"]] for p in tr if p["name"] in cc}
    miss_tr = sorted(p["name"] for p in tr if p["name"] not in cc)
    P = {n: props(n) for n in list(T) + [p["name"] for p in ho]}
    lv = np.array([v[0] for v in T.values()]); sd_all = float(lv.std())
    res, flat, check = {}, {}, []
    for p in ho:
        n = p["name"]
        if n not in cc:
            res[n] = {"missing": True}; continue
        L, pr = cc[n][0], P[n]
        z = np.load(os.path.join(SR, RUNS[0], "H5", n + ".npz"))
        check.append(abs(L - float(z["mu_sd"][0])))
        atN = np.array([T[m][0] for m in T if P[m]["n"] == pr["n"]])
        key = lambda m, full: (P[m]["n"], P[m]["initial"]) + ((P[m]["born"],) if full else ())
        match = [m for m in T if key(m, True) == key(n, True)]
        how = "n, initial state, born orders"
        if len(match) < 2:
            match = [m for m in T if key(m, False) == key(n, False)]; how = "n, initial state"
        if len(match) < 2:
            match = [m for m in T if P[m]["n"] == pr["n"]]; how = "n"
        mm = float(np.mean([T[m][0] for m in match]))
        offs = [kl[r]["H5"][n]["offset"] for r in RUNS]
        r = {**pr, "level": L, "train_min": float(lv.min()), "train_max": float(lv.max()),
             "outside_all": bool(L < lv.min() or L > lv.max()),
             "atN_range": [float(atN.min()), float(atN.max())] if len(atN) else None,
             "outside_atN": bool(len(atN) and (L < atN.min() or L > atN.max())),
             "nearest_in_sd": float(np.min(np.abs(lv - L)) / sd_all),
             "match_by": how, "n_match": len(match), "match_mean": mm, "departure": L - mm,
             "offset": offs, "pred_level": [L + o for o in offs]}
        res[n] = r
    names = [n for n in res if not res[n].get("missing")]
    dep = np.array([res[n]["departure"] for n in names]); om = np.array([np.mean(res[n]["offset"]) for n in names])
    slope = float(np.polyfit(dep, om, 1)[0]); corr = float(np.corrcoef(dep, om)[0, 1])
    # the predicted level against the matched training mean: what the network places, if it places the class level
    pdev = np.array([np.mean(res[n]["pred_level"]) - res[n]["match_mean"] for n in names])
    big = sorted(names, key=lambda n: -abs(np.mean(res[n]["offset"])))[:6]
    s = {"n_train": len(T), "n_train_missing": len(miss_tr), "n_heldout": len(names),
         "train_level_range": [float(lv.min()), float(np.median(lv)), float(lv.max())], "train_level_sd": sd_all,
         "outside_all": int(sum(res[n]["outside_all"] for n in names)),
         "outside_atN": int(sum(res[n]["outside_atN"] for n in names)),
         "median_nearest_in_sd": float(np.median([res[n]["nearest_in_sd"] for n in names])),
         "corr_departure_offset": corr, "slope_offset_on_departure": slope,
         "sd_departure": float(dep.std()), "sd_offset": float(om.std()), "sd_pred_minus_match": float(pdev.std()),
         "level_check_maxabs": float(max(check)), "largest": big}
    print(f"training one-loop pools {len(T)} (missing {len(miss_tr)}), levels [{lv.min():.2f}, {np.median(lv):.2f}, "
          f"{lv.max():.2f}] sd {sd_all:.2f}; held-out {len(names)}: outside overall {s['outside_all']}, at their N "
          f"{s['outside_atN']}; nearest training level median {s['median_nearest_in_sd']:.3f} sd; level check max "
          f"{s['level_check_maxabs']:.2g}")
    print(f"departure from matched training mean vs seed-mean offset: corr {corr:+.2f}, slope {slope:+.2f}; sd departure "
          f"{dep.std():.2f}, sd offset {om.std():.2f}, sd (predicted level - matched mean) {pdev.std():.2f}")
    for n in sorted(names, key=lambda n: -abs(np.mean(res[n]["offset"]))):
        r = res[n]
        print(f"  {n:28s} N={r['n']} {r['initial']:3s} {r['born']:7s}{' qcdborn' if r['qcd_born'] else ''}"
              f"{' loop-induced' if r['loop_induced'] else ''}  level {r['level']:+7.2f}  offset {r['offset'][0]:+.2f}/"
              f"{r['offset'][1]:+.2f}  matched ({r['match_by']}, {r['n_match']}) {r['match_mean']:+7.2f} departure "
              f"{r['departure']:+6.2f}{'  OUTSIDE at N' if r['outside_atN'] else ''}")
    json.dump({"summary": s, "heldout": res, "train": {m: {"level": T[m][0], **P[m]} for m in T},
               "train_missing": miss_tr}, open(os.path.join(HERE, "data", "loop_levels.json"), "w"), indent=1)
    for k in ("n_train", "n_heldout", "outside_all", "outside_atN"):
        flat[k] = s[k]
    for k in ("median_nearest_in_sd", "corr_departure_offset", "slope_offset_on_departure", "sd_departure", "sd_offset",
              "sd_pred_minus_match", "train_level_sd"):
        flat[k] = round(s[k], 2)
    for n in big:
        flat[f"departure.{n}"] = round(res[n]["departure"], 2)
        flat[f"match_n.{n}"] = res[n]["n_match"]
    plot(res, names, T, P)
    print(json.dumps(flat))


def plot(res, names, T, P):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    a.plot([P[m]["n"] - 0.12 + rng.uniform(-0.05, 0.05) for m in T], [T[m][0] for m in T], ".", color=ps.C.blue,
           label=f"training ({len(T)})")
    a.plot([res[n]["n"] + 0.12 for n in names], [res[n]["level"] for n in names], "o", mfc="none", color=ps.C.vermillion,
           label=f"held out ({len(names)})")
    a.plot([res[n]["n"] + 0.12 for n in names], [np.mean(res[n]["pred_level"]) for n in names], "x", color=ps.C.green,
           label="held out, predicted")
    a.set_xlabel(r"particles $N$"); a.set_ylabel(r"$\overline{t}$")
    ps.legend(a, "upper right"); ps.make_room(a)
    dep = [res[n]["departure"] for n in names]
    b.errorbar(dep, [np.mean(res[n]["offset"]) for n in names],
               yerr=[abs(res[n]["offset"][0] - res[n]["offset"][1]) / 2 for n in names], fmt="o", color=ps.C.blue,
               label="held-out pool, seed mean")
    lo, hi = min(dep), max(dep)
    b.plot([lo, hi], [-lo, -hi], "--", color=ps.C.grey, label="offset $=-$departure")
    b.set_xlabel(r"$\overline{t}-\overline{t}_{\rm matched}$"); b.set_ylabel(r"$\overline{q-t}$")
    ps.legend(b, "upper right"); ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "loop_levels"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    (dump if ap.parse_args().dump else analyse)()
