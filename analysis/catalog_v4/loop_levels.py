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
  level (level + offset), and the matched training pools (one rule: same multiplicity, initial state and born orders;
  a pool with none is listed as unmatched): the departure of its level from their mean, against its offset, compared
  with offset = -departure (a network that predicts the matched pools' mean level).
    python3 analysis/catalog_v4/loop_levels.py   -> data/loop_levels.json, figs/loop_levels.{png,pdf}
The last line printed is the JSON of the numbers docs/results.tex quotes.

Tree mode (--trees, both steps): does the thin-class reading above (the largest one-loop offsets on pools with two
matched training pools or fewer) hold on catalog_v4's tree held-out axes H1 (new combinations), H2 (multiplicity), H3
(particle content). The same dump over the training recipe's tree pools and the three axes' recipes (split into
--chunk i --nchunks m runs on the login node; per-pool statistics, so a chunk's numbers are the whole dump's), saved as
data/tree_levels_cc.json; the same single match rule against the training tree pools, unmatched pools listed by name;
per pool the seed-mean absolute offset against its matched class size, read in the one-loop paragraph's two bins (two
matched pools or fewer, unmatched included, against more) and, as there, which of each axis's six largest offsets fall
in the thin bin, against that bin's share of the axis. No fit.
    python3 analysis/catalog_v4/loop_levels.py --trees -> data/tree_levels.json, figs/tree_levels.{png,pdf}
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


TREE_AXES = ("H1", "H2", "H3")
THIN = 2  # the one-loop paragraph's boundary (par:v4_loop_levels: "two matched training pools or fewer")
TOP = 6   # the one-loop paragraph's count of largest offsets (`big` in analyse)


def heldout(axis):
    return yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{axis}.yaml")))["processes"]


def entries(trees=False):
    tr = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]
          if is_loop(p["name"]) != trees]
    ho = [p for a in TREE_AXES for p in heldout(a)] if trees else heldout("H5")
    return tr, ho


def dump(trees=False, chunk=0, nchunks=1):
    import siteconf  # noqa: F401  (site paths)
    import torch
    from omegaconf import OmegaConf, open_dict
    tr, ho = entries(trees)
    parent = os.path.join(os.environ.get("PROJECT_DIR", ROOT), PARENT)
    rec = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))
    rec = {k: v for k, v in rec.items() if k != "processes"}
    rec["processes"] = (tr + ho)[chunk::nchunks]
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
    print(("TREE_LEVELS " if trees else "LOOP_LEVELS ") + json.dumps(out), flush=True)


BASE = {p["name"]: p.get("base", p["name"]) for p in entries()[0] + entries(True)[0] + entries(True)[1]}


def props(name):
    import mg5_pipeline_final as mg
    c = mg.PROCESSES[BASE.get(name, name)]  # a heavy-quark layer entry (v4_x__mt300) has its base's process
    o = mg.order_vector(c)
    ini = sorted(abs(x) for x in c["pdg_ids"][:2])
    st = ("ee" if ini[0] == 11 else "gg" if ini == [21, 21] else
          ("bg" if 5 in ini else "qg") if 21 in ini else "bb" if ini == [5, 5] else "bq" if 5 in ini else "qq")
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
        match = [m for m in T if match_key(P, m) == match_key(P, n)]
        offs = [kl[r]["H5"][n]["offset"] for r in RUNS]
        r = {**pr, "level": L, "train_min": float(lv.min()), "train_max": float(lv.max()),
             "outside_all": bool(L < lv.min() or L > lv.max()),
             "atN_range": [float(atN.min()), float(atN.max())] if len(atN) else None,
             "outside_atN": bool(len(atN) and (L < atN.min() or L > atN.max())),
             "nearest_in_sd": float(np.min(np.abs(lv - L)) / sd_all),
             "matched": match, "n_match": len(match), "offset": offs, "pred_level": [L + o for o in offs]}
        if match:
            mm = float(np.mean([T[m][0] for m in match]))
            r.update({"match_mean": mm, "departure": L - mm})
        res[n] = r
    names = [n for n in res if not res[n].get("missing")]
    mt = [n for n in names if res[n]["n_match"]]
    unmatched = [n for n in names if not res[n]["n_match"]]
    dep = np.array([res[n]["departure"] for n in mt]); om = np.array([np.mean(res[n]["offset"]) for n in mt])
    corr = float(np.corrcoef(dep, om)[0, 1])
    # the comparison the figure draws: offset = -departure is a network that predicts the matched pools' mean level;
    # a pool whose offset is smaller in size than its departure has its level placed closer than that mean
    closer = [n for n in mt if abs(np.mean(res[n]["offset"])) < abs(res[n]["departure"])]
    closer_seed = [int(sum(abs(res[n]["offset"][i]) < abs(res[n]["departure"]) for n in mt)) for i in range(len(RUNS))]
    ee = [n for n in names if res[n]["initial"] == "ee"]
    big = sorted(names, key=lambda n: -abs(np.mean(res[n]["offset"])))[:6]
    s = {"n_train": len(T), "n_train_missing": len(miss_tr), "n_heldout": len(names),
         "train_level_range": [float(lv.min()), float(np.median(lv)), float(lv.max())], "train_level_sd": sd_all,
         "outside_all": int(sum(res[n]["outside_all"] for n in names)),
         "outside_atN": int(sum(res[n]["outside_atN"] for n in names)),
         "median_nearest_in_sd": float(np.median([res[n]["nearest_in_sd"] for n in names])),
         "n_matched": len(mt), "unmatched": unmatched, "corr_departure_offset": corr,
         "median_abs_departure": float(np.median(np.abs(dep))), "median_abs_offset_matched": float(np.median(np.abs(om))),
         "n_closer_than_match": len(closer), "n_closer_per_seed": closer_seed,
         "level_check_maxabs": float(max(check)), "largest": big,
         "n_train_ee": int(sum(P[m]["initial"] == "ee" for m in T)),
         "outside_names": [n for n in names if res[n]["outside_all"]]}
    print(f"training one-loop pools {len(T)} (missing {len(miss_tr)}), levels [{lv.min():.2f}, {np.median(lv):.2f}, "
          f"{lv.max():.2f}] sd {sd_all:.2f}; held-out {len(names)}: outside overall {s['outside_all']}, at their N "
          f"{s['outside_atN']}; nearest training level median {s['median_nearest_in_sd']:.3f} sd; level check max "
          f"{s['level_check_maxabs']:.2g}; outside {s['outside_names']}")
    print(f"matched (same N, initial state, born orders) {len(mt)}, unmatched {unmatched}; departure vs seed-mean offset "
          f"corr {corr:+.2f}; median |departure| {s['median_abs_departure']:.2f}, median |offset| "
          f"{s['median_abs_offset_matched']:.2f}; offset smaller than departure on {len(closer)} (per seed {closer_seed}); "
          f"e+e- training pools {s['n_train_ee']}, held out {len(ee)}")
    for n in sorted(names, key=lambda n: -abs(np.mean(res[n]["offset"]))):
        r = res[n]
        m = (f"matched {r['n_match']} {r['match_mean']:+7.2f} departure {r['departure']:+6.2f}" if r["n_match"]
             else "unmatched")
        print(f"  {n:28s} N={r['n']} {r['initial']:3s} {r['born']:7s}{' qcdborn' if r['qcd_born'] else ''}"
              f"{' loop-induced' if r['loop_induced'] else ''}  level {r['level']:+7.2f}  offset {r['offset'][0]:+.2f}/"
              f"{r['offset'][1]:+.2f}  {m}{'  OUTSIDE at N' if r['outside_atN'] else ''}")
    json.dump({"summary": s, "heldout": res, "train": {m: {"level": T[m][0], **P[m]} for m in T},
               "train_missing": miss_tr}, open(os.path.join(HERE, "data", "loop_levels.json"), "w"), indent=1)
    for k in ("n_train", "n_heldout", "outside_all", "outside_atN", "n_matched", "n_closer_than_match", "n_train_ee"):
        flat[k] = s[k]
    for k in ("median_nearest_in_sd", "corr_departure_offset", "median_abs_departure", "median_abs_offset_matched",
              "train_level_sd"):
        flat[k] = round(s[k], 2)
    flat["n_closer_seed42"], flat["n_closer_seed43"] = closer_seed
    flat["n_unmatched"] = len(unmatched)
    flat["train_level_min"], flat["train_level_max"] = round(float(lv.min()), 2), round(float(lv.max()), 2)
    flat["level_check_maxabs"] = round(s["level_check_maxabs"], 3)
    for n in big:
        flat[f"level.{n}"] = round(res[n]["level"], 2)
        flat[f"match_n.{n}"] = res[n]["n_match"]
        if res[n]["n_match"]:
            flat[f"departure.{n}"] = round(res[n]["departure"], 2)
    plot(res, mt, names, T, P)
    print(json.dumps(flat))


def match_key(P, m):
    # one match rule: the training pools of the same multiplicity, initial state and born orders; a pool with none
    # is listed as unmatched, never re-matched under a looser key
    return (P[m]["n"], P[m]["initial"], P[m]["born"])


def analyse_trees():
    cc = json.load(open(os.path.join(HERE, "data", "tree_levels_cc.json")))
    tr, _ = entries(True)
    kd = json.load(open(os.path.join(HERE, "data", "kshot_level.json")))
    kl, ks = kd["probes"], kd["summary"]
    T = {p["name"]: cc[p["name"]] for p in tr if p["name"] in cc}
    miss_tr = sorted(p["name"] for p in tr if p["name"] not in cc)
    lv = np.array([v[0] for v in T.values()])
    P = {n: props(n) for n in T}
    byk = {}
    for m in T:
        byk.setdefault(match_key(P, m), []).append(m)
    res, s, flat = {}, {"n_train": len(T), "n_train_missing": len(miss_tr), "train_level_sd": float(lv.std())}, {}
    for a in TREE_AXES:
        R, check = {}, []
        for p in heldout(a):
            n = p["name"]
            if n not in cc:
                R[n] = {"missing": True}; continue
            pr = props(n); L = cc[n][0]
            z = np.load(os.path.join(SR, RUNS[0], a, n + ".npz"))
            check.append(abs(L - float(z["mu_sd"][0])))
            match = byk.get(match_key({n: pr}, n), [])
            offs = [kl[r][a][n]["offset"] for r in RUNS]
            r = {**pr, "level": L, "outside_all": bool(L < lv.min() or L > lv.max()),
                 "nearest_in_sd": float(np.min(np.abs(lv - L)) / lv.std()), "matched": match, "n_match": len(match),
                 "offset": offs, "abs_off": float(abs(np.mean(offs)))}
            if match:
                mm = float(np.mean([T[m][0] for m in match]))
                r.update({"match_mean": mm, "departure": L - mm})
            R[n] = r
        names = [n for n in R if not R[n].get("missing")]
        mt = [n for n in names if R[n]["n_match"]]
        thin = [n for n in names if R[n]["n_match"] <= THIN]
        wide = [n for n in names if R[n]["n_match"] > THIN]
        big = sorted(names, key=lambda n: -R[n]["abs_off"])[:TOP]
        med = lambda ns: float(np.median([R[n]["abs_off"] for n in ns])) if ns else None
        closer = [n for n in mt if R[n]["abs_off"] < abs(R[n]["departure"])]
        S = {"n": len(names), "n_missing": len(R) - len(names), "n_matched": len(mt),
             "unmatched": [n for n in names if not R[n]["n_match"]], "n_thin": len(thin), "n_wide": len(wide),
             "median_abs_off_thin": med(thin), "median_abs_off_wide": med(wide),
             "median_abs_off_unmatched": med([n for n in names if not R[n]["n_match"]]),
             "largest": big, "largest_thin": int(sum(n in thin for n in big)),
             "thin_share": len(thin) / len(names),
             "largest_thin_per_seed": [int(sum(R[n]["n_match"] <= THIN for n in
                                               sorted(names, key=lambda n: -abs(R[n]["offset"][i]))[:TOP]))
                                       for i in range(len(RUNS))],
             "median_abs_departure": float(np.median([abs(R[n]["departure"]) for n in mt])),
             "n_closer_than_match": len(closer), "outside_all": int(sum(R[n]["outside_all"] for n in names)),
             "median_nearest_in_sd": float(np.median([R[n]["nearest_in_sd"] for n in names])),
             "level_check_maxabs": float(max(check)),
             "zero_mse": [ks[r][a]["zero"] for r in RUNS], "level_share": [ks[r][a]["level_share"] for r in RUNS]}
        res[a], s[a] = R, S
        print(f"{a}: {S['n']} held out ({S['n_missing']} missing), matched {S['n_matched']}, unmatched "
              f"{len(S['unmatched'])}; thin (<= {THIN} matches) {S['n_thin']}: median |offset| "
              f"{S['median_abs_off_thin']:.3f} (unmatched {S['median_abs_off_unmatched']}), wider {S['n_wide']}: "
              f"{S['median_abs_off_wide']}; {TOP} largest: {S['largest_thin']} thin (per seed "
              f"{S['largest_thin_per_seed']}) against a thin share {S['thin_share']:.2f}; median |departure| "
              f"{S['median_abs_departure']:.2f}, offset smaller on {S['n_closer_than_match']}/{len(mt)}; outside "
              f"{S['outside_all']}, nearest {S['median_nearest_in_sd']:.3f} sd; level check {S['level_check_maxabs']:.2g}; "
              f"zero MSE {S['zero_mse']}, level share {S['level_share']}")
        for n in big:
            r = R[n]
            print(f"  {n:30s} N={r['n']} {r['initial']:3s} {r['born']:7s} |offset| {r['abs_off']:.3f} "
                  f"({r['offset'][0]:+.3f}/{r['offset'][1]:+.3f}) matched {r['n_match']}")
        for k in ("n", "n_matched", "n_thin", "n_wide", "largest_thin", "n_closer_than_match", "outside_all"):
            flat[f"{a}.{k}"] = S[k]
        flat[f"{a}.n_unmatched"] = len(S["unmatched"])
        flat[f"{a}.largest_thin_seed42"], flat[f"{a}.largest_thin_seed43"] = S["largest_thin_per_seed"]
        for k in ("median_abs_off_thin", "median_abs_off_wide", "thin_share", "median_abs_departure",
                  "median_nearest_in_sd"):
            flat[f"{a}.{k}"] = None if S[k] is None else round(S[k], 3)
        flat[f"{a}.level_check_maxabs"] = round(S["level_check_maxabs"], 3)
    flat["n_train"], flat["train_level_sd"] = s["n_train"], round(s["train_level_sd"], 2)
    json.dump({"summary": s, "heldout": res, "train": {m: {"level": T[m][0], **P[m]} for m in T},
               "train_missing": miss_tr}, open(os.path.join(HERE, "data", "tree_levels.json"), "w"), indent=1)
    plot_trees(res)
    print(json.dumps(flat))


def plot_trees(res):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    cols = (ps.C.blue, ps.C.vermillion, ps.C.green)
    for ax, c in zip(TREE_AXES, cols):
        R = {n: r for n, r in res[ax].items() if not r.get("missing")}
        a.plot([r["n_match"] + 1 + rng.uniform(-0.15, 0.15) * (r["n_match"] + 1) for r in R.values()],
               [r["abs_off"] for r in R.values()], ".", color=c, label=f"{ax} ({len(R)})")
        mt = [r for r in R.values() if r["n_match"]]
        b.plot([r["departure"] for r in mt], [np.mean(r["offset"]) for r in mt], ".", color=c, label=ax)
    a.axvline(THIN + 1.5, ls="--", color=ps.C.grey)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"matched training pools $+1$"); a.set_ylabel(r"$|\overline{q-t}|$")
    ps.legend(a, "upper right"); ps.make_room(a)
    lo, hi = b.get_xlim()
    b.plot([lo, hi], [-lo, -hi], "--", color=ps.C.grey, label="offset $=-$departure")
    b.set_xlabel(r"$\overline{t}-\overline{t}_{\rm matched}$"); b.set_ylabel(r"$\overline{q-t}$")
    ps.legend(b, "upper right"); ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "tree_levels"))


def plot(res, mt, names, T, P):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    a.plot([P[m]["n"] - 0.12 + rng.uniform(-0.05, 0.05) for m in T], [T[m][0] for m in T], ".", color=ps.C.blue,
           label=f"training ({len(T)})")
    a.plot([res[n]["n"] + 0.12 for n in names], [res[n]["level"] for n in names], "o", mfc="none", color=ps.C.vermillion,
           label=f"held out ({len(names)})")
    a.plot([res[n]["n"] + 0.12 for n in names], [np.mean(res[n]["pred_level"]) for n in names], "x", color=ps.C.green,
           label="held out, predicted")
    from matplotlib.ticker import MaxNLocator
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.set_xlabel(r"particles $N$"); a.set_ylabel(r"$\overline{t}$")
    ps.legend(a, "upper right"); ps.make_room(a)
    dep = [res[n]["departure"] for n in mt]
    b.errorbar(dep, [np.mean(res[n]["offset"]) for n in mt],
               yerr=[abs(res[n]["offset"][0] - res[n]["offset"][1]) / 2 for n in mt], fmt="o", color=ps.C.blue,
               label="held-out pool, seed mean")
    lo, hi = min(dep), max(dep)
    b.plot([lo, hi], [-lo, -hi], "--", color=ps.C.grey, label="offset $=-$departure")
    b.set_xlabel(r"$\overline{t}-\overline{t}_{\rm matched}$"); b.set_ylabel(r"$\overline{q-t}$")
    ps.legend(b, "upper right"); ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "loop_levels"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--trees", action="store_true", help="the tree held-out axes H1, H2, H3")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    A = ap.parse_args()
    if A.dump:
        dump(A.trees, A.chunk, A.nchunks)
    else:
        (analyse_trees if A.trees else analyse)()
