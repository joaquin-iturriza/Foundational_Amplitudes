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

Top read (in --trees): does an external top carry the tree level error; see top_read.
New process or new mass (in --trees): is a new process's level error a new-process effect or the untrained top mass;
see group_read (key abs_off).
Shape read (in --trees): the floor var(q - t) left after each pool's offset, in the same groups and cells; see
group_read (key shape), figs/tree_levels_shape.{png,pdf}.

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
            "qcd_born": "_qcdborn" in name, "loop_induced": bool(c.get("loopind")), "order": o,
            "top": 6 in [abs(x) for x in c["pdg_ids"]], "heavy_layer": "__mt" in name}


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
            fl = [kl[r][a][n]["floor"] for r in RUNS]
            r = {**pr, "level": L, "outside_all": bool(L < lv.min() or L > lv.max()),
                 "nearest_in_sd": float(np.min(np.abs(lv - L)) / lv.std()), "matched": match, "n_match": len(match),
                 "offset": offs, "abs_off": float(abs(np.mean(offs))), "floor": fl, "shape": float(np.mean(fl))}
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
        for n in big:
            flat[f"{a}.match_n.{n}"] = R[n]["n_match"]
    flat["n_train"], flat["train_level_sd"] = s["n_train"], round(s["train_level_sd"], 2)
    s["top"] = top_read(res, T, P, flat)
    s["new_process"] = group_read(res, flat, "abs_off", "np", "new process or new mass")
    s["shape"] = group_read(res, flat, "shape", "sh", "shape (floor = var(q - t))")
    json.dump({"summary": s, "heldout": res, "train": {m: {"level": T[m][0], **P[m]} for m in T},
               "train_missing": miss_tr}, open(os.path.join(HERE, "data", "tree_levels.json"), "w"), indent=1)
    plot_trees(res)
    plot_top(res, T)
    plot_new_process(res)
    plot_shape(res)
    print(json.dumps(flat))


def top_read(res, T, P, flat):
    """Does an external top carry the tree level error (v3_prep_explore, next step of par:v4_tree_levels). Three reads,
    no fit, no new threshold: per axis and pooled over the three, the seed-mean |offset| of pools with an external top
    against those without; the TOP largest offsets' top count against the top share; and, for top pools, the level
    against the matched training pools (the one match rule) that also have an external top, a property read beside
    the match, never a re-match. The training tree pools' only external tops are the heavy-quark layer's (the top's
    quantum numbers at m_Q = 50, 300, 600 GeV); a held-out pool whose own process is in that layer is also read against
    those siblings' levels (inside their range or not)."""
    sib = {}
    for m in T:
        if P[m]["heavy_layer"]:
            sib.setdefault(BASE[m], []).append(m)
    out, allp = {"train_top": int(sum(P[m]["top"] for m in T)),
                 "train_top_heavy_layer": int(sum(P[m]["top"] and P[m]["heavy_layer"] for m in T))}, []
    for a in TREE_AXES:
        R = {n: r for n, r in res[a].items() if not r.get("missing")}
        allp += [(a, n) for n in R]
        top = [n for n in R if R[n]["top"]]
        no = [n for n in R if not R[n]["top"]]
        out[a] = {"n_top": len(top), "n_no_top": len(no),
                  "median_abs_off_top": float(np.median([R[n]["abs_off"] for n in top])) if top else None,
                  "median_abs_off_no_top": float(np.median([R[n]["abs_off"] for n in no])) if no else None}
        for n in top:
            r = R[n]
            mtop = [m for m in r["matched"] if P[m]["top"]]
            r["n_match_top"] = len(mtop)
            if mtop:
                r["departure_top"] = r["level"] - float(np.mean([T[m][0] for m in mtop]))
            sl = [T[m][0] for m in sib.get(n, [])]
            r["n_sibling"] = len(sl)
            if sl:
                r["inside_sibling_range"] = bool(min(sl) <= r["level"] <= max(sl))
        wt = [n for n in top if R[n]["n_match_top"]]
        if wt:
            out[a].update({
                "n_match_top": len(wt), "n_no_match_top": len(top) - len(wt),
                "median_abs_off_match_top": float(np.median([R[n]["abs_off"] for n in wt])),
                "median_abs_off_no_match_top": float(np.median([R[n]["abs_off"] for n in top if n not in wt])),
                "median_abs_departure_top": float(np.median([abs(R[n]["departure_top"]) for n in wt])),
                "median_abs_departure_all": float(np.median([abs(R[n]["departure"]) for n in wt])),
                "n_departure_top_smaller": int(sum(abs(R[n]["departure_top"]) < abs(R[n]["departure"]) for n in wt)),
                "n_off_smaller_than_departure_top": int(sum(R[n]["abs_off"] < abs(R[n]["departure_top"]) for n in wt))})
        ws = [n for n in top if R[n]["n_sibling"]]
        if ws:
            out[a].update({
                "n_sibling": len(ws),
                "n_inside_sibling_range": int(sum(R[n]["inside_sibling_range"] for n in ws)),
                "median_abs_off_sibling": float(np.median([R[n]["abs_off"] for n in ws])),
                "median_abs_off_no_sibling": float(np.median([R[n]["abs_off"] for n in top if n not in ws]))})
            for N in sorted({R[n]["n"] for n in ws}):  # the same split at one multiplicity (siblings are mostly small N)
                a1 = [R[n]["abs_off"] for n in ws if R[n]["n"] == N]
                a0 = [R[n]["abs_off"] for n in top if n not in ws and R[n]["n"] == N]
                out[a].update({f"n_sibling_N{N}": len(a1), f"median_abs_off_sibling_N{N}": float(np.median(a1)),
                               f"n_no_sibling_N{N}": len(a0),
                               f"median_abs_off_no_sibling_N{N}": float(np.median(a0)) if a0 else None})
    off = {k: res[k[0]][k[1]]["abs_off"] for k in allp}
    big = sorted(allp, key=lambda k: -off[k])[:TOP]
    ntop = sum(res[a][n]["top"] for a, n in allp)
    sq = sum(off[k] ** 2 for k in allp)
    out["pooled"] = {"n": len(allp), "n_top": ntop, "top_share": ntop / len(allp),
                     "largest": [f"{a}:{n}" for a, n in big], "largest_top": int(sum(res[a][n]["top"] for a, n in big)),
                     "largest_top_per_seed": [int(sum(res[a][n]["top"] for a, n in sorted(
                         allp, key=lambda k: -abs(res[k[0]][k[1]]["offset"][i]))[:TOP])) for i in range(len(RUNS))],
                     "top_share_sq_offset": sum(off[k] ** 2 for k in allp if res[k[0]][k[1]]["top"]) / sq}
    O = out["pooled"]
    print(f"top: training tree pools with an external top {out['train_top']} (heavy-quark layer "
          f"{out['train_top_heavy_layer']}); pooled {O['n']} held out, {O['n_top']} with a top (share "
          f"{O['top_share']:.2f}); {TOP} largest: {O['largest_top']} top (per seed {O['largest_top_per_seed']}); top "
          f"share of the summed squared offset {O['top_share_sq_offset']:.2f}")
    for a in TREE_AXES:
        print(f"  {a}: {out[a]}")
    flat.update({"top.train_top": out["train_top"], "top.train_top_heavy_layer": out["train_top_heavy_layer"],
                 "top.n": O["n"], "top.n_top": O["n_top"], "top.top_share": round(O["top_share"], 3),
                 "top.largest_top": O["largest_top"], "top.top_share_sq_offset": round(O["top_share_sq_offset"], 3)})
    flat["top.largest_top_seed42"], flat["top.largest_top_seed43"] = O["largest_top_per_seed"]
    for a in TREE_AXES:
        for k, v in out[a].items():
            flat[f"top.{a}.{k}"] = round(v, 3) if isinstance(v, float) else v
    return out


def massive_legs(name):
    """External legs with a mass in catalog_v4: the property table's massive species (W, Z, H, t, tau, the BSM
    states), the b counted massless (catalog_v4 generates it massless)."""
    import mg5_pipeline_final as mg
    from particle_ids import BSM_PARTICLE_PROPERTIES
    ids = [abs(int(x)) for x in mg.PROCESSES[BASE.get(name, name)]["pdg_ids"]]
    return int(sum(x != 5 and (mg._table_mass(x) > 0 or x in BSM_PARTICLE_PROPERTIES) for x in ids))


NP_GROUPS = (("new combinations", "H1", lambda r: True), ("multiplicity", "H2", lambda r: True),
             ("top, process at another top mass", "H3", lambda r: r.get("n_sibling", 0) > 0),
             ("top, process at no top mass", "H3", lambda r: not r.get("n_sibling", 0)))


NP_KEY = {"new combinations": "H1", "multiplicity": "H2", "top, process at another top mass": "other_mass",
          "top, process at no top mass": "no_mass", "one-loop, virtual": "loop_virtual",
          "one-loop, loop-induced": "loop_induced"}
NEWP = ("new combinations", "multiplicity")
REFS = ("top, process at another top mass", "top, process at no top mass")


def cell_compare(G, key, labs=NEWP, refs=REFS):
    """Per group in `labs`, in each cell of multiplicity and initial state holding both top groups, each pool's
    per-pool quantity `key` against the two top groups' cell medians, read as nearer one or the other in log10; the
    count and the median log10 ratios."""
    cell = lambda r: (r["n"], r["initial"])
    sib, nos = G[refs[0]], G[refs[1]]
    med = lambda rs: float(np.median([r[key] for r in rs]))
    out = {}
    for lab in labs:
        g, O = G[lab], {"cells": {}}
        lr_nos, lr_sib, near_nos, used = [], [], 0, []
        for c in sorted({cell(r) for r in g.values()}):
            A = [r for r in g.values() if cell(r) == c]
            B = [r for r in nos.values() if cell(r) == c]
            C = [r for r in sib.values() if cell(r) == c]
            O["cells"][f"{c[0]}{c[1]}"] = {"n": len(A), "median": med(A), "n_no_mass": len(B),
                                          "median_no_mass": med(B) if B else None, "n_other_mass": len(C),
                                          "median_other_mass": med(C) if C else None}
            if not (B and C):
                continue
            mb, mc = med(B), med(C)
            for r in A:
                x, y = np.log10(r[key] / mb), np.log10(r[key] / mc)
                lr_nos.append(x); lr_sib.append(y); near_nos += abs(x) < abs(y); used.append(r)
        O.update({"n_compared": len(used), "n_cells_compared": int(sum(
                      v["n_no_mass"] > 0 and v["n_other_mass"] > 0 for v in O["cells"].values())),
                  "n_nearer_no_mass": int(near_nos),
                  "median_log_ratio_no_mass": float(np.median(lr_nos)) if used else None,
                  "median_log_ratio_other_mass": float(np.median(lr_sib)) if used else None,
                  f"median_{key}_compared": med(used) if used else None})
        out[lab] = O
    return out


def group_read(res, flat, key, prefix, label, groups=NP_GROUPS, newp_labs=NEWP):
    """New process or new mass, for one per-pool quantity `key` (v3_prep_explore, the user's priorities D178: the
    zero-shot bias first, then the shape on unseen processes). The new combinations (H1) and multiplicity (H2) pools
    are new processes made of particles trained at their masses; the particle content pools (H3, all with a top at
    172.5 GeV, a mass no training tree has) split by top_read into the 35 whose process is in the heavy-quark layer at
    another top mass and the 244 whose process is at none. No fit, no new threshold: per group the median; in each
    cell of multiplicity and initial state that holds the new-process pool and both top groups, the pool's value
    against the two top groups' cell medians (cell_compare); over the new-process pools the rank correlation
    (Spearman) with multiplicity, the born's alpha_s and alpha powers, the massive legs and the matched class size, and
    the median by initial state and by massive legs. key "abs_off": the seed-mean |offset| (with the signed offset's
    median and positive share per group, and the correlation with the departure from the matched class); key "shape":
    the seed-mean floor var(q - t), the MSE left once the pool's own offset is removed (with the level share
    offset^2 / zero-shot MSE per group, and the correlation with |offset|, over the new-process pools and over all)."""
    from scipy.stats import spearmanr
    G = {lab: {n: r for n, r in res[a].items() if not r.get("missing") and sel(r)} for lab, a, sel in groups}
    for g in G.values():
        for n, r in g.items():
            r.setdefault("massive_legs", massive_legs(n))
    med = lambda rs: float(np.median([r[key] for r in rs]))
    out = {"groups": {}}
    for lab, g in G.items():
        v = {"n": len(g), f"median_{key}": med(g.values())}
        if key == "abs_off":
            so = [float(np.mean(r["offset"])) for r in g.values()]
            v.update({"median_signed_off": float(np.median(so)), "positive_share": float(np.mean(np.array(so) > 0)),
                      "median_massive_legs": float(np.median([r["massive_legs"] for r in g.values()]))})
        else:
            v["median_level_share"] = float(np.median([np.mean([o ** 2 / (o ** 2 + f) for o, f in
                                                                zip(r["offset"], r["floor"])]) for r in g.values()]))
        out["groups"][lab] = v
    out.update(cell_compare(G, key, newp_labs))
    newp = [r for lab in newp_labs for r in G[lab].values()]
    y = [r[key] for r in newp]
    pred = {"particles": [r["n"] for r in newp], "alpha_s power": [r["order"][2] for r in newp],
            "alpha power": [r["order"][3] for r in newp], "massive legs": [r["massive_legs"] for r in newp],
            "matched class size": [r["n_match"] for r in newp]}
    if key == "shape":
        pred["abs offset"] = [r["abs_off"] for r in newp]
    rho = {k: float(spearmanr(v, y)[0]) for k, v in pred.items()}
    if key == "abs_off":
        mt = [r for r in newp if r["n_match"]]
        rho["departure from the matched class"] = float(spearmanr([abs(r["departure"]) for r in mt],
                                                                  [r["abs_off"] for r in mt])[0])
    else:
        allr = [r for g in G.values() for r in g.values()]
        rho["abs offset, all groups"] = float(spearmanr([r["abs_off"] for r in allr], [r["shape"] for r in allr])[0])
    out["predictors"] = {"n": len(newp), "spearman": rho,
                         "by_initial": {i: {"n": int(sum(r["initial"] == i for r in newp)),
                                            f"median_{key}": med([r for r in newp if r["initial"] == i])}
                                        for i in sorted({r["initial"] for r in newp})},
                         "by_massive_legs": {int(k): {"n": int(sum(r["massive_legs"] == k for r in newp)),
                                                      f"median_{key}": med([r for r in newp if r["massive_legs"] == k])}
                                             for k in sorted({r["massive_legs"] for r in newp})}}
    print(label + ":")
    for lab, v in out["groups"].items():
        print(f"  {lab:34s} {v}")
    for lab in newp_labs:
        O = out[lab]
        print(f"  {lab}: compared {O['n_compared']} pools in {O['n_cells_compared']} cells; nearer the no-mass group "
              f"{O['n_nearer_no_mass']}; median log10 ratio to no-mass {O['median_log_ratio_no_mass']}, to other-mass "
              f"{O['median_log_ratio_other_mass']}")
        for c, v in O["cells"].items():
            print(f"    {c}: {v}")
    print(f"  predictors over {len(newp)} new-process pools: {rho}")
    print(f"  by initial {out['predictors']['by_initial']}")
    print(f"  by massive legs {out['predictors']['by_massive_legs']}")
    rnd = lambda x: round(x, 3) if isinstance(x, float) else x
    for lab, v in out["groups"].items():
        for k, x in v.items():
            flat[f"{prefix}.{NP_KEY[lab]}.{k}"] = rnd(x)
    for lab in newp_labs:
        for k, x in out[lab].items():
            if k != "cells":
                flat[f"{prefix}.{NP_KEY[lab]}.{k}"] = rnd(x)
    for k, x in rho.items():
        flat[f"{prefix}.rho." + k.replace(", ", "_").replace(" ", "_")] = round(x, 2)
    for i, v in out["predictors"]["by_initial"].items():
        flat[f"{prefix}.initial.{i}.n"], flat[f"{prefix}.initial.{i}.median_{key}"] = v["n"], rnd(v[f"median_{key}"])
    for k, v in out["predictors"]["by_massive_legs"].items():
        flat[f"{prefix}.massive{k}.n"], flat[f"{prefix}.massive{k}.median_{key}"] = v["n"], rnd(v[f"median_{key}"])
    return out


LOOP_GROUPS = NP_GROUPS + (("one-loop, virtual", "H5", lambda r: not r["loop_induced"]),
                           ("one-loop, loop-induced", "H5", lambda r: r["loop_induced"]))


def loop_group_read():
    """The one-loop axis (H5) by the tree reads' method (v3_prep_explore, the user's priorities D178: the zero-shot
    bias first, then the shape). On trees both the level and the shape are largest where the top is held out at its
    mass (par:v4_tree_newproc, par:v4_tree_shape); is the one-loop axis's error the same source? Reads
    data/tree_levels.json (analyse_trees), data/loop_levels.json (analyse) and data/kshot_level.json, no new dump.
    First the counts the split needs: held-out and training one-loop pools with an external top, and training one-loop
    pools of the heavy-quark layer. Then group_read for both quantities (the seed-mean |offset| and the seed-mean floor
    var(q - t)) with the tree read's four groups plus the one-loop pools, virtual and loop-induced apart: the
    group medians, each one-loop pool against the tree top groups' cell medians in its cell of multiplicity and
    initial state (cell_compare), and over the virtual one-loop pools the rank correlations with the listed properties
    and the departure from the matched training level (par:v4_loop_levels). One-loop and tree pools have different
    targets (each its own t = log(|M|^2 F)); both quantities are in the same units of t. No fit, no new threshold.
        python3 analysis/catalog_v4/loop_levels.py --loop-groups -> data/loop_groups.json, figs/loop_groups.{png,pdf}"""
    tree = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    loop = json.load(open(os.path.join(HERE, "data", "loop_levels.json")))
    kl = json.load(open(os.path.join(HERE, "data", "kshot_level.json")))["probes"]
    res = {a: tree["heldout"][a] for a in TREE_AXES}
    H5 = {}
    for n, r in loop["heldout"].items():
        if r.get("missing"):
            H5[n] = r; continue
        offs = [kl[x]["H5"][n]["offset"] for x in RUNS]
        fl = [kl[x]["H5"][n]["floor"] for x in RUNS]
        H5[n] = {**r, **props(n), "offset": offs, "abs_off": float(abs(np.mean(offs))), "floor": fl,
                 "shape": float(np.mean(fl))}
    res["H5"] = H5
    tr = {m: props(m) for m in loop["train"]}
    R = {n: r for n, r in H5.items() if not r.get("missing")}
    cnt = {"n_heldout": len(R), "n_heldout_top": int(sum(r["top"] for r in R.values())),
           "n_heldout_loop_induced": int(sum(r["loop_induced"] for r in R.values())),
           "n_train": len(tr), "n_train_top": int(sum(p["top"] for p in tr.values())),
           "n_train_heavy_layer": int(sum(p["heavy_layer"] for p in tr.values())),
           "n_train_loop_induced": int(sum(p["loop_induced"] for p in tr.values()))}
    print(f"one-loop counts: {cnt}")
    flat = {f"lg.{k}": v for k, v in cnt.items()}
    out = {"counts": cnt,
           "level": group_read(res, flat, "abs_off", "lgo", "one-loop level (|offset|)", LOOP_GROUPS,
                               ("one-loop, virtual",)),
           "shape": group_read(res, flat, "shape", "lgs", "one-loop shape (floor = var(q - t))", LOOP_GROUPS,
                               ("one-loop, virtual",))}
    for q in ("abs_off", "shape"):  # the one-loop virtual group's median over the tree top-at-no-mass group's
        g = out["level" if q == "abs_off" else "shape"]["groups"]
        flat[f"lg.ratio_virtual_to_no_mass.{q}"] = round(g["one-loop, virtual"][f"median_{q}"]
                                                         / g["top, process at no top mass"][f"median_{q}"], 3)
    li = [n for n, r in R.items() if r["loop_induced"]]
    for n in li:
        flat[f"lg.loop_induced.{n}.abs_off"] = round(R[n]["abs_off"], 3)
        flat[f"lg.loop_induced.{n}.shape"] = round(R[n]["shape"], 3)
    json.dump(out, open(os.path.join(HERE, "data", "loop_groups.json"), "w"), indent=1)
    plot_loop_groups(res)
    print(json.dumps(flat))


def plot_loop_groups(res):
    """Fig. v4_loop_groups: per group, the tree read's four and the one-loop virtual pools (the loop-induced pool
    drawn apart), left the seed-mean |offset|, right the seed-mean floor; the bar the group's median."""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    cols = (ps.C.blue, ps.C.green, ps.C.vermillion, ps.C.grey, ps.C.purple, ps.C.orange)
    G = [(lab, [r for r in res[ax].values() if not r.get("missing") and sel(r)]) for lab, ax, sel in LOOP_GROUPS]
    ticks = ["new\ncombi-\nnations", "multi-\nplicity", "top,\nother\nmass", "top,\nno\nmass", "one-\nloop,\nvirtual",
             "loop-\ninduced"]
    for a, key, yl in zip(axs, ("abs_off", "shape"), (r"$|\overline{q-t}|$", r"$\mathrm{var}(q-t)$")):
        for i, ((lab, rs), c) in enumerate(zip(G, cols)):
            a.plot(i + rng.uniform(-0.2, 0.2, len(rs)), [r[key] for r in rs], ".", color=c, alpha=0.6)
            if len(rs) > 1:
                a.plot([i - 0.3, i + 0.3], [np.median([r[key] for r in rs])] * 2, "-", color="k")
        a.set_xticks(range(len(G)), [f"{t}\n({len(rs)})" for t, (lab, rs) in zip(ticks, G)], fontsize=7)
        a.set_yscale("log"); a.set_ylabel(yl)
    ps.save(fig, os.path.join(HERE, "figs", "loop_groups"))


def plot_shape(res):
    """Fig. v4_tree_shape: left, per pool its seed-mean floor against its seed-mean |offset|, coloured by group;
    right, the four groups' floors, the bar the median (as tree_levels_newproc's left panel)."""
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    cols = (ps.C.blue, ps.C.green, ps.C.vermillion, ps.C.grey)
    G = [(lab, [r for r in res[ax].values() if not r.get("missing") and sel(r)]) for lab, ax, sel in NP_GROUPS]
    for (lab, rs), c in zip(G[::-1], cols[::-1]):
        a.plot([r["abs_off"] for r in rs], [r["shape"] for r in rs], ".", color=c, alpha=0.6, label=lab)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$|\overline{q-t}|$"); a.set_ylabel(r"$\mathrm{var}(q-t)$")
    ps.legend(a, "lower right")
    for i, ((lab, rs), c) in enumerate(zip(G, cols)):
        b.plot(i + rng.uniform(-0.2, 0.2, len(rs)), [r["shape"] for r in rs], ".", color=c, alpha=0.6)
        b.plot([i - 0.3, i + 0.3], [np.median([r["shape"] for r in rs])] * 2, "-", color="k")
    b.set_xticks(range(4), ["new\ncombi-\nnations", "multi-\nplicity", "top,\nother\nmass", "top,\nno\nmass"],
                 fontsize=8)
    b.set_yscale("log"); b.set_ylabel(r"$\mathrm{var}(q-t)$")
    ps.save(fig, os.path.join(HERE, "figs", "tree_levels_shape"))


def plot_new_process(res):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    cols = (ps.C.blue, ps.C.green, ps.C.vermillion, ps.C.grey)
    G = [(lab, [r for r in res[ax].values() if not r.get("missing") and sel(r)]) for lab, ax, sel in NP_GROUPS]
    for i, ((lab, rs), c) in enumerate(zip(G, cols)):
        a.plot(i + rng.uniform(-0.2, 0.2, len(rs)), [r["abs_off"] for r in rs], ".", color=c, alpha=0.6)
        a.plot([i - 0.3, i + 0.3], [np.median([r["abs_off"] for r in rs])] * 2, "-", color="k")
    a.set_xticks(range(4), ["new\ncombi-\nnations\n(%d)" % len(G[0][1]), "multi-\nplicity\n(%d)" % len(G[1][1]),
                            "top,\nother\nmass\n(%d)" % len(G[2][1]), "top,\nno\nmass\n(%d)" % len(G[3][1])],
                  fontsize=8)
    a.set_yscale("log"); a.set_ylabel(r"$|\overline{q-t}|$")  # no legend: the groups are the ticks
    for (lab, rs), c in zip(G, cols):
        rs = [r for r in rs if r["n"] == 6]
        ks = sorted({r["initial"] for r in rs})
        x = {k: j for j, k in enumerate(("qq", "qg", "gg", "bq", "bg", "bb", "ee"))}
        off = {0: -0.24, 1: -0.08, 2: 0.08, 3: 0.24}[[g[0] for g in G].index(lab)]
        for k in ks:
            v = [r["abs_off"] for r in rs if r["initial"] == k]
            b.plot(x[k] + off, np.median(v), "o", color=c, label=lab if k == ks[0] else None)
    b.set_xticks(range(7), ["qq", "qg", "gg", "bq", "bg", "bb", "ee"])
    b.set_yscale("log"); b.set_xlabel(r"initial state, $N=6$"); b.set_ylabel(r"median $|\overline{q-t}|$")
    ps.legend(b, "upper right"); ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "tree_levels_newproc"))


AXIS_NAME = {"H1": "new combinations", "H2": "multiplicity", "H3": "particle content"}


def plot_top(res, T):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    R = {n: r for n, r in res["H3"].items() if not r.get("missing")}
    wt = [r for r in R.values() if r.get("n_match_top")]
    nt = [r for r in R.values() if r["n_match"] and not r.get("n_match_top")]
    a.plot([r["departure"] for r in nt], [np.mean(r["offset"]) for r in nt], ".", color=ps.C.grey,
           label=f"no matched top pool ({len(nt)})")
    a.plot([r["departure"] for r in wt], [np.mean(r["offset"]) for r in wt], ".", color=ps.C.vermillion,
           label=f"all matched pools ({len(wt)})")
    a.plot([r["departure_top"] for r in wt], [np.mean(r["offset"]) for r in wt], "o", mfc="none", color=ps.C.blue,
           label="matched top pools")
    yo = [np.mean(r["offset"]) for r in R.values() if r["n_match"]]
    a.plot([-max(yo), -min(yo)], [max(yo), min(yo)], "--", color=ps.C.grey, label="offset $=-$departure")
    a.set_xlabel(r"$\overline{t}-\overline{t}_{\rm matched}$"); a.set_ylabel(r"$\overline{q-t}$")
    ps.legend(a, "upper right"); ps.make_room(a)
    ws = [r for r in R.values() if r.get("n_sibling")]
    rest = [r for r in R.values() if not r.get("n_sibling")]
    rng = np.random.default_rng(0)
    for i, (grp, c) in enumerate(((rest, ps.C.grey), ([r for r in ws if r["inside_sibling_range"]], ps.C.blue),
                                  ([r for r in ws if not r["inside_sibling_range"]], ps.C.vermillion))):
        b.plot(i + rng.uniform(-0.15, 0.15, len(grp)), [r["abs_off"] for r in grp], ".", color=c)
    b.set_xticks([0, 1, 2], ["no sibling\n(%d)" % len(rest), "inside\n(%d)" % sum(r["inside_sibling_range"] for r in ws),
                             "outside\n(%d)" % sum(not r["inside_sibling_range"] for r in ws)])
    b.set_yscale("log"); b.set_ylabel(r"$|\overline{q-t}|$")  # no legend: the groups are the ticks
    ps.save(fig, os.path.join(HERE, "figs", "tree_levels_top"))


def plot_trees(res):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rng = np.random.default_rng(0)
    cols = (ps.C.blue, ps.C.vermillion, ps.C.green)
    for ax, c in zip(TREE_AXES, cols):
        R = {n: r for n, r in res[ax].items() if not r.get("missing")}
        a.plot([r["n_match"] + 1 + rng.uniform(-0.15, 0.15) * (r["n_match"] + 1) for r in R.values()],
               [r["abs_off"] for r in R.values()], ".", color=c)
        mt = [r for r in R.values() if r["n_match"]]
        b.plot([r["departure"] for r in mt], [np.mean(r["offset"]) for r in mt], ".", color=c,
               label=f"{AXIS_NAME[ax]} ({len(R)})")
    a.axvline(THIN + 1.5, ls="--", color=ps.C.grey)
    a.set_xscale("log"); a.set_yscale("log")
    yo = [r["abs_off"] for ax in TREE_AXES for r in res[ax].values() if not r.get("missing")]
    a.set_ylim(min(yo) / 1.5, max(yo) * 1.5)  # trimmed to the data; no legend here (the right panel's colours)
    a.set_xlabel(r"matched training pools $+1$"); a.set_ylabel(r"$|\overline{q-t}|$")
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


def training_parents(r, n_parent, T, P4, cls, rep_of, adds):
    """The parent rule of multiplicity_read: the training class representatives (heavy-quark layer out) with n_parent
    particles from which a member of class r was built by one recorded addition (catalog_v4_processes.yaml `why`)."""
    out = set()
    for m in cls.get(r, [r]):
        why = P4.get(m, {}).get("why", "")
        for s_ in adds:
            if why.endswith(s_) and why[: -len(s_)] in rep_of:
                out.add(rep_of[why[: -len(s_)]]); break
    return sorted(p for p in out if p in T and not T[p]["heavy_layer"] and T[p]["n"] == n_parent)


# colour Casimir and squared electric charge per |pdg| (SM, the 2HDM's H+ 37, h2 35, h3 36 neutral)
CF, CA = 4.0 / 3.0, 3.0
CHARGES = {**{q: (CF, 4.0 / 9.0) for q in (2, 4, 6)}, **{q: (CF, 1.0 / 9.0) for q in (1, 3, 5)}, 21: (CA, 0.0),
           **{l: (0.0, 1.0) for l in (11, 13, 15, 24, 37)}, **{z: (0.0, 0.0) for z in (12, 14, 16, 22, 23, 25, 35, 36)}}


def eikonal_term(pdgs):
    """Candidate 1 of multiplicity_candidates, from the emission-scale target's soft limit: an emitted soft gluon
    (photon) k gives |M_{n+1}|^2 -> 4 pi alpha_s (alpha) * sum over unordered pairs a<b of the rest of
    4 (-T_a.T_b) / q^2_ab |M_n|^2 (q^2_ab = s_ak s_bk / s_ab, the target's emission scale; for e+e- -> q q~ g,
    4 C_F / q^2, the known (x1^2 + x2^2) / ((1-x1)(1-x2)) at its soft end). The target divides out 4 pi alpha and q^2,
    so each soft emission leaves log(4 x colour weight) in the level. Weighting the pairs alike, colour conservation
    gives sum_{a<b} (-T_a.T_b) = (1/2) sum_a C_a, so the term is log(2 sum_a C_a) over the rest's legs (incoming and
    outgoing), log(2 sum_a Q_a^2) for a photon. Returns the sum over the final gluons and photons, None for a species
    outside CHARGES or an emission with no charged rest. Z and H are massive (no soft limit) and get none."""
    if any(abs(p) not in CHARGES for p in pdgs):
        return None
    tot = 0.0
    for k in range(2, len(pdgs)):
        if abs(pdgs[k]) not in (21, 22):
            continue
        rest = [CHARGES[abs(p)][0 if abs(pdgs[k]) == 21 else 1] for i, p in enumerate(pdgs) if i != k]
        if sum(rest) <= 0:
            return None
        tot += float(np.log(2.0 * sum(rest)))
    return tot


def multiplicity_candidates():
    """Candidate target or input changes for the level's dependence on multiplicity (v3_prep_explore, the user's first
    priority D178; par:v4_mult_bias: the multiplicity pools' signed offset follows the level's step from the five
    particle parent). No new run: the per-pool train levels of data/tree_levels.json (analyse_trees, from
    tree_levels_cc.json) and multiplicity_read's parent rule. Candidates, fixed before their numbers were read, each a
    function of the process's particles alone (every process alike, never per beam), none with a fitted scale:
      none        the pretraining default (the emission-scale target, coupling orders, s0 = 1), the reference;
      soft_norm   the soft limit's normalisation, log 4 per emission step beyond the 2->2 core, taken out of the target
                  ((N - 4) log 4; a constant per particle, which moves every step alike and so tests the steps' common
                  part only);
      eikonal     the soft limit's colour or charge weight taken out as well: minus eikonal_term (per final gluon or
                  photon, log(2 sum C_a) or log(2 sum Q_a^2) over the other legs);
      parent_scale the parent's level as an input scale, the network predicting a pool's level less its one-particle-
                  fewer training parents' mean level: the step it must then extrapolate from five to six particles is
                  the step's change along the family, the own step less the parent's own step from its four particle
                  training parents (read on the pools whose parent has one).
    Per candidate, on the training six particle pools with a five particle training parent and on the multiplicity
    held-out pools (H2): the median absolute own step and its spread across classes (one match rule: initial state and
    born orders at six particles; the standard deviation of the class medians, over classes with two pools or more);
    beside them, as the check that a candidate does not widen the level spread elsewhere, the same for the steps from
    four to five particles of the training pools, and the spread (standard deviation) of all training tree levels at
    each multiplicity. No threshold: a candidate shrinks the steps if each of the four numbers at five to six particles
    is below the reference's.
        python3 analysis/catalog_v4/loop_levels.py --mult-candidates -> data/tree_levels_mult_cand.json"""
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from select_catalog_v4 import ADDS
    d = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    T, H = d["train"], {n: r for n, r in d["heldout"]["H2"].items() if not r.get("missing")}
    P4 = yaml.load(open(os.path.join(ROOT, "recipes", "catalog_v4_processes.yaml")),
                   Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["processes"]
    cls = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    rep_of = {m: r for r, ms in cls.items() for m in ms}
    pdg = lambda n: next((P4[m]["pdg_ids"] for m in cls.get(n, [n]) if m in P4), None)
    eik = {n: (eikonal_term(pdg(n)) if pdg(n) is not None else None) for n in list(T) + list(H)}
    shift = {"none": lambda n, r: 0.0, "soft_norm": lambda n, r: -(r["n"] - 4) * np.log(4.0),
             "eikonal": lambda n, r: (None if eik[n] is None else -(r["n"] - 4) * np.log(4.0) - eik[n])}
    key = lambda r: f"{r['initial']} {r['born']}"
    par = lambda n, k: training_parents(n, k, T, P4, cls, rep_of, ADDS)

    def steps(pools, n_child, cand):
        """per pool with n_child particles and a training parent of one fewer: (class, own step) under cand"""
        out, skipped = [], 0
        for n, r in pools.items():
            if r["n"] != n_child or r.get("heavy_layer"):
                continue
            ps = par(n, n_child - 1)
            if not ps:
                continue
            if cand == "parent_scale":
                gps = {p: par(p, n_child - 2) for p in ps}
                if not all(gps.values()):
                    skipped += 1; continue
                own = r["level"] - np.mean([T[p]["level"] for p in ps])
                pst = np.mean([T[p]["level"] - np.mean([T[g]["level"] for g in gps[p]]) for p in ps])
                out.append((key(r), float(own - pst), n)); continue
            f = shift[cand]
            fc, fp = f(n, r), [f(p, T[p]) for p in ps]
            if fc is None or any(x is None for x in fp):
                skipped += 1; continue
            out.append((key(r), float((r["level"] + fc) - np.mean([T[p]["level"] + x for p, x in zip(ps, fp)])), n))
        return out, skipped

    def summ(rows):
        by = {}
        for k, s, _ in rows:
            by.setdefault(k, []).append(s)
        cm = [float(np.median(v)) for v in by.values() if len(v) > 1]
        bi = {}
        for k, s, _ in rows:
            bi.setdefault(k.split()[0], []).append(s)
        return {"median_step_by_initial": {k: float(np.median(v)) for k, v in sorted(bi.items())},"n": len(rows), "median_abs_step": float(np.median([abs(s) for _, s, _ in rows])) if rows else None,
                "median_step": float(np.median([s for _, s, _ in rows])) if rows else None,
                "class_spread": float(np.std(cm)) if len(cm) > 1 else None, "n_classes": len(cm)}

    C = ("none", "soft_norm", "eikonal", "parent_scale")
    R, per = {}, {}
    for c in C:
        tr6, s1 = steps(T, 6, c); ho6, s2 = steps(H, 6, c); tr5, s3 = steps(T, 5, c)
        R[c] = {"train_6": summ(tr6), "heldout_6": summ(ho6), "train_5": summ(tr5),
                "skipped": {"train_6": s1, "heldout_6": s2, "train_5": s3}}
        per[c] = {"train_6": tr6, "heldout_6": ho6}
        if c != "parent_scale":   # the level spread at each multiplicity (parent_scale is an input, the level unchanged)
            lv = {}
            for n, r in T.items():
                x = shift[c](n, r)
                if x is not None and not r["heavy_layer"]:
                    lv.setdefault(r["n"], []).append(r["level"] + x)
            R[c]["level_sd"] = {str(k): float(np.std(v)) for k, v in sorted(lv.items())}
        # a common subset for the reference: the same pools as this candidate reads (eikonal and parent_scale skip some)
        if c != "none":
            for lab, rows in (("train_6", tr6), ("heldout_6", ho6)):
                keep = {n for _, _, n in rows}
                R[c][f"{lab}_ref_same_pools"] = summ([x for x in per["none"][lab] if x[2] in keep])
    # why parent_scale may read no pool: the recorded origin of the five particle training pools (a 2->3 built by
    # one addition to a 2->2 has a four particle parent; "exhaustive" ones have none)
    import collections
    R["origin_of_five_particle_training_pools"] = dict(collections.Counter(
        ("addition" if any(P4.get(m, {}).get("why", "").endswith(a) for m in cls.get(n, [n]) for a in ADDS)
         else "other") for n, r in T.items() if r["n"] == 5 and not r["heavy_layer"]))
    for c in C[1:]:
        R[c]["shrinks"] = all(
            R[c][lab][m] is not None and R[c][f"{lab}_ref_same_pools"][m] is not None
            and R[c][lab][m] < R[c][f"{lab}_ref_same_pools"][m]
            for lab in ("train_6", "heldout_6") for m in ("median_abs_step", "class_spread"))
    for c in C:
        print(f"{c:13s}", {k: v for k, v in R[c].items()})
    print("origin of the five particle training pools:", R["origin_of_five_particle_training_pools"])
    json.dump({"summary": R, "pools": per, "eikonal_term": eik},
              open(os.path.join(HERE, "data", "tree_levels_mult_cand.json"), "w"), indent=1)
    flat = {f"mc.five_origin.{k}": v for k, v in R["origin_of_five_particle_training_pools"].items()}
    for c in C:
        for lab in ("train_6", "heldout_6", "train_5"):
            for m in ("n", "median_abs_step", "class_spread", "median_step"):
                v = R[c][lab][m]
                if v is not None:
                    flat[f"mc.{c}.{lab}.{m}"] = round(v, 3) if isinstance(v, float) else v
        if c != "none":
            for lab in ("train_6", "heldout_6"):
                for m in ("median_abs_step", "class_spread"):
                    v = R[c][f"{lab}_ref_same_pools"][m]
                    if v is not None:
                        flat[f"mc.{c}.{lab}.ref.{m}"] = round(v, 3)
            flat[f"mc.{c}.shrinks"] = int(R[c]["shrinks"])
        for lab in ("train_6", "heldout_6"):
            for k, v in R[c][lab].get("median_step_by_initial", {}).items():
                flat[f"mc.{c}.{lab}.initial.{k}"] = round(v, 3)
        for k, v in R[c].get("level_sd", {}).items():
            flat[f"mc.{c}.level_sd.{k}"] = round(v, 3)
    print(json.dumps(flat))
    return R


def multiplicity_read():
    """The multiplicity axis's signed level bias (v3_prep_explore, the user's first priority D178: the zero-shot bias).
    par:v4_tree_newproc found the multiplicity pools' (H2, all 2->4) seed-mean offset positive on most of them: the
    network places their level too high. Is that an under-extrapolation of the level's fall from five to six
    particles? Reads data/tree_levels.json (analyse_trees, no new dump). Each 2->4 process of catalog_v4 is a 2->3 or
    2->2 process with particles added (recipes/catalog_v4_processes.yaml `why`, the parent rule of
    recipes/select_catalog_v4.py, mapped to class representatives); an H2 pool's family parent stays in training, its
    2->4 children do not. Per H2 pool: its own step, its level minus the mean level of its five particle training
    parents (one particle added, one coupling power); per class of the one match rule (initial state and born orders
    at six particles) the training step, the median of the same step over the training six particle pools of that
    class with a five particle training parent. No fit, no threshold: the rank correlations (Spearman, per seed and on
    the seed mean) of the signed offset with the own step, the class's training step and the departure from the
    matched class, and within class (offset and own step less their class medians); the positive share where the level falls (own step below zero) and where it rises; and two
    predictors with no free parameter set against the offset: a network that predicts the matched class mean
    (offset = -departure) and one that predicts the parent's level plus the class's training step (offset = training
    step - own step). Pools with no five particle training parent (pairs added to a 2->2) or no class step are counted
    and left out of the step reads.
        python3 analysis/catalog_v4/loop_levels.py --multiplicity -> data/tree_levels_mult.json,
                                                                     figs/tree_levels_mult.{png,pdf}"""
    from scipy.stats import spearmanr
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from select_catalog_v4 import ADDS
    d = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    T, H = d["train"], {n: r for n, r in d["heldout"]["H2"].items() if not r.get("missing")}
    P4 = yaml.load(open(os.path.join(ROOT, "recipes", "catalog_v4_processes.yaml")),
                   Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["processes"]
    cls = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    rep_of = {m: r for r, ms in cls.items() for m in ms}

    def parents5(r):
        return training_parents(r, 5, T, P4, cls, rep_of, ADDS)

    key = lambda r: (r["initial"], r["born"])
    steps = {}
    for c, r in T.items():
        ps = parents5(c) if r["n"] == 6 and not r["heavy_layer"] else []
        if ps:
            steps.setdefault(key(r), []).append(r["level"] - float(np.mean([T[p]["level"] for p in ps])))
    cstep = {k: float(np.median(v)) for k, v in steps.items()}
    rows, no_parent, no_class = [], [], []
    for n, r in H.items():
        ps = parents5(n)
        if not ps:
            no_parent.append(n); continue
        if key(r) not in cstep:
            no_class.append(n); continue
        own = r["level"] - float(np.mean([T[p]["level"] for p in ps]))
        rows.append({"name": n, "initial": r["initial"], "born": r["born"], "parents": ps, "own_step": own,
                     "class_step": cstep[key(r)], "n_class_step": len(steps[key(r)]), "departure": r.get("departure"),
                     "offset": r["offset"], "off": float(np.mean(r["offset"]))})
    off = np.array([x["off"] for x in rows]); own = np.array([x["own_step"] for x in rows])
    cs = np.array([x["class_step"] for x in rows])
    rho = lambda a, b: [float(v) for v in spearmanr(a, b)]
    S = {"n_heldout": len(H), "n_read": len(rows), "no_five_particle_parent": no_parent, "no_class_step": no_class,
         "n_train_children": int(sum(len(v) for v in steps.values())), "n_classes_with_step": len(cstep),
         "median_signed_off_all": float(np.median([np.mean(r["offset"]) for r in H.values()])),
         "positive_share_all": float(np.mean([np.mean(r["offset"]) > 0 for r in H.values()])),
         "median_signed_off": float(np.median(off)), "positive_share": float(np.mean(off > 0)),
         "median_own_step": float(np.median(own)), "median_class_step": float(np.median(cs)),
         "median_own_minus_class_step": float(np.median(own - cs)),
         "share_own_below_class_step": float(np.mean(own < cs))}
    for lab, x in (("own_step", own), ("class_step", cs), ("own_minus_class_step", own - cs)):
        S[f"rho_{lab}"] = rho(x, off)
        S[f"rho_{lab}_per_seed"] = [rho(x, [r["offset"][i] for r in rows])[0] for i in range(len(RUNS))]
    dm = [i for i, x in enumerate(rows) if x["departure"] is not None]
    S["rho_departure"] = rho([rows[i]["departure"] for i in dm], off[dm])
    S["rho_departure_per_seed"] = [rho([rows[i]["departure"] for i in dm], [rows[i]["offset"][k] for i in dm])[0]
                                   for k in range(len(RUNS))]
    falls, rises = own < 0, own >= 0
    S.update({"n_falls": int(falls.sum()), "n_rises": int(rises.sum()),
              "positive_share_falls": float(np.mean(off[falls] > 0)) if falls.any() else None,
              "positive_share_rises": float(np.mean(off[rises] > 0)) if rises.any() else None,
              "median_off_falls": float(np.median(off[falls])) if falls.any() else None,
              "median_off_rises": float(np.median(off[rises])) if rises.any() else None})
    predA = np.array([-rows[i]["departure"] for i in dm]); predB = cs - own
    S.update({"median_abs_resid_zero": float(np.median(np.abs(off))),
              "median_abs_resid_match_mean": float(np.median(np.abs(off[dm] - predA))),
              "median_abs_resid_parent_step": float(np.median(np.abs(off - predB))),
              "rho_pred_match_mean": rho(predA, off[dm]), "rho_pred_parent_step": rho(predB, off),
              "n_departure": len(dm)})
    by = {}
    for x in rows:
        by.setdefault(f"{x['initial']} {x['born']}", []).append(x)
    S["by_class"] = {k: {"n": len(v), "median_off": float(np.median([x["off"] for x in v])),
                         "positive_share": float(np.mean([x["off"] > 0 for x in v])),
                         "median_own_step": float(np.median([x["own_step"] for x in v])),
                         "class_step": v[0]["class_step"], "n_class_step": v[0]["n_class_step"]}
                     for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    # within class: each pool's offset and own step less its class's medians (classes of two pools or more, the least
    # a within-class difference needs), so the class's common level and step drop out
    wc = [(x["off"] - S["by_class"][k]["median_off"], x["own_step"] - S["by_class"][k]["median_own_step"])
          for k, v in by.items() if len(v) > 1 for x in v]
    S["n_within_class"] = len(wc)
    S["rho_within_class"] = rho([w[1] for w in wc], [w[0] for w in wc])
    byi = {}
    for x in rows:
        byi.setdefault(x["initial"], []).append(x)
    S["by_initial"] = {k: {"n": len(v), "median_off": float(np.median([x["off"] for x in v])),
                           "positive_share": float(np.mean([x["off"] > 0 for x in v])),
                           "median_own_step": float(np.median([x["own_step"] for x in v])),
                           "median_class_step": float(np.median([x["class_step"] for x in v]))}
                       for k, v in sorted(byi.items())}
    print(f"multiplicity axis: {S['n_read']} of {S['n_heldout']} read (no five particle training parent "
          f"{len(no_parent)}, no class step {len(no_class)}); training children with a parent {S['n_train_children']} in "
          f"{S['n_classes_with_step']} classes")
    for k, v in S.items():
        if k not in ("by_class", "by_initial", "no_five_particle_parent", "no_class_step"):
            print(f"  {k}: {v}")
    for k, v in S["by_class"].items():
        print(f"  class {k:10s} {v}")
    for k, v in S["by_initial"].items():
        print(f"  initial {k:3s} {v}")
    json.dump({"summary": S, "pools": rows}, open(os.path.join(HERE, "data", "tree_levels_mult.json"), "w"), indent=1)
    flat = {}
    for k, v in S.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"mb.{k}"] = round(v, 3) if isinstance(v, float) else v
        elif k.startswith("rho_") and isinstance(v, list):
            if k.endswith("per_seed"):
                flat[f"mb.{k[:-9]}.seed42"], flat[f"mb.{k[:-9]}.seed43"] = [round(x, 2) for x in v]
            else:
                flat[f"mb.{k}"], flat[f"mb.{k}.p"] = round(v[0], 2), float(f"{v[1]:.2g}")
    flat["mb.n_no_parent"], flat["mb.n_no_class"] = len(no_parent), len(no_class)
    for k, v in S["by_initial"].items():
        for kk, x in v.items():
            flat[f"mb.initial.{k}.{kk}"] = round(x, 3) if isinstance(x, float) else x
    plot_mult(rows, dm, predA, predB)
    print(json.dumps(flat))


def plot_mult(rows, dm, predA, predB):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    own = [x["own_step"] for x in rows]
    a.errorbar(own, [x["off"] for x in rows], yerr=[abs(x["offset"][0] - x["offset"][1]) / 2 for x in rows],
               fmt=".", color=ps.C.blue, alpha=0.6, label=f"own step ({len(rows)})")
    a.plot([x["class_step"] for x in rows], [x["off"] for x in rows], "x", color=ps.C.vermillion, alpha=0.5,
           label="class training step")
    a.axhline(0, color=ps.C.grey, lw=0.8)
    a.set_xlabel(r"$\overline{t}_{6}-\overline{t}_{5}$"); a.set_ylabel(r"$\overline{q-t}$")
    ps.legend(a, "upper right"); ps.make_room(a)
    off = np.array([x["off"] for x in rows])
    b.plot(predA, off[dm], ".", color=ps.C.green, alpha=0.6, label="matched class mean")
    b.plot(predB, off, ".", color=ps.C.vermillion, alpha=0.6, label="parent + training step")
    lo, hi = min(predA.min(), predB.min()), max(predA.max(), predB.max())
    b.plot([lo, hi], [lo, hi], "--", color=ps.C.grey, label="offset = prediction")
    b.set_xlabel(r"predicted $\overline{q-t}$"); b.set_ylabel(r"$\overline{q-t}$")
    ps.legend(b, "upper left"); ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "tree_levels_mult"))


LOOP_BORN_CANDIDATES = ("none", "born_rel", "born_rel_pole")


def born_pole(name):
    """|c2|, the universal 1/eps^2 coefficient of a one-loop QCD virtual normalised to its Born (alpha_s/2pi units):
    sum of C_i over the massless coloured legs (tools/nlo_pole_check.double_pole, the generator's own pole check), a
    property of the leg set alone; None when it is 0 (no massless coloured leg)."""
    import mg5_pipeline_final as mg
    import nlo_pole_check as PC
    c = mg.PROCESSES[name]
    bm = c.get("beam_masses") or {}
    if c.get("beam_masses") is None:
        bm = mg.BEAM_MASS
    m = [bm.get(abs(int(p)), 0.0) for p in c["pdg_ids"][:2]] + list(c["m_finals"])
    v = abs(PC.double_pole(c["pdg_ids"], m))
    return v if v > 0 else None


def loop_born_read():
    """A target change for the one-loop axis (v3_prep_explore, the user's first priority D178, the zero-shot bias;
    par:v4_loop_groups: the one-loop axis's error has a source of its own, its level placed low, most in thin classes).
    The stored one-loop QCD virtual is the alpha_s-stripped (c0 + shift) x Born (tools/nlo_virtual_pipeline.py), and the
    target t = log(|M|^2 F) of a virtual and of its Born tree carry the same F (the same legs, the Born's coupling
    powers, mg5_pipeline_final.order_vector), so per event t_virt - t_Born = log|c0 + shift|: the Born's level drops
    out. The Born is not stored per event, so on the existing reads the candidate is read at the pool level: each
    virtual pool's level (data/loop_levels_cc.json) minus its Born tree pool's level (the entry's virt_base, read in
    data/tree_levels_cc.json); a proxy, since the tree pools are sampled with the mixture and the one-loop pools never
    shaped (mg5_pipeline_final, "One-loop (MadLoop) entries are never shaped"), so the two means are over different
    measures. Candidates, fixed before their numbers were read, each from the target's definition, no fitted scale:
      none           the pretraining's target, the reference;
      born_rel       the virtual's level relative to its Born's, L - L_Born (the log of the finite coefficient);
      born_rel_pole  that coefficient in units of its own universal double pole, L - L_Born - log|c2| (born_pole).
    Pools: the virtual (not loop-induced) training and held-out one-loop pools whose Born tree pool is read; the
    loop-induced ones have no Born, the pure-QCD-born ones (virt_base *_qcdborn) no read Born, both listed.
    Measures, per candidate, on the same pools: (1) the held-out pools' departure from the mean level of their
    matched training pools (match_key, the one match rule, among the read ones), median |departure| and the count of
    pools narrowed against none, a one-sided sign test (p < 0.05, the entry's convention); (2) the within-class spread
    of the training levels, per class of two or more read training pools the standard deviation, median over classes
    and the count narrowed, the same test; beside them the overall training spread, so a candidate that only shrinks
    every level is seen. And the read the main session folded in: do the held-out virtual pools' signed offsets follow
    the step from their Born tree (L - L_Born), Spearman per seed, as par:v4_mult_bias read the five-to-six step.
        python3 analysis/catalog_v4/loop_levels.py --loop-born -> data/loop_born.json, figs/loop_born.{png,pdf}"""
    import mg5_pipeline_final as mg
    from scipy.stats import spearmanr, binomtest
    cc = json.load(open(os.path.join(HERE, "data", "loop_levels_cc.json")))
    tc = json.load(open(os.path.join(HERE, "data", "tree_levels_cc.json")))
    kl = json.load(open(os.path.join(HERE, "data", "kshot_level.json")))["probes"]
    tr, ho = entries()
    trn, hon = [p["name"] for p in tr if p["name"] in cc], [p["name"] for p in ho if p["name"] in cc]
    excl, rows = {"loop_induced": [], "no_born_read": []}, {}
    for n in trn + hon:
        c = mg.PROCESSES[n]
        if c.get("loopind"):
            excl["loop_induced"].append(n); continue
        b = c["virt_base"]
        if b not in tc:
            excl["no_born_read"].append(n); continue
        pole = born_pole(n)
        L, B = cc[n][0], tc[b][0]
        rows[n] = {**props(n), "heldout": n in hon, "born_pool": b, "level": L, "born_level": B, "step": L - B,
                   "pole": pole, "transform": cc[n][2], "born_transform": tc[b][2],
                   "cand": {"none": L, "born_rel": L - B, "born_rel_pole": (L - B - np.log(pole)) if pole else None}}
    nopole = [n for n, r in rows.items() if r["pole"] is None]
    R = {n: r for n, r in rows.items() if r["pole"] is not None}
    T = {n: r for n, r in R.items() if not r["heldout"]}
    H = {n: r for n, r in R.items() if r["heldout"]}
    P = {n: r for n, r in R.items()}
    cls = {}
    for m in T:
        cls.setdefault(match_key(P, m), []).append(m)
    out = {"n_train": len(T), "n_heldout": len(H), "excluded": excl, "no_pole": nopole, "cand": {}}
    flat = {"lb.n_train": len(T), "lb.n_heldout": len(H), "lb.n_loop_induced": len(excl["loop_induced"]),
            "lb.n_no_born_read": len(excl["no_born_read"]), "lb.n_no_pole": len(nopole)}
    hm = [n for n in H if cls.get(match_key(P, n))]
    flat["lb.n_signed_train"] = int(sum(T[m]["transform"] != "log" for m in T))
    flat["lb.n_signed_heldout"] = int(sum(H[n]["transform"] != "log" for n in H))
    flat["lb.n_heldout_matched"] = len(hm)
    multi = [k for k, v in cls.items() if len(v) >= 2]
    flat["lb.n_classes_multi"] = len(multi)
    dep, sd = {}, {}
    for c in LOOP_BORN_CANDIDATES:
        dep[c] = {n: H[n]["cand"][c] - float(np.mean([T[m]["cand"][c] for m in cls[match_key(P, n)]])) for n in hm}
        sd[c] = {k: float(np.std([T[m]["cand"][c] for m in cls[k]])) for k in multi}
        g = float(np.std([T[m]["cand"][c] for m in T]))
        o = {"median_abs_departure": float(np.median(np.abs(list(dep[c].values())))),
             "median_class_sd": float(np.median(list(sd[c].values()))), "train_sd": g}
        # the same in units of the candidate's own overall training spread (a candidate that only shrinks every level
        # leaves these unchanged)
        o["departure_in_train_sd"] = o["median_abs_departure"] / g
        o["class_sd_in_train_sd"] = o["median_class_sd"] / g
        if c != "none":
            nd = sum(abs(dep[c][n]) < abs(dep["none"][n]) for n in hm)
            ns = sum(sd[c][k] < sd["none"][k] for k in multi)
            o.update({"n_dep_narrowed": int(nd), "p_dep": float(binomtest(nd, len(hm), 0.5, "greater").pvalue),
                      "n_sd_narrowed": int(ns), "p_sd": float(binomtest(ns, len(multi), 0.5, "greater").pvalue)})
            o["works"] = bool(o["p_dep"] < 0.05 and o["p_sd"] < 0.05)
        out["cand"][c] = o
        for k, v in o.items():
            flat[f"lb.{c}.{k}"] = float(f"{v:.3g}") if isinstance(v, float) else v
        print(f"{c:14s} " + "  ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in o.items()))
    # the folded-in read: the held-out virtual pools' signed offset against their step from the Born tree
    hs = [n for n in H]
    st = np.array([H[n]["step"] for n in hs])
    for i, run in enumerate(RUNS):
        off = np.array([kl[run]["H5"][n]["offset"] for n in hs])
        rho, p = spearmanr(st, off)
        out[f"rho_offset_step.seed{i}"] = [float(rho), float(p)]
        flat[f"lb.rho_offset_step.seed{42 + i}"] = round(float(rho), 2)
        flat[f"lb.p_offset_step.seed{42 + i}"] = round(float(p), 3)
    om = np.array([np.mean([kl[r]["H5"][n]["offset"] for r in RUNS]) for n in hs])
    rho, p = spearmanr(st, om)
    flat["lb.rho_offset_step.mean"], flat["lb.p_offset_step.mean"] = round(float(rho), 2), round(float(p), 3)
    # and against the Born-relative departure (does the network place the matched class's Born-relative mean)
    omm = np.array([np.mean([kl[r]["H5"][n]["offset"] for r in RUNS]) for n in hm])
    for c in ("none", "born_rel"):
        rho, p = spearmanr([dep[c][n] for n in hm], omm)
        flat[f"lb.rho_offset_departure.{c}"] = round(float(rho), 2)
    print(f"held-out signed offset vs step from the Born tree: seeds {out['rho_offset_step.seed0']}, "
          f"{out['rho_offset_step.seed1']}; seed mean rho {flat['lb.rho_offset_step.mean']}")
    for n in sorted(hs, key=lambda n: H[n]["step"]):
        r = H[n]
        d = {c: dep[c].get(n) for c in LOOP_BORN_CANDIDATES}
        ds = "  ".join(f"{c} {v:+6.2f}" if v is not None else f"{c}   --  " for c, v in d.items())
        print(f"  {n:26s} {r['initial']:3s} N={r['n']} step {r['step']:+6.2f} offset {om[hs.index(n)]:+5.2f}  "
              f"departure {ds}")
    out["rows"] = rows
    out["departure"], out["class_sd"] = dep, {c: {str(k): v for k, v in s.items()} for c, s in sd.items()}
    json.dump(out, open(os.path.join(HERE, "data", "loop_born.json"), "w"), indent=1)
    plot_loop_born(H, T, dep, hm, hs, st, om)
    print(json.dumps(flat))


def plot_loop_born(H, T, dep, hm, hs, st, om):
    """Fig. v4_loop_born: left, per matched held-out virtual pool its |departure| from the matched training mean,
    the pretraining's target against the level relative to the Born tree, with the diagonal; right, per held-out
    virtual pool its seed-mean signed offset against its step from the Born tree."""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    a = axs[0]
    x = np.abs([dep["none"][n] for n in hm]); y = np.abs([dep["born_rel"][n] for n in hm])
    a.plot(x, y, ".", color=ps.C.blue, label="Born removed")
    y2 = np.abs([dep["born_rel_pole"][n] for n in hm])
    a.plot(x, y2, "x", color=ps.C.vermillion, ms=4, label="Born and double pole removed")
    lo, hi = 0.8 * min(x.min(), y.min(), y2.min()), 1.25 * max(x.max(), y.max(), y2.max())
    a.plot([lo, hi], [lo, hi], "-", color=ps.C.grey, lw=0.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$|\Delta\overline{t}|$, target as trained"); a.set_ylabel(r"$|\Delta\overline{t}|$, candidate")
    a.legend(fontsize=7)
    b = axs[1]
    b.plot(st, om, ".", color=ps.C.blue)
    b.axhline(0, color=ps.C.grey, lw=0.8)
    b.set_xlabel(r"$\overline{t}-\overline{t}_{\mathrm{Born}}$"); b.set_ylabel(r"$\overline{q-t}$")
    ps.save(fig, os.path.join(HERE, "figs", "loop_born"))


def shape_gap():
    """The zero-shot shape gap (v3_prep_explore, next step after par:v4_tree_shape): is a held-out tree pool's floor
    var(q - t) above the in-distribution floor of the training pools in the same cell of multiplicity and initial
    state by more than the pools' intrinsic difficulty explains. Three reads per cell, each optional but the first:
      net_free  the training pools' result.json per-process validation loss at the best checkpoint times the shared
                standardization's sigma^2 (analysis/catalog_v3/data/train_levels.json, the conversion of
                par:v3_factorised: 0.50 against the scorer's 0.51 on 40 catalog_v3 parents); 200 validation events
                per pool, the MSE with the offset in it, so an upper bound on the training floor
      net_tr    the same scorer as the held-out reads on the training pools' test split (short_reads/<run>/TR, mode
                eval V3_AXES=TR), floor = var(q - t)
      proxy     the shape proxy (analysis/catalog_v3/prep_shape.py --catalog v4, data/prep_shape_v4.json, target
                qperp = the control's target): one small regressor per pool, its held-out MSE
    Per cell: the seed-mean medians over training pools and over the held-out pools of H1, H2, H3, and the log10
    ratio held out over training; the network's excess is its log10 ratio minus the proxy's. The seed spread: the
    per-pool |log10(floor seed 42 / floor seed 43)| of the held-out reads (median), and of the cell medians.
        python3 analysis/catalog_v4/loop_levels.py --shape-gap -> data/shape_gap.json; last line the flat JSON"""
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    tv = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "train_levels.json")))
    T = tl["train"]
    cell = lambda r: f"{r['n']}{r['initial']}"
    free = {m: float(np.mean([tv[r]["mse"][m] for r in RUNS])) for m in T if all(m in tv[r]["mse"] for r in RUNS)}
    free_s = {m: [tv[r]["mse"][m] for r in RUNS] for m in free}
    tr_dir = [os.path.join(SR, r, "TR") for r in RUNS]
    net_tr = {}
    if all(os.path.isdir(d) for d in tr_dir):
        for m in T:
            fl = []
            for d in tr_dir:
                f = os.path.join(d, m + ".npz")
                if not os.path.exists(f):
                    break
                z = np.load(f)
                fl.append(float((z["q"].astype(float) - z["t"].astype(float)).var()))
            if len(fl) == len(RUNS):
                net_tr[m] = fl
    pf = os.path.join(HERE, "data", "prep_shape_v4.json")
    prox = json.load(open(pf)) if os.path.exists(pf) else {}
    pmse = {n: r["qperp"]["mse"] for s_, d in prox.items() if s_ != "skipped" for n, r in d.items()}
    H = {n: r for a in TREE_AXES for n, r in tl["heldout"][a].items() if not r.get("missing")}
    hax = {n: a for a in TREE_AXES for n in tl["heldout"][a]}
    med = lambda v: float(np.median(v)) if len(v) else None
    l10 = lambda a, b: float(np.log10(a / b)) if a and b else None
    cells, flat = {}, {}
    for c in sorted({cell(r) for r in H.values()}):
        tr = [m for m in T if cell(T[m]) == c]
        ho = [n for n in H if cell(H[n]) == c]
        C = {"n_train": len(tr), "n_heldout": len(ho), "n_heldout_by_axis": {a: sum(hax[n] == a for n in ho) for a in TREE_AXES},
             "heldout_floor": med([H[n]["shape"] for n in ho]),
             "heldout_floor_per_seed": [med([H[n]["floor"][i] for n in ho]) for i in range(len(RUNS))],
             "train_free": med([free[m] for m in tr if m in free]),
             "train_free_per_seed": [med([free_s[m][i] for m in tr if m in free]) for i in range(len(RUNS))]}
        C["log_ratio_free"] = l10(C["heldout_floor"], C["train_free"])
        C["log_ratio_free_per_seed"] = [l10(a, b) for a, b in zip(C["heldout_floor_per_seed"], C["train_free_per_seed"])]
        if net_tr:
            C["train_tr"] = med([float(np.mean(net_tr[m])) for m in tr if m in net_tr])
            C["train_tr_per_seed"] = [med([net_tr[m][i] for m in tr if m in net_tr]) for i in range(len(RUNS))]
            C["log_ratio_tr"] = l10(C["heldout_floor"], C["train_tr"])
        if pmse:
            C["n_proxy"] = [sum(m in pmse for m in tr), sum(n in pmse for n in ho)]
            C["proxy_train"] = med([pmse[m] for m in tr if m in pmse])
            C["proxy_heldout"] = med([pmse[n] for n in ho if n in pmse])
            C["log_ratio_proxy"] = l10(C["proxy_heldout"], C["proxy_train"])
            for k in ("free", "tr"):
                if C.get(f"log_ratio_{k}") is not None and C["log_ratio_proxy"] is not None:
                    C[f"excess_{k}"] = C[f"log_ratio_{k}"] - C["log_ratio_proxy"]
        cells[c] = C
    seed_pool = [abs(np.log10(H[n]["floor"][0] / H[n]["floor"][1])) for n in H]
    S = {"n_train": len(T), "n_train_free": len(free), "n_train_tr": len(net_tr), "n_proxy": len(pmse),
         "n_heldout": len(H), "median_seed_spread_pool": med(seed_pool),
         "median_seed_spread_cell_ratio": med([abs(v["log_ratio_free_per_seed"][0] - v["log_ratio_free_per_seed"][1])
                                                for v in cells.values() if v["n_train"] and None not in v["log_ratio_free_per_seed"]]),
         "train_free_all": med(list(free.values())), "heldout_floor_all": med([H[n]["shape"] for n in H])}
    used = {c: v for c, v in cells.items() if v["n_train"] and v["log_ratio_free"] is not None}
    S["n_cells"] = len(used)
    S["n_heldout_in_cells"] = int(sum(v["n_heldout"] for v in used.values()))
    S["median_log_ratio_free"] = med([v["log_ratio_free"] for v in used.values()])
    S["n_cells_free_above"] = int(sum(v["log_ratio_free"] > 0 for v in used.values()))
    # pool-weighted: each held-out pool against its cell's training median (seed mean)
    pl = [np.log10(H[n]["shape"] / used[cell(H[n])]["train_free"]) for n in H if cell(H[n]) in used]
    S["median_pool_log_ratio_free"] = med(pl)
    print(f"shape gap: {len(T)} training tree pools ({len(free)} in result.json, {len(net_tr)} TR reads, {len(pmse)} "
          f"proxy), {len(H)} held out; all-pool medians: training result.json MSE {S['train_free_all']:.3f}, held-out floor "
          f"{S['heldout_floor_all']:.3f}; per-pool seed spread |log10| {S['median_seed_spread_pool']:.3f}")
    print(f"{'cell':6s} {'n_tr':>4s} {'n_ho':>4s} {'tr_free':>8s} {'ho_floor':>8s} {'log10':>6s} {'seeds':>13s}"
          + ("  prox_tr prox_ho log10p excess" if pmse else ""))
    for c, v in cells.items():
        f3 = lambda x: f"{x:8.3f}" if x is not None else "       -"
        ps_ = "/".join(f"{x:+.2f}" if x is not None else "-" for x in v["log_ratio_free_per_seed"])
        line = (f"{c:6s} {v['n_train']:4d} {v['n_heldout']:4d} {f3(v['train_free'])} {f3(v['heldout_floor'])} "
                f"{(v['log_ratio_free'] if v['log_ratio_free'] is not None else float('nan')):+6.2f} {ps_:>13s}")
        if pmse:
            line += f" {f3(v.get('proxy_train'))} {f3(v.get('proxy_heldout'))} {v.get('log_ratio_proxy') or float('nan'):+6.2f} {v.get('excess_free', float('nan')):+6.2f}"
        print(line)
        for k in ("n_train", "n_heldout"):
            flat[f"sg.{c}.{k}"] = v[k]
        for k in ("train_free", "heldout_floor", "log_ratio_free", "proxy_train", "proxy_heldout", "log_ratio_proxy",
                  "excess_free", "train_tr", "log_ratio_tr", "excess_tr"):
            if v.get(k) is not None:
                flat[f"sg.{c}.{k}"] = round(v[k], 3)
    for k, x in S.items():
        flat[f"sg.{k}"] = round(x, 3) if isinstance(x, float) else x
    json.dump({"summary": S, "cells": cells}, open(os.path.join(HERE, "data", "shape_gap.json"), "w"), indent=1)
    print(json.dumps(flat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--trees", action="store_true", help="the tree held-out axes H1, H2, H3")
    ap.add_argument("--multiplicity", action="store_true", help="the multiplicity axis's signed bias (multiplicity_read)")
    ap.add_argument("--mult-candidates", action="store_true",
                    help="candidate target or input changes for the multiplicity step (multiplicity_candidates)")
    ap.add_argument("--loop-groups", action="store_true", help="the one-loop axis by the tree reads (loop_group_read)")
    ap.add_argument("--loop-born", action="store_true",
                    help="the one-loop target relative to its Born tree (loop_born_read)")
    ap.add_argument("--shape-gap", action="store_true", help="the held-out floor against the training pools' (shape_gap)")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    A = ap.parse_args()
    if A.dump:
        dump(A.trees, A.chunk, A.nchunks)
    elif A.mult_candidates:
        multiplicity_candidates()
    elif A.multiplicity:
        multiplicity_read()
    elif A.loop_born:
        loop_born_read()
    elif A.shape_gap:
        shape_gap()
    elif A.loop_groups:
        loop_group_read()
    else:
        (analyse_trees if A.trees else analyse)()
