#!/usr/bin/env python3
"""The flat part's candidates, current against multichannel, on five catalog_v4 classes (plan catalog_coverage step 57,
D288 "build it and run the check"; no pool is built or stored).

Per class, train role, seeds 42 and 43, each policy: the train pool's chunks are generated as the pool build does
(datagen.chunk_tasks + datagen.gen_chunk, the class's catalog_v4 entry without the default share,
mix_compose.control_config), in a temporary folder, and the flat-in-log|M|^2 part's selection is recorded
(mg._flat_logm_select: every candidate's log|M|^2 and which were kept). The policies:
  current      the entry's mixture as built (candidates half splitter, half RAMBO)
  multichannel the same with sampling.multichannel: the candidates' finals from RAMBO, the splitter, one Breit-Wigner
               channel per s-channel Z/W/H/t propagator of the class's diagrams (mg.resonance_channels on
               data/diagrams/<class>.diagrams.json) and the initial-state collinear channel, equal shares in the first
               pass, shares adapted by the 1/density(log|M|^2) weight in the second (mg5_pipeline_final.py)
  quota        the multichannel candidates with the flat part filled to a quota per log|M|^2 bin over the class's pp
               range (sampling.flat_range from recipes/catalog_v4_flat_range.json, tools/flat_range.py; plan step 58,
               D294): rounds of candidates steered to the bins short of their quota until every bin holds it or
               sampling.flat_budget (mg.FLAT_BUDGET) candidates per flat event were labelled; per chunk the quota, the
               kept count per bin, the rounds, the candidates and whether it stopped at the budget with bins short
               are recorded (mg.FLAT_LAST); the range is the in-support pp events' (tools/flat_range.py)
  quota_ladder_res2  quota with two more channels (sampling.mc_mperi, mc_res2; plan step 58's re-check): the
               multiperipheral ladder (every final at pT near the cut, rapidities spread to the acceptance edge) and
               one channel per pair of resonances on disjoint finals, on shell together
  quota_ladder_res2_budget500  the same at sampling.flat_budget 500 (10x; step 65's re-check)
  quota_ladder_both_res2[_budget750]  both ladders as two channels (sampling.mc_mperi_both; plan step 67), at budget
               50 and 750
  quota_ladder_edge_res2  quota_ladder_res2 with the ladder's edge variant (sampling.mc_mperi_edge; plan step 66):
               each event's pT span log-uniform in [1.05, 3.2], the outer end at 5 - d, d log-uniform in [1e-3, 1]
  grid_both_res2[f]_budget12|8  the quota sampler with each channel's uniform numbers through an adaptive grid
               (sampling.flat_grid; plan step 72) at 12 and 8 labels per flat event; with res2f also the joint-resonance
               channel's forward variant (sampling.mc_res2_fwd: the pair along the beam; plan step 75)
Every policy's chunks are timed (wall seconds of datagen.gen_chunk, the whole chunk: bulk, forward and flat parts).
Everything else (6x oversampling, the flat selection, the shares of bulk, forward and flat, the sqrt(s) passes) is the
entry's. Against the class's 13 TeV pp MadEvent events (its PP4 probe's train and test pools, as pp_support.py reads
them, the same files; log10|M|^2 of the label), per seed and policy:
  above_max    the share of the pp events in the catalog's support (sqrt(s_hat) in the class's window and the partonic-
               frame cuts passed, mg.fiducial_pass_mask: pp_support.py's support) whose log10|M|^2 is above the largest
               kept flat event's: the share the sampler can move, judged; above_max_all over every pp event and
               out_of_support (the share outside the support, a floor no sampler reaches) reported beside it
  bins         the kept flat events per 0.25-dex bin of log10|M|^2, over the bins wholly between the kept part's lowest
               value and the pp events' 99.9% quantile; plateau = their median; min_over_plateau = the least of them
               over the plateau
  pass         above_max <= 0.001 and min_over_plateau >= 0.5 (the rule of the proposal, plan step 57), on both seeds
               (the step's "against the two seeds' spread" read as: the rule holds on each seed)
  sources      per run, each pp source (pp_sources): `table` the probe's train + test pools, from which the quota
               policy's range is read, so its above/below shares hold by construction for that policy (flagged
               by_construction); `heldout` the probe's val pool and the class's dsh_ pools (default-share MadEvent
               events), never read for the range: the in-support events outside the kept [min, max] (D269's count,
               target 0) with 3/N, the 95% upper bound on the share when the count is 0; beside them, over every event,
               the out-of-support share split into sqrt(s_hat) outside the window and cuts failed in the partonic frame,
               and the above-max share
  quota_outcome  step 58's rule per class: held-out count 0, no quota shortfall, no budget or no_steer stop, both seeds
Resonance masses and widths: mg.mc_res_params of the entry's physics block.

    python3 analysis/catalog_v4/multichannel_check.py run <class> <probe> [<policy> <seed>]   CPU job
               (scripts/job_multichannel_check.sh, one task per class, policy and seed): -> $SCRATCH/zz_mcheck/
               <class>_<policy>_<seed>.json
    python3 analysis/catalog_v4/multichannel_check.py pull                 local: $SCRATCH/zz_mcheck/<class>.json of each
               class from CC (site run cat) -> data/multichannel_check_reads/
    python3 analysis/catalog_v4/multichannel_check.py run <class> <probe> multichannel+quota <seed>   CPU job, both
               policies in one task on one node (their chunk times compare) -> <class>_multichannel+quota_<seed>.json
    python3 analysis/catalog_v4/multichannel_check.py pull multichannel+quota   local: those per-seed files
    python3 analysis/catalog_v4/multichannel_check.py summary              local: data/multichannel_check_reads/*.json -> data/multichannel_check.json
"""
import glob, json, os, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "multichannel_check_reads")
LOGM_BINS = np.linspace(-20.0, 8.0, 113)          # log10 |M|^2 in 0.25 dex
SEEDS = (42, 43)
# the flat part filled to a quota per bin; the second with the multiperipheral and joint-resonance channels (step 58
# re-check), the third the same at a 10x candidate budget (step 65)
QUOTA_POLICIES = ("quota", "quota_ladder_res2", "quota_ladder_res2_budget500", "quota_ladder_edge_res2",
                  "quota_ladder_both_res2", "quota_ladder_both_res2_budget750", "quota_ladder_both_res2_fwdfree_budget750",
                  "mcmc_both_res2_budget12", "quota_ladder_both_res2_budget12", "grid_both_res2_budget12",
                  "grid_both_res2_budget8", "grid_both_res2f_budget12", "grid_both_res2f_budget8", "grid_r1_budget12")
# the flat-part diversity read (flat_diversity.py) of each quota policy against this one at the same seed in the same job:
# independent draws (the quota sampler with both ladders at budget 50), plan catalog_coverage step 69
DIVERSITY_REF = "quota_ladder_both_res2"
ROLE = "train"
# the step-57 classes (the pp_support worst list, 2026-11-04, and the shape study's class) and their PP4 probes
CLASSES = {"v4_uux_epemg": "pp13_uux_epemg", "v4_uux_epemvevex": "pp13_uux_epemvevex",
           "v4_uu_bbxuu": "pp13v4_uu_bbxuu", "v4_us_ddxsu": "pp13v4_us_ddxsu",
           "v4_udx_csxvevex": "pp13_udx_csxvevex"}
# the e+e- classes (plan step (f), D294's rule on every beam): the four trained classes with the largest share of their
# default-sampled events above the catalog pool's largest log|M|^2 (data/ee_support_reads.json, logm_above), as the five
# above were the pp_support worst list: the three 4-final ones of distinct final families (tau pair + neutrinos 0.27;
# b pair + neutrinos 0.26, the d and c pairs 0.26 being its flavour copies; photon + Higgs + neutrinos 0.22) and the
# 3-final one with the largest share (nu_mu pair + Z, 0.10)
EE_CLASSES = {"v4_epem_taptamvevex": "ee4_epem_taptamvevex", "v4_epem_bbxvevex": "ee4_epem_bbxvevex",
              "v4_epem_ahvevex": "ee4_epem_ahvevex", "v4_epem_vmvmxz": "ee4_epem_vmvmxz"}
RECIPES = ["catalog_v4_base_train.yaml"] + [f"catalog_v4_heldout_{h}.yaml"
                                            for h in ("H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")]


def _entry(name):
    import yaml
    import mg5_pipeline_final as mg
    for r in RECIPES:
        spec = yaml.safe_load(open(os.path.join(ROOT, "recipes", r)))
        ps = {p["name"]: p for p in spec["processes"]}
        if name in ps:
            mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]),
                                           "physics": p.get("physics"), "sampling": p.get("sampling")}
                                          for p in spec["processes"]], default_sampling=spec.get("sampling"))
            return ps[name]
    raise SystemExit(f"{name}: in none of {RECIPES}")


def _find(stem):
    """pp_support.find: the amp_data_cache copy first, then DATA_DIR"""
    for d in (os.path.join(os.environ.get("SCRATCH", "/nonexistent"), "amp_data_cache"), os.environ["DATA_DIR"]):
        g = sorted(glob.glob(os.path.join(d, stem)))
        if g:
            return g[0]


def _pp_rows(stems):
    """the rows of the pools named by `stems` that exist (each through _find), and their file names"""
    rows, files = [], []
    for stem in stems:
        f = _find(stem)
        if f is not None:
            rows.append(np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)); files.append(os.path.basename(f))
    return rows, files


def _pp(stems, cfg, lo, hi, label):
    """log10|M|^2 of the events of the pools `stems` (those that exist), whether each lies in the catalog's support
    (sqrt(s_hat) in [lo, hi] and the class's cuts passed in the partonic frame, pp_support.py's mask), and the two
    parts of the out-of-support share: sqrt(s_hat) outside the window, and inside it but failing the cuts in the
    partonic frame (the lab card's |eta| < 5 does not bound the partonic |eta|). None when no pool exists."""
    import mg5_pipeline_final as mg
    rows, files = _pp_rows(stems)
    if not rows:
        print(f"MCHECK pp {label}: none of {stems}", flush=True)
        return None
    x = np.concatenate(rows)
    k = (x.shape[1] - 1) // 5
    assert x.shape[1] == 5 * k + 1, (label, x.shape)
    P = x[:, :4 * k].reshape(-1, k, 4)
    pdg = x[0, 4 * k:5 * k].astype(int)
    mass = dict(zip((abs(int(q)) for q in cfg["pdg_ids"][2:]), (float(m) for m in cfg["m_finals"])))
    mf = np.array([mass[abs(int(q))] for q in pdg[2:]])
    q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))
    ok = mg.fiducial_pass_mask(mg.to_partonic_frame(P), mf, pdg, cfg["fiducial_cuts"])
    inw = (sq >= lo) & (sq <= hi)
    with np.errstate(divide="ignore"):
        lm = np.log10(np.abs(x[:, -1]))
    fin = np.isfinite(lm)
    print(f"MCHECK pp {label}: {len(lm)} events ({files}), {int((~fin).sum())} with |M|^2 0, "
          f"{float(np.mean(~(ok & inw))):.4f} outside the support", flush=True)
    return {"lm": lm[fin], "sup": (inw & ok)[fin], "files": files, "outside_window": float(np.mean(~inw[fin])),
            "fail_cuts_in_window": float(np.mean((inw & ~ok)[fin]))}


def pp_sources(name, probe):
    """the pp pools of a class, by role in the check: `table` the PP4 probe's train and test pools (what
    tools/flat_range.py's range was read from: shares judged on them hold by construction for the quota policy);
    `heldout` pp events the range never saw: the probe's val pool and the class's default-share MadEvent pools
    (dsh_<x>, train, val and test; catalog_v4's physics, where a reused pp13_ probe carries catalog_v3's)"""
    w = "13000-13000GeV"
    dsh = "dsh_" + name[len("v4_"):]
    if probe.startswith("ee4_"):
        # an e+e- class: its range was read from its default-share pool (pp_support.py pull ee), its EE4 test pool
        # (2000 MadEvent events at the fixed collider energies) is the held-out source
        return {"table": [f"{dsh}_{w}_train_lhe_amplitudes.npy"], "heldout": [f"{probe}_{w}_test_lhe_amplitudes.npy"]}
    # heldout_r1: what step 75's revision is scored on (its range also reads the dsh_ train pool,
    # tools/flat_range.ranges_dsh): the probe's val pool and the dsh_ val and test pools
    return {"table": [f"{probe}_{w}_{r}_lhe_amplitudes.npy" for r in ("train", "test")],
            "heldout": [f"{probe}_{w}_val_lhe_amplitudes.npy"] + [f"{dsh}_{w}_{r}_lhe_amplitudes.npy"
                                                                  for r in ("train", "val", "test")],
            "heldout_r1": [f"{probe}_{w}_val_lhe_amplitudes.npy"] + [f"{dsh}_{w}_{r}_lhe_amplitudes.npy"
                                                                     for r in ("val", "test")]}


def run(name, probe, policies=("current", "multichannel"), seeds=SEEDS):
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flat_diversity as fdv
    p = _entry(name)
    lo, hi = map(float, p["sqrts"]); n = int(p["n_" + ROLE])
    cfg0 = mc.control_config(name)
    pdg = cfg0["pdg_ids"]
    side = os.path.join(ROOT, "data", "diagrams", f"{p.get('base', name)}.diagrams.json")
    res = mg.resonance_channels(json.load(open(side)), pdg)
    rp = mg.mc_res_params(p.get("physics"))
    out = {"class": name, "probe": probe, "window": [lo, hi], "n": n, "resonances": [[list(S), q] for S, q in res],
           "res_params": {str(k): list(v) for k, v in rp.items()}, "logm_bins": LOGM_BINS.tolist(), "runs": {}}
    fs = mg._flat_logm_select
    sel = []
    def fs_rec(logm, n_keep, rng, n_bins=60):
        keep = fs(logm, n_keep, rng, n_bins=n_bins)
        sel.append((np.asarray(logm, float) / np.log(10.0), np.asarray(keep)))
        return keep
    datagen_ok = False
    flat_kept = {}            # policy_seed -> (features, shape vars, bin, chain, edges log10) of the kept flat events
    for policy in policies:
        cfg = dict(cfg0); s = dict(cfg.get("sampling") or {})
        if policy in ("multichannel",) + QUOTA_POLICIES:
            s.update(multichannel=True, mc_resonances=[[list(S), q] for S, q in res], mc_isc=True, mc_res_params=rp)
        if policy in QUOTA_POLICIES:
            from tools.flat_range import ranges, ranges_dsh
            s.update(flat_range=(ranges_dsh() if policy == "grid_r1_budget12" else ranges())[name])
        if policy == "grid_r1_budget12":
            # plan step 75's revision of step 65's sampler (tools/regen_recipes.R1): step 65's switches plus the stall
            # stop, the ladder solves at 1e-12, ladders on every hadronic class, the resonance ladder; the range joined
            # with the dsh_ train pool (tools/flat_range.ranges_dsh)
            from tools.regen_recipes import R1
            s.update(R1)
        if policy.startswith("quota_ladder_res2"):     # step 58's re-check: + the multiperipheral and joint-resonance channels
            s.update(mc_mperi=True, mc_res2=True)
        if policy == "quota_ladder_edge_res2":     # step 66: the same channels with the ladder's edge variant
            s.update(mc_mperi=True, mc_mperi_edge=True, mc_res2=True)
        if policy in ("mcmc_both_res2_budget12", "quota_ladder_both_res2_budget12"):
            # plan catalog_coverage step 69 (cost bound <= 2x: the step-69 design; D317 refused the
            # quota sampler's ~100x): at most 12 labels per flat event, 0.65 + 0.35 x 12 = 4.85 labels per pool event
            # against the old 0.65 + 0.35 x 6 = 2.75 (1.76x); the flat part from Wang-Landau Markov chains
            # (sampling.flat_mcmc), its first round from both ladders' channels, or the quota sampler at the same budget
            s.update(mc_mperi=True, mc_mperi_both=True, mc_res2=True, flat_budget=12)
            if policy.startswith("mcmc"):
                s.update(flat_mcmc=True)
        if policy in ("grid_both_res2_budget12", "grid_both_res2_budget8"):
            # plan catalog_coverage step 72 (after step 69's NO; cost bound <= 2x: the step-69 design): the quota sampler
            # with both ladders and joint resonances, each channel's uniform numbers through an adaptive grid
            # (sampling.flat_grid), at the step-69 budget of 12 labels per flat event and at 8 (cost margin)
            s.update(mc_mperi=True, mc_mperi_both=True, mc_res2=True, flat_grid=True,
                     flat_budget=12 if policy.endswith("12") else 8)
        if policy in ("grid_both_res2f_budget12", "grid_both_res2f_budget8"):
            # plan catalog_coverage step 72's PARTIAL branch (one revision): the step-72 grids plus the joint-resonance
            # channel's forward variant (sampling.mc_res2_fwd: the boson pair along the beam, where the electroweak
            # classes' short top bins sit, data/quota_where/v4_*_grid72.json)
            s.update(mc_mperi=True, mc_mperi_both=True, mc_res2=True, mc_res2_fwd=True, flat_grid=True,
                     flat_budget=12 if policy.endswith("12") else 8)
        if policy.startswith("quota_ladder_both_res2"):   # step 67: both ladders as two channels
            s.update(mc_mperi=True, mc_mperi_both=True, mc_res2=True)
        if policy == "quota_ladder_both_res2_fwdfree_budget750":
            # step (f)'s SHORT branch on e+e-: the default events above the fill have a neutrino, Z or H at 1 - |cos|
            # 1e-3..1e-6 at 1-3 TeV (data/quota_where/v4_epem_*), which the forward component's cut-edge cap (0.99)
            # never reaches; uncut finals down to 1e-8 (below 2 m_Z^4 / s^2 ~ 1e-7 at the 6 TeV top, where a
            # t-channel electron's pole flattens), at the same budget as the control
            s.update(fwd_free_floor=1e-8, flat_budget=750)
        if policy == "quota_ladder_both_res2_budget750":
            # step 66's STRONG branch: the budget from the measured quota/kept of the QCD bins still short inside the pp
            # range, worst 1/0.065 = 15.5 at log10|M|^2 4.70 (uu_bbxuu, edge ladder at budget 50), so 15 x 50
            s.update(flat_budget=750)
        if policy == "quota_ladder_res2_budget500":
            # step 65's NO branch (shortfall reduced, carried by the new channels): the budget raised 10x, the measured
            # quota/kept of the bins still short below the pp max being <= 8.5 on the electroweak classes and <= 9.5 up
            # to log10|M|^2 ~3.85 on the QCD ones (data/multichannel_check_reads/*quota_ladder_res2*)
            s.update(flat_budget=500)
        cfg["sampling"] = s
        for seed in seeds:
            sel.clear(); diag = []; secs = []; qd = []; kept_ev = []
            with mc._Swap(name, cfg):
                if not datagen_ok:
                    datagen.ensure_backend(name); datagen_ok = True
                tasks = mc.chunk_plan(name, lo, hi, n, ROLE, seed)
                mg._flat_logm_select = fs_rec
                try:
                    for t in tasks:
                        mg.MC_LAST.clear(); mg.FLAT_LAST.clear()
                        with tempfile.TemporaryDirectory(prefix="zz_probe_mcheck_") as td:
                            t0 = time.perf_counter()
                            datagen.gen_chunk(dict(t, out_dir=td))
                            secs.append(time.perf_counter() - t0)
                        if policy in QUOTA_POLICIES:
                            d = mg.FLAT_LAST
                            sel.append((np.asarray(d["logm"], float) / np.log(10.0), np.asarray(d["keep"])))
                            qd.append({"edges_log10": (np.asarray(d["edges"]) / np.log(10.0)).tolist(),
                                       "quota": np.asarray(d["quota"]).tolist(),
                                       "kept_per_bin": np.asarray(d["kept_per_bin"]).tolist(),
                                       "rounds": d["rounds"], "n_candidates": int(d["n_candidates"]),
                                       "n_labels": int(d.get("n_labels", d["n_candidates"])),
                                       "n_labels_charged": int(d.get("n_labels_charged", d.get("n_labels", d["n_candidates"]))),
                                       "budget_hit": bool(d["budget_hit"]), "stop": d["stop"],
                                       "kept_per_channel": (np.bincount(np.asarray(d["chan"])[np.asarray(d["keep"])],
                                                                        minlength=len(d["channels"])).tolist()
                                                            if d.get("channels") is not None else None),
                                       "channels": d["channels"], "mcmc": d.get("mcmc"),
                                       "t_propose": d.get("t_propose"), "t_label": d.get("t_label"),
                                       "grid_dims": d.get("grid_dims"),
                                       "cand_per_channel_bin": (np.asarray(d["cand_per_channel_bin"]).tolist()
                                                                if d.get("cand_per_channel_bin") is not None else None),
                                       "kept_per_channel_bin": (np.asarray(d["kept_per_channel_bin"]).tolist()
                                                                if d.get("kept_per_channel_bin") is not None else None)})
                            if d.get("kept_P") is not None:
                                kc = np.asarray(d["kept_chain"], np.int64)   # chain ids are per chunk: offset them
                                kept_ev.append((d["kept_P"], d["kept_sq"], np.where(kc >= 0, kc + 1000000 * len(kept_ev), kc),
                                                fdv.bins_of(np.asarray(d["logm"])[np.asarray(d["keep"])], d["edges"]),
                                                np.asarray(d["edges"]) / np.log(10.0)))
                        elif policy == "multichannel":
                            d = mg.MC_LAST
                            kc = np.bincount(np.asarray(d["chan"])[np.asarray(d["keep"])], minlength=len(d["channels"]))
                            diag.append({"channels": d["channels"], "shares1": d["shares1"],
                                         "shares2": d.get("shares2"), "kept_per_channel": kc.tolist(),
                                         "cand_per_channel": np.bincount(np.asarray(d["chan"]),
                                                                         minlength=len(d["channels"])).tolist()})
                finally:
                    mg._flat_logm_select = fs
            lm = np.concatenate([a for a, _ in sel]); kept = np.concatenate([a[k] for a, k in sel])
            fin = np.isfinite(lm)
            if kept_ev:
                Pk = np.concatenate([x[0] for x in kept_ev]); sqk = np.concatenate([x[1] for x in kept_ev])
                flat_kept[f"{policy}_{seed}"] = (fdv.features(Pk, sqk), fdv.shape_vars(Pk, sqk),
                                                 np.concatenate([x[3] for x in kept_ev]),
                                                 np.concatenate([x[2] for x in kept_ev]), kept_ev[0][4])
            out["runs"][f"{policy}_{seed}"] = {
                "n_chunks": len(tasks), "n_candidates": int(len(lm)), "n_kept": int(len(kept)),
                "n_labels_flat": int(sum(q["n_labels"] for q in qd)) if qd else None,
                "kept_min": float(np.nanmin(kept)), "kept_max": float(np.nanmax(kept)),
                "cand_max": float(np.nanmax(lm[fin])),
                "hist_kept": np.histogram(np.clip(kept, LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), LOGM_BINS)[0].tolist(),
                "hist_cand": np.histogram(np.clip(lm[fin], LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), LOGM_BINS)[0].tolist(),
                "chunks": diag, "quota_chunks": qd, "chunk_seconds": secs,
                "n_events": int(sum(int(t["count"]) for t in tasks)),
                "seconds": float(sum(secs))}
            print(f"MCHECK {name} {policy} seed {seed}: {len(lm)} candidates, {len(kept)} kept, max "
                  f"{np.nanmax(kept):.3f}", flush=True)
    for key, (F, V, b, ch, e10) in flat_kept.items():
        nb = len(e10) - 1
        r = out["runs"][key]
        r["diversity"] = {"edges_log10": e10.tolist(), "per_bin": fdv.per_bin(F, b, ch, nb),
                          "distinct_sqrts": int(len(np.unique(F[:, -1]))), "n": int(len(F))}
        ref = f"{DIVERSITY_REF}_{key.rsplit('_', 1)[1]}"
        if ref in flat_kept and ref != key and len(flat_kept[ref][4]) == len(e10):
            Fr, Vr, br = flat_kept[ref][:3]
            r["diversity"]["vs_ref"] = {"ref": ref, "per_bin": fdv.compare(F, V, b, Fr, Vr, br, nb)}
    src = {k: _pp(v, cfg0, lo, hi, f"{probe} {k}") for k, v in pp_sources(name, probe).items()}
    if src["table"] is None:
        raise SystemExit(f"{probe}: no train/test pool")
    lmp, sup = src["table"]["lm"], src["table"]["sup"]
    if not sup.any():
        raise SystemExit(f"{probe}: no pp event in the catalog's support")
    out["pp"] = {"n": int(len(lmp)), "n_in_support": int(sup.sum()), "out_of_support": float(np.mean(~sup)),
                 "q999": float(np.quantile(lmp, 0.999)), "q999_in_support": float(np.quantile(lmp[sup], 0.999)),
                 "hist": np.histogram(np.clip(lmp, LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), LOGM_BINS)[0].tolist(),
                 "hist_in_support": np.histogram(np.clip(lmp[sup], LOGM_BINS[0], LOGM_BINS[-1] - 1e-9),
                                                 LOGM_BINS)[0].tolist()}
    for key, r in out["runs"].items():
        r["above_max"] = float(np.mean(lmp[sup] > r["kept_max"]))
        r["above_max_all"] = float(np.mean(lmp > r["kept_max"]))
        r["below_min"] = float(np.mean(lmp[sup] < r["kept_min"]))
        r["below_min_all"] = float(np.mean(lmp < r["kept_min"]))
        r["sources"] = {}
        for k, v in src.items():
            if v is None:
                continue
            ls, ss = v["lm"], v["sup"]
            out_in = ss & ((ls > r["kept_max"]) | (ls < r["kept_min"]))
            r["sources"][k] = {"files": v["files"], "n": int(len(ls)), "n_in_support": int(ss.sum()),
                               "n_in_support_outside_kept": int(out_in.sum()),
                               "above_max": float(np.mean(ls[ss] > r["kept_max"])) if ss.any() else None,
                               "below_min": float(np.mean(ls[ss] < r["kept_min"])) if ss.any() else None,
                               "out_of_support_all": float(np.mean(~ss)), "outside_window_all": v["outside_window"],
                               "fail_cuts_in_window_all": v["fail_cuts_in_window"],
                               "above_max_all": float(np.mean(ls > r["kept_max"])),
                               "by_construction": k == "table" and key.startswith("quota")}
    d = os.path.join(os.environ.get("SCRATCH") or "/tmp", "zz_mcheck"); os.makedirs(d, exist_ok=True)
    tag = name if (tuple(policies), tuple(seeds)) == (("current", "multichannel"), SEEDS) else \
        f"{name}_{'+'.join(policies)}_{'+'.join(map(str, seeds))}"
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("MCHECK_DONE", name, json.dumps({k: [r["kept_max"], r["above_max"]] for k, r in out["runs"].items()}))


def _pp_quantile(hist, q):
    """the lower edge of the 0.25-dex bin holding the q quantile of a pp histogram"""
    c = np.cumsum(hist) / max(np.sum(hist), 1)
    return float(LOGM_BINS[int(np.searchsorted(c, q))])


def judge(r, q999, pp_hist=None):
    """the pass rule on one run: above_max (pp events in the support) and the kept bins' least over their median,
    between the kept part's lowest value and the in-support pp events' 99.9% quantile (bins wholly inside)"""
    e = LOGM_BINS
    h = np.asarray(r["hist_kept"])
    inside = (e[:-1] >= r["kept_min"]) & (e[1:] <= q999)
    v = h[inside]
    plateau = float(np.median(v)) if len(v) else None
    mo = float(v.min() / plateau) if len(v) and plateau else None
    low = [float(e[i]) for i in np.nonzero(inside)[0] if plateau and h[i] < 0.5 * plateau]
    # where the failing bins sit: below the pp events' bulk (the bottom edge) or within it, from its median up
    med = _pp_quantile(pp_hist, 0.5) if pp_hist is not None else None
    top = [x for x in low if med is not None and x >= med]
    top_all = [float(e[i]) for i in np.nonzero(inside)[0] if med is not None and e[i] >= med]
    return {"above_max": r["above_max"], "above_max_all": r.get("above_max_all"), "n_bins": int(len(v)),
            "pp_median_bin": med, "bins_from_pp_median": len(top_all), "bins_below_half_from_pp_median": top,
            "min_over_plateau_from_pp_median": float(min(h[i] for i in np.nonzero(inside)[0] if e[i] >= med) / plateau)
            if med is not None and top_all and plateau else None,
            "plateau": plateau, "min_over_plateau": mo, "bins_below_half": low,
            "pass": bool(r["above_max"] <= 0.001 and mo is not None and mo >= 0.5),
            "above_max_by_construction": bool(r.get("sources", {}).get("table", {}).get("by_construction")),
            "below_min": r.get("below_min"), "seconds": r.get("seconds"), "n_events": r.get("n_events"),
            "n_candidates": r.get("n_candidates"), **quota_fill(r), **heldout_read(r)}


def heldout_read(r):
    """D269's terms ("0% of pp events outside"): the held-out in-support pp events (never read for the range table)
    outside the kept flat part's [min, max], target 0, and the 95% upper bound on that share when the count is 0 of N
    (3/N, the rule of three); the table's own shares beside it, marked as holding by construction for the quota policy;
    the out-of-support shares over every event of each source (what no sampler in the support reaches: sqrt(s_hat)
    outside the window, or the cuts failed in the partonic frame)"""
    src = r.get("sources") or {}
    out = {"pp_sources": src}
    h = src.get("heldout")
    if h:
        n, c = h["n_in_support"], h["n_in_support_outside_kept"]
        out.update(heldout_n_in_support=n, heldout_outside_kept=c,
                   heldout_upper95=(3.0 / n if n and c == 0 else None), heldout_share=(c / n if n else None))
    return out


def where(name, probe, cut_log10):
    """where the held-out in-support pp events above `cut_log10` (the quota policy's kept max) sit, against those
    below it: per group the 10/50/90% quantiles, in the partonic frame, of sqrt(s_hat) over the window's lower edge,
    the least pT of the cut legs (massless, not neutrinos), the least Delta R between them, the least pair mass over
    sqrt(s_hat), the largest |eta|, the pairs within 5 widths of m_Z or m_W, the largest and the median pT of the cut legs (step 58's NO branch: the quota bins left short sit at the top of the range)"""
    import mg5_pipeline_final as mg
    import mix_compose as mc
    p = _entry(name)
    lo, hi = map(float, p["sqrts"])
    cfg = mc.control_config(name)
    rows, files = _pp_rows(pp_sources(name, probe)["heldout"])
    x = np.concatenate(rows)
    k = (x.shape[1] - 1) // 5
    P = x[:, :4 * k].reshape(-1, k, 4)
    pdg = x[0, 4 * k:5 * k].astype(int)
    mass = dict(zip((abs(int(q)) for q in cfg["pdg_ids"][2:]), (float(m) for m in cfg["m_finals"])))
    mf = np.array([mass[abs(int(q))] for q in pdg[2:]])
    Q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(1), 0, None))
    Pp = mg.to_partonic_frame(P)
    sup = mg.fiducial_pass_mask(Pp, mf, pdg, cfg["fiducial_cuts"]) & (sq >= lo) & (sq <= hi)
    with np.errstate(divide="ignore"):
        lm = np.log10(np.abs(x[:, -1]))
    F = Pp[:, 2:]
    leg = (mf == 0) & ~np.isin(np.abs(pdg[2:]), (12, 14, 16))
    pt = np.hypot(F[..., 1], F[..., 2])
    pz, pa = F[..., 3], np.sqrt((F[..., 1:] ** 2).sum(-1))
    eta = 0.5 * np.log(np.clip(pa + pz, 1e-300, None) / np.clip(pa - pz, 1e-300, None))
    phi = np.arctan2(F[..., 2], F[..., 1])
    ii = np.nonzero(leg)[0]
    feats = {"sqrts_over_lo": sq / lo, "max_abs_eta": np.abs(eta[:, ii]).max(1) if len(ii) else None,
             "min_pt": pt[:, ii].min(1) if len(ii) else None}
    pr = [(a, b) for j, a in enumerate(ii) for b in ii[j + 1:]]
    if pr:
        dr = [np.hypot(eta[:, a] - eta[:, b], (phi[:, a] - phi[:, b] + np.pi) % (2 * np.pi) - np.pi) for a, b in pr]
        feats["min_dR"] = np.min(dr, 0)
    allp = [(a, b) for a in range(k - 2) for b in range(a + 1, k - 2)]
    mij = [np.sqrt(np.clip((F[:, a, 0] + F[:, b, 0]) ** 2 - ((F[:, a, 1:] + F[:, b, 1:]) ** 2).sum(1), 0, None))
           for a, b in allp]
    feats["min_pair_mass_over_sqrts"] = np.min(mij, 0) / sq
    # pairs of finals within 5 widths of the Z or W pole (mg.MC_RES, the catalog's cards; any pair, neutrinos too):
    # resonances on shell together. The 5 widths are a diagnostic choice, as tests/test_multichannel.py's
    (mz, gz), (mw, gw) = mg.MC_RES[23], mg.MC_RES[24]
    feats["n_pairs_near_Z_or_W"] = np.sum([(np.abs(m - mz) < 5 * gz) | (np.abs(m - mw) < 5 * gw) for m in mij], 0)
    # every final's polar angle in the partonic frame, neutrinos and massive finals too (step (f)'s e+e- SHORT read:
    # forward boson emission off a t-channel electron is not bounded by the cuts on these finals), and the polar
    # angle of the pair of finals whose mass is nearest m_Z, with that distance in Z widths
    cth = np.abs(F[..., 3]) / np.clip(np.sqrt((F[..., 1:] ** 2).sum(-1)), 1e-300, None)
    feats["max_abs_cos_any_final"] = cth.max(1)
    feats["min_1_minus_abs_cos_any_final_log10"] = np.log10(np.clip(1.0 - cth.max(1), 1e-12, None))
    if allp:
        j = np.argmin(np.abs(np.asarray(mij) - mz), 0)
        a_, b_ = np.asarray(allp)[j, 0], np.asarray(allp)[j, 1]
        r_ = np.arange(len(j))
        S = F[r_, a_] + F[r_, b_]
        feats["zpair_abs_cos"] = np.abs(S[:, 3]) / np.clip(np.sqrt((S[:, 1:] ** 2).sum(-1)), 1e-300, None)
        feats["zpair_mass_minus_mz_over_width"] = (np.asarray(mij)[j, r_] - mz) / gz
    feats["sqrts"] = sq
    if len(ii):
        feats["max_pt"] = pt[:, ii].max(1)      # the hardest cut leg (the ladder channel's pT span)
        feats["median_leg_pt"] = np.median(pt[:, ii], 1)
    groups = {"above": sup & (lm > cut_log10), "below": sup & (lm <= cut_log10)}
    out = {"class": name, "files": files, "cut_log10": cut_log10, "window": [lo, hi],
           "n": {g: int(m.sum()) for g, m in groups.items()}, "quantiles_10_50_90": {}}
    for f, v in feats.items():
        if v is not None:
            out["quantiles_10_50_90"][f] = {g: np.quantile(v[m], [0.1, 0.5, 0.9]).round(4).tolist() if m.any() else None
                                            for g, m in groups.items()}
    return out


def quota_fill(r):
    """the quota policy's fill over the chunks: events short of the quota (unfilled bins) over the flat part, the bins
    short in any chunk (log10 lower edges), the rounds and candidates per flat event kept; {} for other policies"""
    qc = r.get("quota_chunks") or []
    if not qc:
        return {}
    q = sum(sum(c["quota"]) for c in qc)
    short = sum(int(np.maximum(np.asarray(c["quota"]) - np.asarray(c["kept_per_bin"]), 0).sum()) for c in qc)
    bins = sorted({round(c["edges_log10"][j], 2) for c in qc for j in range(len(c["quota"]))
                   if c["kept_per_bin"][j] < c["quota"][j]})
    # fill per bin: sum over chunks of min(kept, quota) over the sum of the quota, keyed by the bin's log10 lower edge
    fq, fk = {}, {}
    for c in qc:
        for j, (k, x) in enumerate(zip(c["kept_per_bin"], c["quota"])):
            e = round(c["edges_log10"][j], 2)
            fq[e] = fq.get(e, 0) + x; fk[e] = fk.get(e, 0) + min(k, x)
    fill = {str(e): round(fk[e] / fq[e], 4) for e in sorted(fq) if fq[e] > 0}
    return {"fill_per_bin": fill, "quota_events": int(q), "quota_short": int(short), "quota_short_share": short / max(q, 1),
            "bins_short_log10": bins, "chunks_short": int(sum(any(k < x for k, x in zip(c["kept_per_bin"], c["quota"]))
                                                             for c in qc)),
            "n_chunks": len(qc), "rounds_max": max(len(c["rounds"]) for c in qc),
            "chunks_budget_hit": int(sum(bool(c.get("budget_hit")) for c in qc)),
            "chunks_no_steer": int(sum(c.get("stop") == "no_steer" for c in qc)),
            "flat_cand_per_kept": sum(c["n_candidates"] for c in qc) / max(q, 1)}


def summary():
    out = {"source": "analysis/catalog_v4/multichannel_check.py summary over data/multichannel_check_reads/",
           "rule": "above_max <= 0.001 and min_over_plateau >= 0.5 on both seeds", "classes": {}}
    merged = {}
    for f in sorted(glob.glob(os.path.join(READS, "*.json"))):     # one file per class, or per (class, policy, seed)
        d = json.load(open(f))
        m = merged.setdefault(d["class"], dict(d, runs={}))
        assert m["pp"]["n"] == d["pp"]["n"] and m["resonances"] == d["resonances"], f
        m["runs"].update(d["runs"])
    for d in merged.values():
        q = d["pp"]["q999_in_support"]
        c = {"probe": d["probe"], "resonances": d["resonances"], "res_params": d.get("res_params"), "pp_n": d["pp"]["n"],
             "pp_out_of_support": d["pp"]["out_of_support"], "pp_q999_in_support": q}
        for pol in ("current", "multichannel") + QUOTA_POLICIES:
            if not all(f"{pol}_{s}" in d["runs"] for s in SEEDS):
                continue
            # every seed run for the policy (42 and 43 always; more where a step added seeds, plan step 67)
            ss = sorted({int(k[len(pol) + 1:]) for k in d["runs"] if k.startswith(pol + "_") and k[len(pol) + 1:].isdigit()})
            js = {s: judge(d["runs"][f"{pol}_{s}"], q, d["pp"]["hist_in_support"]) for s in ss}
            r0 = d["runs"][f"{pol}_{SEEDS[0]}"]
            ch = r0.get("quota_chunks") or r0["chunks"]      # a quota policy's per-channel counts are in its quota chunks
            kc = np.sum([x["kept_per_channel"] for x in ch], 0).tolist() if ch else None
            if pol in QUOTA_POLICIES:
                # step 58's outcome (D269): 0 held-out in-support pp events outside the kept range on both seeds, no
                # quota shortfall, no chunk stopped at the budget or with nothing to steer by
                c[pol + "_outcome"] = {
                    "rule": "heldout_outside_kept == 0 and quota_short == 0 and chunks_budget_hit == 0 and "
                            "chunks_no_steer == 0, on both seeds; the table-pool shares hold by construction",
                    "yes": all(j.get("heldout_outside_kept") == 0 and j.get("quota_short") == 0 and
                               j.get("chunks_budget_hit") == 0 and j.get("chunks_no_steer") == 0 for j in js.values()),
                    "heldout": {s: [j.get("heldout_outside_kept"), j.get("heldout_n_in_support"), j.get("heldout_upper95")]
                                for s, j in js.items()}}
            c[pol] = {"seeds": js, "pass": all(j["pass"] for j in js.values()),
                      "kept_max": [d["runs"][f"{pol}_{s}"]["kept_max"] for s in ss],
                      "channels": ch[0]["channels"] if ch else None, "kept_per_channel_seed42": kc}
        out["classes"][d["class"]] = c
        print(d["class"], {pol: (c[pol]["pass"], [(round(j["above_max"], 4), j["min_over_plateau"] and
                                                    round(j["min_over_plateau"], 3)) for j in c[pol]["seeds"].values()])
                           for pol in ("current", "multichannel") + QUOTA_POLICIES if pol in c})
    json.dump(out, open(os.path.join(DATA, "multichannel_check.json"), "w"), indent=1)


# step 67's read (its passed design, 2026-11-06): the guard of both ladders at budget 50 and the coverage at 750
STEP67 = {"old": "quota_ladder_res2", "edge": "quota_ladder_edge_res2", "both": "quota_ladder_both_res2",
          "both750": "quota_ladder_both_res2_budget750", "fwdfree750": "quota_ladder_both_res2_fwdfree_budget750"}


# step 66's e+e- reads (the budget-750 control's earlier run on the same seeds, plan step (f)); a later read of the same
# policy and seed replaces it in the merged summary, so the reproduction check reads this file directly
STEP66_EE_FILE = "{c}_current+multichannel+quota_ladder_both_res2+quota_ladder_both_res2_budget750_{s}.json"


def _edge(bins):
    """the lower edge of the first in-range bin with fill < 0.95 per seed (None: every in-range bin >= 0.95)"""
    return [min((b["lo"] for b in bins[s] if b["in_range"] and b["fill"] < 0.95), default=None) for s in sorted(bins)]


def _fwdfree_read(c, d, P, o):
    """the outcome of step (f)'s forward check (design of 2026-11-06 ~22:20), budget 750 with the forward component's
    uncut finals down to sampling.fwd_free_floor against the same policy without it: per seed the edge (first in-range
    bin < 0.95) of each, its shift in dex; YES = held-out 0 (e+e-) and every in-range bin >= 0.95 on both seeds with
    no no_steer stop and margin bins filled, NEAR = the same with a margin bin short, PARTIAL = not that but the edge
    up >= 1 dex on both seeds, NO = less; control_reproduces_step66: the control's kept-per-bin and held-out count
    equal step 66's read of the same seed (else the run is void)"""
    per = {}
    for k in ("both750", "fwdfree750"):
        js = d[STEP67[k]]["seeds"]
        per[k] = {s: [{"lo": float(b), "in_range": bool(float(b) <= o["pp_in_support_max_log10"]), "fill": v}
                      for b, v in js[s]["fill_per_bin"].items()] for s in js}
    eb, ef = _edge(per["both750"]), _edge(per["fwdfree750"])
    top = o["pp_in_support_max_log10"] + 0.25
    shift = [(f if f is not None else top) - b if b is not None else 0.0 for b, f in zip(eb, ef)]
    v = o.get("verdict_fwdfree_750")
    verdict = v if v in ("YES", "NEAR") else ("PARTIAL" if all(x >= 1.0 for x in shift) else "NO")
    rep = []
    for s in sorted(d[STEP67["both750"]]["seeds"]):
        f = os.path.join(READS, STEP66_EE_FILE.format(c=c, s=s))
        if not os.path.exists(f):
            rep.append(None); continue
        r66 = json.load(open(f))["runs"].get(f"{STEP67['both750']}_{s}")
        rnow = d[STEP67["both750"]]["seeds"][s]
        j66 = judge(r66, d["pp_q999_in_support"]) if r66 else None
        rep.append(bool(j66 and j66["fill_per_bin"] == rnow["fill_per_bin"] and
                        j66.get("heldout_outside_kept") == rnow.get("heldout_outside_kept")))
    return {"edge_both750": eb, "edge_fwdfree750": ef, "edge_shift_dex": shift, "verdict": verdict,
            "control_reproduces_step66": rep,
            "void": any(x is False for x in rep)}


def coverage():
    """data/multichannel_check.json -> data/quota_coverage.json: per class and policy, the fill per quota bin
    (sum over chunks of min(kept, quota) / sum of the quota; seeds' mean, min and max), each bin IN RANGE when its lower
    edge is <= the class's pp in-support max (the flat range table's upper end minus its margin, tools/flat_range.py),
    else MARGIN; the guard of the both-ladder policy at budget 50 (paired per seed: mean shortfall difference against
    the old ladder <= max(0.002, 2 SE), every bin's mean fill difference >= -max(0.06, 2 SE), SE of the paired
    differences over the seeds); the budget-750 verdict: YES = held-out outside kept <= 2 on every seed, every in-range
    bin >= 0.95 on every seed, no no_steer stop, and margin bins filled too; NEAR = the same with a margin bin short;
    SHORT = an in-range bin < 0.95 or held-out outside kept >= 3"""
    from tools.flat_range import ranges, MARGIN_DEX
    rng = ranges()
    src = json.load(open(os.path.join(DATA, "multichannel_check.json")))["classes"]
    out = {"source": "analysis/catalog_v4/multichannel_check.py coverage over data/multichannel_check.json",
           "classes": {}}
    for c, d in src.items():
        # step 67's pp classes carry the old ladder; step (f)'s e+e- classes have no ladder policy (mc_channels turns
        # the ladders off on e+e- beams), so a class is read when it has the both-ladder policy at 50 and 750 or the
        # old ladder beside both
        if not (all(STEP67[k] in d for k in ("old", "both")) or all(STEP67[k] in d for k in ("both", "both750"))):
            continue
        pp_max = rng[c][1] / np.log(10.0) - MARGIN_DEX
        # the class's default-sampled in-support max (13 TeV pp events, or the e+e- default-share pool at its fixed
        # energy); the key keeps its step-67 name
        o = {"pp_in_support_max_log10": pp_max, "beam": "ee" if c.startswith("v4_epem_") else "pp", "policies": {}}
        # the cost read in the same job, seed, node and commit: per-replay seconds of the old sampling (current) and
        # plain multichannel beside the quota policies' (whose seconds are in "policies")
        o["seconds_same_job"] = {pol: [d[pol]["seeds"][s]["seconds"] for s in sorted(d[pol]["seeds"])]
                                 for pol in ("current", "multichannel") if pol in d}
        for k, pol in STEP67.items():
            if pol not in d:
                continue
            js = d[pol]["seeds"]
            bins = sorted({float(b) for j in js.values() for b in j["fill_per_bin"]})
            f = {b: [js[s]["fill_per_bin"].get(f"{b:g}", js[s]["fill_per_bin"].get(str(b))) for s in sorted(js)]
                 for b in bins}
            o["policies"][k] = {
                "policy": pol, "seeds": sorted(js),
                "bins": [{"lo": b, "in_range": bool(b <= pp_max), "mean": float(np.mean(v)), "min": float(min(v)),
                          "max": float(max(v))} for b, v in f.items() if None not in v],
                "shortfall": [js[s]["quota_short_share"] for s in sorted(js)],
                "heldout_outside_kept": [js[s]["heldout_outside_kept"] for s in sorted(js)],
                "heldout_n_in_support": js[sorted(js)[0]]["heldout_n_in_support"],
                "chunks_budget_hit": [js[s]["chunks_budget_hit"] for s in sorted(js)],
                "chunks_no_steer": [js[s]["chunks_no_steer"] for s in sorted(js)],
                "seconds": [js[s]["seconds"] for s in sorted(js)]}
        def paired(a, b):           # b minus a, per seed: mean and 2 SE
            x = np.asarray(b, float) - np.asarray(a, float)
            return float(x.mean()), float(2 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
        P = o["policies"]
        g = {}
        for ref in ("old", "edge"):
            if ref not in P or "both" not in P or P[ref]["seeds"] != P["both"]["seeds"]:
                continue
            m, se2 = paired(P[ref]["shortfall"], P["both"]["shortfall"])
            fr = {s: d[STEP67[ref]]["seeds"][s]["fill_per_bin"] for s in P[ref]["seeds"]}
            fb = {s: d[STEP67["both"]]["seeds"][s]["fill_per_bin"] for s in P["both"]["seeds"]}
            per_bin = []
            for b in sorted(set.intersection(*(set(v) for v in list(fr.values()) + list(fb.values()))), key=float):
                dm, dse = paired([fr[s][b] for s in sorted(fr)], [fb[s][b] for s in sorted(fb)])
                per_bin.append({"lo": float(b), "diff": dm, "two_se": dse,
                                "ok": bool(dm >= -max(0.06, dse if dse == dse else 0.0))})
            short_ok = bool(m <= max(0.002, se2 if se2 == se2 else 0.0))
            bad = [x for x in per_bin if not x["ok"]]
            g[ref] = {"shortfall_diff": m, "shortfall_two_se": se2, "shortfall_ok": short_ok,
                      "worst_bin": min(per_bin, key=lambda x: x["diff"]) if per_bin else None, "bins_failed": bad,
                      "pass": bool(short_ok and not bad)}
        o["guard_both_vs"] = g
        for key, suf in (("both750", "750"), ("fwdfree750", "fwdfree_750")):
            if key not in P:
                continue
            q = P[key]
            # e+e- (2000 held-out events): RESOLVED = 0 outside (the design of step (f)); pp (~53000): <= 2 (step 67)
            held_ok = all(h is not None and h <= (0 if o["beam"] == "ee" else 2) for h in q["heldout_outside_kept"])
            in_ok = all(x["min"] >= 0.95 for x in q["bins"] if x["in_range"])
            mar_ok = all(x["min"] >= 0.95 for x in q["bins"] if not x["in_range"])
            o["verdict_" + suf] = ("YES" if held_ok and in_ok and mar_ok and not any(q["chunks_no_steer"]) else
                                   "NEAR" if held_ok and in_ok and not any(q["chunks_no_steer"]) else "SHORT")
            o["short_bins_" + suf] = [x for x in q["bins"] if x["min"] < 0.95]
            cur = o["seconds_same_job"].get("current")
            if cur and len(cur) == len(q["seconds"]) and None not in cur + q["seconds"]:
                o["cost_" + suf + "_over_current"] = [b / a for a, b in zip(cur, q["seconds"])]
        if "fwdfree750" in P and "both750" in P:
            o["fwdfree_vs_both750"] = _fwdfree_read(c, d, P, o)
        out["classes"][c] = o
        print(c, o.get("verdict_750"), o.get("fwdfree_vs_both750", {}).get("verdict"),
              {r: v["pass"] for r, v in g.items()}, o.get("short_bins_750"))
    json.dump(out, open(os.path.join(DATA, "quota_coverage.json"), "w"), indent=1)


def need(name, policies):
    """the job's time limit for `policies` (one seed, one node) on class `name`: per policy, the slowest per-replay seconds
    measured for the class at budget 50 over every read (any quota policy with the ladder, any node), times the policy's
    budget over 50; summed, times 1.5 (plan step 67: a limit sized from one node's time ran out at 4 h, the job needing
    4.2 h on a node 1.46x slower than step 66's) -> prints NEED_SECONDS and the slurm limit HH:MM:00"""
    s50 = []
    for f in glob.glob(os.path.join(READS, f"{name}_*.json")):
        for k, r in json.load(open(f))["runs"].items():
            if k.startswith("quota_ladder") and "budget" not in k and r.get("chunk_seconds"):
                s50.append(sum(r["chunk_seconds"]))
    t = 0.0
    if not s50:
        # a class with no read (the e+e- classes of step (f)): its old train-pool time from catalog_v4's generation logs
        # (sampling_cost.py, data/sampling_cost_gen.json) times, per policy, the largest ratio of that policy's replay
        # seconds over the old train time measured on any class with a read of it
        gen = json.load(open(os.path.join(DATA, "sampling_cost_gen.json")))["classes"]
        if name not in gen:
            sys.exit(f"need: no read and no generation time of {name}: time it from its own first jobs")
        rat = {"current": 1.0}     # the old sampling is what the generation logs timed
        for f in glob.glob(os.path.join(READS, "v4_*_*.json")):
            c = json.load(open(f))
            if c.get("class") not in gen:
                continue
            for k, r in c["runs"].items():
                if r.get("chunk_seconds"):
                    pol = k.rsplit("_", 1)[0]
                    rat[pol] = max(rat.get(pol, 0.0), sum(r["chunk_seconds"]) / gen[c["class"]]["train"]["seconds"])
        for pol in policies:
            if pol not in rat:
                sys.exit(f"need: no measured ratio for {pol}")
            t += gen[name]["train"]["seconds"] * rat[pol]
        s50 = [gen[name]["train"]["seconds"]]
        print("RATIOS " + json.dumps({p: round(rat[p], 1) for p in policies}))
    for pol in policies if not t else ():
        b = int(pol.rsplit("budget", 1)[1]) if "budget" in pol else 50
        t += max(s50) * b / 50.0
    lim = int(np.ceil(1.5 * t / 600.0)) * 600
    print(f"NEED_SECONDS {t:.0f} (slowest budget-50 replay {max(s50):.0f} s over {len(s50)} reads) "
          f"LIMIT {lim // 3600:02d}:{lim % 3600 // 60:02d}:00")


PULL_REMOTE = r'''
import json, os, sys
print("MCHECK_READ " + open(os.path.join(os.environ["SCRATCH"], "zz_mcheck", sys.argv[1] + ".json")).read().strip())
'''


# step 69's read (plan catalog_coverage, design of 2026-11-07): the Markov-chain flat part at 12 labels per flat event
# against the quota sampler at the same budget and at 50 (the diversity reference), same class, seed, job and commit
STEP69 = {"mcmc12": "mcmc_both_res2_budget12", "quota12": "quota_ladder_both_res2_budget12",
          "quota50": "quota_ladder_both_res2"}
STEP69_TAG = "current+quota_ladder_both_res2+quota_ladder_both_res2_budget12+mcmc_both_res2_budget12"
DIV_DUP_EXCESS = 0.02    # near-duplicate share (weighted over in-range bins) above the reference's, at most
DIV_DUP_TOP = 0.05       # near-duplicate share in any of the top 40% of in-range bins, at most
DIV_NN_MED = 0.8         # median over in-range bins of the NN-distance ratio against the reference at equal count, at least
DIV_NN_Q10 = 0.5         # its 10% quantile over in-range bins, at least
COST_MAX = 2.0           # train-pool replay seconds over the old generation-log seconds (the step-69 design's bound)


def step69(tag=STEP69_TAG, pols=STEP69, test="mcmc12", out_name="mcmc_check.json", source="step69"):
    """data/multichannel_check_reads/<class>_<STEP69_TAG>_<seed>.json -> data/mcmc_check.json: per class, seed and
    step-69 policy the fill per in-range quota bin (sum over chunks of min(kept, quota) / sum of quota; in range when
    the bin's lower edge <= the class's pp in-support max, tools/flat_range.py), held-out pp events in support outside
    the kept range, cost (train replay s over the old generation-log s, data/sampling_cost.json, and over the old
    sampling's replay in the same job), labels per flat event, and the flat part's diversity over in-range bins
    (flat_diversity.py); then the design's verdict:
    YES = on every class and seed the chains fill every in-range bin >= 0.95, held-out outside kept <= max(0, step 67's
    budget-750 count), cost <= COST_MAX, and diversity holds (near-duplicates <= reference + DIV_DUP_EXCESS, <= DIV_DUP_TOP
    in each top-40% in-range bin, NN ratio median >= DIV_NN_MED and q10 >= DIV_NN_Q10);
    QUOTA12 = the quota sampler at 12 meets the same fill, held-out and cost rules on every class and seed (chains not
    needed); NO = the chains' diversity fails on a class, or their share of in-range bins >= 0.95 is not above the quota
    sampler's at 12 on the 2->4 classes on both seeds; COST = fill and diversity as YES but cost > COST_MAX on a class
    (the chains' overhead, not their labels: fix it and re-run); PARTIAL = otherwise with diversity holding and the
    chains' share above the quota sampler's at 12 (where read of the short bins, one revision)"""
    from tools.flat_range import ranges, MARGIN_DEX
    rg = ranges()
    cost = json.load(open(os.path.join(DATA, "sampling_cost.json")))["test_classes"]
    cov = json.load(open(os.path.join(DATA, "quota_coverage.json")))["classes"]
    out = {"source": f"analysis/catalog_v4/multichannel_check.py {source}", "test": test, "rules": {
        "dup_excess": DIV_DUP_EXCESS, "dup_top": DIV_DUP_TOP, "nn_med": DIV_NN_MED, "nn_q10": DIV_NN_Q10,
        "cost_max": COST_MAX}, "classes": {}}
    for c in CLASSES:
        pp_max = rg[c][1] / np.log(10.0) - MARGIN_DEX
        h750 = (cov.get(c, {}).get("policies", {}).get("both750") or {}).get("heldout_outside_kept")
        o = {"pp_in_support_max_log10": pp_max, "heldout_750": h750, "seeds": {}}
        for s in SEEDS:
            f = os.path.join(READS, f"{c}_{tag}_{s}.json")
            if not os.path.exists(f):
                continue
            R = json.load(open(f))["runs"]
            cur = R[f"current_{s}"]["seconds"]
            o["seeds"][s] = {}
            for k, pol in pols.items():
                r = R[f"{pol}_{s}"]
                qc = r["quota_chunks"]
                e = np.asarray(qc[0]["edges_log10"]); lo_e = e[:-1]
                q = np.sum([x["quota"] for x in qc], 0); kp = np.sum([np.minimum(x["kept_per_bin"], x["quota"]) for x in qc], 0)
                fill = kp / np.maximum(q, 1)
                inr = lo_e <= pp_max
                n_flat = int(q.sum())
                row = {"fill_in_range": fill[inr].round(4).tolist(), "lo_in_range": lo_e[inr].round(3).tolist(),
                       "every_bin_ge95": bool((fill[inr] >= 0.95).all()), "share_ge95": float((fill[inr] >= 0.95).mean()),
                       "bins_short_lo": lo_e[inr & (fill < 0.95)].round(3).tolist(),
                       "heldout_outside_kept": r["sources"]["heldout"]["n_in_support_outside_kept"],
                       "cost_vs_old_log": r["seconds"] / cost[c]["old_train_s"], "cost_vs_current_same_job": r["seconds"] / cur,
                       "labels_per_flat": sum(x["n_labels"] for x in qc) / max(n_flat, 1),
                       "mcmc": [x.get("mcmc") for x in qc[:1]],
                       "t_propose_s": sum(x.get("t_propose") or 0.0 for x in qc), "t_label_s": sum(x.get("t_label") or 0.0 for x in qc)}
                dv = r.get("diversity")
                if dv:
                    pb = [x for x, ok in zip(dv["per_bin"], inr) if ok and "dup" in x]
                    n = np.array([x["n"] for x in pb], float); dup = np.array([x["dup"] for x in pb])
                    top = pb[int(0.6 * len(pb)):]
                    row["dup"] = float((n * dup).sum() / max(n.sum(), 1))
                    row["dup_top_max"] = float(max((x["dup"] for x in top), default=0.0))
                    row["distinct_sqrts_share"] = dv["distinct_sqrts"] / max(dv["n"], 1)
                    row["top_source_share_med"] = float(np.median([x["top_source_share"] for x in pb])) if pb else None
                    vr = [x for x, ok in zip((dv.get("vs_ref") or {}).get("per_bin", []), inr) if ok and "nn_ratio" in x]
                    if vr:
                        rr = np.array([x["nn_ratio"] for x in vr])
                        row.update(nn_ratio_med=float(np.median(rr)), nn_ratio_q10=float(np.quantile(rr, 0.1)),
                                   ks_med={v: float(np.median([x[v] for x in vr])) for v in ("ks_sqrts", "ks_mpair", "ks_maxeta")})
                o["seeds"][s][k] = row
        out["classes"][c] = o
    def ok_fill(row, c):
        h = out["classes"][c]["heldout_750"]
        hmax = max([0] + [x for x in (h or []) if x is not None])
        return row["every_bin_ge95"] and row["heldout_outside_kept"] <= hmax
    rows = [(c, s, d) for c, o in out["classes"].items() for s, d in o["seeds"].items()]
    if not rows or any(len(o["seeds"]) < len(SEEDS) for o in out["classes"].values()):
        out["verdict"] = None
    else:
        def div_ok(m, ref):
            return (m["dup"] <= ref["dup"] + DIV_DUP_EXCESS and m["dup_top_max"] <= DIV_DUP_TOP and
                    m.get("nn_ratio_med", 0) >= DIV_NN_MED and m.get("nn_ratio_q10", 0) >= DIV_NN_Q10)
        div = all(div_ok(d[test], d["quota50"]) for _, _, d in rows)
        fill = all(ok_fill(d[test], c) for c, _, d in rows)
        cheap = all(d[test]["cost_vs_old_log"] <= COST_MAX for _, _, d in rows)
        q12 = all(ok_fill(d["quota12"], c) and d["quota12"]["cost_vs_old_log"] <= COST_MAX for c, _, d in rows)
        above = all(d[test]["share_ge95"] > d["quota12"]["share_ge95"] for c, _, d in rows
                    if c not in ("v4_uux_epemg",))
        out["checks"] = {"diversity": div, "fill": fill, "cost": cheap, "quota12_suffices": q12, "above_quota12": above}
        out["verdict"] = ("QUOTA12" if q12 else "NO" if not div or not above else "YES" if fill and cheap
                          else "COST" if fill else "PARTIAL")
    json.dump(out, open(os.path.join(DATA, out_name), "w"), indent=1)
    print(source.upper(), test, out.get("verdict"), out.get("checks"))


# step 72's read (plan catalog_coverage): the channel-grid quota sampler at 12 and at 8 labels per flat event against the
# quota sampler at 12 (matched budget) and at 50 (diversity reference), same class, seed, job and commit; step69's rules
STEP72 = {"grid12": "grid_both_res2_budget12", "grid8": "grid_both_res2_budget8",
          "quota12": "quota_ladder_both_res2_budget12", "quota50": "quota_ladder_both_res2"}
STEP72_TAG = "current+quota_ladder_both_res2+quota_ladder_both_res2_budget12+grid_both_res2_budget12+grid_both_res2_budget8"


# plan step 75 (step 72's PARTIAL branch, its one revision re-run once): the grids with the forward joint-resonance
# channel at 12 and 8, read by step 72's rules against the same quota references, -> data/grid_fwd_check.json
STEP75 = {"grid12": "grid_both_res2f_budget12", "grid8": "grid_both_res2f_budget8",
          "quota12": "quota_ladder_both_res2_budget12", "quota50": "quota_ladder_both_res2"}
STEP75_TAG = "current+quota_ladder_both_res2+quota_ladder_both_res2_budget12+grid_both_res2f_budget12+grid_both_res2f_budget8"


def step72(tag=STEP72_TAG, pols=STEP72, out="grid_check", source="step72"):
    """data/grid_check.json: step69's per-class rows and verdict (YES / QUOTA12 / NO / COST / PARTIAL) once with the
    grids at 12 as the tested policy and once at 8 (verdict_by_test); YES at 8 or 12 = the grids fill every in-range
    bin >= 0.95 at <= COST_MAX on every class and seed; the top-level verdict is YES when either passes (passed_at the
    lower budget), else the grids at 12's"""
    v = {}
    for t in ("grid12", "grid8"):
        step69(tag, pols, t, f"{out}_{t}.json", source)
        v[t] = json.load(open(os.path.join(DATA, f"{out}_{t}.json")))
    vb = {t: v[t].get("verdict") for t in v}
    # the design's rule: YES when the grids at 8 or at 12 pass (the lower budget preferred); otherwise the grids at 12
    # decide (COST, PARTIAL, NO), the budget-8 row read beside it
    top = (None if None in vb.values() else "YES" if "YES" in vb.values() else vb["grid12"])
    res = dict(v["grid12"], test=None, verdict=top, passed_at=("grid8" if vb["grid8"] == "YES" else
                                                                 "grid12" if vb["grid12"] == "YES" else None),
               verdict_by_test=vb, checks_by_test={t: v[t].get("checks") for t in v})
    res.pop("checks", None)
    json.dump(res, open(os.path.join(DATA, f"{out}.json"), "w"), indent=1)
    for t in v:
        os.remove(os.path.join(DATA, f"{out}_{t}.json"))
    print(source.upper(), res["verdict"], res["passed_at"], res["verdict_by_test"])


# plan step 75's revision r1 checked before the catalog (design 2026-11-07): the worst classes of every failure kind of
# the step-65 regeneration (data/regen_g12_profile.json, regen_where.json) and step 75's two controls, each with its PP4
# probe (data/pp_support_reads.json)
R1_CLASSES = {"v4_bbx_aazz": "pp13v4_bbx_aazz", "v4_dbx_hhtxu": "pp13v4_dbx_hhtxu", "v4_ddx_ghwpwm": "pp13_ddx_ghwpwm",
              "v4_dg_adwpwm": "pp13_dg_adwpwm", "v4_dbx_ahtxu": "pp13v4_dbx_ahtxu", "v4_bb_bbvevex": "pp13v4_bb_bbvevex",
              "v4_bbx_htaptam": "pp13v4_bbx_htaptam", "v4_udx_vevexwp": "pp13_udx_vevexwp", "v4_bb_bb": "pp13v4_bb_bb",
              "v4_ddx_ttx": "pp13v4_ddx_ttx", "v4_ddx_btxwpz": "pp13v4_ddx_btxwpz", "v4_ug_auuux": "pp13v4_ug_auuux",
              "v4_uux_ddxwpwm": "pp13v4_uux_ddxwpwm", "v4_bbx_ccxh": "pp13v4_bbx_ccxh", "v4_udx_cggsx": "pp13_udx_cggsx",
              "v4_uc_cuvevex": "pp13v4_uc_cuvevex", "v4_us_ddxsu": "pp13v4_us_ddxsu", "v4_uu_bbxuu": "pp13v4_uu_bbxuu",
              # design review 2026-11-07 (FIX 2): held-out outside on a class without a dsh_ pool, whose range r1's
              # dsh_ join leaves as it was (step 65: 2 events)
              "v4_uux_bgtxwp": "pp13v4_uux_bgtxwp"}
R1_TAG = "current+grid_both_res2f_budget12+grid_r1_budget12"
R1_FILL = 0.95       # steps 72/75/65's bin bar
R1_COST = 2.0        # CLAUDE.md: a new method's cost over the one it replaces above 2x is a finding


def step75r1():
    """data/r1_check.json: per class and seed, for r1 (grid_r1_budget12) and the step-65 policy (grid_both_res2f_budget12)
    in the same job: fill of every in-range quota bin at >= R1_FILL of quota (kept and quota summed over the chunks;
    each policy against its own range table, tools/flat_range.ranges_dsh for r1, ranges for step 65), step 65's bar
    with the 0.5-dex margin included (design review 2026-11-07, FIX 1); exempt only a bin above the class's kinematic
    reach, whose lower edge lies above every candidate any channel of any policy drew for the class at either seed
    (cand_max over the job files), listed apart; the core bins ([range lo + 0.5 dex, range hi - 0.5 dex]) and margin
    bins reported separately; the held-out in-support events outside the kept [min, max] on heldout_r1 (PP4 val + dsh_
    val + dsh_ test); the cost (policy seconds / current seconds, same job), labels and stops; verdict PASS when r1
    fills every non-exempt bin, leaves 0 outside and costs <= R1_COST on every class and seed, else the failing kinds"""
    sys.path.insert(0, ROOT)
    from tools.flat_range import MARGIN_DEX, ranges, ranges_dsh
    rg = {"grid_r1_budget12": ranges_dsh(), "grid_both_res2f_budget12": ranges()}
    L = np.log(10.0)
    rows, fails = {}, {"fill": [], "outside": [], "cost": [], "missing": []}
    for c in R1_CLASSES:
        # the class's kinematic reach: the highest candidate of any policy and channel at either seed
        reach = [r.get("cand_max") for seed in SEEDS
                 for f in [os.path.join(READS, f"{c}_{R1_TAG}_{seed}.json")] if os.path.exists(f)
                 for r in json.load(open(f))["runs"].values() if r.get("cand_max") is not None]
        reach = max(reach) if reach else None
        for seed in SEEDS:
            f = os.path.join(READS, f"{c}_{R1_TAG}_{seed}.json")
            if not os.path.exists(f):
                fails["missing"].append(f"{c}/{seed}"); continue
            runs = json.load(open(f))["runs"]
            cur = runs.get(f"current_{seed}", {}).get("seconds")
            for pol in rg:
                r = runs.get(f"{pol}_{seed}")
                if r is None:
                    fails["missing"].append(f"{c}/{pol}/{seed}"); continue
                qc = r["quota_chunks"]; e = np.asarray(qc[0]["edges_log10"])
                q = np.sum([x["quota"] for x in qc], 0).astype(float); k = np.sum([x["kept_per_bin"] for x in qc], 0)
                fill = np.where(q > 0, k / np.maximum(q, 1), 1.0)
                lo, hi = rg[pol][c][0] / L + MARGIN_DEX, rg[pol][c][1] / L - MARGIN_DEX
                core = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9) & (q > 0)
                marg = ~core & (q > 0)
                exempt = (q > 0) & (e[:-1] > reach) if reach is not None else np.zeros_like(core)
                scored = (q > 0) & ~exempt
                out = r["sources"].get("heldout_r1", {}).get("n_in_support_outside_kept")
                d = {"bins": int(scored.sum()), "fill_share": float((fill[scored] >= R1_FILL).mean()) if scored.any() else 1.0,
                     "short": [[round(e[i], 3), round(e[i + 1], 3), round(float(fill[i]), 3), "core" if core[i] else "margin"]
                               for i in np.nonzero(scored & (fill < R1_FILL))[0]],
                     "exempt_above_reach": [[round(e[i], 3), round(e[i + 1], 3)] for i in np.nonzero(exempt)[0]],
                     "class_reach_log10": reach, "core_bins": int(core.sum()), "core_fill_share": float((fill[core] >= R1_FILL).mean()) if core.any() else 1.0,
                     "core_short": [[round(e[i], 3), round(e[i + 1], 3), round(float(fill[i]), 3)] for i in np.nonzero(core & (fill < R1_FILL))[0]],
                     "margin_fill": [[round(e[i], 3), round(float(fill[i]), 3)] for i in np.nonzero(marg)[0]],
                     "outside_heldout_r1": out, "outside_by_source": {k2: v.get("n_in_support_outside_kept") for k2, v in r["sources"].items()},
                     "kept_max": r["kept_max"], "seconds": r["seconds"], "cost_vs_current": r["seconds"] / cur if cur else None,
                     "labels_flat": r.get("n_labels_flat"), "stops": dict(zip(*np.unique([x["stop"] for x in qc], return_counts=True))) if qc else {}}
                d["stops"] = {str(a): int(b) for a, b in d["stops"].items()}
                rows.setdefault(c, {})[f"{pol}_{seed}"] = d
                if pol == "grid_r1_budget12":
                    if d["fill_share"] < 1.0:
                        fails["fill"].append(f"{c}/{seed}")
                    if out is None or out > 0:
                        fails["outside"].append(f"{c}/{seed}")
                    if d["cost_vs_current"] is None or d["cost_vs_current"] > R1_COST:
                        fails["cost"].append(f"{c}/{seed}")
    verdict = ("INCOMPLETE" if fails["missing"] else "PASS" if not any(fails[k] for k in ("fill", "outside", "cost"))
               else "FAIL: " + ", ".join(k for k in ("fill", "outside", "cost") if fails[k]))
    res = {"design": "plan catalog_coverage step 75 revision r1 (design 2026-11-07)", "verdict": verdict, "fails": fails,
           "classes": rows}
    json.dump(res, open(os.path.join(DATA, "r1_check.json"), "w"), indent=1)
    print("STEP75R1", verdict, json.dumps(fails))


def step75r2_where():
    """data/r1_where.json (plan step 75, r2's analysis of the r1 check): per class and seed of R1_CLASSES, for r1
    (grid_r1_budget12), each margin bin r1_check.json lists short: quota, kept, candidates and kept per channel summed
    over the chunks, and how the chunks stopped (quota, budget, stall); per class its cost split: label and proposal
    seconds of r1 and of the old sampling (current) over the chunks, labels, candidates per kept event, and per
    channel candidates and kept events (all bins)"""
    chk = json.load(open(os.path.join(DATA, "r1_check.json")))
    out = {}
    for c in R1_CLASSES:
        for seed in SEEDS:
            f = os.path.join(READS, f"{c}_{R1_TAG}_{seed}.json")
            if not os.path.exists(f):
                continue
            runs = json.load(open(f))["runs"]
            r, cur = runs[f"grid_r1_budget12_{seed}"], runs.get(f"current_{seed}", {})
            qc = r["quota_chunks"]; e = np.asarray(qc[0]["edges_log10"])
            chs = next((x["channels"] for x in qc if x.get("channels")), None) or []
            ch = [" ".join(str(x) for x in (k if isinstance(k, list) else [k])) for k in chs]
            rec = {"channels": ch, "stops": {}, "short_bins": []}
            for x in qc:
                rec["stops"][x["stop"]] = rec["stops"].get(x["stop"], 0) + 1
            for b in chk["classes"][c][f"grid_r1_budget12_{seed}"]["short"]:
                j = int(np.argmin(np.abs(e[:-1] - b[0])))
                rec["short_bins"].append({
                    "lo": round(float(e[j]), 3), "fill": b[2], "kind": b[3],
                    "quota": int(sum(x["quota"][j] for x in qc)), "kept": int(sum(x["kept_per_bin"][j] for x in qc)),
                    "cand_by_channel": dict(zip(ch, np.sum([np.asarray(x["cand_per_channel_bin"])[:, j] for x in qc if x.get("cand_per_channel_bin")], 0).tolist())) if chs else {},
                    "kept_by_channel": dict(zip(ch, np.sum([np.asarray(x["kept_per_channel_bin"])[:, j] for x in qc if x.get("kept_per_channel_bin")], 0).tolist())) if chs else {},
                    "chunks_short": int(sum(x["kept_per_bin"][j] < x["quota"][j] for x in qc))})
            tsum = lambda run, k: float(sum(x.get(k) or 0.0 for x in run.get("quota_chunks") or []))
            rec["cost"] = {"vs_current": chk["classes"][c][f"grid_r1_budget12_{seed}"]["cost_vs_current"],
                           "r1_s": r["seconds"], "current_s": cur.get("seconds"),
                           "r1_t_label": tsum(r, "t_label"), "r1_t_propose": tsum(r, "t_propose"),
                           "current_t_label": tsum(cur, "t_label"), "current_t_propose": tsum(cur, "t_propose"),
                           "r1_labels": int(sum(x["n_labels"] for x in qc)),
                           "current_labels": int(sum(x.get("n_labels") or 0 for x in cur.get("quota_chunks") or [])),
                           "r1_cand_by_channel": dict(zip(ch, np.sum([np.asarray(x["cand_per_channel_bin"]).sum(1) for x in qc if x.get("cand_per_channel_bin")], 0).tolist())) if chs else {},
                           "r1_kept_by_channel": dict(zip(ch, np.sum([x["kept_per_channel"] for x in qc if x.get("kept_per_channel")], 0).tolist())) if chs else {}}
            out[f"{c}/{seed}"] = rec
    with open(os.path.join(DATA, "r1_where.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("STEP75R2_WHERE", len(out))


def pull(policies=None, classes=None, seeds=SEEDS):
    """each class's $SCRATCH/zz_mcheck/<class>.json from CC (analysis/common site_python) -> data/multichannel_check_reads/;
    with `policies` (e.g. "multichannel+quota"), the per-seed files <class>_<policies>_<seed>.json of each seed"""
    from analysis.common import site_python
    os.makedirs(READS, exist_ok=True)
    tags = [n for n in CLASSES] if policies is None else [f"{n}_{policies}_{s}" for n in (classes or {**CLASSES, **EE_CLASSES})
                                                           for s in seeds]
    for tag in tags:
        d = site_python(PULL_REMOTE, [tag], "MCHECK_READ", timeout=300)
        json.dump(d, open(os.path.join(READS, f"{tag}.json"), "w"), separators=(",", ":"))
        print("pulled", tag, sorted(d["runs"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append("--help")
    if sys.argv[1] == "run":   # <policy> may join several with '+' (run on one node, so their times compare)
        run(sys.argv[2], sys.argv[3], *((tuple(sys.argv[4].split("+")), (int(sys.argv[5]),)) if len(sys.argv) > 5 else ()))
    elif sys.argv[1] == "where":   # where <class> <probe> <cut_log10>
        print("MCHECK_WHERE " + json.dumps(where(sys.argv[2], sys.argv[3], float(sys.argv[4]))))
    elif sys.argv[1] == "pull":
        # pull [<policies> [<class> <seed,seed,...>]]: extra seeds of one class (plan step 67)
        pull(sys.argv[2] if len(sys.argv) > 2 else None, *(([sys.argv[3]], tuple(int(x) for x in sys.argv[4].split(",")))
                                                          if len(sys.argv) > 4 else ()))
    elif sys.argv[1] == "step69":
        step69()
    elif sys.argv[1] == "step72":
        step72()
    elif sys.argv[1] == "step75":
        step72(STEP75_TAG, STEP75, "grid_fwd_check", "step75")
    elif sys.argv[1] == "step75r1-pull":
        pull(R1_TAG, R1_CLASSES)
    elif sys.argv[1] == "step75r1":
        step75r1()
    elif sys.argv[1] == "step75r2-where":
        step75r2_where()
    elif sys.argv[1] == "summary":
        summary()
    elif sys.argv[1] == "coverage":
        coverage()
    elif sys.argv[1] == "need":      # need <class> <policy+policy...>
        need(sys.argv[2], sys.argv[3].split("+"))
    else:   # an unknown argument (e.g. --help) once rewrote data/multichannel_check.json through summary()
        sys.exit("usage: multichannel_check.py run <class> <probe> [<policies> <seed>] | pull [<policies>] | "
                 "where <class> <probe> <cut_log10> | summary")
