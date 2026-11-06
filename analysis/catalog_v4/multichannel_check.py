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
                  "quota_ladder_both_res2", "quota_ladder_both_res2_budget750", "quota_ladder_both_res2_fwdfree_budget750")
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
    return {"table": [f"{probe}_{w}_{r}_lhe_amplitudes.npy" for r in ("train", "test")],
            "heldout": [f"{probe}_{w}_val_lhe_amplitudes.npy"] + [f"{dsh}_{w}_{r}_lhe_amplitudes.npy"
                                                                  for r in ("train", "val", "test")]}


def run(name, probe, policies=("current", "multichannel"), seeds=SEEDS):
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
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
    for policy in policies:
        cfg = dict(cfg0); s = dict(cfg.get("sampling") or {})
        if policy in ("multichannel",) + QUOTA_POLICIES:
            s.update(multichannel=True, mc_resonances=[[list(S), q] for S, q in res], mc_isc=True, mc_res_params=rp)
        if policy in QUOTA_POLICIES:
            from tools.flat_range import ranges
            s.update(flat_range=ranges()[name])
        if policy.startswith("quota_ladder_res2"):     # step 58's re-check: + the multiperipheral and joint-resonance channels
            s.update(mc_mperi=True, mc_res2=True)
        if policy == "quota_ladder_edge_res2":     # step 66: the same channels with the ladder's edge variant
            s.update(mc_mperi=True, mc_mperi_edge=True, mc_res2=True)
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
            sel.clear(); diag = []; secs = []; qd = []
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
                                       "budget_hit": bool(d["budget_hit"]), "stop": d["stop"],
                                       "kept_per_channel": np.bincount(np.asarray(d["chan"])[np.asarray(d["keep"])],
                                                                       minlength=len(d["channels"])).tolist(),
                                       "channels": d["channels"]})
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
            out["runs"][f"{policy}_{seed}"] = {
                "n_chunks": len(tasks), "n_candidates": int(len(lm)), "n_kept": int(len(kept)),
                "kept_min": float(np.nanmin(kept)), "kept_max": float(np.nanmax(kept)),
                "cand_max": float(np.nanmax(lm[fin])),
                "hist_kept": np.histogram(np.clip(kept, LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), LOGM_BINS)[0].tolist(),
                "hist_cand": np.histogram(np.clip(lm[fin], LOGM_BINS[0], LOGM_BINS[-1] - 1e-9), LOGM_BINS)[0].tolist(),
                "chunks": diag, "quota_chunks": qd, "chunk_seconds": secs,
                "n_events": int(sum(int(t["count"]) for t in tasks)),
                "seconds": float(sum(secs))}
            print(f"MCHECK {name} {policy} seed {seed}: {len(lm)} candidates, {len(kept)} kept, max "
                  f"{np.nanmax(kept):.3f}", flush=True)
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
        out["classes"][c] = o
        print(c, o.get("verdict_750"), {r: v["pass"] for r, v in g.items()}, o.get("short_bins_750"))
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
    elif sys.argv[1] == "summary":
        summary()
    elif sys.argv[1] == "coverage":
        coverage()
    elif sys.argv[1] == "need":      # need <class> <policy+policy...>
        need(sys.argv[2], sys.argv[3].split("+"))
    else:   # an unknown argument (e.g. --help) once rewrote data/multichannel_check.json through summary()
        sys.exit("usage: multichannel_check.py run <class> <probe> [<policies> <seed>] | pull [<policies>] | "
                 "where <class> <probe> <cut_log10> | summary")
