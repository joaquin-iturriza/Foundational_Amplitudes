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
Resonance masses and widths: mg.mc_res_params of the entry's physics block.

    python3 analysis/catalog_v4/multichannel_check.py run <class> <probe> [<policy> <seed>]   CPU job
               (scripts/job_multichannel_check.sh, one task per class, policy and seed): -> $SCRATCH/zz_mcheck/
               <class>_<policy>_<seed>.json
    python3 analysis/catalog_v4/multichannel_check.py pull                 local: $SCRATCH/zz_mcheck/<class>.json of each
               class from CC (site run cat) -> data/multichannel_check_reads/
    python3 analysis/catalog_v4/multichannel_check.py summary              local: data/multichannel_check_reads/*.json -> data/multichannel_check.json
"""
import glob, json, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "multichannel_check_reads")
LOGM_BINS = np.linspace(-20.0, 8.0, 113)          # log10 |M|^2 in 0.25 dex
SEEDS = (42, 43)
ROLE = "train"
# the step-57 classes (the pp_support worst list, 2026-11-04, and the shape study's class) and their PP4 probes
CLASSES = {"v4_uux_epemg": "pp13_uux_epemg", "v4_uux_epemvevex": "pp13_uux_epemvevex",
           "v4_uu_bbxuu": "pp13v4_uu_bbxuu", "v4_us_ddxsu": "pp13v4_us_ddxsu",
           "v4_udx_csxvevex": "pp13_udx_csxvevex"}
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


def _pp(probe, cfg, lo, hi):
    """log10|M|^2 of the probe's train and test events (pp_support.py's files) and whether each lies in the catalog's
    support: sqrt(s_hat) in [lo, hi] and the class's cuts passed in the partonic frame (pp_support.py's mask)"""
    import mg5_pipeline_final as mg
    rows, files = [], []
    for role in ("train", "test"):
        f = _find(f"{probe}_13000-13000GeV_{role}_lhe_amplitudes.npy")
        if f is None:
            raise SystemExit(f"{probe}: no {role} pool")
        rows.append(np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)); files.append(f)
    x = np.concatenate(rows)
    k = (x.shape[1] - 1) // 5
    assert x.shape[1] == 5 * k + 1, (probe, x.shape)
    P = x[:, :4 * k].reshape(-1, k, 4)
    pdg = x[0, 4 * k:5 * k].astype(int)
    mass = dict(zip((abs(int(q)) for q in cfg["pdg_ids"][2:]), (float(m) for m in cfg["m_finals"])))
    mf = np.array([mass[abs(int(q))] for q in pdg[2:]])
    q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))
    ok = mg.fiducial_pass_mask(mg.to_partonic_frame(P), mf, pdg, cfg["fiducial_cuts"])
    with np.errstate(divide="ignore"):
        lm = np.log10(np.abs(x[:, -1]))
    fin = np.isfinite(lm)
    print(f"MCHECK pp {probe}: {len(lm)} events ({[os.path.basename(f) for f in files]}), {int((~fin).sum())} with "
          f"|M|^2 0, {float(np.mean(~(ok & (sq >= lo) & (sq <= hi)))):.4f} outside the support", flush=True)
    return lm[fin], ((sq >= lo) & (sq <= hi) & ok)[fin]


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
        if policy == "multichannel":
            s.update(multichannel=True, mc_resonances=[[list(S), q] for S, q in res], mc_isc=True, mc_res_params=rp)
        cfg["sampling"] = s
        for seed in seeds:
            sel.clear(); diag = []
            with mc._Swap(name, cfg):
                if not datagen_ok:
                    datagen.ensure_backend(name); datagen_ok = True
                tasks = mc.chunk_plan(name, lo, hi, n, ROLE, seed)
                mg._flat_logm_select = fs_rec
                try:
                    for t in tasks:
                        mg.MC_LAST.clear()
                        with tempfile.TemporaryDirectory(prefix="zz_probe_mcheck_") as td:
                            datagen.gen_chunk(dict(t, out_dir=td))
                        if policy == "multichannel":
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
                "chunks": diag}
            print(f"MCHECK {name} {policy} seed {seed}: {len(lm)} candidates, {len(kept)} kept, max "
                  f"{np.nanmax(kept):.3f}", flush=True)
    lmp, sup = _pp(probe, cfg0, lo, hi)
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
    d = os.path.join(os.environ.get("SCRATCH") or "/tmp", "zz_mcheck"); os.makedirs(d, exist_ok=True)
    tag = name if (tuple(policies), tuple(seeds)) == (("current", "multichannel"), SEEDS) else \
        f"{name}_{'+'.join(policies)}_{'+'.join(map(str, seeds))}"
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("MCHECK_DONE", name, json.dumps({k: [r["kept_max"], r["above_max"]] for k, r in out["runs"].items()}))


def judge(r, q999):
    """the pass rule on one run: above_max (pp events in the support) and the kept bins' least over their median,
    between the kept part's lowest value and the in-support pp events' 99.9% quantile (bins wholly inside)"""
    e = LOGM_BINS
    h = np.asarray(r["hist_kept"])
    inside = (e[:-1] >= r["kept_min"]) & (e[1:] <= q999)
    v = h[inside]
    plateau = float(np.median(v)) if len(v) else None
    mo = float(v.min() / plateau) if len(v) and plateau else None
    low = [float(e[i]) for i in np.nonzero(inside)[0] if plateau and h[i] < 0.5 * plateau]
    return {"above_max": r["above_max"], "above_max_all": r.get("above_max_all"), "n_bins": int(len(v)),
            "plateau": plateau, "min_over_plateau": mo, "bins_below_half": low,
            "pass": bool(r["above_max"] <= 0.001 and mo is not None and mo >= 0.5)}


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
        for pol in ("current", "multichannel"):
            js = {s: judge(d["runs"][f"{pol}_{s}"], q) for s in SEEDS}
            ch = d["runs"][f"{pol}_{SEEDS[0]}"]["chunks"]
            kc = np.sum([x["kept_per_channel"] for x in ch], 0).tolist() if ch else None
            c[pol] = {"seeds": js, "pass": all(j["pass"] for j in js.values()),
                      "kept_max": [d["runs"][f"{pol}_{s}"]["kept_max"] for s in SEEDS],
                      "channels": ch[0]["channels"] if ch else None, "kept_per_channel_seed42": kc}
        out["classes"][d["class"]] = c
        print(d["class"], {pol: (c[pol]["pass"], [(round(j["above_max"], 4), j["min_over_plateau"] and
                                                    round(j["min_over_plateau"], 3)) for j in c[pol]["seeds"].values()])
                           for pol in ("current", "multichannel")})
    json.dump(out, open(os.path.join(DATA, "multichannel_check.json"), "w"), indent=1)


def pull():
    import subprocess
    os.makedirs(READS, exist_ok=True)
    for name in CLASSES:
        out = subprocess.run(["site", "--timeout", "300", "run", "ccin2p3", "FA", "--",
                              f'echo MCHECK_BEGIN; cat "$SCRATCH/zz_mcheck/{name}.json"; echo; echo MCHECK_END'],
                             capture_output=True, text=True, timeout=400).stdout
        body = out.split("MCHECK_BEGIN", 1)[-1].split("MCHECK_END", 1)[0].strip()
        d = json.loads(body)
        json.dump(d, open(os.path.join(READS, f"{name}.json"), "w"), separators=(",", ":"))
        print("pulled", name, sorted(d["runs"]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], *(((sys.argv[4],), (int(sys.argv[5]),)) if len(sys.argv) > 5 else ()))
    elif sys.argv[1] == "pull":
        pull()
    else:
        summary()
