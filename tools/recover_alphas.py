"""Write <pool>.alphas.npy, each event's alpha_s (the AQCDUP it was labelled at), for lhe_pp pools built before
mg5_pipeline_final.build_lhe_pp saved it (D93: the coupling-order factor of the target, data.target_physics_orders,
divides it out). MadEvent is rerun with the pool's own recorded run (seed, iseed, events requested, cores, from its
.recipe.json), the events are shuffled and sliced exactly as build_lhe_pp did, and the momenta are checked against the
stored pool row by row before anything is written: a pool whose momenta do not reproduce is reported and left alone.
Pools are found in $DATA_DIR (val, test) and $SCRATCH/amp_data_cache (train). CPU only; one MadEvent run per process,
--workers at once.
    python tools/recover_alphas.py recipes/catalog_daniel_all.yaml [--workers 48] [--only NAME ...]
"""
import argparse, glob, json, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pools_of(name):
    roots = [os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")]
    return sorted(p for r in roots for p in glob.glob(os.path.join(r, f"{name}_*_lhe_amplitudes.npy")))


def recover(name):
    import mg5_pipeline_final as mg
    t0 = time.time()
    todo = [p for p in pools_of(name) if not os.path.exists(p[:-len(".npy")] + ".alphas.npy")]
    if not todo:
        return name, "done already", []
    recs = {p: json.load(open(p + ".recipe.json")) for p in todo}
    runs = {(r["effective_seed"] if "effective_seed" in r else r["seed"], r["lhe"]["iseed"], r["lhe"]["n_requested"],
             int(r.get("lhe_run", {}).get("mg_cores", 1))) for r in recs.values()}
    if len(runs) != 1:
        return name, f"pools from different MadEvent runs {sorted(runs)}", []
    seed, iseed, n_req, cores = runs.pop()
    work = os.path.join(mg.lhe_work_dir(), "recover_alphas")
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, f"{name}.mg5.log")
    open(log, "w").close()
    ev, _ = mg.lhe_prepare(name, work, log)
    try:
        lhe, _ = mg.lhe_run_madevent(name, ev, n_req, iseed, log, cores)
        mom, aqcd, _ = mg.read_lhe_pp(lhe, mg.PROCESSES[name]["pdg_ids"])
    finally:
        shutil.rmtree(ev, ignore_errors=True)
    order = np.random.default_rng(np.random.SeedSequence([seed, iseed])).permutation(len(mom))
    mom, aqcd = mom[order], aqcd[order]
    npart = int(mg.PROCESSES[name]["nfinal"]) + 2
    out, bad = [], []
    for p in todo:
        lo, hi = recs[p]["lhe"]["block"]
        stored = np.load(p, mmap_mode="r")[:, :npart * 4]
        again = mom[lo:hi].reshape(hi - lo, npart * 4)
        if stored.shape != again.shape or not np.allclose(stored, again, rtol=1e-9, atol=1e-9):
            bad.append(p)
            continue
        np.save(p[:-len(".npy")] + ".alphas.npy", np.asarray(aqcd[lo:hi], dtype=np.float64))
        out.append(p)
    msg = f"{len(out)} written, {len(bad)} not reproduced, {time.time() - t0:.0f} s"
    return name, msg, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    import mg5_pipeline_final as mg
    names = [p["name"] for p in yaml.safe_load(open(a.spec))["processes"]]
    names = [n for n in names if (not a.only or n in a.only) and mg.is_lhe_pp(n)]
    print(f"{len(names)} lhe_pp processes, {a.workers} at once", flush=True)
    fails = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(recover, n): n for n in names}
        for f in as_completed(futs):
            try:
                n, msg, bad = f.result()
            except Exception as e:                     # one process failing never stops the rest
                n, msg, bad = futs[f], f"error {e!r}"[:300], ["?"]
            print(f"[ALPHAS] {n}: {msg}", flush=True)
            if bad:
                fails.append((n, bad))
    print(f"RECOVER_ALPHAS_DONE {len(names) - len(fails)}/{len(names)}", flush=True)
    for n, bad in fails:
        print(f"  not recovered: {n} {bad}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
