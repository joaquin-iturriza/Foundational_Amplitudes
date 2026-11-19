"""Check of the lhe label rule 'virt' (mg5_pipeline_final.lhe_label_virt) on a one-loop class's own catalog pool: N rows
of the class's catalog test pool (the first N) relabelled through the rule's path (the class's MadLoop standalone in
subprocess chunks, nlo_virtual_pipeline.label_events) and compared with the stored labels, which the same evaluator
wrote from the same momenta (exact up to MadLoop's stability rescue); prints one VIRT_LABEL_CHECK JSON line per
class: points, unusable, the relative difference to the stored label (max, median, count above 1e-6), MadLoop
seconds per point and core, and the alpha_s the rule records (the catalog's one-loop convention). CPU.
    python tools/virt_label_check.py --n 200 --workers 1 v4_epem_bbx_nlo [...]
    python tools/virt_label_check.py --n 200 --from-run <run dir> --save <dir> uubar_ttbar_nlo
        (the train and val pools that run loaded, from its out_0.log, instead of the test pool; --save writes per pool
         an npz of sqrt(s), the stored and the relabelled value and ok, for a figure)
"""
import argparse, glob, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--from-run", default=None, help="check the [train]/[val] pools this run's out_0.log names")
ap.add_argument("--save", default=None, help="directory for one npz per pool checked")
ap.add_argument("classes", nargs="+")
a = ap.parse_args()
def pools_of(nlo):
    """[(role, path)]: the class's first test pool, or with --from-run the train and val pools that run loaded"""
    if a.from_run:
        import re
        log = open(os.path.join(a.from_run, "out_0.log")).read()
        return [(r, re.search(r"\[%s\] %s: \d+ events.*? from (\S+\.npy)" % (r, re.escape(nlo)), log).group(1))
                for r in ("train", "val")]
    pools = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{nlo}_*_test_*amplitudes.npy")))
    if not pools:
        raise FileNotFoundError(f"no test pool of {nlo} in $DATA_DIR")
    return [("test", pools[0])]


for nlo, (role, pool) in [(c, rp) for c in a.classes for rp in pools_of(c)]:
    rec = {"class": nlo, "role": role}
    pools = [pool]
    try:
        d = np.load(pools[0], mmap_mode="r")
        npart = len(mg.PROCESSES[nlo]["pdg_ids"])
        rows = np.asarray(d[:a.n], dtype=np.float64)
        m = rows[:, :npart * 4].reshape(-1, npart, 4)
        t0 = time.time()
        v, ok, born, al = mg.lhe_label_virt(nlo, m, workers=a.workers)
        sec = time.time() - t0
        stored = rows[:, -1]
        rel = np.abs(v - stored) / np.maximum(np.abs(stored), 1e-300)
        rec.update(pool=os.path.basename(pools[0]), n=int(len(m)), unusable=int((~ok).sum()),
                   max_rel=float(rel[ok].max()) if ok.any() else None,
                   median_rel=float(np.median(rel[ok])) if ok.any() else None,
                   n_rel_above_1e_6=int((rel[ok] > 1e-6).sum()), n_negative=int((v[ok] < 0).sum()),
                   s_per_point_core=round(sec * a.workers / len(m), 4), wall_s=round(sec, 1),
                   alphas_range=[float(al.min()), float(al.max())])
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            tot = m[:, 0] + m[:, 1]
            sq = np.sqrt(np.maximum(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(1), 0))
            np.savez(os.path.join(a.save, f"{nlo}_{role}.npz"), stored=stored, relabelled=v, ok=ok, sqrts=sq, pool=pools[0])
            q, lr = np.quantile(sq, [0, .25, .5, .75, 1]), np.log(np.abs(v) / np.abs(stored))
            rec["median_log_ratio_by_sqrts_quartile"] = [float(np.median(lr[ok & (sq >= q[i]) & (sq <= q[i + 1])]))
                                                         for i in range(4)]
        bp = pools[0][:-len(".npy")] + ".born.npy"
        if os.path.exists(bp):
            b0 = np.load(bp, mmap_mode="r")[:a.n]
            rec["born_max_rel"] = float(np.max(np.abs(born[ok] - b0[ok]) / np.maximum(np.abs(b0[ok]), 1e-300)))
    except Exception as e:  # listed, not dropped: the other classes still checked
        rec["error"] = repr(e)[:400]
    print("VIRT_LABEL_CHECK " + json.dumps(rec), flush=True)
