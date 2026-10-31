"""Check of the lhe label rule 'virt' (mg5_pipeline_final.lhe_label_virt) on a one-loop class's own catalog pool: N rows
of the class's catalog test pool (the first N) relabelled through the rule's path (the class's MadLoop standalone in
subprocess chunks, nlo_virtual_pipeline.label_events) and compared with the stored labels, which the same evaluator
wrote from the same momenta (exact up to MadLoop's stability rescue); prints one VIRT_LABEL_CHECK JSON line per
class: points, unusable, the relative difference to the stored label (max, median, count above 1e-6), MadLoop
seconds per point and core, and the alpha_s the rule records (the catalog's one-loop convention). CPU.
    python tools/virt_label_check.py --n 200 --workers 1 v4_epem_bbx_nlo [...]
"""
import argparse, glob, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("classes", nargs="+")
a = ap.parse_args()
for nlo in a.classes:
    rec = {"class": nlo}
    try:
        pools = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{nlo}_*_test_*amplitudes.npy")))
        if not pools:
            raise FileNotFoundError(f"no test pool of {nlo} in $DATA_DIR")
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
        bp = pools[0][:-len(".npy")] + ".born.npy"
        if os.path.exists(bp):
            b0 = np.load(bp, mmap_mode="r")[:a.n]
            rec["born_max_rel"] = float(np.max(np.abs(born[ok] - b0[ok]) / np.maximum(np.abs(b0[ok]), 1e-300)))
    except Exception as e:  # listed, not dropped: the other classes still checked
        rec["error"] = repr(e)[:400]
    print("VIRT_LABEL_CHECK " + json.dumps(rec), flush=True)
