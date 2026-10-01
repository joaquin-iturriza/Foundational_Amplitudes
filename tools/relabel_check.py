"""Relabel stored pool rows with today's compiled backend and compare with the stored |M|^2: the same
labeller the generator uses (CppDriverPipe, rows mapped to MadGraph slots by row_to_slot_perm,
per-event alpha_s(sqrt s)). Prints the median and max relative difference, and their trend in sqrt(s).
Tells whether two pools of one process carry the same function of the momenta. CPU; runs where the
pools and the compiled backend are.
    python tools/relabel_check.py <process> <pool.npy> [<pool.npy> ...] [--n 2000]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("process"); ap.add_argument("pools", nargs="+"); ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
cfg = mg.PROCESSES[a.process]
sa = f"{mg.WORK_DIR}/{a.process}_standalone"
backend, _, driver, _ = mg.detect_compiled_backend(sa)
assert backend == "cpp" and driver, f"no compiled C++ backend at {sa}"
perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
amz = float(cfg.get("alphas_mz", 0.118))
for path in a.pools:
    X = np.asarray(np.load(path, mmap_mode="r")[:a.n])
    npart = (X.shape[1] - 1) // 5
    mom = X[:, :npart * 4].reshape(-1, npart, 4)
    tot = mom[:, 0] + mom[:, 1]
    s = np.sqrt(np.maximum(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(1), 0))
    with mg.CppDriverPipe(driver, sa) as pipe:
        me2 = np.asarray(pipe.compute([(m, np.asarray(cfg["pdg_ids"])) for m in mom], perm=perm,
                                      alphas=mg.compute_alphas(s, alphas_mz=amz)), float)
    # ratio of magnitudes, so a signed (one-loop) pool compares too; a sign flip is counted apart
    lr = np.log(np.abs(me2) / np.abs(X[:, -1]))
    flips = int((np.sign(me2) != np.sign(X[:, -1])).sum())
    q = np.quantile(s, [0, .25, .5, .75, 1])
    by_s = [float(np.median(lr[(s >= q[i]) & (s <= q[i + 1])])) for i in range(4)]
    print(json.dumps(dict(pool=path, n=len(X), median_log_ratio=float(np.median(lr)),
                          max_abs_log_ratio=float(np.abs(lr).max()), sign_flips=flips, median_by_sqrts_quartile=by_s)))
