"""The on-shell mass of every leg of a pool, read from its momenta (sqrt(E^2 - p^2), median and spread over events), with
the leg's PDG id: e.g. the b mass of the ladder's e+e- -> b b~ one-loop probe quoted in the draft's Sec. 2.
    python analysis/data_saving/pool_mass.py ee_bb_nlo_25-1000GeV_val_amplitudes.npy [--site ccin2p3]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from readers import site_python  # noqa: E402

REMOTE = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import siteconf
x = np.load(os.path.join(siteconf.DATA_DIR, sys.argv[1]), mmap_mode="r")[:2000]
n = (x.shape[1] - 1) // 5
p = np.asarray(x[:, :4 * n]).reshape(-1, n, 4)
m = np.sqrt(np.abs(p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)))
pdg = np.asarray(x[0, 4 * n:5 * n]).astype(int).tolist()
print("POOLMASS " + json.dumps(dict(pool=sys.argv[1], n_events=len(p), pdg=pdg,
      median=np.median(m, 0).tolist(), min=m.min(0).tolist(), max=m.max(0).tolist())))
'''
ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("--site", default="ccin2p3")
a = ap.parse_args()
print(json.dumps(site_python(REMOTE, [a.pool], "POOLMASS", site=a.site), indent=1))
