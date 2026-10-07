"""Generation cost of synthetic processes on one core, through the pools' own path (mg5_pipeline_final's
mixture sampler with the synthetic labeller), split into phase-space sampling and labelling, to compare with the
real pools' per-core rates (docs/results.tex, Generation cost). CPU only, a minute or two:
    python tools/bench_synthetic.py [--n 5000] [--first 0 --count 19]
One line per process: final-state size, diagrams, helicities, stored events per second, the labeller's share.
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mg5_pipeline_final as mg  # noqa: E402
from tools import synthetic_amplitudes as syn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--count", type=int, default=19)
a = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "1")
rows = []
for i in range(a.first, a.first + a.count):
    name = f"syn_{i:05d}"
    mg.register_synthetic(name)
    cfg = mg.PROCESSES[name]
    st = syn.structure(name)
    pol = mg.sampling_policy(cfg, "train")
    t_lab = [0.0]

    def label(ev, sq, name=name):
        t = time.perf_counter(); out = syn.label_events(name, ev); t_lab[0] += time.perf_counter() - t
        return out
    cuts = mg.FIDUCIAL_CUTS if mg.FIDUCIAL_CUTS_ENABLED else None
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    mg.build_mixture_dataset(a.n, st["sqrts"][0], st["sqrts"][1], list(st["m_legs"][2:]), st["pdg_ids"], rng, cuts,
                             pol, label)
    dt = time.perf_counter() - t0
    rows.append((name, st["nfinal"], len(st["diagrams"]), len(st["numerators"]), len(st["perms"]), a.n / dt, t_lab[0] / dt))
    print(f"BENCH {name} 2->{st['nfinal']} K={len(st['diagrams'])} H={len(st['numerators'])} perms={len(st['perms'])} "
          f"{a.n / dt:9.0f} stored ev/s  labeller {100 * t_lab[0] / dt:4.1f}%", flush=True)
for n in (2, 3, 4):
    r = [x[5] for x in rows if x[1] == n]
    if r:
        print(f"SUMMARY 2->{n}: {len(r)} processes, stored ev/s median {np.median(r):.0f}, range {min(r):.0f}-{max(r):.0f}")
