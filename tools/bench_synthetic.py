"""Generation cost on one core, through the pools' own path (mg5_pipeline_final's mixture sampler,
build_mixture_dataset, with the default mixture parameters), split into phase-space sampling and labelling, to
compare synthetic processes with the real pools' per-core rates (docs/results.tex, Generation cost). CPU only:

    python tools/bench_synthetic.py [--n 5000] [--first 0 --count 19]      # v1 pilot names syn_<first>..
    python tools/bench_synthetic.py --hm [--versions 1 2 3 4 5] [--n 2000]  # 2->5 and 2->6 per version: the 8
        # high-multiplicity names of recipes/transfer_finale_synhm*.yaml, syn5[v<V>]_00000..03, syn6[v<V>]_00000..03
    python tools/bench_synthetic.py --real uubar_Zgggg uubar_Zggggg udbar_Wgggg udbar_Wggggg [--hm] [--n 2000]
        # the same measurement on MadGraph processes: compiles the C++ standalone if missing (datagen.ensure_backend,
        # timed apart), then labels through the pipe driver as prebuild does. Needs MadGraph: a site CPU node
        # (scripts/job_bench_generation.sh).

One line per process: final-state size, diagrams (K, the drawn ones; K|Pi|, with their identical-leg images),
helicities, stored events per second, the labeller's share; a summary per (version, multiplicity).
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mg5_pipeline_final as mg  # noqa: E402
from tools import synthetic_amplitudes as syn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000, help="stored events per process")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--count", type=int, default=19)
ap.add_argument("--hm", action="store_true", help="the 8 high-multiplicity names per version")
ap.add_argument("--versions", type=int, nargs="*", default=list(syn.SYN_VERSIONS))
ap.add_argument("--real", nargs="*", default=None, help="MadGraph processes (catalog names) instead")
a = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "1")
cuts = mg.FIDUCIAL_CUTS if mg.FIDUCIAL_CUTS_ENABLED else None


def timed(name, cfg, label, sqrts, pdgs, m_finals):
    """Stored events/s of one build_mixture_dataset call, and the labeller's share of it."""
    t_lab = [0.0]

    def lab(ev, sq):
        t = time.perf_counter(); out = label(ev, sq); t_lab[0] += time.perf_counter() - t
        return out
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    mg.build_mixture_dataset(a.n, sqrts[0], sqrts[1], list(m_finals), pdgs, rng, cuts, mg.sampling_policy(cfg, "train"),
                             lab)
    dt = time.perf_counter() - t0
    return a.n / dt, t_lab[0] / dt


rows = []
if a.real is not None:
    import datagen
    for name in a.real:
        cfg = mg.PROCESSES[name]
        t0 = time.perf_counter()
        datagen.ensure_backend(name)
        t_build = time.perf_counter() - t0
        sa_dir = f"{mg.WORK_DIR}/{mg.standalone_name(name)}_standalone"
        backend, subproc_dirs, driver_bin, eff_dir = mg.detect_compiled_backend(sa_dir)
        if backend != "cpp" or driver_bin is None:
            raise SystemExit(f"{name}: the mixture path needs the C++ backend, found {backend}")
        perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
        amz = float(cfg.get("alphas_mz", 0.118))
        lo = max(25.0, 1.05 * sum(cfg["m_finals"]))
        with mg.CppDriverPipe(driver_bin, eff_dir) as pipe:
            label = lambda ev, sq: pipe.compute(ev, perm=perm, alphas=mg.compute_alphas(np.asarray(sq, float), alphas_mz=amz))
            rate, share = timed(name, cfg, label, (np.ceil(lo), 1000.0), cfg["pdg_ids"], cfg["m_finals"])
        rows.append(("real", cfg["nfinal"], rate))
        print(f"BENCH {name} 2->{cfg['nfinal']} MadGraph C++ {rate:9.1f} stored ev/s  labeller {100 * share:4.1f}%  "
              f"(backend ready in {t_build:.0f} s)", flush=True)
if a.real is None or a.hm:   # --real with --hm: both on the same node, so the rates compare
    if a.hm:
        names = [(v, f"syn{n}{'' if v == 1 else f'v{v}'}_{k:05d}") for v in a.versions for n in (5, 6) for k in range(4)]
    else:
        names = [(1, f"syn_{i:05d}") for i in range(a.first, a.first + a.count)]
    for v, name in names:
        cfg = mg.register_synthetic(name)
        st = syn.structure(name)
        rate, share = timed(name, cfg, lambda ev, sq, name=name: syn.label_events(name, ev), st["sqrts"],
                            st["pdg_ids"], st["m_legs"][2:])
        K, nP = len(st["diagrams"]), len(st["perms"])
        rows.append((v, st["nfinal"], rate))
        print(f"BENCH {name} v{v} 2->{st['nfinal']} {st['pdg_ids'][:2]}->{st['pdg_ids'][2:]} K={K} K|Pi|={K * nP} "
              f"H={len(st['numerators'])} {rate:9.0f} stored ev/s  labeller {100 * share:4.1f}%", flush=True)
for v in dict.fromkeys(r[0] for r in rows):
    for n in sorted({r[1] for r in rows if r[0] == v}):
        r = [x[2] for x in rows if x[0] == v and x[1] == n]
        lab = "MadGraph" if v == "real" else f"v{v}"
        print(f"SUMMARY {lab} 2->{n}: {len(r)} processes, stored ev/s median {np.median(r):.0f}, "
              f"range {min(r):.0f}-{max(r):.0f}")
