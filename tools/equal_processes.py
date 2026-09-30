"""Which catalog trees are the SAME dataset: |M|^2 of one is a constant multiple of the
other's at every phase-space point, up to relabelling the legs. Per-dataset
standardization removes the constant, so two such processes are one dataset to the model
(massless flavour relabels, an overall charge or colour factor, uu~>ga vs uu~>aa, ...).

Processes sharing a multiplicity and a final-mass multiset share phase-space points: one
set per group, drawn in the common sqrt(s) window with the fiducial cuts. Each process is
evaluated on those points under every mass-preserving assignment of the points' legs to its
own rows (and both beam orientations), with alpha_s(sqrt s) from its own alpha_s(M_Z), as
in generation. A pair is equal when the log-ratio is constant for some assignment: its
spread over the points is at the level of double-precision round-off (TOL), which no
genuine difference in couplings or propagators reaches. Near pairs are listed with their
spread so the closest non-equal ones are visible too.

Trees only (one-loop and loop-induced entries are not compared). Runs on the site where the
backends are compiled, as a CPU job (scripts/job_equal_processes.sh).

    python tools/equal_processes.py [--n 48] [--workers 16] [--out analysis/process_equality]
"""
import argparse, itertools, json, os, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datagen
import mg5_pipeline_final as mg

RECIPES = ["recipes/catalog_v2_train.yaml", "recipes/catalog_v2_holdout.yaml"]
TOL = 1e-9     # spread of log|M_B|^2 - log|M_A|^2 at round-off level
NEAR = 1e-2    # also report pairs this close, as the nearest non-equal ones


def m_finals_of(cfg):
    if "m_finals" in cfg:
        return [float(m) for m in cfg["m_finals"]]
    m = cfg["m_final"]
    return [float(x) for x in m] if isinstance(m, (list, tuple)) else [float(m)] * cfg["nfinal"]


def tree_processes():
    out = {}
    for r in RECIPES:
        for p in yaml.safe_load(open(r))["processes"]:
            n = p["name"]
            if datagen.is_virt(n) or n.endswith("_loop") or n.endswith("_nlo"):
                continue
            out[n] = (float(p["sqrts"][0]), float(p["sqrts"][1]))
    return out


def assignments(masses, canon):
    """Every bijection rows -> canonical final slots that preserves mass."""
    n = len(masses)
    return [perm for perm in itertools.permutations(range(n))
            if all(masses[r] == canon[perm[r]] for r in range(n))]


def evaluate(task):
    """|M|^2 of one process on its group's points, one row per (beam flip, assignment)."""
    name, P, sqrts, canon = task
    cfg = mg.PROCESSES[name]
    sa_dir = datagen.ensure_backend(name)
    backend, _, driver_bin, eff_dir = mg.detect_compiled_backend(sa_dir)
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    alphas = mg.compute_alphas(sqrts, alphas_mz=float(cfg.get("alphas_mz", 0.118)))
    pdg = np.asarray(cfg["pdg_ids"])
    maps = [(flip, a) for flip in (False, True) for a in assignments(m_finals_of(cfg), canon)]
    out = np.empty((len(maps), len(P)))
    with mg.CppDriverPipe(driver_bin, eff_dir) as pipe:
        for k, (flip, a) in enumerate(maps):
            beams = [1, 0] if flip else [0, 1]
            rows = beams + [2 + a[r] for r in range(len(a))]
            ev = [(p[rows], pdg) for p in P]
            out[k] = pipe.compute(ev, perm=perm, alphas=alphas)
    return name, [[bool(f), list(map(int, a))] for f, a in maps], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--out", default="analysis/process_equality")
    args = ap.parse_args()

    procs = tree_processes()
    groups = defaultdict(list)
    for n in procs:
        cfg = mg.PROCESSES[n]
        groups[(cfg["nfinal"], tuple(sorted(m_finals_of(cfg))))].append(n)
    print(f"{len(procs)} trees in {len(groups)} (multiplicity, final masses) groups", flush=True)

    # Backends first, serially: parallel workers must never race to compile one.
    for n in procs:
        datagen.ensure_backend(n)

    tasks, meta = [], {}
    for gi, (key, names) in enumerate(sorted(groups.items())):
        if len(names) < 2:
            continue
        nfinal, canon = key
        lo = max(procs[n][0] for n in names)
        hi = min(procs[n][1] for n in names)
        lo = max(lo, 1.2 * sum(canon) + 1.0)
        if lo >= hi:
            print(f"  skip {key}: empty common window", flush=True)
            continue
        rng = np.random.default_rng(1000 + gi)
        cut_pdg = [11, -11] + [21 if m == 0.0 else 6 for m in canon]
        cuts = mg.FIDUCIAL_CUTS if mg.FIDUCIAL_CUTS_ENABLED else None
        if nfinal == 2:
            ev, sq = mg.sample_2to2_phase_space(args.n, lo, hi, list(canon), cut_pdg, rng=rng, cuts=cuts)
        else:
            ev, sq = mg.sample_nbody_phase_space(args.n, lo, hi, list(canon), cut_pdg, rng=rng, cuts=cuts)
        P = np.stack([e[0] for e in ev])
        meta[key] = dict(names=names, window=[lo, hi])
        tasks += [(n, P, np.asarray(sq, float), list(canon)) for n in names]

    with Pool(args.workers) as pool:
        res = {name: (maps, amps) for name, maps, amps in pool.imap_unordered(evaluate, tasks)}

    pairs = []
    for key, g in meta.items():
        for a, b in itertools.combinations(sorted(g["names"]), 2):
            A = res[a][1][0]                      # a on its first assignment
            B = res[b][1]                         # b on every assignment
            ok = (A > 0) & (B > 0).all(0)
            if ok.sum() < 8:
                continue
            lr = np.log(B[:, ok]) - np.log(A[ok])
            spread = lr.std(axis=1)
            k = int(np.argmin(spread))
            pairs.append(dict(a=a, b=b, spread=float(spread[k]), log_ratio=float(lr[k].mean()),
                              map=res[b][0][k], group=[key[0], list(key[1])]))

    # equivalence classes from the equal pairs
    parent = {n: n for n in procs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for p in pairs:
        if p["spread"] < TOL:
            parent[find(p["a"])] = find(p["b"])
    classes = defaultdict(list)
    for n in procs:
        classes[find(n)].append(n)
    equal = sorted((sorted(c) for c in classes.values() if len(c) > 1), key=lambda c: c[0])

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/equal_processes.json", "w") as f:
        json.dump(dict(tol=TOL, near=NEAR, n_points=args.n, classes=equal,
                       pairs=sorted(pairs, key=lambda p: p["spread"])), f, indent=1)
    print(f"\n{len(equal)} classes of equal processes (spread < {TOL:g}):")
    for c in equal:
        print("  " + "  ".join(c))
    print(f"\nnear, not equal ({TOL:g} <= spread < {NEAR:g}):")
    for p in sorted(pairs, key=lambda p: p["spread"]):
        if TOL <= p["spread"] < NEAR:
            print(f"  {p['a']:22s} {p['b']:22s} spread {p['spread']:.2e}")
    print(f"\nwrote {args.out}/equal_processes.json")


if __name__ == "__main__":
    main()
