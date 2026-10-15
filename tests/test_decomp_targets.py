"""tools/decomp_targets.py against the production labelling (CPU, needs MadGraph; no torch).

For each process a small pool is built by the production path (datagen.ensure_backend, then
mg5_pipeline_final.build_dataset_variable_energy, the C++ driver at per-event alpha_s(sqrt s)), and the
decomposition is checked against it: the coherent sum over colour flows and over diagram amplitudes must
reproduce the stored |M|^2 and MadGraph's own sigmaKin to machine precision. The standalones and pools go
under $DECOMP_TEST_DIR (default /tmp/decomp_test); MG5_WORK_DIR is pointed there before the pipeline is
imported, so no production standalone is touched.

    MG5_BIN=~/mg5amcnlo/bin/mg5_aMC python tests/test_decomp_targets.py [ee_uu uubar_gg ee_uug] [--n 2000]

--head (needs torch, no MadGraph once the sidecars exist): the training-side conventions of decomp_head. For each
sidecar, net outputs that represent its X exactly (decomp_head.exact_raw) must give back the standardized
ln|M|^2 through the coherent readout and a zero parts loss, in both bases; a perturbed output must not.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("DECOMP_TEST_DIR", "/tmp/decomp_test")
os.environ["MG5_WORK_DIR"] = os.path.join(BASE, "mg5work")
os.makedirs(os.environ["MG5_WORK_DIR"], exist_ok=True)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import numpy as np  # noqa: E402
if "--head" not in sys.argv:
    import datagen  # noqa: E402
    import mg5_pipeline_final as mg  # noqa: E402
    import decomp_targets as dt  # noqa: E402

WINDOWS = {"ee_uu": (91, 1000), "uubar_gg": (25, 1000), "ee_uug": (91, 1000), "ee_ttbar": (346, 1000)}


def check(process, n, rtol=1e-12):
    sa_dir = datagen.ensure_backend(process)
    backend, subs, drv, eff = mg.detect_compiled_backend(sa_dir)
    pool = os.path.join(BASE, f"{process}_pool.npy")
    if not os.path.exists(pool):
        lo, hi = WINDOWS[process]
        cfg = dict(mg.PROCESSES[process])
        mg.build_dataset_variable_energy(n, lo, hi, eff, backend, subs, drv, cfg, pool,
                                         rng=np.random.default_rng(7), role="train")
    d = dt.run_pool(process, pool, os.path.join(BASE, f"{process}_pool.decomp.npz"),
                    os.path.join(BASE, "decomp_sa"), rtol=rtol)
    stored = np.load(pool)[:, -1]
    out = {}
    for basis in ("colour", "diagram"):
        rel = np.abs(d[f"sum_{basis}"] - stored) / np.abs(stored)
        out[basis] = (float(rel.max()), float(np.median(rel)))
        assert rel.max() < rtol, f"{process} {basis}: max rel {rel.max():.2e}"
    # the Gram matrices are Hermitian and positive semidefinite (sums of a a^dagger over helicities)
    for X in (d["Xc"], d["Xd"]):
        assert np.allclose(X, np.conj(np.transpose(X, (0, 2, 1))), rtol=1e-12, atol=0)
        ev = np.linalg.eigvalsh(X)
        assert (ev.min(axis=1) >= -1e-10 * ev.max(axis=1)).all()
    return d, out


def check_head(process):
    import torch
    import decomp_head as dh
    z = np.load(os.path.join(BASE, f"{process}_pool.decomp.npz"))
    M = z["stored"]
    lm = np.log(M)
    m, sd = lm.mean(), lm.std()
    y0 = (lm - m) / sd
    worst = {}
    for basis, X, K in (("colour", z["Xc"], z["G"].astype(complex)), ("diagram", z["Xd"], z["K"])):
        nb = X.shape[1]
        logc = np.log(np.trace(K).real)
        y = torch.as_tensor(np.concatenate([y0[:, None], dh.features(X, M)], 1))
        raw = torch.as_tensor(dh.exact_raw(X, m, logc))
        pid = torch.zeros(len(M), dtype=torch.long)
        Kt = torch.as_tensor((K / np.trace(K).real)[None], dtype=torch.complex128)
        lc, s = torch.tensor([logc], dtype=torch.float64), torch.tensor([sd], dtype=torch.float64)
        yp, lhat, gram = dh.readout(raw, Kt, lc, s, pid, nb)
        e_tot = float((yp[:, 0] - y[:, 0]).abs().max())
        e_parts = float(dh.parts_loss(lhat, gram, y, s, pid, nb).max())
        yp2, lh2, g2 = dh.readout(raw + 0.01 * torch.randn_like(raw), Kt, lc, s, pid, nb)
        e_pert = float(dh.parts_loss(lh2, g2, y, s, pid, nb).mean())
        assert e_tot < 1e-10 and e_parts < 1e-20 and e_pert > 1e-6, (basis, e_tot, e_parts, e_pert)
        worst[basis] = (e_tot, e_parts, e_pert)
    return worst


def main():
    if "--head" in sys.argv:
        for p in [a for a in sys.argv[1:] if not a.startswith("--")] or ["ee_uu", "uubar_gg", "ee_uug"]:
            w = check_head(p)
            print(f"PASS head {p}: " + "; ".join(f"{b} max|y_pred - y| {a:.1e}, max parts loss {c:.1e} "
                                                f"(perturbed {d:.1e})" for b, (a, c, d) in w.items()), flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("processes", nargs="*", default=["ee_uu", "uubar_gg", "ee_uug"])
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    for p in a.processes:
        d, out = check(p, a.n)
        print(f"PASS {p}: n_col {d['n_col']}, n_amp {d['n_amp']}; "
              + "; ".join(f"{b} max rel {m:.1e} (median {md:.1e})" for b, (m, md) in out.items()), flush=True)


if __name__ == "__main__":
    main()
