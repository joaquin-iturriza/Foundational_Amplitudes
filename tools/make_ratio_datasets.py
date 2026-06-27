#!/usr/bin/env python3
"""
make_ratio_datasets.py — Build the virtual/born RATIO amplitude datasets.

The full e+e- -> uu / ttbar data lives in data/eeuu.dat and data/eett.dat, with
columns:

    id  muR2  p1E p1px p1py p1pz  p2E..p2pz  p3E..p3pz  p4E..p4pz  born  virt_fin

This writes, for each process, a dataset in the standard 21-column amplitude
format used by experiment.py:

    [ p1E..p4pz (16) ,  PDG ids (4) ,  amplitude (1) ]

with amplitude = virt_fin / born  (the "ratio form": virtual correction divided
by the Born amplitude).  Momenta and PDG ids are taken to match the existing
virtual datasets (ee_*_nlo_virt_e4.npy) exactly, against which we sanity-check.

The ratio can be negative, so downstream preprocessing must use the signed-log
('signedlog') transform — experiment.py now swaps 'log'->'signedlog' automatically
when a dataset has non-positive amplitudes (see preprocessing.resolve_amp_trafos).

Run on Jean Zay (reads two 484 MB text files):
    python make_ratio_datasets.py
"""

import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# process: (dat file, PDG ids, output basename, existing virt npy for cross-check)
PROCESSES = {
    "eeuu": ("eeuu.dat", [11, -11, 2, -2], "ee_uu_nlo_virt_ratio",   "ee_uu_nlo_virt_e4.npy"),
    "eett": ("eett.dat", [11, -11, 6, -6], "ee_ttbar_nlo_virt_ratio","ee_ttbar_nlo_virt_e4.npy"),
}

# .dat column indices (0-based, after the leading '# id muR2' header is skipped)
COL_MOM   = slice(2, 18)   # p1E..p4pz  (16 cols)
COL_BORN  = 18
COL_VIRT  = 19


def build(proc, dat_name, pdg, out_base, virt_npy):
    dat_path = os.path.join(DATA, dat_name)
    print(f"\n[{proc}] loading {dat_path} ...", flush=True)
    raw = np.loadtxt(dat_path, comments="#")          # (N, 20)
    if raw.ndim != 2 or raw.shape[1] < 20:
        raise SystemExit(f"unexpected shape {raw.shape} for {dat_path}")
    N = raw.shape[0]

    momenta = raw[:, COL_MOM]                          # (N, 16)
    born    = raw[:, COL_BORN]
    virt    = raw[:, COL_VIRT]

    # Born is |M|^2 >= 0; drop any non-positive (would make the ratio ill-defined).
    bad = born <= 0
    if bad.any():
        print(f"[{proc}] WARNING: {bad.sum()} rows with born<=0 dropped.")
        keep = ~bad
        momenta, born, virt = momenta[keep], born[keep], virt[keep]
        N = keep.sum()

    ratio = virt / born                                # (N,)

    pdg_block = np.tile(np.array(pdg, dtype=raw.dtype), (N, 1))   # (N, 4)
    out = np.concatenate([momenta, pdg_block, ratio[:, None]], axis=1)  # (N, 21)

    # ── sanity cross-check against the existing virtual dataset ───────────────
    vpath = os.path.join(DATA, virt_npy)
    if os.path.exists(vpath):
        v = np.load(vpath, mmap_mode="r")
        m = min(len(v), 2000)
        mom_match = np.allclose(np.asarray(v[:m, :16]), momenta[:m], rtol=1e-6, atol=1e-6)
        pdg_match = np.array_equal(np.asarray(v[0, 16:20]), np.array(pdg, dtype=v.dtype))
        # existing virt amp == virt_fin * const  -> ratio_of_amps should be constant
        const = np.asarray(v[:m, -1]) / virt[:m]
        const_ok = np.allclose(const, const[0], rtol=1e-4)
        print(f"[{proc}] cross-check: momenta_match={mom_match}  pdg_match={pdg_match}  "
              f"virt_const_stable={const_ok} (const≈{const[0]:.6g})")
        if not (mom_match and pdg_match):
            print(f"[{proc}] !! row alignment/layout mismatch vs {virt_npy} — inspect before use.")
    else:
        print(f"[{proc}] (no existing {virt_npy} to cross-check against)")

    out_path = os.path.join(DATA, out_base + ".npy")
    np.save(out_path, out)
    print(f"[{proc}] wrote {out_path}  shape={out.shape}  "
          f"ratio: min={ratio.min():.4g} max={ratio.max():.4g} "
          f"frac<0={np.mean(ratio < 0):.4f}")


def main():
    for proc, (dat, pdg, out_base, virt_npy) in PROCESSES.items():
        build(proc, dat, pdg, out_base, virt_npy)
    print("\nDone. New datasets:")
    for _, (_, _, out_base, _) in PROCESSES.items():
        print(f"  {out_base}")


if __name__ == "__main__":
    main()
