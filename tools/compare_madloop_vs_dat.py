#!/usr/bin/env python3
"""
compare_madloop_vs_dat.py
=========================
Point-by-point consistency check of MadGraph/MadLoop virtual finite parts
against the collaborators' GoSam+Sherpa reference (.dat files).

For each phase-space point in eeuu.dat / eett.dat we:
  - feed the EXACT momenta into MadLoop (the f2py module matrix2py.so built from
    the [virt=QCD] standalone) at mu_R^2 = s  (the .dat convention),
  - read the full Laurent array  ANS(0:3) = (Born, finite, 1/eps, 1/eps^2),
    where MadLoop's finite/1eps/2eps already include the alpha_s/(2*pi) factor,
  - form  R_ML = finite / Born / (alpha_s/2pi),  the SAME quantity the .dat stores
    as  R_dat = virt_fin / born,
  - compare R_ML vs R_dat point-by-point (this is coupling-independent, so the EW
    e^4 prefactor and the exact alpha_s value cancel and need not be matched).

The IR poles are also reported as a scheme cross-check:
    eeuu (massless q):  2eps/born/ao2pi = -2 C_F = -2.667,
                        1eps/born/ao2pi = -3 C_F = -4.000,
                        finite          =  C_F(pi^2-8) = 2.4928 (kinematics-indep.)
    eett (massive q):   double pole -> 0 (no collinear sing.), single soft pole only.

Usage (on Jean Zay, login node is fine -- MadLoop is CPU-only):
    python tools/compare_madloop_vs_dat.py \
        --so-dir $WORK/nlo_consistency/ee_uu_virt_SA/SubProcesses/P0_epem_uux \
        --dat    data/eeuu.dat \
        --dat-order 11 -11 2 -2 \
        --proc-order -11 11 2 -2 \
        --n 500
"""

import argparse
import os
import sys
import numpy as np

# .dat columns: id  muR2  p1E..p4pz(16)  born  virt_fin
COL_MUR2 = 1
COL_MOM  = slice(2, 18)
COL_BORN = 18
COL_VIRT = 19


def load_module(so_dir):
    """Import the matrix2py f2py module living in <so_dir> and initialise it."""
    import glob
    so_dir = os.path.abspath(so_dir)
    if not glob.glob(os.path.join(so_dir, "matrix2py*.so")):
        sys.exit(f"matrix2py*.so not found in {so_dir} -- run `make matrix2py.so` there first.")
    sys.path.insert(0, so_dir)
    # MadLoop resolves MadLoop5_resources / ident_card relative to CWD -> run from so_dir
    os.chdir(so_dir)
    import matrix2py  # noqa: E402
    param_card = os.path.abspath(os.path.join(so_dir, os.pardir, os.pardir, "Cards", "param_card.dat"))
    init = getattr(matrix2py, "ml5_0_initialise", None) or getattr(matrix2py, "initialise")
    init(param_card)
    full = getattr(matrix2py, "ml5_0_get_me_full", None) or getattr(matrix2py, "get_me_full")
    return matrix2py, full


def reorder(mom4, dat_order, proc_order):
    """Permute a (4particles,4) momenta block from .dat ordering to MadGraph's
    process ordering, by matching signed PDG ids."""
    dat_order = list(dat_order)
    idx = [dat_order.index(p) for p in proc_order]
    return mom4[idx, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-dir", required=True, help="P0_* dir containing matrix2py.so")
    ap.add_argument("--dat", required=True, help=".dat reference file")
    ap.add_argument("--dat-order", type=int, nargs=4, required=True,
                    help="signed PDG ids in .dat momentum order, e.g. 11 -11 2 -2")
    ap.add_argument("--proc-order", type=int, nargs=4, required=True,
                    help="signed PDG ids in MadGraph process order, e.g. -11 11 2 -2")
    ap.add_argument("--n", type=int, default=500, help="number of points to test")
    ap.add_argument("--alphas", type=float, default=0.118, help="alpha_s (cancels in the ratio)")
    ap.add_argument("--dump", type=int, default=0, help="print per-point detail for first DUMP points")
    ap.add_argument("--csv", default=None, help="write per-point CSV here (sqrts,costh,born_dat,born_ML,virt_dat,fin_ML,R_dat,R_ML,e1,e2)")
    args = ap.parse_args()
    args.dat = os.path.abspath(args.dat)   # resolve before we chdir into so_dir

    matrix2py, get_me_full = load_module(args.so_dir)
    ao2pi = args.alphas / (2.0 * np.pi)

    raw = np.loadtxt(args.dat, comments="#", max_rows=args.n + 1)
    raw = raw[:args.n]
    n = raw.shape[0]
    print(f"Loaded {n} points from {args.dat}")
    print(f"dat order {args.dat_order} -> proc order {args.proc_order}\n")

    R_ml   = np.full(n, np.nan)
    R_dat  = raw[:, COL_VIRT] / raw[:, COL_BORN]
    sgl    = np.full(n, np.nan)
    dbl    = np.full(n, np.nan)
    born_ratio = np.full(n, np.nan)   # MadLoop Born / dat born (coupling-dep, just informational)
    rc_bad = 0
    csv_rows = []

    if args.dump:
        print(f"{'i':>3} {'sqrt_s':>8} {'cos_th':>7} {'born_dat':>11} {'born_ML':>11} "
              f"{'bML/bdat':>9} {'R_dat':>9} {'R_ML':>9} {'1eps_ML':>8} {'rc':>3}")

    for i in range(n):
        mom4_dat = raw[i, COL_MOM].reshape(4, 4)        # (particle, [E,px,py,pz]) in .dat order
        mom4 = reorder(mom4_dat, args.dat_order, args.proc_order)
        P = np.asfortranarray(mom4.T)                   # (0:3, nexternal)
        pin = mom4[0] + mom4[1]
        s = pin[0]**2 - pin[1]**2 - pin[2]**2 - pin[3]**2
        ans, rc = get_me_full(P, args.alphas, s, -1)
        born, fin, e1, e2 = ans[0], ans[1], ans[2], ans[3]
        if born == 0.0:
            continue
        R_ml[i]  = fin / born / ao2pi
        sgl[i]   = e1 / born / ao2pi
        dbl[i]   = e2 / born / ao2pi
        born_ratio[i] = born / raw[i, COL_BORN]
        if rc not in (1,):   # MadLoop return code 1 == stable/accurate
            rc_bad += 1
        # cos(theta) between incoming particle 1 (.dat p1) and outgoing particle 3 (.dat p3)
        p_in1 = mom4_dat[0, 1:]
        p_out = mom4_dat[2, 1:]
        cth = np.dot(p_in1, p_out) / (np.linalg.norm(p_in1) * np.linalg.norm(p_out))
        if args.csv is not None:
            csv_rows.append((np.sqrt(s), cth, raw[i, COL_BORN], born,
                             raw[i, COL_VIRT], fin, R_dat[i], R_ml[i], sgl[i], dbl[i]))
        if i < args.dump:
            print(f"{i:3d} {np.sqrt(s):8.2f} {cth:7.3f} {raw[i,COL_BORN]:11.4e} {born:11.4e} "
                  f"{born/raw[i,COL_BORN]:9.4e} {R_dat[i]:9.4f} {R_ml[i]:9.4f} {sgl[i]:8.3f} {rc:3d}")

    good = np.isfinite(R_ml) & np.isfinite(R_dat)
    rel = np.abs(R_ml[good] - R_dat[good]) / np.abs(R_dat[good])

    print("=== finite-part ratio  R = virt_fin/born  (MadLoop vs .dat) ===")
    print(f"  points compared : {good.sum()}   (MadLoop rc!=1 on {rc_bad})")
    print(f"  R_dat  mean      : {R_dat[good].mean():.6f}")
    print(f"  R_ML   mean      : {R_ml[good].mean():.6f}")
    print(f"  max  rel. diff   : {rel.max():.3e}")
    print(f"  mean rel. diff   : {rel.mean():.3e}")
    print(f"  median rel. diff : {np.median(rel):.3e}")
    print()
    print("=== IR poles (MadLoop, /born/ao2pi) -- scheme cross-check ===")
    print(f"  single 1/eps : mean {np.nanmean(sgl):.6f}  std {np.nanstd(sgl):.3e}")
    print(f"  double 1/eps2: mean {np.nanmean(dbl):.6f}  std {np.nanstd(dbl):.3e}")
    print()
    print("=== MadLoop Born / .dat born (should be a constant = e^4 coupling factor) ===")
    br = born_ratio[np.isfinite(born_ratio)]
    print(f"  mean {br.mean():.6e}   std/mean {br.std()/br.mean():.3e}   1/mean {1/br.mean():.4f}")

    if args.csv is not None and csv_rows:
        hdr = "sqrts,costh,born_dat,born_ML,virt_dat,fin_ML,R_dat,R_ML,e1,e2"
        np.savetxt(args.csv, np.array(csv_rows), delimiter=",", header=hdr, comments="")
        print(f"\nwrote {len(csv_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
