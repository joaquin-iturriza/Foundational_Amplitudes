#!/usr/bin/env python3
r"""
nlo_virtual_pipeline.py — generate NLO QCD *virtual* datasets with MadGraph in
the locked convention (tools/nlo_conventions.py), pole-certified per process.

Mirrors the LO pipeline (mg5_pipeline_final.py) but for the one-loop virtual:
  build_virt_standalone(proc)   -> $WORK/mg5amcnlo/<proc>_virt_standalone
                                   ( [virt=QCD] standalone + GET_ME_FULL f2py
                                     wrapper + locked param_card + matrix2py.so )
  pole_certify(proc)            -> universal IR-pole check (nlo_pole_check)
  generate_virt_dataset(proc..) -> 21-col virt_e4 dataset by phase-space sampling

Amplitude column stored (matches the existing *_nlo_virt_e4.npy semantics):
    virt_e4 = ( c0 + mass_scheme_shift ) * born_MG
            = finite_MG / (alpha_s/2pi) + shift * born_MG
i.e. the absolute one-loop finite part (NO alpha_s prefactor), MadGraph couplings,
with the heavy-quark pole->references mass-scheme shift applied for massive finals.

Reuses mg5_pipeline_final for: RAMBO sampler, alpha_s, MG5_BIN/WORK_DIR.
CPU-only (login-node safe). Build is a one-time cost per process.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import mg5_pipeline_final as mg          # RAMBO sampler, alpha_s, paths
import nlo_conventions as C
import nlo_pole_check as PC

# ---------------------------------------------------------------------------
# Process table (extend freely; mass>0 final quarks get the mass-scheme shift)
# ---------------------------------------------------------------------------
VIRT_PROCESSES = {
    "ee_uu":    {"mg5": "generate e+ e- > u u~ [virt=QCD]", "pdg_ids": [11, -11,  2, -2], "m_finals": [0.0, 0.0]},
    "ee_dd":    {"mg5": "generate e+ e- > d d~ [virt=QCD]", "pdg_ids": [11, -11,  1, -1], "m_finals": [0.0, 0.0]},
    "ee_ss":    {"mg5": "generate e+ e- > s s~ [virt=QCD]", "pdg_ids": [11, -11,  3, -3], "m_finals": [0.0, 0.0]},
    "ee_cc":    {"mg5": "generate e+ e- > c c~ [virt=QCD]", "pdg_ids": [11, -11,  4, -4], "m_finals": [0.0, 0.0]},
    "ee_ttbar": {"mg5": "generate e+ e- > t t~ [virt=QCD]", "pdg_ids": [11, -11,  6, -6], "m_finals": [172.5, 172.5]},
}

# Fortran subroutine appended to each standalone's f2py_wrapper.f so the f2py
# module exposes the full Laurent array (born, finite, 1/eps, 1/eps^2).
GET_ME_FULL_F = r"""

      SUBROUTINE GET_ME_FULL(P, ALPHAS, MU_R2, NHEL, ANS, RETURNCODE)
C     ANS(0)=Born  ANS(1)=finite  ANS(2)=1/eps  ANS(3)=1/eps^2 ; MU_R2 = mu_R^2.
      IMPLICIT NONE
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      REAL*8 PMASS(NEXTERNAL)
      INCLUDE 'ngraphs.inc'
      INCLUDE 'nsquaredSO.inc'
      INTEGER I
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER MATELEM_ARRAY_DIM
      REAL*8 , ALLOCATABLE :: MATELEM(:,:)
      INTEGER RETURNCODE
      INTEGER NSQUAREDSO_LOOP
      REAL*8 , ALLOCATABLE :: PREC_FOUND(:)
      DOUBLE PRECISION ANS(0:3)
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS, MU_R2
CF2PY INTENT(OUT) :: ANS(0:3)
CF2PY INTENT(OUT) :: RETURNCODE
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: MU_R2
      INTEGER NLOOPCHOSEN
      CHARACTER*20 CHOSEN_LOOP_SO_INDICES(NSQUAREDSO)
      LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/ML5_0_CHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS

      CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)
      CALL ML5_0_GET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
      ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
      CALL ML5_0_GET_NSQSO_LOOP(NSQUAREDSO_LOOP)
      ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))
      INCLUDE 'pmass.inc'
      NLOOPCHOSEN=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLOOPCHOSEN=NLOOPCHOSEN+1
          WRITE(CHOSEN_LOOP_SO_INDICES(NLOOPCHOSEN),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO
      CALL UPDATE_AS_PARAM2(MU_R2, ALPHAS)
      CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,-1.0D0, PREC_FOUND,
     $  RETURNCODE)
      DO I=0,3
        ANS(I) = MATELEM(I,0)
      ENDDO
      DEALLOCATE(MATELEM)
      DEALLOCATE(PREC_FOUND)
      END
"""


def virt_standalone_dir(process):
    return f"{mg.WORK_DIR}/{process}_virt_standalone"


def find_p0(standalone_dir):
    hits = sorted(glob.glob(f"{standalone_dir}/SubProcesses/P0_*"))
    if not hits:
        raise FileNotFoundError(f"no P0_* subprocess in {standalone_dir}")
    return hits[0]


def patch_param_card_slha(card_path, patches):
    """Patch an SLHA param_card: the value is the last token before the '# NAME'
    comment; NAME is matched case-insensitively against the patch keys."""
    want = {k.lower(): v for k, v in patches.items()}
    out, done = [], set()
    for line in open(card_path):
        if "#" in line and line.split("#", 1)[0].strip():
            code, comment = line.split("#", 1)
            name = comment.strip().split()[0].lower() if comment.strip() else ""
            if name in want and name not in done:
                toks = code.rstrip("\n").rstrip().split()
                toks[-1] = f"{want[name]:.6e}"
                out.append(" " + " ".join(toks) + " #" + comment)
                done.add(name)
                continue
        out.append(line)
    open(card_path, "w").writelines(out)
    missing = set(want) - done
    if missing:
        print(f"  [WARN] param_card keys not found: {sorted(missing)}")


def build_virt_standalone(process, force=False):
    """Generate the [virt=QCD] standalone, inject GET_ME_FULL, patch the
    param_card to the locked convention, and build matrix2py.so. Idempotent."""
    cfg = VIRT_PROCESSES[process]
    sa = virt_standalone_dir(process)
    p0_glob = glob.glob(f"{sa}/SubProcesses/P0_*/matrix2py*.so")
    if p0_glob and not force:
        print(f"[BUILD] {process}: standalone+module exist ({sa}) — skipping.")
        return sa
    if force and os.path.exists(sa):
        import shutil
        shutil.rmtree(sa)

    print(f"[BUILD] {process}: generating [virt=QCD] standalone at {sa}")
    script = f"import model loop_sm\n{cfg['mg5']}\noutput standalone {sa}\n"
    r = subprocess.run([mg.MG5_BIN], input=script, text=True)
    if r.returncode != 0 or not os.path.isdir(sa):
        sys.exit(f"[BUILD] MG5 generation failed for {process}")

    p0 = find_p0(sa)
    wrapper = f"{p0}/f2py_wrapper.f"
    if "GET_ME_FULL" not in open(wrapper).read():
        with open(wrapper, "a") as f:
            f.write(GET_ME_FULL_F)
    patch_param_card_slha(f"{sa}/Cards/param_card.dat", C.PARAM_CARD_PATCHES)

    print(f"[BUILD] {process}: compiling matrix2py.so ...")
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    log = f"{p0}/build_f2py.log"
    with open(log, "w") as lf:
        rc = subprocess.run(["make", "matrix2py.so"], cwd=p0, env=env,
                            stdout=lf, stderr=lf).returncode
    if rc != 0 or not glob.glob(f"{p0}/matrix2py*.so"):
        sys.exit(f"[BUILD] matrix2py.so build failed for {process} (see {log})")
    print(f"[BUILD] {process}: ready.")
    return sa


def pole_certify(process, n=100, seed=7):
    """Run the universal pole check on the process's standalone (separate
    Python process: the f2py module chdir's and is a singleton)."""
    cfg = VIRT_PROCESSES[process]
    p0 = find_p0(virt_standalone_dir(process))
    proc_order = [cfg["pdg_ids"][1], cfg["pdg_ids"][0]] + cfg["pdg_ids"][2:]  # e+ e- ...
    masses = [0.0, 0.0] + list(cfg["m_finals"])
    cmd = [sys.executable, f"{HERE}/nlo_pole_check.py", "--so-dir", p0,
           "--proc-order", *map(str, proc_order), "--m", *map(str, masses),
           "--n", str(n), "--seed", str(seed)]
    print(f"[CERTIFY] {process}: pole check ...")
    out = subprocess.run(cmd, capture_output=True, text=True,
                         env=dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")).stdout
    tail = [l for l in out.splitlines() if any(k in l for k in
            ("DOUBLE", "SINGLE", "predicted", "MadLoop", "PASS", "FAIL", "not predicted"))]
    print("  " + "\n  ".join(tail))
    return "FAIL" not in out


def generate_virt_dataset(process, sqrts_min, sqrts_max, n_events, out_file,
                          seed=42, mass_shift=True, alphas_mz=0.118):
    """Sample phase space, evaluate the virtual, apply the heavy-quark mass
    shift, and write a 21-col virt_e4 dataset (momenta in [e-,e+,finals] order)."""
    import nlo_madloop as ML
    cfg = VIRT_PROCESSES[process]
    pdg = cfg["pdg_ids"]
    m_finals = cfg["m_finals"]
    nfinal = len(m_finals)
    npart = nfinal + 2

    rng = np.random.default_rng(seed)
    events, sqrts = mg.sample_nbody_phase_space(
        n_events, sqrts_min, sqrts_max, m_finals, pdg, rng=rng)
    # heavy final quarks (for the mass-scheme shift): unique nonzero masses
    heavy_m = next((m for m in m_finals if m and m > 0), 0.0)

    p0 = find_p0(virt_standalone_dir(process))
    get_me_full = ML.load(p0)   # chdir into p0
    swap = [1, 0] + list(range(2, npart))   # [e-,e+,..] -> MadGraph [e+,e-,..]

    mom_store = np.empty((n_events, npart * 4))
    amp = np.empty(n_events)
    bad = 0
    for i, (mom, _) in enumerate(events):
        r = ML.evaluate(get_me_full, mom[swap], alphas=alphas_mz)
        born, c0, s = r["born"], r.get("c0"), r["s"]
        if c0 is None:
            bad += 1; born, c0 = 0.0, 0.0
        shift = C.heavy_quark_scheme_shift(s, heavy_m) if mass_shift else 0.0
        amp[i] = (c0 + shift) * born            # virt_e4 (absolute, no alpha_s)
        mom_store[i] = mom.flatten()            # store [e-,e+,finals] order
        if (i + 1) % 50_000 == 0:
            print(f"  [AMP] {i+1:,}/{n_events:,}", flush=True)

    pdg_block = np.tile(np.array(pdg, float), (n_events, 1))
    arr = np.concatenate([mom_store, pdg_block, amp[:, None]], axis=1)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    np.save(out_file, arr)
    print(f"[DATA] {process}: saved {arr.shape} -> {out_file}  "
          f"virt_e4 in [{amp.min():.3e},{amp.max():.3e}]  bad={bad}  "
          f"mass_shift={'on(m=%.1f)'%heavy_m if (mass_shift and heavy_m) else 'off'}")
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("process", choices=list(VIRT_PROCESSES))
    ap.add_argument("--build", action="store_true", help="build the standalone")
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--certify", action="store_true", help="run the pole check")
    ap.add_argument("--generate", action="store_true", help="generate a dataset")
    ap.add_argument("--sqrts-min", type=float, default=None)
    ap.add_argument("--sqrts-max", type=float, default=1000.0)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-mass-shift", action="store_true")
    args = ap.parse_args()

    if args.build or args.force_build:
        build_virt_standalone(args.process, force=args.force_build)
    if args.certify:
        ok = pole_certify(args.process)
        print(f"[CERTIFY] {args.process}: {'PASS' if ok else 'FAIL'}")
    if args.generate:
        cfg = VIRT_PROCESSES[args.process]
        smin = args.sqrts_min
        if smin is None:
            smin = 1.05 * sum(cfg["m_finals"]) if sum(cfg["m_finals"]) > 0 else 50.0
        out = args.out or f"{mg.OUTPUT_DIR}/{args.process}_nlo_virt_e4_{smin:.0f}-{args.sqrts_max:.0f}GeV.npy"
        generate_virt_dataset(args.process, smin, args.sqrts_max, args.n, out,
                              mass_shift=not args.no_mass_shift)


if __name__ == "__main__":
    main()
