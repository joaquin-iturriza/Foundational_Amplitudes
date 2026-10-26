#!/usr/bin/env python3
r"""
mixed_order_virt_check.py — why catalog_v4's one-loop entries with a mixed-order born fail the pole check (CPU).

catalog_v4 generates the one-loop entries with the trees' full coupling orders (QED<=n). Eight fail
tools/nlo_pole_check.py where catalog_v3 (pure-QCD born, no QED order) passed:
  u u~ > c c~ z, d d~ > b b~ z  (QED<=3)   double pole identically 0 against -16/3
  d d~ > c c~, d d~ > b b~      (QED<=2)   exact at 95% of points, a few rc 236 points off by up to 25%
  b b~ > c c~                   (QED<=2)   exact at 95%, 12% off at one rc 236 point
  b g > b w+ w-, b b~ > ta+ vt w-, b b~ > h w+ w-   all points rc 216, the double pole off by ~4e-3
This builds, per process, MadLoop standalones of the same born in several forms and reads MadLoop's answer per
squared split order (MATELEM(0:3, k): k = 0 the sum, k >= 1 each squared order), so the poles of each piece can be
compared with the born of the same piece (the universal double pole is -sum C_i times the born of the SAME order):

  v4        <born> QED<=n [virt=QCD]                (catalog_v4, reproduces the failure)
  v3        <born> [virt=QCD]                       (catalog_v3: MadGraph's minimal QED, pure-QCD born)
  sqK       <born> QED<=n QED^2==K [virt=QCD]       (one squared order of the born x loop interference at a time:
                                                     the QCD-only, the mixed and the EW-only born pieces)
  sqrvirt   <born> QED<=n [sqrvirt=QCD]             (the loop squared, no born interference)
  qcdqed    <born> QED<=n [virt=QCD QED]            (loop_qcd_qed_sm: the QCD and EW loops of every order)
  amcnlo    <born> QED<=n [QCD], aMC@NLO output, check_poles  (MadGraph's own pole check against its FKS
                                                     integrated counterterms, colour correlations included)
  top173    v4's string with catalog_v3's internal top (MT = ymt = 173, WT 1.4915), for the three sub-percent ones

The five mixed-order borns are generated at the pure-QCD born in catalog_v4 now (v4_<x>_qcdborn, the user 2026-10-26);
the second run (MOV_ARGS="--only v4_bg_bwpwm v4_bbx_tapvtwm v4_bbx_hwpwm") is the three sub-percent ones with every
variant. check_poles is the Fortran test `compile FO` builds per SubProcesses/P*/ (not an aMC@NLO shell command).

plus a stability scan on the v4 standalone of the processes with outlier points (MadLoopParams.dat: the default,
MLStabThres 1e-5, CTModeRun 4 = quadruple precision everywhere, MLReductionLib 1 / 6 / 7 alone): a point whose poles
move between them was a precision failure, not a definition one. MadGraph's generation output is kept per variant
and every WARNING line is printed (does it say the mixed-order virtual is incomplete for this squared order?).

Everything is built under $SCRATCH/zz_mixed_order_virt (nothing of the production backends is touched). Results:
$SCRATCH/zz_mixed_order_virt/results.json; the log lines are prefixed MOV.

    python tools/mixed_order_virt_check.py run --workers 8
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_mixed_order_virt")
MAXSO = 20
N_POLES = 50     # points of aMC@NLO's check_poles rerun

# the five failing entries: born string, the QED order of catalog_v4, the squared QED orders of the born pieces
PROCS = {
    "v4_uux_ccxz": {"born": "u u~ > c c~ z", "qed": 3, "sq": [2, 4, 6], "stab": False},
    "v4_ddx_bbxz": {"born": "d d~ > b b~ z", "qed": 3, "sq": [2, 4, 6], "stab": False},
    "v4_ddx_ccx":  {"born": "d d~ > c c~",   "qed": 2, "sq": [0, 2, 4], "stab": True},
    "v4_ddx_bbx":  {"born": "d d~ > b b~",   "qed": 2, "sq": [0, 2, 4], "stab": True},
    "v4_bbx_ccx":  {"born": "b b~ > c c~",   "qed": 2, "sq": [0, 2, 4], "stab": True},
    # the sub-percent rc 216 pattern (all points "stable" in double precision, the double pole off by ~4e-3 in every
    # MadLoopParams setting of the first run, so not precision): single-order borns (QED<=n is their minimal order), so
    # the split-order variant is the one squared order; the top variant puts back catalog_v3's internal top (MT = ymt =
    # 173, WT 1.4915), the one other change from v3, where they passed (with a 4.7 GeV b); aMC@NLO's check_poles
    # compares MadLoop's poles with its own FKS prediction per squared order
    "v4_bg_bwpwm":    {"born": "b g > b w+ w-",    "qed": 2, "sq": [4], "stab": False, "top173": True},
    "v4_bbx_tapvtwm": {"born": "b b~ > ta+ vt w-", "qed": 3, "sq": [6], "stab": False, "top173": True},
    "v4_bbx_hwpwm":   {"born": "b b~ > h w+ w-",   "qed": 3, "sq": [6], "stab": False, "top173": True},
}
V3_TOP = {"MT": 173.0, "ymt": 173.0, "WT": 1.4915}     # catalog_v3's internal top (CLAUDE.md, masses and widths)

STAB = {   # MadLoopParams.dat key -> value, per configuration of the stability scan
    "default": {},
    "thres1e-5": {"#MLStabThres": "1.000000d-05"},
    "quad": {"#CTModeRun": "4"},
    "red_cuttools": {"#MLReductionLib": "1"},
    "red_ninja": {"#MLReductionLib": "6"},
    "red_collier": {"#MLReductionLib": "7"},
}

GET_ME_SPLIT_F = r"""

      SUBROUTINE GET_ME_SPLIT(P, ALPHAS, MU_R2, ANSO, NDIM, NSOL,
     $  CHOSEN, RETURNCODE)
C     ANSO(0:3,K): K=0 the sum, K>=1 each squared split order (born, finite, 1/eps, 1/eps^2)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'nsquaredSO.inc'
      INTEGER I, K, NDIM, NSOL, RETURNCODE
      REAL*8 P(0:3,NEXTERNAL)
      DOUBLE PRECISION ALPHAS, MU_R2
      DOUBLE PRECISION ANSO(0:3,0:20)
      INTEGER CHOSEN(20)
      REAL*8 , ALLOCATABLE :: MATELEM(:,:)
      REAL*8 , ALLOCATABLE :: PREC_FOUND(:)
      LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/ML5_0_CHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: MU_R2
CF2PY INTENT(OUT) :: ANSO(0:3,0:20)
CF2PY INTENT(OUT) :: NDIM
CF2PY INTENT(OUT) :: NSOL
CF2PY INTENT(OUT) :: CHOSEN(20)
CF2PY INTENT(OUT) :: RETURNCODE
      CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)
      CALL ML5_0_GET_ANSWER_DIMENSION(NDIM)
      ALLOCATE(MATELEM(0:3,0:NDIM))
      CALL ML5_0_GET_NSQSO_LOOP(NSOL)
      ALLOCATE(PREC_FOUND(0:NSOL))
      DO K=0,20
        DO I=0,3
          ANSO(I,K) = 0D0
        ENDDO
      ENDDO
      DO I=1,20
        CHOSEN(I) = -1
      ENDDO
      DO I=1,MIN(NSQUAREDSO,20)
        CHOSEN(I) = 0
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) CHOSEN(I) = 1
      ENDDO
      CALL UPDATE_AS_PARAM2(MU_R2, ALPHAS)
      CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,-1.0D0,PREC_FOUND,
     $  RETURNCODE)
      DO K=0,MIN(NDIM,20)
        DO I=0,3
          ANSO(I,K) = MATELEM(I,K)
        ENDDO
      ENDDO
      DEALLOCATE(MATELEM)
      DEALLOCATE(PREC_FOUND)
      END
"""


def log(msg):
    print(f"MOV {msg}", flush=True)


def variants(name, P):
    b, n = P["born"], P["qed"]
    v = {"v4": (f"generate {b} QED<={n} [virt=QCD]", "loop_sm-no_b_mass")}
    if P.get("only_v4"):
        return v
    v["v3"] = (f"generate {b} [virt=QCD]", "loop_sm-no_b_mass")
    for k in P["sq"]:
        v[f"sq{k}"] = (f"generate {b} QED<={n} QED^2=={k} [virt=QCD]", "loop_sm-no_b_mass")
    v["sqrvirt"] = (f"generate {b} QED<={n} [sqrvirt=QCD]", "loop_sm-no_b_mass")
    v["qcdqed"] = (f"generate {b} QED<={n} [virt=QCD QED]", qcdqed_model())
    if P.get("top173"):
        v["top173"] = (f"generate {b} QED<={n} [virt=QCD]", "loop_sm-no_b_mass", V3_TOP)
    return v


def qcdqed_model():
    import mg5_pipeline_final as mg
    mdir = os.path.join(os.path.dirname(os.path.dirname(mg.MG5_BIN)), "models", "loop_qcd_qed_sm")
    return "loop_qcd_qed_sm-no_b_mass" if os.path.exists(os.path.join(mdir, "restrict_no_b_mass.dat")) \
        else "loop_qcd_qed_sm"


def warnings_of(text):
    return [l.strip() for l in text.splitlines()
            if re.search(r"WARNING|not supported|cannot|Error|error|mixed|squared order", l)][:40]


def build(name, var, string, model, patches=None):
    """MG5 standalone + the split-order wrapper + the locked param card (and the variant's patches) + matrix2py.so;
    returns a status dict"""
    import mg5_pipeline_final as mg
    import nlo_conventions as C
    import nlo_virtual_pipeline as NV
    sa = os.path.join(OUT, f"{name}__{var}")
    st = {"proc": name, "var": var, "mg5": string, "model": model, "dir": sa, "patches": patches or {}}
    done = glob.glob(f"{sa}/SubProcesses/P0_*/matrix2py*.so")
    if done and open(os.path.join(sa, ".mg5_string")).read().strip() == string:
        st["built"] = True
        st["warnings"] = json.load(open(os.path.join(sa, ".warnings.json")))
        return st
    t0 = time.time()
    if os.path.isdir(sa) and os.path.basename(sa).startswith(name + "__"):   # our own earlier attempt only
        import shutil
        shutil.rmtree(sa)
    script = f"import model {model}\n{string}\noutput standalone {sa}\n"
    r = subprocess.run([mg.MG5_BIN], input=script, text=True, capture_output=True, timeout=5400)
    txt = r.stdout + r.stderr
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"{name}__{var}.mg5.log"), "w").write(txt)
    st["warnings"] = warnings_of(txt)
    p0s = sorted(glob.glob(f"{sa}/SubProcesses/P0_*"))
    if not p0s:
        st.update(built=False, why="MG5 produced no P0_* (see the .mg5.log)", tail=txt.strip().splitlines()[-15:])
        return st
    open(os.path.join(sa, ".mg5_string"), "w").write(string + "\n")
    json.dump(st["warnings"], open(os.path.join(sa, ".warnings.json"), "w"))
    p0 = p0s[0]
    with open(f"{p0}/f2py_wrapper.f", "a") as f:
        f.write(NV.GET_ME_FULL_F + GET_ME_SPLIT_F)
    NV.patch_param_card_slha(f"{sa}/Cards/param_card.dat", {**C.PARAM_CARD_PATCHES, "WT": 1.480327, **(patches or {})})
    # the squared orders MadGraph wrote into the code, for reading the columns
    so = []
    for fn in sorted(glob.glob(f"{p0}/*.f")):
        for l in open(fn, errors="replace"):
            if re.search(r"(QCD|QED)\^?2?\s*[=<>]|SQSO|squared order", l, re.I) and l.lstrip().upper().startswith("C"):
                so.append(f"{os.path.basename(fn)}: {l.rstrip()}")
    st["so_comments"] = so[:40]
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    f2py = os.path.join(os.path.dirname(sys.executable), "f2py")
    with open(f"{p0}/build_f2py.log", "w") as lf:
        rc = subprocess.run(["make", "matrix2py.so", f"F2PY={f2py}"], cwd=p0, env=env, stdout=lf, stderr=lf).returncode
    # MadGraph's own standalone check (a random point, every squared order printed)
    with open(f"{p0}/check_build.log", "w") as lf:
        subprocess.run(["make", "check"], cwd=p0, env=env, stdout=lf, stderr=lf)
    if os.path.exists(f"{p0}/check"):
        try:
            c = subprocess.run(["./check"], cwd=p0, capture_output=True, text=True, timeout=600)
            st["check_sa"] = [l.rstrip() for l in c.stdout.splitlines() if l.strip()][-60:]
        except subprocess.TimeoutExpired:
            st["check_sa"] = ["timeout"]
    st["built"] = rc == 0 and bool(glob.glob(f"{p0}/matrix2py*.so"))
    st["build_s"] = round(time.time() - t0)
    if not st["built"]:
        st["why"] = "matrix2py.so build failed (build_f2py.log)"
    return st


def set_params(sa, params):
    """write the scan's MadLoopParams.dat values into this variant's own cards (default values restored first)"""
    cards = {os.path.realpath(os.path.join(sa, *p, "MadLoopParams.dat"))
             for p in (("Cards",), ("SubProcesses", "MadLoop5_resources"))}
    for card in sorted(c for c in cards if os.path.isfile(c)):
        bak = card + ".zz_default"
        if not os.path.exists(bak):
            open(bak, "w").write(open(card).read())
        L = open(bak).read().split("\n")
        for i, l in enumerate(L[:-1]):
            if l.strip() in params:
                L[i + 1] = params[l.strip()]
        open(card, "w").write("\n".join(L))


def eval_one(sa, string, pdg, m_finals, n, seed):
    """one subprocess per standalone (the f2py module is a singleton): per point, every column of MATELEM"""
    import mg5_pipeline_final as mg
    import nlo_conventions as C
    import nlo_pole_check as PC
    import nlo_virtual_pipeline as NV
    perm = mg.row_to_slot_perm(pdg, string)
    masses = [0.0, 0.0] + list(m_finals)
    p0 = sorted(glob.glob(f"{sa}/SubProcesses/P0_*"))[0]
    rng = np.random.default_rng(seed)
    if len(masses) == 4:
        sq = rng.uniform(50.0 if sum(masses[2:]) == 0 else 1.05 * sum(masses[2:]), 1000.0, n)
        pts = [PC.sample_2body(s, masses, rng, pdg) for s in sq]
    else:
        ev, _ = mg.sample_nbody_phase_space(n, 1.05 * sum(masses[2:]) or 50.0, 1000.0, masses[2:], pdg, rng=rng,
                                            cuts=mg.FIDUCIAL_CUTS if mg.FIDUCIAL_CUTS_ENABLED else None,
                                            beam_mass={})
        pts = [mom for mom, _ in ev]
    pts = [np.asarray(mom)[perm] for mom in pts]
    slot_pdgs = [pdg[j] for j in perm]
    slot_m = [masses[j] for j in perm]
    with NV._zero_top_width(os.path.join(sa, "Cards", "param_card.dat")):
        sys.path.insert(0, p0)
        os.chdir(p0)
        import matrix2py
        init = getattr(matrix2py, "ml5_0_initialise", None) or getattr(matrix2py, "initialise")
        init(os.path.join(sa, "Cards", "param_card.dat"))
        split = matrix2py.get_me_split
        rows = []
        for mom in pts:
            P = np.asfortranarray(mom.T)
            pin = mom[0] + mom[1]
            s = float(pin[0] ** 2 - pin[1] ** 2 - pin[2] ** 2 - pin[3] ** 2)
            anso, ndim, nsol, chosen, rc = split(P, 0.118, s)
            rows.append({"sqrts": s ** 0.5, "rc": int(rc), "ndim": int(ndim), "nsol": int(nsol),
                         "chosen": [int(c) for c in chosen if c >= 0],
                         "ans": np.asarray(anso)[:, :int(ndim) + 1].T.tolist()})
    c2p = PC.double_pole(slot_pdgs, slot_m)
    return {"c2_pred": c2p, "rows": rows}


def summarise(tag, res):
    """per column k: c2 against its own born and against the total born; the worst points; return codes"""
    ao2pi = 0.118 / (2 * np.pi)
    rows = res["rows"]
    A = np.array([r["ans"] for r in rows])          # (npt, ncol, 4)
    rc = np.array([r["rc"] for r in rows])
    out = {"c2_pred": res["c2_pred"], "rc_counts": {str(k): int(v) for k, v in zip(*np.unique(rc, return_counts=True))},
           "chosen": rows[0]["chosen"], "ncol": A.shape[1], "cols": []}
    born_tot = A[:, 0, 0]
    for k in range(A.shape[1]):
        b, e2, e1, fin = A[:, k, 0], A[:, k, 3], A[:, k, 2], A[:, k, 1]
        col = {"k": k, "born_med": float(np.median(np.abs(b))), "born_over_total": float(np.median(np.abs(b) / np.abs(born_tot))),
               "fin_med": float(np.median(np.abs(fin)))}
        with np.errstate(all="ignore"):
            own = e2 / b / ao2pi
            tot = e2 / born_tot / ao2pi
        good = np.isfinite(own) & (np.abs(b) > 1e-12 * np.abs(born_tot))
        if good.any():
            d = np.abs(own[good] - res["c2_pred"]) / abs(res["c2_pred"] or 1)
            col.update(c2_own_med=float(np.median(own[good])), c2_own_maxrel=float(d.max()),
                       n_bad_1e3=int((d > 1e-3).sum()))
        col.update(c2_tot_med=float(np.nanmedian(tot)), c2_tot_min=float(np.nanmin(tot)), c2_tot_max=float(np.nanmax(tot)))
        out["cols"].append(col)
    # the worst points of the sum against the prediction, with their return codes
    with np.errstate(all="ignore"):
        c2 = A[:, 0, 3] / born_tot / ao2pi
    d = np.abs(c2 - res["c2_pred"]) / abs(res["c2_pred"] or 1)
    out["worst"] = [{"sqrts": rows[i]["sqrts"], "rc": int(rc[i]), "c2": float(c2[i]),
                     "cols_e2_over_born": [float(x) for x in (A[i, :, 3] / np.where(A[i, :, 0] != 0, A[i, :, 0], np.nan) / ao2pi)]}
                    for i in np.argsort(-np.nan_to_num(d, nan=1e9))[:5]]
    out["n_bad_1e3"] = int(np.nansum(d > 1e-3))
    log(f"{tag}: c2_pred {res['c2_pred']:+.5f}  cols {out['ncol']} chosen {out['chosen']}  rc {out['rc_counts']}  "
        f"sum: bad(>1e-3) {out['n_bad_1e3']}/{len(rows)}")
    for c in out["cols"]:
        log(f"{tag}:   col {c['k']}: |born| med {c['born_med']:.3e} ({c['born_over_total']:.3f} of the sum)  "
            f"c2/own born med {c.get('c2_own_med', float('nan')):+.5f} maxrel {c.get('c2_own_maxrel', float('nan')):.2e}  "
            f"c2/total born med {c['c2_tot_med']:+.5f} [{c['c2_tot_min']:+.4f}, {c['c2_tot_max']:+.4f}]")
    for w in out["worst"][:3]:
        log(f"{tag}:   worst sqrt(s) {w['sqrts']:.1f} rc {w['rc']} c2 {w['c2']:+.5f} per col {['%+.4f' % x for x in w['cols_e2_over_born']]}")
    return out


def run_eval(st, name, cfg, params_tag, n, seed):
    set_params(st["dir"], STAB[params_tag])
    cmd = [sys.executable, __file__, "eval", "--dir", st["dir"], "--mg5", st["mg5"], "--n", str(n), "--seed", str(seed),
           "--pdg", *map(str, cfg["pdg_ids"]), "--m", *map(str, cfg["m_finals"])]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400,
                       env=dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib"))
    js = [l for l in r.stdout.splitlines() if l.startswith("EVALJSON ")]
    set_params(st["dir"], {})
    if not js:
        return {"error": (r.stdout[-1500:] + r.stderr[-2500:])}
    return json.loads(js[-1][9:])


def amcnlo_check_poles(name, P):
    """aMC@NLO output of <born> QED<=n [QCD] and MadGraph's check_poles (its FKS-predicted poles)"""
    import mg5_pipeline_final as mg
    d = os.path.join(OUT, f"{name}__amcnlo")
    st = {"proc": name, "var": "amcnlo", "mg5": f"generate {P['born']} QED<={P['qed']} [QCD]", "dir": d}
    if not os.path.isdir(d):
        script = f"import model loop_sm-no_b_mass\n{st['mg5']}\noutput {d}\n"
        r = subprocess.run([mg.MG5_BIN], input=script, text=True, capture_output=True, timeout=5400)
        txt = r.stdout + r.stderr
        open(os.path.join(OUT, f"{name}__amcnlo.mg5.log"), "w").write(txt)
        st["warnings"] = warnings_of(txt)
    if not os.path.exists(os.path.join(d, "bin", "aMCatNLO")):
        st["why"] = "no aMC@NLO output"
        return st
    # check_poles is not a command of the aMC@NLO shell: it is a Fortran test that `compile FO` builds and runs in
    # every SubProcesses/P*/ with a virtual (amcatnlo_run_interface.compile_dir; input: points, tolerance). `compile FO
    # -f` builds the libraries and runs it at the default tolerance (FKS_params.dat IRPoleCheckThreshold 1e-5; it
    # raises when more than 10% fail, the logs stay); then check_poles is rerun here on N_POLES points, each P* dir
    # reporting MadLoop's against the FKS double and single pole per squared order.
    cmdf = os.path.join(d, "zz_compile_fo.cmd")
    open(cmdf, "w").write("compile FO -f\n")
    try:
        r = subprocess.run([os.path.join(d, "bin", "aMCatNLO"), cmdf], cwd=d, text=True, capture_output=True,
                           timeout=7200, input="\n" * 20)
        txt = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        txt = (e.stdout or "") if isinstance(e.stdout, str) else "timeout"
    open(os.path.join(OUT, f"{name}__amcnlo.compile.log"), "w").write(txt)
    st["compile"] = [l.rstrip() for l in txt.splitlines()
                     if re.search(r"pole|Pole|cancel|PASS|FAIL|Error|error|tolerance|not recognized", l)][-20:]
    st["check_poles"] = []
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    for pd in sorted(glob.glob(os.path.join(d, "SubProcesses", "P*_*"))):
        if not os.path.isdir(pd) or not glob.glob(os.path.join(pd, "V*")):
            continue
        tag = os.path.basename(pd)
        if not os.path.exists(os.path.join(pd, "check_poles")):
            with open(os.path.join(pd, "zz_make_check_poles.log"), "w") as lf:
                subprocess.run(["make", "check_poles"], cwd=pd, env=env, stdout=lf, stderr=lf)
        if not os.path.exists(os.path.join(pd, "check_poles")):
            st["check_poles"].append(f"{tag}: check_poles did not build (zz_make_check_poles.log)")
            continue
        try:
            c = subprocess.run(["./check_poles"], cwd=pd, input=f"{N_POLES}\n-1\n", text=True, capture_output=True,
                               timeout=3600)
            out = c.stdout + c.stderr
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else "timeout"
        open(os.path.join(OUT, f"{name}__amcnlo.{tag}.check_poles.log"), "w").write(out)
        npass, nfail = out.count(" PASSED"), out.count(" FAILED")
        st["check_poles"].append(f"{tag}: PASSED {npass} FAILED {nfail} (tolerance: FKS_params.dat default)")
        st["check_poles"] += [f"{tag}: {l.rstrip()}" for l in out.splitlines()
                              if re.search(r"pole|POLE|Pole|MadFKS|MadLoop|ratio|FAILED|amp_split|order", l)][:60]
    return st


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--n", type=int, default=100)
    a.add_argument("--n-2to2", type=int, default=200)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--only", nargs="*", default=None)
    a.add_argument("--no-amcnlo", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--dir", required=True)
    e.add_argument("--mg5", required=True)
    e.add_argument("--pdg", type=int, nargs="+", required=True)
    e.add_argument("--m", type=float, nargs="+", required=True)
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.cmd == "eval":
        res = eval_one(args.dir, args.mg5, args.pdg, args.m, args.n, args.seed)
        print("EVALJSON " + json.dumps(res), flush=True)
        return

    import yaml
    v4 = yaml.load(open(os.path.join(ROOT, "recipes", "catalog_v4_loops.yaml")),
                   Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["virt"]
    os.makedirs(OUT, exist_ok=True)
    names = args.only or list(PROCS)
    jobs = [(nm, var, *sm) for nm in names for var, sm in variants(nm, PROCS[nm]).items()]
    log(f"{len(jobs)} standalones to build under {OUT}, {args.workers} workers")
    builds = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build, *j): j for j in jobs}
        if not args.no_amcnlo:
            for nm in names:
                if not PROCS[nm].get("only_v4"):
                    futs[ex.submit(amcnlo_check_poles, nm, PROCS[nm])] = (nm, "amcnlo", None, None)
        for f in cf.as_completed(futs):
            nm, var = futs[f][:2]
            try:
                st = f.result()
            except Exception as exc:      # one variant's failure (MG5 refusing a syntax, a timeout) is a result
                st = {"proc": nm, "var": var, "built": False, "why": repr(exc)}
            builds[(nm, var)] = st
            log(f"BUILT {nm} {var}: built={st.get('built')} {st.get('why', '')} ({st.get('build_s', '-')} s) {st.get('mg5', '')}")
            for w in st.get("warnings", [])[:15]:
                log(f"  WARN {nm} {var}: {w}")
            for w in st.get("compile", []):
                log(f"  COMPILE_FO {nm}: {w}")
            for w in st.get("check_poles", []):
                log(f"  CHECK_POLES {nm}: {w}")
    results = {"builds": {f"{k[0]}__{k[1]}": v for k, v in builds.items()}, "evals": {}}
    tasks = []
    for (nm, var), st in builds.items():
        if not st.get("built") or var == "amcnlo":
            continue
        cfg = dict(v4[nm] if nm in v4 else v4[nm + "_qcdborn"])   # the five mixed-order borns are v4_<x>_qcdborn now
        n = args.n_2to2 if len(cfg["pdg_ids"]) == 4 else args.n
        scan = list(STAB) if (var == "v4" and PROCS[nm]["stab"]) else ["default"]
        for ptag in scan:
            tasks.append((nm, var, ptag, st, cfg, n))
    # variants evaluate in parallel; one standalone's scan stays sequential (its MadLoopParams.dat is shared)
    by_dir = {}
    for t in tasks:
        by_dir.setdefault(t[3]["dir"], []).append(t)

    def run_dir(ts):
        out = []
        for nm, var, ptag, st, cfg, n in ts:
            t0 = time.time()
            res = run_eval(st, nm, cfg, ptag, n, args.seed)
            out.append((nm, var, ptag, res, time.time() - t0))
        return out
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for f in cf.as_completed([ex.submit(run_dir, ts) for ts in by_dir.values()]):
            for nm, var, ptag, res, dt in f.result():
                tag = f"{nm} {var} {ptag}"
                if "error" in res:
                    log(f"{tag}: EVAL ERROR {res['error'][-600:]!r}")
                    results["evals"][tag] = res
                    continue
                s = summarise(tag, res)
                s["eval_s"] = round(dt)
                results["evals"][tag] = {"summary": s, "rows": res["rows"]}
    for (nm, var), st in sorted(builds.items()):
        for l in (st.get("check_sa") or [])[-25:]:
            log(f"  CHECK_SA {nm} {var}: {l}")
        for l in (st.get("so_comments") or [])[:10]:
            log(f"  SO {nm} {var}: {l}")
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), default=str)
    log(f"DONE results {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
