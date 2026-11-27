#!/usr/bin/env python3
"""Bakar's slide-11 end-user test of a KIT surrogate with MadGraph (kit_benchmark, D571, D575):

  "We run importance-sampled integration and two-stage unweighting with MadGraph. Tree level: cross-section bias
   sigma_hat/sigma - 1, when the surrogate is used without correction; second-stage efficiency eps2 = <r>/r_max, with
   r = |M|^2/|M_hat|^2, on events distributed as |M_hat|^2 [...] eps2 is reported with its r_max quantile."

    python tools/kit_mc_madgraph.py gen <ee_uug|ee_uugg> --dir D --n N [--seed S] [--cores C] [--surrogate RUN]
    python tools/kit_mc_madgraph.py predict <ee_uug|ee_uugg> --dir D <run_dir>
    python tools/kit_mc_madgraph.py metrics --dir D --out <json> [--boot B]
    python tools/kit_mc_madgraph.py figure <json> --out <stem>          (local)

gen: MadEvent (MG5_aMC, model sm, the KIT file's settings: sqrt s 1 TeV, fixed alpha_s 0.118 at mu_R = M_Z where the
card's alpha_s(M_Z) = 0.118 is the value used, 1/alpha 132.507) integrates e+ e- -> u u~ g (g) by its importance-sampled
multi-channel VEGAS and unweights: N events distributed as the exact |M|^2 inside the benchmark's bulk, "s_ij/s > 0.01
for every singular pair" (every pair holding a gluon; set in MadEvent's dummy_cuts, every run-card cut off). The exact
|M|^2 of each event comes from the same process's standalone and must reproduce KIT's stored me2 on its bulk_test events.
metrics (events x_i distributed as |M|^2, u_i = |M_hat|^2/|M|^2 at x_i, r_i = 1/u_i):
  bias  sigma_hat/sigma - 1 = mean(u) - 1 (sigma_hat = int |M_hat|^2 = sigma E_|M|^2[u]), its 16-84% bootstrap band and
        the statistical error std(u)/sqrt N
  eps2  on events distributed as |M_hat|^2 = these events weighted by u: <r> = sum(u r)/sum(u) = 1/mean(u), r_max = max r,
        eps2 = <r>/r_max; with r_max at the u-weighted q-quantile of r (q = 0.99, 0.999, 0.9999) and the share of sigma in
        events above that cap (the bias that cap leaves; events weighted by u r = 1, i.e. the exact measure)
Without --surrogate this differs from slide 11: MadEvent's grid and first unweighting are driven by the exact |M|^2, and
the |M_hat|^2-distributed sample is that sample reweighted by u.
gen --surrogate RUN (slide 11 as written): MadEvent integrates and unweights |M_hat|^2. Each P dir's matrix-element call
(DSIG's SMATRIX and the vectorised SMATRIX_MULTI, vector_size VEC) is followed by KITSURR, which multiplies MadEvent's
value by u = |M_hat|^2/|M|^2 from a `serve` process (one per madevent process, over two FIFOs): the integrand becomes
alpha_c |M_hat|^2, alpha_c MadEvent's own multi-channel weights (a partition of unity, so the integral is int |M_hat|^2
exactly), and the grid, sigma_hat and the unweighted events are those of |M_hat|^2. |M_hat|^2: the run rebuilt once as
tools/kit_predict.predict does (float64, 1 CPU thread, each event in its own frame), each batch re-read through the
experiment's own init_data; |M|^2 by the process's standalone. Events -> <proc>_surr.h5 (exact |M|^2 per event as above).
metrics on a --surrogate file (events y_i distributed as |M_hat|^2, r_i = |M|^2/|M_hat|^2 by `predict`):
  bias  sigma_hat/sigma - 1 with sigma_hat MadEvent's integral of |M_hat|^2 and sigma the exact run's (<proc>.h5), MC error
        from the two runs' quoted errors
  eps2  <r>/r_max, r_max the maximum and the q-quantiles of r, with the share of sigma above each cap
        (E[max(r - r_q, 0)]/E[r])
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import kit_hardps as kh  # noqa: E402

PROC = {"ee_uug": "e+ e- > u u~ g", "ee_uugg": "e+ e- > u u~ g g"}
QS = (0.99, 0.999, 0.9999)
CUT = """      logical FUNCTION dummy_cuts(P)
C     kit_mc_madgraph: the KIT bulk, s_ij/s > {y} for every pair of finals holding a gluon (legs {g} on)
      IMPLICIT NONE
      include 'genps.inc'
      include 'nexternal.inc'
      REAL*8 P(0:3,nexternal)
      INTEGER I, J
      REAL*8 S, M2
      dummy_cuts=.true.
      S=(P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2
      DO I=NINCOMING+1,NEXTERNAL-1
        DO J=MAX(I+1,{g}),NEXTERNAL
          M2=(P(0,I)+P(0,J))**2-(P(1,I)+P(1,J))**2
     &      -(P(2,I)+P(2,J))**2-(P(3,I)+P(3,J))**2
          IF (M2.LT.{y}D0*S) THEN
            dummy_cuts=.false.
            RETURN
          ENDIF
        ENDDO
      ENDDO
      RETURN
      END
"""
RUN = {"lpp1": 0, "lpp2": 0, "ebeam1": 500.0, "ebeam2": 500.0, "fixed_ren_scale": "True", "fixed_fac_scale": "True",
       "scale": 91.188, "dsqrt_q2fact1": 91.188, "dsqrt_q2fact2": 91.188, "ptj": 0.0, "etaj": -1.0, "drjj": 0.0,
       "mmjj": 0.0, "ptheavy": 0.0, "use_syst": "False", "cut_decays": "False"}


VEC = int(os.environ.get("KIT_MC_VEC", 1024))   # MadEvent's vector_size: events per surrogate call
KITSURR = """      SUBROUTINE KITSURR(P, OUT, N)
C     kit_mc_madgraph --surrogate: OUT(i) *= |M_hat|^2/|M|^2 at P(:,:,i), from the serve process ($KIT_SURR_CMD) over
C     two FIFOs: sent M, NEXTERNAL (int32) and P of the M events with OUT /= 0, read M ratios (float64)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER N, I, M, PID, GETPID
      DOUBLE PRECISION P(0:3,NEXTERNAL,N), OUT(N)
      DOUBLE PRECISION Q(0:3,NEXTERNAL,N), U(N)
      INTEGER IDX(N)
      LOGICAL INIT
      CHARACTER*1024 CMD
      CHARACTER*64 FI, FO
      DATA INIT/.FALSE./
      SAVE INIT
      IF (.NOT.INIT) THEN
        CALL GET_ENVIRONMENT_VARIABLE('KIT_SURR_CMD', CMD)
        IF (LEN_TRIM(CMD).EQ.0) STOP 'KITSURR: KIT_SURR_CMD unset'
        PID = GETPID()
        WRITE(FI,'(A,I0,A)') 'kitsurr_', PID, '.in'
        WRITE(FO,'(A,I0,A)') 'kitsurr_', PID, '.out'
        CALL EXECUTE_COMMAND_LINE('rm -f '//TRIM(FI)//' '//TRIM(FO)
     &    //' && mkfifo '//TRIM(FI)//' '//TRIM(FO))
        CALL EXECUTE_COMMAND_LINE(TRIM(CMD)//' '//TRIM(FI)//' '
     &    //TRIM(FO), WAIT=.FALSE.)
        OPEN(91, FILE=FI, ACCESS='STREAM', FORM='UNFORMATTED',
     &    ACTION='WRITE')
        OPEN(92, FILE=FO, ACCESS='STREAM', FORM='UNFORMATTED',
     &    ACTION='READ')
        INIT = .TRUE.
      ENDIF
      M = 0
      DO I = 1, N
        IF (OUT(I).NE.0D0) THEN
          M = M + 1
          IDX(M) = I
          Q(:,:,M) = P(:,:,I)
        ENDIF
      ENDDO
      IF (M.EQ.0) RETURN
      WRITE(91) M, NEXTERNAL
      WRITE(91) Q(:,:,1:M)
      FLUSH(91)
      READ(92) U(1:M)
      DO I = 1, M
        OUT(IDX(I)) = OUT(IDX(I))*U(I)
      ENDDO
      END

      SUBROUTINE KITSURR1(P, ANS)
C     the scalar call (DSIG's SMATRIX): one event
      IMPLICIT NONE
      DOUBLE PRECISION P(*), ANS, A(1)
      A(1) = ANS
      CALL KITSURR(P, A, 1)
      ANS = A(1)
      END
"""


def h5(d, proc, surrogate=False):
    return os.path.join(d, f"{proc}{'_surr' if surrogate else ''}.h5")


def patch_surrogate(ev):
    """Every P dir of the MadEvent output `ev`: KITSURR after each SMATRIX call of auto_dsig<k>.f (module docstring)."""
    import glob
    import re
    pdirs = sorted(glob.glob(os.path.join(ev, "SubProcesses", "P*")))
    assert pdirs, f"no P dir in {ev}"
    for pd in pdirs:
        fs = sorted(glob.glob(os.path.join(pd, "auto_dsig[0-9]*.f")))
        assert fs, f"no auto_dsig<k>.f in {pd}"
        for f in fs:
            src = open(f).read()
            src, n1 = re.subn(r"(\n +CALL SMATRIX\d+\(P1,RHEL, RCOL,CHANNEL,1, DSIGUU, SELECTED_HEL\(1\),\n +\$ +SELECTED_COL\(1\)\))",
                              r"\1\n      CALL KITSURR1(P1, DSIGUU)", src)
            src, n2 = re.subn(r"(\n!\$OMP END PARALLEL\n)", r"\1      CALL KITSURR(P_MULTI, OUT, VECSIZE_USED)\n", src)
            assert n1 == 1 and n2 == 1, f"{f}: SMATRIX calls found scalar {n1}, vector {n2} (one each expected)"
            open(f, "w").write(src)
        open(os.path.join(pd, "auto_dsig_kitsurr.f"), "w").write(KITSURR)   # linked by the makefile's auto_dsig*.f
    return pdirs


def cut_pass(mom, g0):
    """The coded dummy_cuts in numpy: (N,) True where s_ij/s >= 0.01 for every final pair (i, j>=g0), mom (N, n, 4)."""
    m2 = lambda p: p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)
    s, n = m2(mom[:, 0] + mom[:, 1]), mom.shape[1]
    return np.all([m2(mom[:, i] + mom[:, j]) >= 0.01 * s for i in range(2, n - 1) for j in range(max(i + 1, g0), n)],
                  axis=0)


def lhe_weights(path):
    """XWGTUP of every event of an LHE file (the line after each <event>)."""
    import gzip
    w, nxt = [], False
    with gzip.open(path, "rt") as f:
        for line in f:
            if nxt:
                w.append(float(line.split()[2])); nxt = False
            nxt = nxt or line.lstrip().startswith("<event")
    return np.asarray(w)


def sigma_err_log(log):
    """(sigma, error) in pb of MadEvent's last 'Cross-section : X +- Y pb' line of a generate_events log."""
    import re
    txt = open(log, errors="replace").read()
    m = re.findall(r"Cross-section :\s*([-+0-9.eE]+) \+- ([-+0-9.eE]+) pb", txt)
    if not m:   # MadEvent's own reason (e.g. 'Survey return zero cross section') instead of a bare assert
        why = [l.strip() for l in txt.splitlines() if re.search(r"zero cross section|Zero result|No points passed|Error", l)]
        sys.exit(f"{log}: MadEvent gave no cross section: {' | '.join(why[:4]) or 'no reason in the log'}")
    return float(m[-1][0]), float(m[-1][1])


def standalone_exact(drv, sa, pdgs, gen, alpha_s):
    """(pipe, f): f(momenta (N, n, 4), KIT order) -> exact |M|^2 by the process's standalone; pipe to close."""
    import mg5_pipeline_final as mg
    perm_kit = kh.beam_perm(pdgs)
    slot = mg.row_to_slot_perm([pdgs[i] for i in perm_kit], [f"generate {gen}"])
    pipe = mg.CppDriverPipe(drv, sa).__enter__()
    return pipe, lambda m: np.asarray(pipe.compute([(x[perm_kit], None) for x in m], perm=slot, alphas=alpha_s))


def cmd_gen(a):
    import h5py
    import mg5_pipeline_final as mg
    meta = kh.metadata(a.proc)
    assert meta["sqrts"] == 1000.0 and meta["alpha_s"] == 0.118 and meta["alpha_inv"] == 132.507, meta
    assert "s_ij/s > 0.01 for every singular pair" in meta["bulk"], meta["bulk"]
    gen = PROC[a.proc]
    pdgs = [p["pdg"] for p in meta["particles"]]
    # MadGraph's leg order is the generate line's: the KIT file's order, which KITSURR's momenta are read in
    assert pdgs == [mg.MG5_PDG[t] for t in gen.replace(">", "").split()], (pdgs, gen)
    d = os.path.abspath(a.dir)
    tag = f"{a.proc}_surr" if a.surrogate else a.proc
    # KIT_MC_WORK_IN_DIR=1: work under <dir> on the shared disk (a CC htc node with no local tmp disk, --tmp=0)
    work = os.path.join(d if os.environ.get("KIT_MC_WORK_IN_DIR") else (os.environ.get("TMPDIR") or d), f"mg_{tag}")
    ev, sa = os.path.join(work, "ev"), os.path.join(work, "sa")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, f"{tag}_mg5.log")
    mg._mg5_script(f"import model sm\ngenerate {gen}\noutput {ev}\noutput standalone_cpp {sa}\nquit\n", log)
    n = len(pdgs)
    g = pdgs.index(21) + 1                          # first gluon leg (MadGraph order = the generate line's)
    with h5py.File(kh.h5_path(a.proc), "r") as K:   # the coded cut is KIT's bulk: every bulk_test event passes it
        bt = K["bulk_test/bulk/momenta"][:]
        assert cut_pass(bt, g - 1).all(), f"{(~cut_pass(bt, g - 1)).sum()} KIT bulk_test events fail the coded cut"
    f = os.path.join(ev, "SubProcesses", "dummy_fct.f")
    src = open(f).read()
    import re
    m = re.search(r"^ +logical +FUNCTION +dummy_cuts\(P\).*?^ +end *$", src, flags=re.I | re.S | re.M)
    open(f, "w").write(src[:m.start()] + CUT.format(y="0.01", g=g).rstrip("\n") + src[m.end():])
    shutil.copy2(f"{ev}/Cards/param_card.dat", f"{sa}/Cards/param_card.dat")
    mg.compile_backends(sa, n)
    backend, _, drv, _ = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and drv, f"no C++ driver in {sa}"
    pipe, exact = standalone_exact(drv, sa, pdgs, gen, meta["alpha_s"])
    with h5py.File(kh.h5_path(a.proc), "r") as K:
        b = K["bulk_test/bulk"]
        chk = exact(b["momenta"][:1000])
        rel = float(np.max(np.abs(chk / b["me2"][:1000] - 1)))
        print(f"CHECK exact |M|^2 vs KIT bulk_test me2 (1000 events): max |ours/KIT - 1| = {rel:.2e}", flush=True)
        assert rel < 1e-6, "the standalone does not reproduce KIT's me2"
        if a.surrogate:   # the server's in-memory |M_hat|^2 is kit_predict.predict's, event by event
            mom = b["momenta"][:1000]
            srv = Predictor(a.surrogate, a.proc).log10(mom)
            tmp = os.path.join(d, f"chk_{a.proc}.h5")
            with h5py.File(tmp, "w") as o:
                o.attrs["metadata"] = K.attrs["metadata"]
                o["bulk_test/bulk/momenta"] = mom
            from tools.kit_predict import predict
            _, pth = predict(a.surrogate, ["bulk_test"], os.path.join(d, f"chk_{a.proc}"), h5=tmp)
            with h5py.File(pth, "r") as P:
                dl = float(np.max(np.abs(srv - P["bulk_test/bulk/log10_me2"][:])))
            print(f"CHECK server log10|M_hat|^2 vs kit_predict on 1000 bulk_test events: max |delta| = {dl:.2e}", flush=True)
            assert dl < 1e-9, "the server's surrogate is not kit_predict's"
    run_card = {**RUN, "nevents": a.n, "iseed": a.seed}
    env = dict(os.environ)
    if a.surrogate:
        patch_surrogate(ev)
        with open(f"{ev}/Cards/run_card.dat", "a") as rc:   # hidden parameters: absent from the card until set
            rc.write(f" {VEC} = vector_size ! kit_mc_madgraph: events per surrogate call\n 1 = nb_warp\n")
        env["KIT_SURR_CMD"] = (f"{sys.executable} {os.path.abspath(__file__)} serve {a.proc} "
                               f"{os.path.abspath(a.surrogate)} {sa} --stats {os.path.join(d, tag + '_serve.jsonl')}")
        env["OMP_NUM_THREADS"] = "1"     # SMATRIX_MULTI's OpenMP loop: one thread per madevent process
        env["KIT_THREADS"] = "1"         # the surrogate at 1 thread, as the package's predictor
    mg.patch_card(f"{ev}/Cards/run_card.dat", run_card)
    mg._set_me5_config(f"{ev}/Cards/me5_configuration.txt", {"run_mode": 2, "nb_core": a.cores,
                                                             "automatic_html_opening": "False"})
    with open(log, "a") as lf:
        subprocess.run([f"{ev}/bin/generate_events", "run_01", "-f"], cwd=ev, stdout=lf, stderr=subprocess.STDOUT,
                       check=True, env=env)
    if a.surrogate:
        vs = open(f"{ev}/Source/vector.inc").read()
        assert re.search(rf"WARP_SIZE={VEC}\)", vs), f"vector_size {VEC} not compiled in: {vs}"
    banner = sorted(__import__("glob").glob(f"{ev}/Events/run_01/*_banner.txt"))[0]
    sigma = mg._banner_xsec(banner)
    sig_log, err = sigma_err_log(log)
    mom, aqcd, _ = mg.read_lhe_pp(f"{ev}/Events/run_01/unweighted_events.lhe.gz", pdgs)
    w = lhe_weights(f"{ev}/Events/run_01/unweighted_events.lhe.gz")
    assert len(w) == len(mom) and np.allclose(w, w[0], rtol=1e-6), f"LHE weights not equal: {w.min()}..{w.max()}"
    assert cut_pass(mom, g - 1).all(), "a MadEvent event fails the coded cut"
    assert np.allclose(aqcd, meta["alpha_s"], rtol=1e-6), f"MadEvent alpha_s {aqcd.min()}..{aqcd.max()}"
    me2 = exact(mom)
    pipe.__exit__(None, None, None)
    with h5py.File(kh.h5_path(a.proc), "r") as K, h5py.File(h5(d, a.proc, a.surrogate) + ".tmp", "w") as o:
        b = K["bulk_test/bulk"]
        o.attrs["metadata"] = K.attrs["metadata"]
        for k in b:
            o[f"bulk_test/bulk/{k}"] = b[k][:5]    # kit_hardps.log10_name reads the target names there
        o["mc/events/momenta"], o["mc/events/me2"], o["mc/events/log10_me2"] = mom, me2, np.log10(me2)
        o.attrs.update(sigma_pb=sigma, sigma_err_pb=err, n=len(me2), check_rel=rel,
                       generator="MadEvent " + str(mg._mg5_version()), surrogate=a.surrogate or "")
    os.replace(h5(d, a.proc, a.surrogate) + ".tmp", h5(d, a.proc, a.surrogate))
    shutil.rmtree(work, ignore_errors=True)
    print(f"GEN {tag}: {len(me2)} events, sigma {sigma} +- {err} pb -> {h5(d, a.proc, a.surrogate)}")


class Predictor:
    """log10 |M_hat|^2 of a KIT run on events in memory, as tools/kit_predict.predict computes it (rebuilt once with
    its best weights, each event in its own frame, the residual against a unit placeholder truth); each call re-reads
    only the scored role through the experiment's init_data (the train pool and the run's statistics unchanged)."""

    def __init__(self, run, proc):
        import h5py
        import torch
        import yaml
        from tools.rebuild_run import rebuild
        torch.set_num_threads(int(os.environ.get("KIT_THREADS", 1)))
        self.meta = kh.metadata(proc)
        self.pool = os.path.join(os.environ["SCRATCH"], f"kitsurr_{os.getpid()}_{os.urandom(4).hex()}.npy")
        self.n = 1
        with h5py.File(kh.h5_path(proc), "r") as f:    # one bulk event to build the run on
            self._write(f["bulk_test/bulk/momenta"][:1])
        self.exp, _ = rebuild(run, "test", self.pool, 1, event_frame=True)
        self.recipe = str(self.exp.cfg.data.processes_file)
        self.name = yaml.safe_load(open(self.recipe))["processes"][0]["name"]

    def _write(self, mom):
        np.save(self.pool, kh.group_rows({"momenta": mom}, "me2", self.meta))
        np.save(self.pool[:-4] + ".alphas.npy", np.full(len(mom), self.meta["alpha_s"]))

    def log10(self, mom):
        import torch
        import yaml
        import datagen
        mom = np.asarray(mom, dtype=np.float64)
        if len(mom) != self.n:      # the recipe's test count is the pool's size
            rec = yaml.safe_load(open(self.recipe))
            rec["processes"][0]["n_test"] = self.n = len(mom)
            yaml.safe_dump(rec, open(self.recipe, "w"))
        self._write(mom)
        real = datagen.ensure_split_set
        datagen.ensure_split_set = lambda specs, role, seed, dest_dir=None, require_cache=False: (
            {self.name: self.pool} if role == "test" else
            real(specs, role=role, seed=seed, dest_dir=dest_dir, require_cache=require_cache))
        try:
            self.exp.init_data()
            self.exp._init_dataloader()
        finally:
            datagen.ensure_split_set = real
        with torch.no_grad():
            pred, truth, _ = self.exp._collect_predictions(self.exp.test_loader)
        res = (np.asarray(pred, np.float64).reshape(-1) - np.asarray(truth, np.float64).reshape(-1)) \
            * float(self.exp.prepd_std[0])
        assert len(res) == len(mom), (len(res), len(mom))
        out = np.full(len(mom), np.nan)
        out[np.asarray(self.exp._role_perm["test"])[:len(res)]] = res
        assert np.isfinite(out).all(), "an event without a prediction"
        return out / np.log(10.0)


def cmd_serve(a):
    """The surrogate behind KITSURR: reads (M, n int32, M*n*4 float64) from fifo_in, writes M ratios u = |M_hat|^2/|M|^2."""
    import time
    import mg5_pipeline_final as mg
    t0 = time.time()
    pred = Predictor(a.run, a.proc)
    meta = kh.metadata(a.proc)
    pdgs = [p["pdg"] for p in meta["particles"]]
    _, _, drv, _ = mg.detect_compiled_backend(a.sa)
    pipe, exact = standalone_exact(drv, a.sa, pdgs, PROC[a.proc], meta["alpha_s"])
    st = {"pid": os.getpid(), "cwd": os.getcwd(), "startup_s": time.time() - t0, "calls": 0, "events": 0,
          "surrogate_s": 0.0, "exact_s": 0.0}
    fin = open(a.fifo_in, "rb")
    fout = open(a.fifo_out, "wb")
    try:
        while True:
            h = fin.read(8)
            if len(h) < 8:
                break
            m, n = (int(x) for x in np.frombuffer(h, np.int32))
            assert n == len(pdgs), (n, pdgs)
            buf = b""
            while len(buf) < m * n * 32:
                c = fin.read(m * n * 32 - len(buf))
                assert c, "KITSURR closed mid-batch"
                buf += c
            mom = np.frombuffer(buf, np.float64).reshape(m, n, 4)
            t = time.time()
            hat = 10.0 ** pred.log10(mom)
            st["surrogate_s"] += time.time() - t
            t = time.time()
            ex = exact(mom)
            st["exact_s"] += time.time() - t
            u = hat / ex
            assert np.isfinite(u).all() and (u > 0).all(), "a non-finite or non-positive |M_hat|^2/|M|^2"
            fout.write(np.ascontiguousarray(u, np.float64).tobytes())
            fout.flush()
            st["calls"] += 1
            st["events"] += m
    finally:
        pipe.__exit__(None, None, None)
        for f in (a.fifo_in, a.fifo_out, pred.pool, pred.pool[:-4] + ".alphas.npy"):
            try:
                os.remove(f)
            except OSError:
                pass
        if a.stats:
            with open(a.stats, "a") as f:
                f.write(json.dumps({**st, "wall_s": time.time() - t0}) + "\n")


def cmd_predict(a):
    from tools.kit_predict import predict
    out = os.path.join(os.path.abspath(a.dir), "pred_surr" if a.surr else "pred", os.path.basename(a.run.rstrip("/")))
    _, path = predict(a.run, ["mc"], out, h5=h5(os.path.abspath(a.dir), a.proc, a.surr))
    print(f"PREDICTED {a.run} -> {path}")


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])


def mc_metrics(me2, me2_hat, boot=1000, seed=0):
    """slide-11 metrics on events distributed as me2 (module docstring)"""
    u = me2_hat / me2
    r = 1 / u
    rng = np.random.default_rng(seed)
    bs = [u[rng.integers(0, len(u), len(u))].mean() - 1 for _ in range(boot)]
    mean_r = 1 / u.mean()
    out = {"n": int(len(u)), "bias": float(u.mean() - 1), "bias_band": [float(np.quantile(bs, .16)), float(np.quantile(bs, .84))],
           "bias_stat": float(u.std() / np.sqrt(len(u))), "mean_r": float(mean_r),
           "eps2": {"max": float(mean_r / r.max())}, "r_max": {"max": float(r.max())}, "sigma_above": {}}
    for q in QS:
        rq = wquantile(r, u, q)
        out["r_max"][str(q)] = rq
        out["eps2"][str(q)] = float(mean_r / rq)
        out["sigma_above"][str(q)] = float(np.mean(np.maximum(r - rq, 0) * u))   # share of sigma over the cap
    return out


def surr_metrics(me2, me2_hat, sh, eh, se, ee):
    """slide-11 metrics on events distributed as me2_hat (module docstring): bias from the two integrals, eps2 of r"""
    r = me2 / me2_hat
    out = {"n": int(len(r)), "bias": sh / se - 1, "bias_err": sh / se * float(np.hypot(eh / sh, ee / se)),
           "mean_r": float(r.mean()), "eps2": {"max": float(r.mean() / r.max())}, "r_max": {"max": float(r.max())},
           "sigma_above": {}}
    for q in QS:
        rq = float(np.quantile(r, q))
        out["r_max"][str(q)] = rq
        out["eps2"][str(q)] = float(r.mean() / rq)
        out["sigma_above"][str(q)] = float(np.mean(np.maximum(r - rq, 0)) / r.mean())
    return out


def cmd_metrics(a):
    import glob
    import h5py
    res = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "pred", "*", "*", "predictions.h5")) +
                    glob.glob(os.path.join(a.dir, "pred", "*", "predictions.h5"))):
        run = [s for s in p.split(os.sep) if s.startswith("kit_")][0]
        proc = "ee_uugg" if "_uugg_" in run else "ee_uug"
        with h5py.File(h5(a.dir, proc), "r") as f, h5py.File(p, "r") as P:
            me2 = f["mc/events/me2"][:].astype(np.float64)
            hat = 10 ** P["mc/events/log10_me2"][:].astype(np.float64)
            res[run] = {"process": proc, "sigma_pb": float(f.attrs["sigma_pb"]), "check_rel": float(f.attrs["check_rel"]),
                        **mc_metrics(me2, hat, a.boot)}
        print("CLAIM " + json.dumps({run: res[run]}))
    for p in sorted(glob.glob(os.path.join(a.dir, "pred_surr", "*", "predictions.h5"))):
        run = os.path.basename(os.path.dirname(p))
        proc = "ee_uugg" if "_uugg_" in run else "ee_uug"
        with h5py.File(h5(a.dir, proc, True), "r") as f, h5py.File(h5(a.dir, proc), "r") as fe, h5py.File(p, "r") as P:
            assert os.path.basename(str(f.attrs["surrogate"]).rstrip("/")) == run, (f.attrs["surrogate"], run)
            me2 = f["mc/events/me2"][:].astype(np.float64)
            hat = 10 ** P["mc/events/log10_me2"][:].astype(np.float64)
            sh, eh = float(f.attrs["sigma_pb"]), float(f.attrs["sigma_err_pb"])
            se, ee = sigma_err_log(os.path.join(a.dir, f"{proc}_mg5.log"))     # the exact run's
            assert abs(se / float(fe.attrs["sigma_pb"]) - 1) < 1e-4, (se, fe.attrs["sigma_pb"])
            res[run + "/surrogate_sample"] = {"process": proc, "sigma_hat_pb": sh, "sigma_hat_err_pb": eh, "sigma_pb": se,
                                              "sigma_err_pb": ee, **surr_metrics(me2, hat, sh, eh, se, ee)}
        print("CLAIM " + json.dumps({run + "/surrogate_sample": res[run + "/surrogate_sample"]}))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


def _selfcheck():
    rng = np.random.default_rng(1)
    m = rng.lognormal(0, 2, 20000)
    r = mc_metrics(m, 1.1 * m, boot=50)
    assert abs(r["bias"] - 0.1) < 1e-12 and abs(r["eps2"]["max"] - 1) < 1e-12
    q = surr_metrics(m, m / 1.1, 1.1, 0.0, 1.0, 0.0)
    assert abs(q["bias"] - 0.1) < 1e-12 and abs(q["eps2"]["max"] - 1) < 1e-12 and q["sigma_above"]["0.99"] == 0
    noisy = m * rng.lognormal(0, 0.1, len(m))
    r = mc_metrics(m, noisy, boot=50)
    assert r["eps2"]["0.99"] > r["eps2"]["max"] and 0 < r["eps2"]["max"] < 1
    print("selfcheck ok")


def cmd_figure(a):
    import plot_style as ps
    ps.use()
    d = json.load(open(a.json))
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a1, a2), (a3, cell) = axs
    runs = sorted(d, key=lambda k: d[k]["process"])
    lab = {"ee_uug": r"$e^+e^-\to u\bar ug$", "ee_uugg": r"$e^+e^-\to u\bar ugg$"}
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange]
    for i, k in enumerate(runs):
        e = d[k]
        a1.errorbar([i], [abs(e["bias"])], yerr=[[abs(e["bias"]) - min(abs(x) for x in e["bias_band"])],
                    [max(abs(x) for x in e["bias_band"]) - abs(e["bias"])]], fmt="o", color=cols[i], label=lab[e["process"]])
        a1.plot([i], [e["bias_stat"]], "_", color=cols[i], ms=14)
        x = [1 - float(q) for q in QS] + [1 / e["n"]]
        a2.plot(x, [e["eps2"][str(q)] for q in QS] + [e["eps2"]["max"]], "o-", color=cols[i])
    a1.plot([], [], "_", color=ps.C.grey, ms=14, label="MC error")
    a1.set_yscale("log"); a1.set_xticks(range(len(runs)), [lab[d[k]["process"]] for k in runs])
    a1.set_ylabel(r"$|\hat\sigma/\sigma-1|$"); a1.set_xlim(-0.5, len(runs) - 0.5)
    a2.set_xscale("log"); a2.set_xlabel(r"$1-q$")
    a2.set_ylabel(r"$\epsilon_2=\langle r\rangle/r_{\max}$")
    if a.tt:   # the one-loop virtual: the threshold scan's exact quadrature per sqrt s (kit_threshold_scan collect)
        t = json.load(open(a.tt))
        b, vex = np.asarray(t["beta"]), np.asarray(t["sigma_v_exact"])
        sv = t["surrogates"][a.tt_run]
        a3.plot(b, np.asarray(sv["sigma_v"]) / vex - 1, "o-", color=ps.C.green, ms=3,
                label=r"$t\bar t$ virtual: $\sigma[\hat V]/\sigma[V]-1$")
        r = np.array([x if x is not None else np.nan for x in sv["ratio"]])
        a3.plot(b, r, "s--", color=ps.C.orange, ms=3, label=r"$t\bar t$: $\hat\sigma_{\rm NLO}/\sigma_{\rm NLO}-1$")
        a3.axhline(0, color="k", lw=0.8)
        a3.set_xscale("log"); a3.set_xlabel(r"$\beta=\sqrt{1-4m_t^2/s}$"); a3.set_ylabel("relative bias")
    else:
        a3.set_axis_off()
    ps.legend_cell(cell, [a1, a2, a3], fontsize=ps.LEGEND_MIN_PT)
    print("FIGURE", ps.save(fig, a.out))


def _figure_tree(ps, d, sur, lab, cols, out):
    fig, (a1, a2, cell) = ps.figure(ncols=3)
    for i, k in enumerate(sur):
        e, ex = d[k], d[k.split("/")[0]]
        c = cols[e["process"]]
        a1.errorbar([i - 0.1], [e["bias"]], yerr=[e["bias_err"]], fmt="o", color=c, label=lab[e["process"]])
        a1.errorbar([i + 0.1], [ex["bias"]], yerr=[ex["bias_stat"]], fmt="s", mfc="none", color=c)
        x = [1 - float(q) for q in QS] + [1 / e["n"]]
        a2.plot(x, [e["eps2"][str(q)] for q in QS] + [e["eps2"]["max"]], "o-", color=c)
    a1.plot([], [], "o", color=ps.C.grey, label="MadEvent on surrogate")
    a1.plot([], [], "s", mfc="none", color=ps.C.grey, label="reweighted exact events")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(range(len(sur)), [lab[d[k]["process"]] for k in sur]); a1.set_xlim(-0.5, len(sur) - 0.5)
    a1.set_ylabel(r"$\hat\sigma/\sigma-1$")
    a2.set_xscale("log"); a2.set_xlabel(r"$1-q$"); a2.set_ylabel(r"$\epsilon_2=\langle r\rangle/r_{\max}$")
    ps.legend_cell(cell, [a1, a2], fontsize=ps.LEGEND_MIN_PT)
    print("FIGURE", ps.save(fig, out + "_tree"))


def cmd_figure_exact(a):
    """Slide 11 as written: (tree) sigma_hat/sigma - 1 per process and eps2 against 1 - q on MadEvent's |M_hat|^2 events;
    (tt) the integrated virtual's bias per sqrt s, threshold scan and bulk window."""
    import plot_style as ps
    ps.use()
    d = {}
    for j in a.json:
        d.update(json.load(open(j)))
    lab = {"ee_uug": r"$e^+e^-\to u\bar ug$", "ee_uugg": r"$e^+e^-\to u\bar ugg$"}
    cols = {"ee_uug": ps.C.blue, "ee_uugg": ps.C.vermillion}
    sur = sorted((k for k in d if k.endswith("/surrogate_sample")), key=lambda k: d[k]["process"])
    if sur:
        _figure_tree(ps, d, sur, lab, cols, a.out)
    if a.tt_bulk:
        fig, (b1, cell) = ps.figure(ncols=2)
        for path, mk, name in ((a.tt_scan, "s", "threshold scan"), (a.tt_bulk, "o", "bulk window")):
            if not path:
                continue
            t = json.load(open(path))
            sv = t["surrogates"][a.tt_run]
            y = np.asarray(sv["sigma_v"]) / np.asarray(t["sigma_v_exact"]) - 1
            b1.plot(t["beta"], y, mk + "-", ms=3, color=ps.C.green if mk == "o" else ps.C.orange, label=name)
        b1.axhline(0, color="k", lw=0.8)
        b1.set_yscale("symlog", linthresh=1e-4)    # threshold -0.3 and bulk 1e-3 on one axis
        b1.set_xscale("log"); b1.set_xlabel(r"$\beta=\sqrt{1-4m_t^2/s}$"); b1.set_ylabel(r"$\sigma[\hat V]/\sigma[V]-1$")
        ps.legend_cell(cell, [b1], fontsize=ps.LEGEND_MIN_PT)
        print("FIGURE", ps.save(fig, a.out + "_tt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("proc", choices=PROC); g.add_argument("--dir", required=True)
    g.add_argument("--n", type=int, default=100000); g.add_argument("--seed", type=int, default=11)
    g.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    g.add_argument("--surrogate", help="a KIT run: MadEvent integrates and unweights its |M_hat|^2 (module docstring)")
    sv = sub.add_parser("serve"); sv.add_argument("proc", choices=PROC); sv.add_argument("run"); sv.add_argument("sa")
    sv.add_argument("--stats"); sv.add_argument("fifo_in"); sv.add_argument("fifo_out")
    p = sub.add_parser("predict"); p.add_argument("proc", choices=PROC); p.add_argument("--dir", required=True)
    p.add_argument("run"); p.add_argument("--surr", action="store_true", help="the --surrogate run's events")
    m = sub.add_parser("metrics"); m.add_argument("--dir", required=True); m.add_argument("--out", required=True)
    m.add_argument("--boot", type=int, default=1000)
    fg = sub.add_parser("figure"); fg.add_argument("json"); fg.add_argument("--out", required=True); fg.add_argument("--tt", help="the threshold scan's JSON (kit_threshold_scan collect)"); fg.add_argument("--tt-run")
    fx = sub.add_parser("figure_exact"); fx.add_argument("json", nargs="*"); fx.add_argument("--out", required=True)
    fx.add_argument("--tt-scan"); fx.add_argument("--tt-bulk"); fx.add_argument("--tt-run")
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    {"gen": cmd_gen, "serve": cmd_serve, "predict": cmd_predict, "metrics": cmd_metrics, "figure": cmd_figure, "figure_exact": cmd_figure_exact,
     "selfcheck": lambda a: _selfcheck()}[a.cmd](a)


if __name__ == "__main__":
    main()
