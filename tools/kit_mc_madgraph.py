#!/usr/bin/env python3
"""Bakar's slide-11 end-user test of a KIT surrogate with MadGraph (kit_benchmark, D571, D575):

  "We run importance-sampled integration and two-stage unweighting with MadGraph. Tree level: cross-section bias
   sigma_hat/sigma - 1, when the surrogate is used without correction; second-stage efficiency eps2 = <r>/r_max, with
   r = |M|^2/|M_hat|^2, on events distributed as |M_hat|^2 [...] eps2 is reported with its r_max quantile."

    python tools/kit_mc_madgraph.py gen <ee_uug|ee_uugg> --dir D --n N [--seed S] [--cores C]
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
Differs from slide 11: MadEvent's grid and first unweighting are driven by the exact |M|^2, and the |M_hat|^2-distributed
sample is that sample reweighted by u, not a second MadEvent run on the surrogate (same sigma_hat and eps2 in
expectation; the |M_hat|^2 tail is sampled where the exact one puts events).
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


def h5(d, proc):
    return os.path.join(d, f"{proc}.h5")


def cmd_gen(a):
    import h5py
    import mg5_pipeline_final as mg
    meta = kh.metadata(a.proc)
    assert meta["sqrts"] == 1000.0 and meta["alpha_s"] == 0.118 and meta["alpha_inv"] == 132.507, meta
    assert "s_ij/s > 0.01 for every singular pair" in meta["bulk"], meta["bulk"]
    gen = PROC[a.proc]
    d = os.path.abspath(a.dir)
    work = os.path.join(os.environ.get("TMPDIR") or d, f"mg_{a.proc}")
    ev, sa = os.path.join(work, "ev"), os.path.join(work, "sa")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, f"{a.proc}_mg5.log")
    mg._mg5_script(f"import model sm\ngenerate {gen}\noutput {ev}\noutput standalone_cpp {sa}\nquit\n", log)
    pdgs = [p["pdg"] for p in meta["particles"]]
    n = len(pdgs)
    g = pdgs.index(21) + 1                          # first gluon leg (MadGraph order = the generate line's)
    f = os.path.join(ev, "SubProcesses", "dummy_fct.f")
    src = open(f).read()
    import re
    m = re.search(r"^ +logical +FUNCTION +dummy_cuts\(P\).*?^ +end *$", src, flags=re.I | re.S | re.M)
    open(f, "w").write(src[:m.start()] + CUT.format(y="0.01", g=g).rstrip("\n") + src[m.end():])
    mg.patch_card(f"{ev}/Cards/run_card.dat", {**RUN, "nevents": a.n, "iseed": a.seed})
    mg._set_me5_config(f"{ev}/Cards/me5_configuration.txt", {"run_mode": 2, "nb_core": a.cores,
                                                             "automatic_html_opening": "False"})
    with open(log, "a") as lf:
        subprocess.run([f"{ev}/bin/generate_events", "run_01", "-f"], cwd=ev, stdout=lf, stderr=subprocess.STDOUT,
                       check=True)
    banner = sorted(__import__("glob").glob(f"{ev}/Events/run_01/*_banner.txt"))[0]
    sigma = mg._banner_xsec(banner)
    shutil.copy2(f"{ev}/Cards/param_card.dat", f"{sa}/Cards/param_card.dat")
    mg.compile_backends(sa, n)
    backend, _, drv, _ = mg.detect_compiled_backend(sa)
    assert backend == "cpp" and drv, f"no C++ driver in {sa}"
    # MadGraph rows: e+ e- then finals in KIT order is the generate line's order; the driver's row convention is the
    # pools' (beam_perm), so feed rows in that order
    perm_kit = kh.beam_perm(pdgs)
    fa_pdgs = [pdgs[i] for i in perm_kit]
    slot = mg.row_to_slot_perm(fa_pdgs, [f"generate {gen}"])
    mom, aqcd, _ = mg.read_lhe_pp(f"{ev}/Events/run_01/unweighted_events.lhe.gz", pdgs)

    def exact(m_kit):
        with mg.CppDriverPipe(drv, sa) as pipe:
            return np.asarray(pipe.compute([(x[perm_kit], None) for x in m_kit], perm=slot, alphas=meta["alpha_s"]))
    with h5py.File(kh.h5_path(a.proc), "r") as K:
        b = K["bulk_test/bulk"]
        chk = exact(b["momenta"][:1000])
        rel = float(np.max(np.abs(chk / b["me2"][:1000] - 1)))
        print(f"CHECK exact |M|^2 vs KIT bulk_test me2 (1000 events): max |ours/KIT - 1| = {rel:.2e}", flush=True)
        assert rel < 1e-6, "the standalone does not reproduce KIT's me2"
        me2 = exact(mom)
        assert np.allclose(aqcd, meta["alpha_s"], rtol=1e-6), f"MadEvent alpha_s {aqcd.min()}..{aqcd.max()}"
        with h5py.File(h5(d, a.proc) + ".tmp", "w") as o:
            o.attrs["metadata"] = K.attrs["metadata"]
            for k in b:
                o[f"bulk_test/bulk/{k}"] = b[k][:5]    # kit_hardps.log10_name reads the target names there
            o["mc/events/momenta"], o["mc/events/me2"], o["mc/events/log10_me2"] = mom, me2, np.log10(me2)
            o.attrs.update(sigma_pb=sigma, n=len(me2), check_rel=rel, generator="MadEvent " + str(mg._mg5_version()))
    os.replace(h5(d, a.proc) + ".tmp", h5(d, a.proc))
    shutil.rmtree(work, ignore_errors=True)
    print(f"GEN {a.proc}: {len(me2)} events, sigma {sigma} pb -> {h5(d, a.proc)}")


def cmd_predict(a):
    from tools.kit_predict import predict
    out = os.path.join(os.path.abspath(a.dir), "pred", os.path.basename(a.run.rstrip("/")))
    _, path = predict(a.run, ["mc"], out, h5=h5(os.path.abspath(a.dir), a.proc))
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
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


def _selfcheck():
    rng = np.random.default_rng(1)
    m = rng.lognormal(0, 2, 20000)
    r = mc_metrics(m, 1.1 * m, boot=50)
    assert abs(r["bias"] - 0.1) < 1e-12 and abs(r["eps2"]["max"] - 1) < 1e-12
    noisy = m * rng.lognormal(0, 0.1, len(m))
    r = mc_metrics(m, noisy, boot=50)
    assert r["eps2"]["0.99"] > r["eps2"]["max"] and 0 < r["eps2"]["max"] < 1
    print("selfcheck ok")


def cmd_figure(a):
    import plot_style as ps
    ps.use()
    d = json.load(open(a.json))
    panels = ps.panels(2)
    (f1, a1), (f2, a2) = panels
    runs = sorted(d, key=lambda k: d[k]["process"])
    lab = {"ee_uug": r"$e^+e^-\to u\bar ug$", "ee_uugg": r"$e^+e^-\to u\bar ugg$"}
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange]
    for i, k in enumerate(runs):
        e = d[k]
        a1.errorbar([i], [abs(e["bias"])], yerr=[[abs(e["bias"]) - min(abs(x) for x in e["bias_band"])],
                    [max(abs(x) for x in e["bias_band"]) - abs(e["bias"])]], fmt="o", color=cols[i], label=lab[e["process"]])
        a1.plot([i], [e["bias_stat"]], "_", color=cols[i], ms=14)
        x = [1 - float(q) for q in QS] + [1 / e["n"]]
        a2.plot(x, [e["eps2"][str(q)] for q in QS] + [e["eps2"]["max"]], "o-", color=cols[i], label=lab[e["process"]])
    a1.plot([], [], "_", color=ps.C.grey, ms=14, label="MC error")
    a1.set_yscale("log"); a1.set_xticks(range(len(runs)), [lab[d[k]["process"]] for k in runs])
    a1.set_ylabel(r"$|\hat\sigma/\sigma-1|$"); a1.set_xlim(-0.5, len(runs) - 0.5)
    a2.set_xscale("log"); a2.set_xlabel(r"$1-q$ ($r_{\max}$ at the $q$-quantile of $r$; leftmost: the maximum)")
    a2.set_ylabel(r"$\epsilon_2=\langle r\rangle/r_{\max}$")
    a1.legend(); a2.legend()
    ps.save_panels(panels, a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("proc", choices=PROC); g.add_argument("--dir", required=True)
    g.add_argument("--n", type=int, default=100000); g.add_argument("--seed", type=int, default=11)
    g.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p = sub.add_parser("predict"); p.add_argument("proc", choices=PROC); p.add_argument("--dir", required=True)
    p.add_argument("run")
    m = sub.add_parser("metrics"); m.add_argument("--dir", required=True); m.add_argument("--out", required=True)
    m.add_argument("--boot", type=int, default=1000)
    fg = sub.add_parser("figure"); fg.add_argument("json"); fg.add_argument("--out", required=True)
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    {"gen": cmd_gen, "predict": cmd_predict, "metrics": cmd_metrics, "figure": cmd_figure,
     "selfcheck": lambda a: _selfcheck()}[a.cmd](a)


if __name__ == "__main__":
    main()
