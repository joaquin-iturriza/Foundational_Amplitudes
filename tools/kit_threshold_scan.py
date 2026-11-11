#!/usr/bin/env python3
"""The e+e- -> t t~ threshold scan of kit_benchmark (Gudrun): sigma(sqrt s) at LO and NLO QCD around 2 m_t, exact and
with the surrogate's one-loop virtual, like Fig. 1 of arXiv:1506.06864 at fixed order.

    python tools/kit_threshold_scan.py build   [--dir D]           # MadLoop standalone, its check on KIT points, scan file
    python tools/kit_threshold_scan.py nlo     [--dir D] --part k/K # MG5_aMC fixed-order LO and NLO cross sections
    python tools/kit_threshold_scan.py predict [--dir D] <run_dir> ...   # surrogate log10 V on the scan points
    python tools/kit_threshold_scan.py collect [--dir D] --out <json>    # the cross sections per sqrt s
    python tools/kit_threshold_scan.py figure  <json> --out <stem>       # local: the figure pair

Settings are the KIT file's (metadata of ee_tt.h5): MadGraph loop_sm defaults with m_t = 173 GeV, alpha_s = 0.118
fixed, 1/alpha = 132.507; the virtual is MadLoop's finite part of 2 Re(M0* M1) at mu_R = sqrt s (KIT's virt_finite).
Scan: sqrt s = 2 m_t / sqrt(1 - beta^2) at N_BETA values of beta log-spaced in [BETA_LO, BETA_HI]; at each, N_COS
Gauss-Legendre nodes in cos(theta) of the top (phi = 0; unpolarised beams, |M|^2 does not depend on phi), so
    sigma = GEV2_PB / (2 s) * beta / (16 pi) * sum_k w_k |M|^2(cos theta_k)
(two-body phase space beta / (32 pi^2) dOmega; |M|^2 at one-loop QCD is a quadratic in cos theta, so the quadrature is
exact). The exact NLO is MG5_aMC's fixed-order e+ e- > t t~ [QCD] (real emission and its subtraction included) at
mu_R = M_Z, where alpha_s = 0.118 (at this order sigma depends on mu_R only through alpha_s). The surrogate's NLO
replaces the exact virtual by the network's:
    sigma_NLO^surr = sigma_NLO^exact + sigma[V_surr] - sigma[V_exact]   (both V at mu_R = sqrt s, alpha_s = 0.118)
Checks: build compares the standalone's Born and finite part with KIT's stored values on its bulk_test and threshold
points (they must agree, else the scan's V is not KIT's); collect compares the quadrature's sigma_LO with MG5_aMC's.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

GEV2_PB = 0.3893794e9
MT, MZ, ALPHA_S = 173.0, 91.188, 0.118
N_BETA, BETA_LO, BETA_HI, N_COS = 32, 1e-3, 0.6, 16
PROC = "kit_ee_tt_virt"          # the standalone's name under mg5_pipeline_final.WORK_DIR (its own: m_t = 173)
DEFAULT_DIR = os.path.join("runs", "kit", "threshold_scan")


def grid():
    """sqrt s (N_BETA,), beta, cos nodes and weights (N_COS,), momenta (N_BETA * N_COS, 4, 4) in KIT order (e+ along
    +z, e-, t, t~), the cos index fastest."""
    beta = np.geomspace(BETA_LO, BETA_HI, N_BETA)
    sq = 2 * MT / np.sqrt(1 - beta ** 2)
    c, w = np.polynomial.legendre.leggauss(N_COS)
    s_ = np.sqrt(1 - c ** 2)
    mom = []
    for e, b in zip(sq / 2, beta):
        p = b * e
        for ci, si in zip(c, s_):
            t = [e, p * si, 0.0, p * ci]
            mom.append([[e, 0, 0, e], [e, 0, 0, -e], t, [e, -t[1], 0.0, -t[3]]])
    return sq, beta, c, w, np.asarray(mom, dtype=np.float64)


def sigma_pb(sq, beta, w, me2):
    """sigma per sqrt s from |M|^2 at the grid's nodes, (N_BETA * N_COS,) -> (N_BETA,)"""
    m = np.asarray(me2, dtype=np.float64).reshape(len(sq), len(w))
    return GEV2_PB / (2 * sq ** 2) * beta / (16 * np.pi) * (m @ w)


def madloop():
    """evaluate(mom (N, 4, 4)) -> dict of arrays born, fin, e1, e2, rc: MadLoop at alpha_s 0.118, mu_R^2 = s."""
    import nlo_madloop as ML
    import nlo_virtual_pipeline as NV
    NV.VIRT_PROCESSES[PROC] = {"mg5": "generate e+ e- > t t~ [virt=QCD]", "pdg_ids": [-11, 11, 6, -6],
                               "m_finals": [MT, MT], "model": "loop_sm",
                               "param_card_patches": {"MT": MT, "ymt": MT}}
    sa = NV.build_virt_standalone(PROC)
    cwd = os.getcwd()
    full = ML.load(NV.find_p0(sa))
    os.chdir(cwd)

    def evaluate(mom):
        rs = [ML.evaluate(full, m, alphas=ALPHA_S) for m in mom]
        return {k: np.array([r[k] for r in rs]) for k in ("born", "fin", "e1", "e2", "rc")}
    return evaluate


def kit_file():
    from tools import kit_hardps as kh
    return kh.h5_path("ee_tt")


def cmd_build(a):
    import h5py
    ev = madloop()
    os.makedirs(a.dir, exist_ok=True)
    check = {}
    with h5py.File(kit_file(), "r") as f:
        meta = f.attrs["metadata"]
        groups = ["bulk_test/bulk"] + [f"hard_test/{k}" for k in sorted(f["hard_test"]) if k.startswith("threshold")]
        for gname in groups:
            g = f[gname]
            r = ev(g["momenta"][:a.n_check])
            rel = {k: float(np.max(np.abs(r[src] / g[k][:a.n_check] - 1))) for k, src in
                   (("born", "born"), ("virt_finite", "fin"))}
            check[gname] = rel
            print(f"CHECK {gname}: max |ours/KIT - 1| born {rel['born']:.2e}, virt_finite {rel['virt_finite']:.2e}")
        bulk = {k: f["bulk_test/bulk"][k][:1000] for k in f["bulk_test/bulk"]}
    worst = max(max(v.values()) for v in check.values())
    assert worst < a.tol, f"the standalone does not reproduce KIT's values (worst {worst:.2e} >= {a.tol})"
    sq, beta, c, w, mom = grid()
    r = ev(mom)
    assert (r["rc"] // 100 != 4).all(), "an unstable MadLoop point on the scan"
    assert (r["fin"] > 0).all(), "a non-positive virtual on the scan (log10 V undefined)"
    path = os.path.join(a.dir, "ee_tt.h5")
    with h5py.File(path + ".tmp", "w") as f:
        f.attrs["metadata"] = meta
        for k, v in bulk.items():          # kit_hardps.log10_name reads the target names from bulk_test/bulk
            f[f"bulk_test/bulk/{k}"] = v
        for i in range(len(sq)):
            sl = slice(i * len(c), (i + 1) * len(c))
            g = f.create_group(f"scan/sqrts_{i:02d}")
            g["momenta"] = mom[sl]
            g["born"], g["virt_finite"] = r["born"][sl], r["fin"][sl]
            g["virt_eps1"], g["virt_eps2"] = r["e1"][sl], r["e2"][sl]
            g["log10_abs_born"], g["log10_abs_virt_finite"] = np.log10(r["born"][sl]), np.log10(r["fin"][sl])
            g["singular_var"] = np.full(len(c), beta[i])
    os.replace(path + ".tmp", path)
    json.dump({"sqrts": sq.tolist(), "beta": beta.tolist(), "cos": c.tolist(), "w": w.tolist(), "check": check,
               "n_check": a.n_check}, open(os.path.join(a.dir, "scan_grid.json"), "w"), indent=1)
    print(f"SCAN_BUILT {path}: {len(sq)} sqrt s x {len(c)} cos nodes; worst check {worst:.2e}")


AMC_CARD = """set lpp1 0
set lpp2 0
set ebeam1 {e:.10f}
set ebeam2 {e:.10f}
set fixed_ren_scale True
set fixed_fac_scale True
set mur_ref_fixed {mz}
set muf_ref_fixed {mz}
set req_acc_FO {acc}
set MT {mt}
set ymt {mt}
done
"""


def amc_sigma(out, sq, order, acc):
    """(sigma, error) in pb of MG5_aMC fixed order at sqrt s, order LO or NLO, from its own output dir `out`."""
    import mg5_pipeline_final as mg
    script = f"launch {out}\nfixed_order=ON\norder={order}\ndone\n" + AMC_CARD.format(e=sq / 2, mz=MZ, acc=acc, mt=MT)
    p = subprocess.run([mg.MG5_BIN], input=script, text=True, capture_output=True, timeout=3600)
    m = re.findall(r"Total cross section:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)\s*pb", p.stdout)
    if not m:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-2000:])
        raise RuntimeError(f"MG5_aMC gave no cross section at sqrt s {sq} ({order})")
    return float(m[-1][0]), float(m[-1][1])


def cmd_nlo(a):
    import mg5_pipeline_final as mg
    k, K = (int(x) for x in a.part.split("/"))
    sq = grid()[0]
    out = os.path.join(os.path.abspath(a.dir), f"amc_{k}")
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        r = subprocess.run([mg.MG5_BIN], input=f"import model loop_sm\ngenerate e+ e- > t t~ [QCD]\noutput {out}\n",
                           text=True)
        assert r.returncode == 0 and os.path.isdir(out), "MG5_aMC output failed"
    os.makedirs(os.path.join(a.dir, "nlo"), exist_ok=True)
    for i in range(k, len(sq), K):
        res = os.path.join(a.dir, "nlo", f"sqrts_{i:02d}.json")
        if os.path.exists(res):
            continue
        lo = amc_sigma(out, sq[i], "LO", a.acc)
        nlo = amc_sigma(out, sq[i], "NLO", a.acc)
        json.dump({"i": i, "sqrts": sq[i], "lo": lo, "nlo": nlo}, open(res, "w"))
        print(f"AMC sqrt s {sq[i]:.6f}: LO {lo[0]:.6g} +- {lo[1]:.2g}, NLO {nlo[0]:.6g} +- {nlo[1]:.2g} pb", flush=True)
    print(f"AMC_DONE part {a.part}")


def cmd_predict(a):
    os.environ["KIT_HARDPS_DIR"] = os.path.abspath(a.dir)
    from tools import kit_predict
    for run in a.runs:
        out = os.path.join(a.dir, "pred", os.path.basename(run.rstrip("/")))
        _, path = kit_predict.predict(run, ["scan"], out)
        print(f"PREDICTED {run} -> {path}")


def cmd_collect(a):
    import h5py
    g = json.load(open(os.path.join(a.dir, "scan_grid.json")))
    sq, beta, w = (np.asarray(g[k]) for k in ("sqrts", "beta", "w"))
    idx = [f"scan/sqrts_{i:02d}" for i in range(len(sq))]
    with h5py.File(os.path.join(a.dir, "ee_tt.h5"), "r") as f:
        born = np.concatenate([f[k]["born"][:] for k in idx])
        vex = np.concatenate([f[k]["virt_finite"][:] for k in idx])
    out = {"sqrts": sq.tolist(), "beta": beta.tolist(), "check": g["check"],
           "sigma_lo_quad": sigma_pb(sq, beta, w, born).tolist(),
           "sigma_v_exact": sigma_pb(sq, beta, w, vex).tolist(), "surrogates": {}}
    nlo = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "nlo", "sqrts_*.json")))]
    got = {r["i"]: r for r in nlo}
    out["amc"] = {k: [got[i][k] if i in got else None for i in range(len(sq))] for k in ("lo", "nlo")}
    for p in sorted(glob.glob(os.path.join(a.dir, "pred", "*", "predictions.h5"))):
        with h5py.File(p, "r") as f:
            v = np.concatenate([10 ** f[k]["log10_abs_virt_finite"][:] for k in idx])
            run = str(f.attrs["run_dir"])
        sv = sigma_pb(sq, beta, w, v)
        out["surrogates"][os.path.basename(run)] = {"run_dir": run, "sigma_v": sv.tolist()}
    json.dump(out, open(a.out, "w"), indent=1)
    lo_q = np.asarray(out["sigma_lo_quad"])
    for i in range(len(sq)):
        amc = out["amc"]["lo"][i]
        dev = f"{lo_q[i] / amc[0] - 1:+.1e} ({amc[1] / amc[0]:.0e} MC)" if amc else "no MG5_aMC"
        print(f"sqrt s {sq[i]:.4f} beta {beta[i]:.2e}: sigma_LO quad {lo_q[i]:.5g} pb vs MG5_aMC {dev}; "
              f"sigma[V] {out['sigma_v_exact'][i]:.5g}; " +
              " ".join(f"{k[-30:]} {s['sigma_v'][i] / out['sigma_v_exact'][i] - 1:+.2e}" for k, s in out["surrogates"].items()))
    print(f"COLLECTED {a.out}")


def cmd_figure(a):
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(a.json))
    sq = np.asarray(d["sqrts"])
    ok = np.array([x is not None for x in d["amc"]["nlo"]])
    lo = np.array([x[0] if x else np.nan for x in d["amc"]["lo"]])
    nlo = np.array([x[0] if x else np.nan for x in d["amc"]["nlo"]])
    vex = np.asarray(d["sigma_v_exact"])
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(sq[ok], lo[ok], "k--", label="LO")
    ax.plot(sq[ok], nlo[ok], "k-", label="NLO")
    for (name, s), col in zip(d["surrogates"].items(), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        lab = a.labels.get(name, name) if a.labels else name
        sur = nlo + np.asarray(s["sigma_v"]) - vex
        ax.plot(sq[ok], sur[ok], color=col, ls=":", label=f"NLO, {lab}")
        rx.plot(sq[ok], (sur / nlo - 1)[ok], color=col, label=lab)
    rx.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$\sigma(e^+e^-\to t\bar t)$ [pb]")
    rx.set_ylabel(r"$\sigma^{\rm surr}_{\rm NLO}/\sigma_{\rm NLO}-1$")
    rx.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=150)
    print(f"FIGURE {a.out}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-check", type=int, default=200)
    b.add_argument("--tol", type=float, default=1e-8)
    n = sub.add_parser("nlo")
    n.add_argument("--part", default="0/1")
    n.add_argument("--acc", type=float, default=1e-3)
    p = sub.add_parser("predict")
    p.add_argument("runs", nargs="+")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    f = sub.add_parser("figure")
    f.add_argument("json")
    f.add_argument("--out", required=True)
    f.add_argument("--labels", type=json.loads, default=None, help='{"run name": "legend label"}')
    for s in (b, n, p, c):
        s.add_argument("--dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    {"build": cmd_build, "nlo": cmd_nlo, "predict": cmd_predict, "collect": cmd_collect, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
