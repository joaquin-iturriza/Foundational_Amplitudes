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
import time

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
    full = ML.load(NV.find_p0(sa))   # stays in the P0 dir: MadLoop finds its resources from the cwd at each call

    def evaluate(mom):
        rs = [ML.evaluate(full, m, alphas=ALPHA_S) for m in mom]
        return {k: np.array([r[k] for r in rs]) for k in ("born", "fin", "e1", "e2", "rc")}
    return evaluate


def kit_file():
    from tools import kit_hardps as kh
    return kh.h5_path("ee_tt")


def cmd_build(a):
    import h5py
    a.dir = os.path.abspath(a.dir)
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
    """{sigma, err, scale_up, scale_dn} (pb, scale band in %) of MG5_aMC fixed order at sqrt s (order LO or NLO) from its
    own output dir `out`, or {failed: reason} when MG5_aMC stops (its soft-limit tests fail near threshold: beta 1e-3)."""
    import mg5_pipeline_final as mg
    ncore = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    script = (f"set nb_core {ncore}\nlaunch {out}\nfixed_order=ON\norder={order}\ndone\n"
              + AMC_CARD.format(e=sq / 2, mz=MZ, acc=acc, mt=MT))
    t0 = time.time()
    p = subprocess.run([mg.MG5_BIN], input=script, text=True, capture_output=True, timeout=3600)
    m = re.findall(r"Total cross section:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)\s*pb", p.stdout)
    if not m:
        dbg = [d for d in glob.glob(os.path.join(out, "run_*_debug.log")) if os.path.getmtime(d) >= t0]  # aMCatNLOError
        why = re.findall(r"(aMCatNLOError[^\n]*|Soft test[^\n]*FAILED[^\n]*)",
                         p.stdout + p.stderr + (open(dbg[-1]).read() if dbg else ""))
        return {"failed": "; ".join(dict.fromkeys(why))[:500] or "no cross section in MG5_aMC's output"}
    r = {"sigma": float(m[-1][0]), "err": float(m[-1][1])}
    runs = sorted(glob.glob(os.path.join(out, "Events", "run_*", "summary.txt")), key=os.path.getmtime)
    band = scale_band(p.stdout) or (scale_band(open(runs[-1]).read()) if runs and order == "NLO" else None)
    if band:
        r["scale_up"], r["scale_dn"] = band
    return r


# e+e- -> t t~ through gamma and Z at LO (loop_sm defaults: G_F, 1/alpha = 132.507, M_Z, Gamma_Z; m_W from them)
GF, AEWM1, GZ, CF = 1.16639e-5, 132.507, 2.441404, 4 / 3


def lo_vector_axial(sq):
    """(beta, sigma_V, sigma_A) in pb at LO: the vector (beta (3 - beta^2)/2) and axial (beta^3) parts of the top current."""
    a = 1 / AEWM1
    mw2 = MZ ** 2 / 2 + np.sqrt(MZ ** 4 / 4 - a * np.pi * MZ ** 2 / (GF * np.sqrt(2)))
    sw2 = 1 - mw2 / MZ ** 2
    s = np.asarray(sq, float) ** 2
    b = np.sqrt(1 - 4 * MT ** 2 / s)
    chi = s / (s - MZ ** 2 + 1j * MZ * GZ) / (4 * sw2 * (1 - sw2))
    qe, qt, ve, ae, vt, at = -1, 2 / 3, -0.5 + 2 * sw2, -0.5, 0.5 - 4 / 3 * sw2, 0.5
    cv = qe ** 2 * qt ** 2 + 2 * qe * qt * ve * vt * chi.real + (ve ** 2 + ae ** 2) * vt ** 2 * abs(chi) ** 2
    ca = (ve ** 2 + ae ** 2) * at ** 2 * abs(chi) ** 2
    pt = 4 * np.pi * a ** 2 / (3 * s) * GEV2_PB * 3
    return b, pt * b * (3 - b * b) / 2 * cv, pt * b ** 3 * ca


def r1_vector(v):
    """R_V^(1) / R_V^(0): the exact O(alpha_s) vector-current correction in units of C_F alpha_s / pi
    (Chetyrkin, Kuehn, Steinhauser, hep-ph/9606230, eq. 34)."""
    from scipy.special import spence                  # spence(1 - x) = Li2(x)
    li2 = lambda x: spence(1 - x)
    v = np.asarray(v, float)
    p, lr = (1 - v) / (1 + v), np.log((1 + v) / (1 - v))
    r1 = (3 - v ** 2) / 2 * ((1 + v ** 2) * (li2(p ** 2) + 2 * li2(p) + lr * np.log((1 + v) ** 3 / (8 * v ** 2)))
                             + 3 * v * np.log((1 - v ** 2) / (4 * v)) - v * np.log(v)
                             + (33 + 22 * v ** 2 - 7 * v ** 4) / (8 * (3 - v ** 2)) * lr + (15 * v - 9 * v ** 3) / (4 * (3 - v ** 2)))
    return r1 / (v * (3 - v ** 2) / 2)


def r1_axial_threshold(v):
    """R_A^(1) / R_A^(0) near threshold, pi^2 (1 + v^2) / (2 v) - 2, relative error O(v^2) (hep-ph/9705254, eq. 15).
    shortcut: the threshold expansion, not the exact axial result; used where the axial part is < 1e-3 of sigma_LO."""
    v = np.asarray(v, float)
    return np.pi ** 2 * (1 + v ** 2) / (2 * v) - 2


def alpha_s_run(mu, n=200):
    """alpha_s(mu), two-loop running with 5 flavours from alpha_s(M_Z) = ALPHA_S (RK4 in ln mu^2)."""
    b0, b1 = 23 / (12 * np.pi), 116 / (24 * np.pi ** 2)
    f = lambda x: -b0 * x * x - b1 * x ** 3
    x, h = ALPHA_S, np.log(mu ** 2 / MZ ** 2) / n
    for _ in range(n):
        k1 = f(x); k2 = f(x + h * k1 / 2); k3 = f(x + h * k2 / 2); k4 = f(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def nlo_analytic(sq, as_var=None):
    """(K, scale_up %, scale_dn %) of the analytic fixed-order NLO: K = sigma_NLO / sigma_LO at alpha_s(M_Z) = ALPHA_S,
    vector exact, axial at its threshold expansion; the band varies mu_R in {M_Z/2, 2 M_Z} as MG5_aMC does: sigma_LO
    does not move and the correction scales with alpha_s(mu) / alpha_s(M_Z) - 1 = as_var (up, down), by default
    two-loop running (MG5_aMC's own running differs by ~0.5% of alpha_s: collect takes as_var from its band)."""
    b, sv, sa = lo_vector_axial(sq)
    corr = CF / np.pi * (sv * r1_vector(b) + sa * r1_axial_threshold(b)) / (sv + sa)      # per unit alpha_s
    k = 1 + ALPHA_S * corr
    as_var = as_var or [alpha_s_run(m) / ALPHA_S - 1 for m in (MZ / 2, 2 * MZ)]
    var = [x * (k - 1) / k for x in as_var]
    return k, 100 * np.maximum(*var), -100 * np.minimum(*var)


def scale_band(text):
    """(+%, -%) of MG5_aMC's scale-variation envelope ("Dynamical_scale_choice -2 (envelope of 9 values):" then
    "5.102e-01 pb  +9.3% -7.5%"), None when the text has none."""
    b = re.findall(r"envelope of \d+ values\):\s*[0-9.eE+-]+ pb\s*\+([0-9.]+)%\s*-([0-9.]+)%", text)
    return (float(b[-1][0]), float(b[-1][1])) if b else None


def cmd_nlo(a):
    import mg5_pipeline_final as mg
    k, K = (int(x) for x in a.part.split("/"))
    a.dir = os.path.abspath(a.dir)
    sq = grid()[0]
    out = os.path.join(os.path.abspath(a.dir), f"amc_{k}")
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        r = subprocess.run([mg.MG5_BIN], input=f"import model loop_sm\ngenerate e+ e- > t t~ [QCD]\noutput {out}\n",
                           text=True)
        assert r.returncode == 0 and os.path.isdir(out), "MG5_aMC output failed"
    os.makedirs(os.path.join(a.dir, "nlo"), exist_ok=True)
    for i in (a.only if a.only else range(k, len(sq), K)):
        res = os.path.join(a.dir, "nlo", f"sqrts_{i:02d}.json")
        if os.path.exists(res):
            continue
        lo = amc_sigma(out, sq[i], "LO", a.acc)
        nlo = amc_sigma(out, sq[i], "NLO", a.acc)
        json.dump({"i": i, "sqrts": sq[i], "lo": lo, "nlo": nlo}, open(res, "w"))
        print(f"AMC sqrt s {sq[i]:.6f}: LO {lo}, NLO {nlo}", flush=True)
    print(f"AMC_DONE part {a.part}")


def cmd_predict(a):
    from tools import kit_predict
    for run in a.runs:
        out = os.path.join(a.dir, "pred", os.path.basename(run.rstrip("/")))
        _, path = kit_predict.predict(run, ["scan"], out, h5=os.path.join(a.dir, "ee_tt.h5"))
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
    ok = lambda r: r if r and "sigma" in r else None      # a point MG5_aMC could not run (its tests failed): no exact NLO
    out["amc"] = {k: [ok(got[i][k]) if i in got else None for i in range(len(sq))] for k in ("lo", "nlo")}
    # below the lowest beta where MG5_aMC's NLO runs, the exact NLO is the analytic one (K x the quadrature's sigma_LO)
    kf, _, _ = nlo_analytic(sq)
    lo_q = np.asarray(out["sigma_lo_quad"])
    amc_b = [beta[i] for i, r in enumerate(out["amc"]["nlo"]) if r]
    # alpha_s(mu)/alpha_s(M_Z) - 1 at mu = M_Z/2, 2 M_Z as MG5_aMC runs it: its band over the correction's share (K-1)/K
    share = {i: 1 - lo_q[i] / r["sigma"] for i, r in enumerate(out["amc"]["nlo"]) if r and "scale_up" in r}
    as_var = ([float(np.median([out["amc"]["nlo"][i]["scale_up"] / 100 / f for i, f in share.items()])),
               -float(np.median([out["amc"]["nlo"][i]["scale_dn"] / 100 / f for i, f in share.items()]))]
              if share else None)
    out["alpha_s_var"] = as_var
    kf, kup, kdn = nlo_analytic(sq, as_var)
    out["analytic_check"] = [{"beta": float(beta[i]), "ratio": float(kf[i] * lo_q[i] / r["sigma"] - 1),
                              "mc_err": r["err"] / r["sigma"], "scale_up": [float(kup[i]), r.get("scale_up")],
                              "scale_dn": [float(kdn[i]), r.get("scale_dn")]} for i, r in enumerate(out["amc"]["nlo"]) if r]
    for c in out["analytic_check"]:
        print("CLAIM " + json.dumps({"analytic_vs_mg5_aMC": c}))
    for i in range(len(sq)):
        if out["amc"]["nlo"][i] is None and (not amc_b or beta[i] < min(amc_b)):
            out["amc"]["nlo"][i] = {"sigma": float(kf[i] * lo_q[i]), "err": 0.0, "scale_up": float(kup[i]),
                                    "scale_dn": float(kdn[i]), "source": "analytic"}
    for p in sorted(glob.glob(os.path.join(a.dir, "pred", "*", "predictions.h5"))):
        with h5py.File(p, "r") as f:
            v = np.concatenate([10 ** f[k]["log10_abs_virt_finite"][:] for k in idx])
            run = str(f.attrs["run_dir"])
        sv = sigma_pb(sq, beta, w, v)
        out["surrogates"][os.path.basename(run)] = {"run_dir": run, "sigma_v": sv.tolist()}
    nlo_s = np.array([r["sigma"] if r else np.nan for r in out["amc"]["nlo"]])
    nlo_e = np.array([r["err"] if r else np.nan for r in out["amc"]["nlo"]])
    up = np.array([r.get("scale_up", np.nan) if r else np.nan for r in out["amc"]["nlo"]]) / 100
    dn = np.array([r.get("scale_dn", np.nan) if r else np.nan for r in out["amc"]["nlo"]]) / 100
    has = np.isfinite(nlo_s)
    noband = [i for i in np.flatnonzero(has) if not np.isfinite(up[i])]
    assert not noband, f"NLO points without a scale band (MG5_aMC's envelope not read): {noband}"
    vex = np.asarray(out["sigma_v_exact"])
    for name, sv in out["surrogates"].items():
        ratio = (np.asarray(sv["sigma_v"]) - vex) / nlo_s          # sigma_NLO^surr / sigma_NLO^exact - 1
        sv["ratio"] = [float(x) if np.isfinite(x) else None for x in ratio]
        sv["unresolved"] = [bool(h and abs(r) < 2 * e / n) for h, r, e, n in zip(has, ratio, nlo_e, nlo_s)]
        sv["in_scale_band"] = [bool(h and -d <= r <= u) for h, r, u, d in zip(has, ratio, up, dn)]
        for lvl in (0.01, 0.10):
            # the smallest beta from which |ratio| stays below lvl at every larger beta with an exact NLO
            bad = [i for i in np.flatnonzero(has) if abs(ratio[i]) >= lvl]
            good = [i for i in np.flatnonzero(has) if not bad or i > max(bad)]
            sv[f"beta_{int(lvl * 100)}pct"] = float(beta[min(good)]) if good else None
        print("CLAIM " + json.dumps({"run": name, "beta_1pct": sv["beta_1pct"], "beta_10pct": sv["beta_10pct"],
                                     "lowest_beta_with_exact_nlo": float(beta[has].min()) if has.any() else None}))
    json.dump(out, open(a.out, "w"), indent=1)
    lo_q = np.asarray(out["sigma_lo_quad"])
    for i in range(len(sq)):
        amc = out["amc"]["lo"][i]
        dev = f"{lo_q[i] / amc['sigma'] - 1:+.1e} ({amc['err'] / amc['sigma']:.0e} MC)" if amc else "no MG5_aMC"
        print(f"sqrt s {sq[i]:.4f} beta {beta[i]:.2e}: sigma_LO quad {lo_q[i]:.5g} pb vs MG5_aMC {dev}; "
              f"sigma[V] {out['sigma_v_exact'][i]:.5g}; " +
              " ".join(f"{label(k)} {s['sigma_v'][i] / out['sigma_v_exact'][i] - 1:+.2e}" for k, s in out["surrogates"].items()))
    print(f"COLLECTED {a.out}")


def label(run):
    """A short legend label from a KIT run name: what its protocol (A, A_ft, B, B_bulk) trained on."""
    names = {"A": "from scratch", "A_ft": "fine-tuned", "B": "hard points added", "B_bulk": "bulk continuation"}
    m = re.match(r"kit_(A_ft|B_bulk|A|B)_", run)
    inputs = {"softpair": ", log inputs", "softpairthr": r", log inputs + $\log\beta^2$",
              "vbeta": r", log inputs + $\log\beta^2$, target $V\beta$",
              # D405 fine-tunes: parent pretraining r5, + LO top pairs log-uniform in beta, + the beta input
              "ftr5own": ", parent r5", "ftr5thrown": ", parent r5 + threshold tops",
              "ftr5thrbetaown": r", parent r5 + threshold tops, $\log\beta^2$"}
    # step 18 (D435): log|M|^2 = log A + g, A the asymptotic factors (1/beta, log^2(s/m^2)); A1 on the log inputs, A2 on
    # plain momenta with the frames' boost capped at gamma <= 4
    asym = {"softpairthr_asym_plainlog": r", log inputs + $\log\beta^2$, asymptotic factor",
            "asym_cont_plainlog": r", asymptotic factor, no log inputs, $\gamma\leq4$"}
    tag = next((t for t in asym if run.endswith("_" + t)), None)
    if tag:
        inputs[run.rsplit("_", 1)[-1]] = asym[tag]
    n = re.search(r"_n(\d+)_", run)
    size = {"100000": r", $10^5$ events", "900000": r", $9\times10^5$ events"}.get(n.group(1), "") if n else ""
    return names[m.group(1)] + size + inputs.get(run.rsplit("_", 1)[-1], "") if m else run


def cmd_figure(a):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(a.json))
    sq = np.asarray(d["beta"])            # x axis: beta = sqrt(1 - 4 m_t^2 / s), log-spaced like the scan
    ok = np.array([x is not None for x in d["amc"]["nlo"]])
    ana = [b for b, x in zip(sq, d["amc"]["nlo"]) if x and x.get("source") == "analytic"]
    get = lambda k, key, scale=1.0: np.array([x.get(key, np.nan) * scale if x else np.nan for x in d["amc"][k]])
    lo, nlo, err = get("lo", "sigma"), get("nlo", "sigma"), get("nlo", "err")
    up, dn = get("nlo", "scale_up", 0.01), get("nlo", "scale_dn", 0.01)
    fig, (ax, rx) = ps.figure(1, 2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.fill_between(sq[ok], (nlo * (1 - dn))[ok], (nlo * (1 + up))[ok], color="0.85", lw=0, label="NLO scale band")
    ax.plot(sq[ok], lo[ok], "k--", label="LO")
    ax.plot(sq[ok], nlo[ok], "k-", label="NLO")
    rx.fill_between(sq[ok], -dn[ok], up[ok], color="0.85", lw=0)
    rx.fill_between(sq[ok], (-2 * err / nlo)[ok], (2 * err / nlo)[ok], color="0.6", lw=0)
    ax.fill_between([], [], [], color="0.6", lw=0, label=r"$\pm2\,\delta_{\rm MC}$ of the NLO (lower panel)")
    # one categorical colour per model, the same in both panels (a sequential ramp made them indistinguishable)
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky]
    sur = {k: v for k, v in d["surrogates"].items() if not a.runs or k in a.runs}
    assert len(sur) <= len(cols), f"{len(sur)} models, {len(cols)} colours: pick them with --runs"
    for (name, s), col in zip(sur.items(), cols):
        r = np.array([x if x is not None else np.nan for x in s["ratio"]])
        ax.plot(sq[ok], (nlo * (1 + r))[ok], color=col, ls=":", label=label(name))
        rx.plot(sq[ok], r[ok], color=col)
    rx.axhline(0, color="k", lw=0.8)
    if ana:      # left of this line the exact NLO is the analytic O(alpha_s) one, right of it MG5_aMC's
        for x, lab in ((ax, "exact NLO from the analytic formula (left) or MadGraph (right)"), (rx, None)):
            x.axvline(np.sqrt(max(ana) * sq[len(ana)]), color="0.4", lw=0.8, ls="-.", label=lab)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma(e^+e^-\to t\bar t)$ [pb]")
    rx.set_ylabel(r"$\sigma^{\rm surr}_{\rm NLO}/\sigma_{\rm NLO}-1$")
    rx.set_xlabel(r"$\beta=\sqrt{1-4m_t^2/s}$")
    # the legend covered the data in a corner: a strip of its own (<out>_legend), shown above the figure
    ps.legend_strip(ax, a.out + "_legend", ncol=2)
    print(f"FIGURE {ps.save(fig, a.out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-check", type=int, default=200)
    b.add_argument("--tol", type=float, default=1e-8)
    n = sub.add_parser("nlo")
    n.add_argument("--part", default="0/1")
    n.add_argument("--acc", type=float, default=1e-3)
    n.add_argument("--only", type=int, nargs="*", default=None, help="these sqrt s indices only (the smoke run)")
    p = sub.add_parser("predict")
    p.add_argument("runs", nargs="+")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    f = sub.add_parser("figure")
    f.add_argument("json")
    f.add_argument("--out", required=True)
    f.add_argument("--runs", nargs="*", help="these surrogates only (run names), at most six")
    for s in (b, n, p, c):
        s.add_argument("--dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    {"build": cmd_build, "nlo": cmd_nlo, "predict": cmd_predict, "collect": cmd_collect, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
