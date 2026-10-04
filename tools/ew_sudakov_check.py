#!/usr/bin/env python3
r"""Feasibility check for the Sudakov star arm (docs/results.tex sec:ladder hand-off): does a MadLoop electroweak
one-loop target carry the Sudakov logarithm log^2(s/M_W^2)? Builds two standalones of e+e- -> d d~ through the
one-loop pipeline (tools/nlo_virtual_pipeline.py: GET_ME_FULL wrapper, locked param_card):
    ee_dd_ewcheck   [virt=QED], model loop_qcd_qed_sm    (the candidate probe)
    ee_dd           [virt=QCD], model loop_sm            (the current probe's target, for contrast)
and evaluates each at fixed scattering angle for sqrt(s) from 100 GeV to 3 TeV, mu_R^2 = s (the pools' scale).
The relative correction delta = finite/born is fitted as a + b L + c L^2, L = ln(s/M_W^2), over sqrt(s) >= 200 GeV:
an electroweak Sudakov target has c < 0 of order alpha/(4 pi) times a few (~1e-3); the QCD vertex at mu = sqrt(s)
should be flat (the Born times a constant), which this also checks. Writes analysis/transfer/ew_sudakov_check.json.
CPU only:  python tools/ew_sudakov_check.py
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

MW = 80.419
SQRTS = np.geomspace(100.0, 3000.0, 30)
COS = (0.0, 0.5, -0.5)
CASES = {"ee_dd_ewcheck": {"mg5": "generate e+ e- > d d~ [virt=QED]", "model": "loop_qcd_qed_sm",
                           "pdg_ids": [11, -11, 1, -1], "m_finals": [0.0, 0.0]},
         "ee_dd": None}   # the pipeline's own [virt=QCD] entry


def one(name):
    """Build (idempotent) and evaluate one standalone; prints one JSON line. One standalone per process (the f2py
    module and MadLoop's globals are singletons)."""
    import nlo_virtual_pipeline as V
    import nlo_madloop as ML
    if CASES[name] is not None:
        V.VIRT_PROCESSES[name] = CASES[name]
    sa = V.build_virt_standalone(name)
    get = ML.load(V.find_p0(sa))
    rows = []
    for c in COS:
        s_ = np.sqrt(1 - c * c)
        for rs in SQRTS:
            E = rs / 2
            # MadGraph slot order e+ e- d d~
            p = np.array([[E, 0, 0, E], [E, 0, 0, -E], [E, E * s_, 0, E * c], [E, -E * s_, 0, -E * c]])
            r = ML.evaluate(get, p)
            rows.append({"cos": c, "sqrts": rs, "born": r["born"], "fin": r["fin"], "e1": r["e1"], "e2": r["e2"],
                         "rc": r["rc"]})
    print("RESULT " + json.dumps({"name": name, "rows": rows}), flush=True)


def main():
    out = {}
    for name in CASES:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(r.stdout[-3000:], r.stderr[-3000:])
            out[name] = {"error": "no result (build or evaluation failed), log above"}
            continue
        rows = json.loads(line[0][7:])["rows"]
        fits = {}
        for c in COS:
            R = [x for x in rows if x["cos"] == c and x["sqrts"] >= 200 and x["born"] and x["rc"] // 100 != 4]
            L = np.log(np.array([x["sqrts"] for x in R]) ** 2 / MW ** 2)
            d = np.array([x["fin"] / x["born"] for x in R])
            coef = np.polyfit(L, d, 2)                     # c, b, a
            fits[str(c)] = {"c_L2": coef[0], "b_L": coef[1], "a": coef[2],
                            "delta_200": d[0], "delta_max": d[-1], "spread": float(d.max() - d.min())}
            print(f"{name:14s} cos={c:+.1f}: delta {d[0]:+.4f} at 200 GeV -> {d[-1]:+.4f} at 3 TeV;"
                  f" fit c={coef[0]:+.2e} L^2, b={coef[1]:+.2e} L, a={coef[2]:+.3f}")
        out[name] = {"rows": rows, "fits": fits}
    json.dump(out, open(os.path.join(ROOT, "analysis", "transfer", "ew_sudakov_check.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        one(sys.argv[2])
    else:
        main()
