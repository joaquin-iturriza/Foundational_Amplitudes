#!/usr/bin/env python
"""Offline mixture-fraction sweep: does a static sampling density have an optimum,
and where? f = antenna fraction in {0(=uniform), .25, .5, .75, 1(=antenna)}; every run
is the same base22, same 400k events, same held-out deep-IR test — only the sampling
density differs. We plot the overall held-out error vs f; the minimum is the optimal
static density for the chosen test weighting.

Two weightings reported (the 'optimal' depends on what you weight):
  log-flat per decade : mean over y_min decades of the per-decade MSE(Δln|M|^2). Treats
                        every decade of the singularity equally (our chosen objective).
  equal-per-event     : the raw overall MSE on the (antenna-dense) test set.
This is the OFFLINE allocation question (best static density for final accuracy), the
counterpart to the online adaptive scheme (variance-optimal proposal at fixed objective).
CPU only.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

EDGES = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
TAG_F = [("uniform", 0.0), ("mix025", 0.25), ("mixture", 0.5), ("mix075", 0.75), ("antenna", 1.0)]


def per_decade_mse(y, resid):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (y >= lo) & (y < hi)
        out.append(float(np.mean(resid[m] ** 2)) if m.sum() > 20 else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--npz_prefix", default="deep_eval_")
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "deep_sampling_fraction_sweep"))
    ap.add_argument("--summary_out", default=os.path.join(here, "fraction_sweep_summary.json"))
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    fs, logflat, per_event, decades = [], [], [], []
    for tag, f in TAG_F:
        p = os.path.join(args.eval_dir, f"{args.npz_prefix}{tag}.npz")
        if not os.path.exists(p):
            print(f"  missing {p}; skipping f={f}"); continue
        d = np.load(p)
        resid = d["pred_logamp"] - d["true_logamp"]
        dec = per_decade_mse(d["y_min"], resid)
        fs.append(f)
        decades.append(dec)
        logflat.append(float(np.nanmean(dec)))          # equal weight per decade
        per_event.append(float(np.mean(resid ** 2)))    # equal weight per event (antenna-dense)
    fs = np.array(fs); logflat = np.array(logflat); per_event = np.array(per_event)
    decades = np.array(decades)

    f_opt_lf = fs[int(np.argmin(logflat))]
    f_opt_pe = fs[int(np.argmin(per_event))]

    fig, (axL, axR) = ps.figure(ncols=2)

    axL.plot(fs, logflat, "o-", color=ps.C.vermillion, label="log-flat over decades")
    axL.plot(fs, per_event, "s--", color=ps.C.blue, label="equal per event")
    axL.set_yscale("log")
    axL.set_xlabel(r"antenna fraction $f$")
    axL.set_ylabel(r"MSE$(\Delta\ln|\mathcal{M}|^2)$")
    axL.legend(loc="upper center")
    ps.process_label(axL, args.process, loc="lower left")

    ramp = ps.sequence(len(fs))
    cen = np.sqrt(EDGES[:-1] * EDGES[1:])
    for i, f in enumerate(fs):
        axR.plot(cen, decades[i], "o-", color=ramp[i], label=f"$f={f:g}$")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"$y_{\min}$")
    axR.set_ylabel(r"MSE$(\Delta\ln|\mathcal{M}|^2)$ per decade")
    axR.legend(loc="lower left", ncol=2)

    ps.save(fig, args.out_base)

    summ = dict(f=fs.tolist(), logflat_mse=logflat.tolist(), per_event_mse=per_event.tolist(),
                f_opt_logflat=float(f_opt_lf), f_opt_per_event=float(f_opt_pe),
                per_decade={f"f{f:g}": decades[i].tolist() for i, f in enumerate(fs)})
    with open(args.summary_out, "w") as fjson:
        json.dump(summ, fjson, indent=1)
    print(f"wrote {args.summary_out}")
    print(f"\n  {'f':>6} {'log-flat MSE':>14} {'per-event MSE':>14}")
    for i, f in enumerate(fs):
        mark = "  <- opt(logflat)" if f == f_opt_lf else ""
        print(f"  {f:>6g} {logflat[i]:>14.4g} {per_event[i]:>14.4g}{mark}")


if __name__ == "__main__":
    main()
