#!/usr/bin/env python
"""Diagram-conditioning A/B: compare the diagram-ON runs ("diag") against the
existing best encoding baseline ("combo") across seeds, at the same HPs.

Per the CLAUDE.md A/B protocol the headline metric is the per-process geomean of
the NON-regularised val loss on log-amplitudes (proc_val_losses_no_reg); combo's
reg_lambda is ~1e-10 so reg vs no_reg differ negligibly, but we report both plus
the combined val_loss. Reports seed mean +/- std and the diag/combo ratio
(<1 = diagrams help). Usage: python compare_models/aggregate_diag.py [dir]
"""
import json, glob, math, statistics, sys

D = sys.argv[1] if len(sys.argv) > 1 else "compare_models/multiseed25"
VARIANTS = ["combo", "diag"]   # baseline, treatment


def gmean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def ms(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


rows = {}
print(f"{'variant':10s} {'n':>2s} | {'val_loss':>22s} | {'proc-geomean(reg)':>22s} | "
      f"{'proc-geomean(no_reg)':>22s}")
print("-" * 86)
for v in VARIANTS:
    files = sorted(glob.glob(f"{D}/{v}_seed*.json"))
    vls, pgs, pgs_nr = [], [], []
    for f in files:
        try:
            d = json.load(open(f))
            vls.append(float(d["val_loss"]))
            pgs.append(gmean(list(d.get("proc_val_losses", {}).values())))
            pgs_nr.append(gmean(list(d.get("proc_val_losses_no_reg", {}).values())))
        except Exception:
            continue
    if not vls:
        print(f"{v:10s}  0 | (no runs yet)")
        continue
    vm, vs = ms(vls); pm, ps = ms(pgs); nm, ns = ms(pgs_nr)
    rows[v] = (vm, pm, nm)
    print(f"{v:10s} {len(vls):2d} | {vm:.3e} +/- {vs:.1e} | {pm:.5e} +/- {ps:.1e} | "
          f"{nm:.5e} +/- {ns:.1e}")

if "combo" in rows and "diag" in rows:
    bv, bp, bn = rows["combo"]
    dv, dp, dn = rows["diag"]
    print(f"\ndiag / combo (seed-mean) [<1 = diagrams help]:")
    print(f"  val_loss             {dv / bv:8.3f}")
    print(f"  proc-geomean(reg)    {dp / bp:8.3f}")
    print(f"  proc-geomean(no_reg) {dn / bn:8.3f}   <-- headline metric")
    verdict = "DIAGRAMS HELP" if dn < bn else "no improvement"
    print(f"\nverdict: {verdict} (no_reg geomean {dn:.3e} vs {bn:.3e})")
