"""Generation rate of the synthetic versions against MadGraph on one node: stored events per second per core at 2->5 and
2->6, the median over each version's four processes (MadGraph: u ubar -> Z+4g, u dbar -> W+4g; Z+5g, W+5g), with the
range as an error bar. Data: the SUMMARY lines of one lxplus CPU job (tools/bench_synthetic.py --real ... --hm,
2000 stored events per process), analysis/transfer/rates/bench_FA-lxplus-11d4c736.txt.
    python analysis/transfer/synth_rates.py     -> figs/synth_rates
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

src = os.path.join(ROOT, "analysis", "transfer", "rates", "bench_FA-lxplus-11d4c736.txt")
R = {}
for l in open(src):
    m = re.match(r"SUMMARY (\S+) 2->(\d): \d+ processes, stored ev/s median (\d+), range (\d+)-(\d+)", l)
    if m:
        R[(m.group(1), int(m.group(2)))] = tuple(int(x) for x in m.groups()[2:])
VERS = ["v1", "v2r2", "v3r2", "v4r2", "v5r2", "v6", "v6t"]
for v in VERS + ["MadGraph"]:
    print(f"{v:9s}" + "".join(f"  2->{n}: {R[(v, n)][0]:6d} ({R[(v, n)][1]}-{R[(v, n)][2]})" for n in (5, 6)))
fig, ax = ps.figure()
for n, c in ((5, ps.C.blue), (6, ps.C.vermillion)):
    x = range(len(VERS))
    med = [R[(v, n)][0] for v in VERS]
    lo = [R[(v, n)][0] - R[(v, n)][1] for v in VERS]
    hi = [R[(v, n)][2] - R[(v, n)][0] for v in VERS]
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o-", color=c, capsize=2, label=rf"synthetic $2\to{n}$")
    ax.axhline(R[("MadGraph", n)][0], color=c, ls="--", label=rf"MadGraph $2\to{n}$")
ax.set_yscale("log")
ax.set_xticks(range(len(VERS)))
ax.set_xticklabels(["v1", "v2", "v3", "v4", "v5", "v6", "v6t"])
ax.set_xlabel("generator version")
ax.set_ylabel("stored events / s / core")
ps.legend(ax, "lower left")
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "synth_rates"))
